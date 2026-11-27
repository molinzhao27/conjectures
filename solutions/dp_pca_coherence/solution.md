# Rank-Eight Counterexamples to Graph Coherence Stability under Random Edge Union

**Target paper:** Tommaso d'Orsi and Gleb Novikov, *Tight Differentially Private PCA via Matrix Coherence*, SODA 2026 / [arXiv:2510.26679](https://arxiv.org/abs/2510.26679) (October 2025), Section 1, Conjectures `conj:coherence-simple` (Conjecture 1.7) and `conj:coherence-normalized` (Conjecture 1.8).  
**Companion files:** [`solutions/dp_pca_coherence/explore.py`](explore.py), [`solutions/dp_pca_coherence/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjectures `conj:coherence-simple` and `conj:coherence-normalized`: BOTH FALSE** under the intended edge-probability regime $np \ge \operatorname{polylog}(n)$ (equivalently $1/2 \ge p \ge \operatorname{polylog}(n)/n$).
- **Simultaneous rank-8 counterexample (Theorem 1):** At constant edge probability $p = 1/5$, there is an explicit family of simple, unweighted, zero-diagonal graphs $K$ of rank $r = 8$ on $n = 8t + \lfloor t^{3/4} \rfloor$ vertices such that:
  - Both input coherences satisfy $\operatorname{coh}_8(A_K) = \operatorname{coh}_8(\bar{A}_K) = \frac{n}{8t} = 1 + o(1)$.
  - For the random edge union $\hat{K} = K \cup G(n, 1/5)$, **both** the raw adjacency matrix $A_{\hat{K}}$ and the normalized adjacency matrix $\bar{A}_{\hat{K}}$ simultaneously satisfy
    $$
    \operatorname{coh}_8(A_{\hat{K}}) = \Omega(n^{1/4}) \qquad \text{and} \qquad \operatorname{coh}_8(\bar{A}_{\hat{K}}) = \Omega(n^{1/4})
    $$
    with probability $1 - o(1)$, exceeding $O(\operatorname{coh}_8(\text{input})) + \operatorname{polylog}(n)$ for every fixed polylogarithm.
- **Note on a literal LaTeX typo in arXiv:2510.26679:** Both printed conjectures literally state $1/2 \ge p \ge \operatorname{polylog}(n)$, which is vacuously empty for large $n$ under any growing positive power of $\log n$. Because our construction operates at constant edge density $p = 1/5$, it exceeds every polylogarithmic expected-degree threshold $np \ge \operatorname{polylog}(n)$ and refutes the intended statements cleanly.

---

## 1. Setup, Coherence Definition, and Input Graph

Let $M \in \mathbb{R}^{n \times n}$ be symmetric with leading rank-$r$ singular projector $P_r$ (when the $r$th and $(r+1)$th singular values satisfy $\sigma_r(M) > \sigma_{r+1}(M)$, the left and right singular projectors coincide and equal the orthogonal projector onto the top-$r$ eigenspace ordered by absolute eigenvalue). By `def:coherence` of arXiv:2510.26679,
$$
\operatorname{coh}_r(M) := \frac{n}{r} \max_{i,j} |(P_r)_{ij}| = \frac{n}{r} \max_j (P_r)_{jj},
$$
where the second equality follows from Cauchy–Schwarz $|(P_r)_{ij}| \le \sqrt{(P_r)_{ii}(P_r)_{jj}}$ for any orthogonal projector $P_r^2 = P_r = P_r^\top$. For a simple graph $K$, $A_K$ is its zero-diagonal adjacency matrix and $\bar{A}_K := D(A_K)^{-1/2} A_K D(A_K)^{-1/2}$, with $D(A_K)^{-1/2}_{jj} := 0$ whenever $D(A_K)_{jj} = 0$.

**Construction of the input graph $K$.** For an integer $t \to \infty$, set $s := \lfloor t^{3/4} \rfloor$ and $n := 8t + s$. Start with a $4$-clique $K_4$ and attach one private leaf to each clique vertex. Blow up each of these $8$ vertices into an independent set of size $t$, replacing each edge by a complete bipartite graph $K_{t,t}$, and append a set $S$ of $s$ isolated vertices. Call the four clique classes the *centers* and the four leaf classes the *leaves*.

**Lemma 1 (Input rank and coherence).** *For every $t \ge 1$, $\operatorname{rank}(A_K) = \operatorname{rank}(\bar{A}_K) = 8$ and*
$$
\operatorname{coh}_8(A_K) = \operatorname{coh}_8(\bar{A}_K) = \frac{n}{8t} = 1 + o(1).
$$

*Proof.* In the orthonormal basis of the $8$ normalized class indicators, $A_K$ acts as $t B$ where
$$
B = \begin{pmatrix} J_4 - I_4 & I_4 \\ I_4 & 0 \end{pmatrix},
$$
and vanishes on the orthogonal complement. Since $B(x, y) = 0$ gives $x = 0$ from the bottom block and then $y = 0$ from the top ($\det B = 1$), $\operatorname{rank}(A_K) = 8$. Degrees are $4t$ on centers, $t$ on leaves, and $0$ on $S$, so $\bar{A}_K$ is an invertible diagonal congruence of $A_K$ on the exact same $8$-dimensional range. Hence both leading rank-$8$ projectors equal the orthogonal projector onto the $8$-class indicator span, which has diagonal $1/t$ on the $8t$ non-isolates and $0$ on $S$, giving $\operatorname{coh}_8 = n/(8t) = 1 + o(1)$. $\square$

---

## 2. Deterministic Population Quotient at $p = 1/5$

Write $\hat{A} := A_{\hat{K}}$ for $\hat{K} = K \cup G(n, p)$ at $p = 1/5$, and set
$$
H := (1 - p) A_K + p J_n, \qquad D_H := \operatorname{diag}(H \mathbf{1}),
$$
so $\mathbb{E}[\hat{A}] = H - p I_n$ exactly. Define the two deterministic comparison matrices
$$
T_{\mathrm{raw}} := \frac{1}{t} H, \qquad T_{\mathrm{norm}} := D_H^{-1/2} H D_H^{-1/2},
$$
to be compared against $\hat{A}/t$ and $\bar{A}_{\hat{K}}$ respectively (scaling $\hat{A}$ by $1/t$ preserves its singular subspaces). Put $\epsilon := \sqrt{s/t} = \Theta(t^{-1/8})$.

**Lemma 2 (Exact 9-dimensional reduction and 8th–9th singular gap).** *At $p = 1/5$, both $T = T_{\mathrm{raw}}$ and $T = T_{\mathrm{norm}}$ vanish outside the $9$-dimensional span of the class indicators (including $\mathbf{1}_S/\sqrt{s}$), and decompose orthogonally into:*
1. *A $6$-dimensional difference subspace carrying three identical $2 \times 2$ blocks $(1-p)\begin{pmatrix} -c^2 & cl \\ cl & 0 \end{pmatrix}$, yielding $6$ singular values uniformly bounded away from $0$.*
2. *A $3$-dimensional block on the normalized indicators of all centers, all leaves, and $S$:*
   $$
   \operatorname{diag}(c, l, i)
   \begin{pmatrix}
   16/5 & 8/5 & 2\epsilon/5 \\
   8/5 & 4/5 & 2\epsilon/5 \\
   2\epsilon/5 & 2\epsilon/5 & \epsilon^2/5
   \end{pmatrix}
   \operatorname{diag}(c, l, i),
   $$
   *where $(c, l, i) = (1, 1, 1)$ for $T_{\mathrm{raw}}$ and*
   $$
   (c, l, i) = \Bigl((24/5 + \epsilon^2/5)^{-1/2},\, (12/5 + \epsilon^2/5)^{-1/2},\, (8/5 + \epsilon^2/5)^{-1/2}\Bigr)
   $$
   *for $T_{\mathrm{norm}}$.*

*In both cases, $T$ has $7$ macroscopic singular values $\Theta(1)$ and two small outlier eigenvalues $\lambda_+ > 0 > \lambda_-$ satisfying*
$$
\lambda_\pm = \pm \epsilon b + o(\epsilon), \qquad
\sigma_8(T) - \sigma_9(T) = \lambda_+ + \lambda_- = \epsilon^2\Bigl(d - \frac{h^2}{a}\Bigr) + o(\epsilon^2),
$$
*with limiting gap coefficient $d - h^2/a \to 16/125$ (raw) and $5/72$ (normalized). Moreover, the unit eigenvector for $\lambda_+$ (the $8$th singular vector of $T$) has squared mass $1/2 + o(1)$ on $S$.*

*Proof.* On the $3$-dimensional subspace spanned by the normalized indicators of all $4t$ centers, all $4t$ leaves, and $S$, $A_K/t$ acts as $\begin{pmatrix} 3 & 1 \\ 1 & 0 \end{pmatrix} \oplus (0)$ and $J_n/t$ acts as $(2, 2, \epsilon)^\top(2, 2, \epsilon)$. Setting $p = 1/5$ yields the displayed $3 \times 3$ matrix, whose upper-left $2 \times 2$ block collapses to rank $1$: $(4/5) v v^\top$ with $v := (2c, l)^\top$ (since $\det\begin{pmatrix} 3+p & 1+3p \\ 1+3p & 4p \end{pmatrix} = -(5p - 1)(p - 1) = 0$ at $p = 1/5$). The row sums of $H/t$ are $24/5 + \epsilon^2/5$ (centers), $12/5 + \epsilon^2/5$ (leaves), and $8/5 + \epsilon^2/5$ (isolates), giving $(c, l, i)$ for $T_{\mathrm{norm}}$. On the $6$-dimensional difference subspace, $J_n$ vanishes and $J_4 - I_4$ acts as $-I$ on each of three orthonormal vectors orthogonal to $\mathbf{1}_4$, producing three copies of $(1-p)\begin{pmatrix} -c^2 & cl \\ cl & 0 \end{pmatrix}$ with nonzero limiting determinant $-(1-p)^2 c^2 l^2$.

In the orthonormal basis $\{v/\|v\|,\, u := (-l, 2c)^\top/\|v\|,\, \mathbf{1}_S/\sqrt{s}\}$, writing $w := (c, l)^\top$, the $3 \times 3$ block becomes
$$
L = \begin{pmatrix}
a & 0 & \epsilon h \\
0 & 0 & \epsilon b \\
\epsilon h & \epsilon b & \epsilon^2 d
\end{pmatrix},
\quad
a = \frac{4}{5}\|v\|^2, \quad h = \frac{2i}{5}\frac{v^\top w}{\|v\|}, \quad b = \frac{2i}{5} u^\top w, \quad d = \frac{1}{5} i^2,
$$
where $a, b$ have strictly positive limits. By Weyl's inequality against $\operatorname{diag}(a, 0, 0)$, the large eigenvalue is $\lambda_0 = a + O(\epsilon)$. Eliminating the second and third coordinates of its eigenvector gives
$$
\lambda_0 - a = \frac{\epsilon^2 h^2}{\lambda_0 - \epsilon^2 d - \epsilon^2 b^2/\lambda_0} = \frac{\epsilon^2 h^2}{a} + o(\epsilon^2).
$$
From $\det L = -a \epsilon^2 b^2$, the remaining two eigenvalues satisfy $\lambda_+ \lambda_- = -a\epsilon^2 b^2 / \lambda_0 < 0$ (so $\lambda_+ > 0 > \lambda_-$), and from $\operatorname{tr} L = a + \epsilon^2 d$,
$$
\lambda_+ + \lambda_- = \epsilon^2\Bigl(d - \frac{h^2}{a}\Bigr) + o(\epsilon^2), \qquad
d - \frac{h^2}{a} = \frac{i^2}{5}\left[1 - \left(\frac{2c^2 + l^2}{4c^2 + l^2}\right)^2\right].
$$
Evaluating at $(c, l, i) = (1, 1, 1)$ gives $\frac{1}{5}(1 - 9/25) = 16/125$; evaluating at the normalized limits $c^2 = 5/24, l^2 = 5/12, i^2 = 5/8$ gives $\frac{1}{8}(1 - (4/6)^2) = 5/72$. From their product and sum, $\lambda_\pm = \pm \epsilon b + o(\epsilon)$, so $\sigma_8(T) = \lambda_+ > |\lambda_-| = \sigma_9(T)$ with gap $\Theta(\epsilon^2)$. Finally, a unit eigenvector $(x, y, z)^\top$ of $L$ for $\lambda_+$ satisfies $x = -\epsilon h z / (a - \lambda_+) = O(\epsilon)$ and $y = \epsilon b z / \lambda_+ = (1 + o(1)) z$, forcing $z^2 = 1/2 + o(1)$. Because $z$ is the coefficient along $\mathbf{1}_S/\sqrt{s}$, the $8$th singular vector carries squared mass $1/2 + o(1)$ on $S$. $\square$

---

## 3. Concentration and Perturbation to the Random Union

**Theorem 1 (Simultaneous refutation of `conj:coherence-simple` and `conj:coherence-normalized`).** *For $p = 1/5$ and $K$ constructed above ($n = 8t + \lfloor t^{3/4} \rfloor$), with probability $1 - o(1)$ as $t \to \infty$,*
$$
\operatorname{coh}_8(A_{\hat{K}}) \ge \frac{n}{8s}\Bigl(\frac{1}{2} - o(1)\Bigr) = \Omega(n^{1/4})
\qquad \text{and} \qquad
\operatorname{coh}_8(\bar{A}_{\hat{K}}) \ge \frac{n}{8s}\Bigl(\frac{1}{2} - o(1)\Bigr) = \Omega(n^{1/4}).
$$

*Proof.* **Step 1 (Spectral and degree concentration).** Set $W := \hat{A} - \mathbb{E}[\hat{A}]$. Its upper-triangular off-diagonal entries are independent, centered, and lie in intervals of length at most $1$. For any fixed unit vector $x \in S^{n-1}$, $x^\top W x = 2\sum_{i < j} x_i x_j W_{ij}$ is a sum of independent bounded centered variables whose squared interval lengths sum to $4\sum_{i < j} x_i^2 x_j^2 \le 2$, so Hoeffding's inequality gives $\Pr(|x^\top W x| > u) \le 2 e^{-u^2}$. A maximal $1/4$-separated subset $\mathcal{N} \subset S^{n-1}$ is a $1/4$-net of size at most $9^n$ (by disjoint ball volumes) and satisfies $\|W\| \le 2\max_{y \in \mathcal{N}} |y^\top W y|$. Choosing $u = 2\sqrt{n}$ gives $\Pr(\|W\| > 4\sqrt{n}) \le 2 \cdot 9^n e^{-4n} \to 0$, hence $\|\hat{A} - H\| \le \|W\| + p = O(\sqrt{n})$ w.h.p.

For degrees, scalar Hoeffding at each vertex and a union bound over $n$ vertices give $\max_j |\hat{d}_j - (D_H)_{jj}| = O(\sqrt{n \log n})$ w.h.p. (including the shift $p$). Because $(D_H)_{jj} \ge pn = \Theta(t)$, all random degrees satisfy $\hat{d}_j = \Theta(t)$. With $\hat{D} := \operatorname{diag}(\hat{d})$, the mean value theorem on $x \mapsto x^{-1/2}$ yields $\|\hat{D}^{-1/2} - D_H^{-1/2}\| = O(\sqrt{n\log n}/t^{3/2})$. Since $\|\hat{D}^{-1/2}\|, \|D_H^{-1/2}\| = O(t^{-1/2})$ and $\|H\| \le n$, telescoping $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} - D_H^{-1/2} H D_H^{-1/2}$ costs $O(\sqrt{n}/t + \sqrt{n\log n}/t) = O(\sqrt{\log t / t})$. Thus on a single event of probability $1 - o(1)$, both $\hat{A}/t - T_{\mathrm{raw}}$ and $\bar{A}_{\hat{K}} - T_{\mathrm{norm}}$ have operator norm at most
$$
\eta = O\!\left(\sqrt{\frac{\log t}{t}}\right) = o(t^{-1/4}) = o(\epsilon^2).
$$

**Step 2 (Eigenvector perturbation and coherence lower bound).** Write $M = T + E$ with $\|E\| \le \eta$. By Weyl's inequality, every singular value and every ordered signed eigenvalue moves by at most $\eta$. Because $\sigma_8(T) - \sigma_9(T) = \Theta(\epsilon^2) \gg \eta$, the strict $8$th–$9$th singular gap persists, and the $8$th singular value of $M$ corresponds to the perturbed positive eigenvalue $\hat{\lambda}_+$ near $\lambda_+$. Moreover, $\lambda_+$ is separated from all other signed eigenvalues of $T$ (including $0$ and $\lambda_-$) by $\Omega(\epsilon)$. Let $\hat{v}$ be a unit eigenvector of $M$ for $\hat{\lambda}_+$; then $\|(T - \lambda_+ I)\hat{v}\| \le \|E\hat{v}\| + |\hat{\lambda}_+ - \lambda_+| \le 2\eta$. Expanding $\hat{v}$ in an eigenbasis of $T$ shows that its component orthogonal to the population eigenvector $v_+$ has norm $O(\eta / \epsilon) = o(1)$. Therefore,
$$
\sum_{j \in S} \hat{v}_j^2 = \frac{1}{2} + o(1).
$$
By pigeonhole, some vertex $j^* \in S$ has $\hat{v}_{j^*}^2 \ge (1/2 - o(1))/s$. Because $\hat{v}$ lies in the leading rank-$8$ singular subspace of $M$, $(P_8)_{j^* j^*} \ge \hat{v}_{j^*}^2$, whence
$$
\operatorname{coh}_8(M) = \frac{n}{8}\max_j (P_8)_{jj} \ge \frac{n}{8s}\Bigl(\frac{1}{2} - o(1)\Bigr) = \Omega(n^{1/4})
$$
simultaneously for $A_{\hat{K}}$ and $\bar{A}_{\hat{K}}$ with probability $1 - o(1)$. $\square$

---

## 4. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/dp_pca_coherence/explore.py
```

The script [`explore.py`](explore.py) verifies five control graphs across $p \in \{0.08, 0.2, 0.5\}$, verifies the exact $9$-dimensional quotient reconstruction against the full $n \times n$ matrices at $t = 32$, tracks the population quotient convergence up to $t = 2^{40}$, and samples full random unions at $t \in \{32, 128, 384\}$.

### Population quotient convergence ($p = 1/5$)

| $t$ | Raw isolate mass (limit $0.5$) | Raw scaled gap (limit $16/125 = 0.128$) | Norm. isolate mass (limit $0.5$) | Norm. scaled gap (limit $5/72 \approx 0.069444$) |
|---:|---:|---:|---:|---:|
| $16$ | $0.616161$ | $0.126767$ | $0.619630$ | $0.063770$ |
| $2^{12}$ | $0.561376$ | $0.127693$ | $0.568348$ | $0.067952$ |
| $2^{24}$ | $0.522187$ | $0.127962$ | $0.525525$ | $0.069255$ |
| $2^{40}$ | $0.505581$ | $0.127998$ | $0.506482$ | $0.069433$ |

### Sampled random graph unions ($p = 1/5$)

| $t$ | $n$ | Input $\operatorname{coh}_8$ | Raw union $\operatorname{coh}_8$ (isolate mass) | Norm. union $\operatorname{coh}_8$ (isolate mass) |
|---:|---:|---:|---:|---:|
| $32$ | $269$ | $1.0508$ | $2.7207$ ($0.1026$) | $2.9201$ ($0.3450$) |
| $128$ | $1062$ | $1.0371$ | $2.7863$ ($0.3122$) | $3.5876$ ($0.4619$) |
| $384$ | $3158$ | $1.0280$ | $3.7213$ ($0.4398$) | $4.2239$ ($0.5123$) |

All assertions passed.

---

## References

- T. d'Orsi and G. Novikov, *Tight Differentially Private PCA via Matrix Coherence*, SODA 2026 / [arXiv:2510.26679](https://arxiv.org/abs/2510.26679), October 2025.
