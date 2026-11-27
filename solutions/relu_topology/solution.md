# Complete Proof of Beshkov's Tight Topological Expressivity Conjecture (arXiv:2610.13931)

- **Conjecture Source**: Kosio Beshkov, *"Constructive Bounds for the Topological Expressivity of Neural Networks"*, [arXiv:2610.13931](https://arxiv.org/abs/2610.13931) (Oct 2026), Section *"Conjectured tight and lower bounds"*.
- **Domain**: Machine Learning Theory / Algebraic Topology of ReLU Neural Networks.
- **Verdict**: **TRUE** for all dimensions $d \ge 1$ under the homeomorphism / homotopy-equivalence reading of the conjecture (and for Betti numbers on any compact polyhedral/semialgebraic domain with continuous PWL/semialgebraic $h$, or in Čech cohomology $\check{H}^d(\Phi(X); G) = 0$ for every compact topological space $X$; singular $\beta_d$ for arbitrary non-triangulable wild compacta is not covered).
- **Attribution / Priority Note**: No completed proof has appeared in the literature (as of Nov 26, 2026, arXiv:2610.13931 remains v1). An independent public preregistration ([the-omega-institute/trureturing#25155](https://github.com/the-omega-institute/trureturing/issues/25155), opened Nov 1, 2026, open with no completed proof or Lean file) outlines a similar Part A construction, whereas our Part B homotopy-theoretic / Straight-Ray Escape + Alexander Duality proof (ruling out not just homeomorphisms but all degree-nonzero maps $S^d \to \Phi(X) \to S^d$ and all homotopy equivalences for arbitrary $X$ and arbitrary $h$) is completely independent and stronger.

---

## 1. Exact Statement & Typographical Note

In Section *"Conjectured tight and lower bounds"* of [arXiv:2610.13931](https://arxiv.org/abs/2610.13931), Beshkov conjectures that for $x = (x_1, \dots, x_d) \in \mathbb{R}^d$, the 1-hidden-layer ReLU network $P = (p_1, \dots, p_{d+1}) : \mathbb{R}^d \to \mathbb{R}^{d+1}$ of architecture $(d, d+1, d+1)$ defined by
$$
\begin{aligned}
h_i(x) &= \max(0, x_i) \qquad (i = 1, \dots, d), \\
h_{d+1}(x) &= \max\Bigl(0,\, 1 - \sum_{i=1}^d x_i\Bigr), \\
p_i(x) &= \max\Bigl(0,\, h_i(x) + 2\Bigl(1 - \sum_{j=1}^{d+1} h_j(x)\Bigr)\Bigr) \qquad (i = 1, \dots, d+1),
\end{aligned}
$$
*(correcting two minor display typos in the preprint: $\sum_{i=1}^{d+1} x_i \to \sum_{i=1}^d x_i$ in $h_{d+1}$ and a missing closing parenthesis in $p_i$)* expresses a space homeomorphic/homotopy equivalent to $S^d$ for all $d \ge 1$, and that **no** ReLU architecture of shape $(d, d, d+1)$ can express such a sphere.

---

## 2. Theorem 1 (Part A — Exact Homeomorphism for $(d, d+1, d+1)$)

**Theorem 1.** *For every integer $d \ge 1$, let $\Delta_0^{d+1} := \operatorname{conv}\{0, e_1, \dots, e_{d+1}\} = \{z \in \mathbb{R}_{\ge 0}^{d+1} : \sum_{j=1}^{d+1} z_j \le 1\}$ be the standard $(d+1)$-simplex at the origin. Define $S(x) := \sum_{j=1}^{d+1} h_j(x)$ and $\Omega := \{x \in \mathbb{R}^d : S(x) < 2\}$. Then:*
1. *$P(\mathbb{R}^d) = \partial \Delta_0^{d+1} \cong S^d$ (exact homeomorphism).*
2. *$\overline{\Omega} = \{x \in \mathbb{R}^d : S(x) \le 2\}$ is a compact convex polytope satisfying $\{x \in \mathbb{R}^d : P(x) \neq 0\} \subseteq \Omega \subset (-1, 2)^d$, with $P \equiv 0$ outside $\Omega$. Consequently, for every domain $X$ with $\overline{\Omega} \subseteq X \subseteq \mathbb{R}^d$ (in particular $X = \overline{\Omega}$, $X = [-1,2]^d$, or $X = \mathbb{R}^d$),*
   $$P(X) = P(\overline{\Omega}) = P(\mathbb{R}^d) = \partial \Delta_0^{d+1} \cong S^d.$$
3. *Therefore, the wedge-sum bound tightens to $\bigl(d, \sum_{k=1}^d \beta_k(k+1), \sum_{k=1}^d \beta_k(k+1)\bigr)$ and the product bound tightens to $\bigl(\sum_{n=1}^N d_n, \sum_{n=1}^N (d_n+1), \sum_{n=1}^N (d_n+1)\bigr)$, both realizing exact homeomorphisms to the corresponding wedge sums and products of spheres.*

### Proof

Factor $P = g \circ h$, where $h : \mathbb{R}^d \to \mathbb{R}_{\ge 0}^{d+1}$ is the hidden layer and $g_i(y) = \max\bigl(0,\, y_i + 2(1 - \sum_{j=1}^{d+1} y_j)\bigr)$.

**Step 1 (Hidden layer image).** Define
$$
H := \Bigl\{y \in \mathbb{R}_{\ge 0}^{d+1} : \sum_{j=1}^{d+1} y_j \ge 1,\ \text{and}\ \Bigl(\sum_{j=1}^{d+1} y_j > 1 \implies \min_{1 \le j \le d+1} y_j = 0\Bigr)\Bigr\}.
$$
We show $h(\mathbb{R}^d) = H$, and every $y \in H$ with $\sum_{j=1}^{d+1} y_j \le 2$ has a preimage $x \in \overline{\Omega}$.

- *($h(\mathbb{R}^d) \subseteq H$)*: For any $x \in \mathbb{R}^d$, $y = h(x) \in \mathbb{R}_{\ge 0}^{d+1}$. Since $\max(0, u) \ge u$,
  $$\sum_{j=1}^{d+1} y_j = \sum_{i=1}^d \max(0, x_i) + \max\Bigl(0, 1 - \sum_{i=1}^d x_i\Bigr) \ge \sum_{i=1}^d x_i + \Bigl(1 - \sum_{i=1}^d x_i\Bigr) = 1.$$
  If $\min_j y_j > 0$, then every ReLU argument is strictly positive, so every $\max(0, \cdot)$ equals its second argument and $\sum_{j=1}^{d+1} y_j = 1$. Taking the contrapositive yields $\sum y_j > 1 \implies \min_j y_j = 0$.

- *($H \subseteq h(\mathbb{R}^d)$)*: Given $y \in H$ with $S := \sum_{j=1}^{d+1} y_j \ge 1$:
  - *Case 1 ($S = 1$ or $y_{d+1} = 0$)*: Set $x := (y_1, \dots, y_d) \in \mathbb{R}_{\ge 0}^d$. Then $h_i(x) = y_i$ for $1 \le i \le d$, and $1 - \sum_{i=1}^d x_i = 1 - (S - y_{d+1}) = y_{d+1} - (S - 1)$. If $S = 1$, this equals $y_{d+1} \ge 0$; if $y_{d+1} = 0$ and $S > 1$, this equals $1 - S < 0$. In either case $h_{d+1}(x) = y_{d+1}$, so $h(x) = y$.
  - *Case 2 ($S > 1$ and $y_{d+1} > 0$)*: Since $y \in H$, there exists $k \in \{1, \dots, d\}$ with $y_k = 0$. Set $x_i := y_i \ge 0$ for $i \in \{1, \dots, d\} \setminus \{k\}$ and $x_k := 1 - S < 0$. Then $h_i(x) = y_i$ for $i \neq k$, $h_k(x) = 0 = y_k$, and
    $$1 - \sum_{i=1}^d x_i = 1 - (S - y_{d+1}) - (1 - S) = y_{d+1} > 0,$$
    so $h_{d+1}(x) = y_{d+1}$, giving $h(x) = y$.

  Whenever $S \le 2$, this preimage satisfies $S(x) = S \le 2$, so $x \in \overline{\Omega}$.

**Step 2 (Output layer image).** We show $g(H) = g(H \cap \{S \le 2\}) = \partial \Delta_0^{d+1}$.

- *($g(H) \subseteq \partial \Delta_0^{d+1}$)*: Let $y \in H$, $S = \sum_{j=1}^{d+1} y_j \ge 1$, and $t := 2(S - 1) \ge 0$, so $S = 1 + t/2$ and $z_i := g_i(y) = \max(0, y_i - t) \ge 0$.
  - If $S = 1$, then $t = 0$ and $z = y \in \Delta_{\mathrm{std}}^d = \{u \in \mathbb{R}_{\ge 0}^{d+1} : \sum u_j = 1\} \subset \partial \Delta_0^{d+1}$.
  - If $S > 1$, then $t > 0$ and $\min_j y_j = 0$, hence $\min_j z_j = \max(0, -t) = 0$. Let $A := \{j : z_j > 0\} = \{j : y_j > t\}$ and $m := |A|$. If $m = 0$, then $z = 0 \in \partial \Delta_0^{d+1}$. If $m \ge 1$, then
    $$\sum_{j=1}^{d+1} z_j = \sum_{j \in A} (y_j - t) \le S - m t = 1 + \frac{t}{2} - m t = 1 - \Bigl(m - \frac{1}{2}\Bigr) t < 1,$$
    since $m - \frac{1}{2} \ge \frac{1}{2} > 0$ and $t > 0$. Since $z \in \mathbb{R}_{\ge 0}^{d+1}$, $\min_j z_j = 0$, and $\sum z_j < 1$, we have $z \in \partial \Delta_0^{d+1}$.

- *($\partial \Delta_0^{d+1} \subseteq g(H \cap \{S \le 2\})$)*: Let $z \in \partial \Delta_0^{d+1}$ and $\Sigma := \sum_{j=1}^{d+1} z_j \in [0, 1]$.
  - If $\Sigma = 1$: take $y := z \in H$. Then $S = 1 \le 2$, $t = 0$, and $g(y) = z$.
  - If $\Sigma = 0$ ($z = 0$): take $y := 2 e_1$. Since $d + 1 \ge 2$, $y_{d+1} = 0$, $S = 2$, $y \in H$, $t = 2$, and $g(y) = 0$.
  - If $0 < \Sigma < 1$: since $z \in \partial \Delta_0^{d+1}$ and $\Sigma < 1$, we must have $\min_j z_j = 0$. Let $A := \{j : z_j > 0\}$ and $m := |A| \in \{1, \dots, d\}$. Set
    $$t := \frac{2(1 - \Sigma)}{2m - 1} > 0, \qquad y_j := \begin{cases} z_j + t & j \in A, \\ 0 & j \notin A. \end{cases}$$
    Because $m \le d$, $\{1, \dots, d+1\} \setminus A \neq \emptyset$, so $\min_j y_j = 0$. Moreover,
    $$S = \sum_{j=1}^{d+1} y_j = \Sigma + m t = \Sigma + \frac{2m(1 - \Sigma)}{2m - 1} = 1 + \frac{1 - \Sigma}{2m - 1} = 1 + \frac{t}{2} \in (1, 2).$$
    Hence $y \in H \cap \{S \le 2\}$, $2(S - 1) = t$, and $g_j(y) = \max(0, y_j - t) = z_j$ for all $j = 1, \dots, d+1$.

**Step 3 (Compact convex support).** $S(x) = \sum_{i=1}^d x_i^+ + (1 - \sum_{i=1}^d x_i)^+$ is convex and continuous with $S(0) = 1 < 2$, so $\Omega = \{S < 2\}$ is nonempty and convex and its closure is $\overline{\Omega} = \{S \le 2\}$, which is closed and convex. For every $k \in \{1, \dots, d\}$:
- $S(x) \ge x_k^+ \ge x_k$, so $S(x) < 2 \implies x_k < 2$;
- writing $x_i = x_i^+ - x_i^-$, $S(x) \ge \sum_{i=1}^d x_i^+ + (1 - \sum_{i=1}^d x_i^+ + \sum_{i=1}^d x_i^-) = 1 + \sum_{i=1}^d x_i^- \ge 1 + x_k^-$, so $S(x) < 2 \implies x_k > -1$.

Hence $\Omega \subset (-1, 2)^d$ and $\overline{\Omega} \subset [-1, 2]^d$ is a compact convex polytope. Whenever $S(x) \ge 2$, we have $t = 2(S(x) - 1) = S(x) + (S(x) - 2) \ge S(x) \ge h_i(x)$ for every $i$, so $P(x) = 0$. Combining Steps 1–3 gives $P(X) = P(\overline{\Omega}) = P(\mathbb{R}^d) = \partial \Delta_0^{d+1} \cong S^d$ for every $\overline{\Omega} \subseteq X \subseteq \mathbb{R}^d$.

**Step 4 (Wedge sums and products).** For wedge sums, the shifted networks $x \mapsto P^{(k)}(x_1 - 3r, x_2, \dots, x_k)$ ($r = 0, 1, \dots, N-1$) have pairwise disjoint open supports contained in $(3r - 1, 3r + 2) \times (-1, 2)^{d-1}$ and vanish identically outside them. Placing their outputs in disjoint coordinate blocks glues their images $\partial \Delta_0^{k+1} \cong S^k$ exclusively at the origin $0$, yielding an exact homeomorphism to $\bigvee_{k=1}^d (\bigvee^{\beta_k} S^k)$ with architecture $(d, \sum_{k=1}^d \beta_k(k+1), \sum_{k=1}^d \beta_k(k+1))$. Stacking $N$ independent copies on disjoint input blocks yields $\prod_{n=1}^N \partial \Delta_0^{d_n+1} \cong \prod_{n=1}^N S^{d_n}$ with architecture $(\sum d_n, \sum(d_n+1), \sum(d_n+1))$. $\blacksquare$

---

## 3. Theorem 2 (Part B — Impossibility for $(d, d, d+1)$)

**Theorem 2.** *Let $d \ge 1$, let $X$ be any topological space, and let $\Phi : X \to \mathbb{R}^{d+1}$ be any map of the form*
$$\Phi(x) = R(W_2 h(x) + b_2) \qquad \text{or} \qquad \Phi(x) = W_2 h(x) + b_2,$$
*where $R(u) = \max(0, u)$ is componentwise ReLU, $W_2 \in \mathbb{R}^{(d+1)\times d}$, $b_2 \in \mathbb{R}^{d+1}$, and $h : X \to \mathbb{R}^d$ is arbitrary (in particular, any $(d, d, d+1)$ ReLU network, or any deeper network whose final hidden layer has width $\le d$). Then:*
1. *$\Phi(X)$ is neither homeomorphic nor homotopy equivalent to $S^d$. More strongly, there do not exist continuous maps $f : S^d \to \Phi(X)$ and $g : \Phi(X) \to S^d$ with $\deg(g \circ f) \neq 0$.*
2. *For every compact topological space $X$ and continuous $h$, the Čech cohomology vanishes: $\check{H}^d(\Phi(X); \mathbb{Z}) = 0$. Moreover, whenever $X \subseteq \mathbb{R}^d$ is a compact polyhedron or compact semialgebraic set, or $X = \mathbb{R}^d$ (or any closed polyhedron in $\mathbb{R}^d$), and $h$ is a ReLU network (or any continuous PWL/semialgebraic map), singular homology also vanishes:*
   $$H_d(\Phi(X); G) = 0 \qquad \text{for every abelian coefficient group } G.$$
   *(For arbitrary wild non-triangulable compact domains $X$, singular $H_d(\Phi(X))$ is not controlled by Alexander duality alone.)*

### Proof

Since $W_2 \in \mathbb{R}^{(d+1)\times d}$ has rank at most $d$, there exists $n \in \mathbb{R}^{d+1} \setminus \{0\}$ with $W_2^\top n = 0$. Setting $\alpha := n \cdot b_2$, the affine image $W_2 \mathbb{R}^d + b_2$ lies inside the affine hyperplane $H := \{u \in \mathbb{R}^{d+1} : n \cdot u = \alpha\}$. Thus $\Phi(X) \subseteq R(H)$ (with output ReLU) or $\Phi(X) \subseteq H$ (linear output).

**Lemma (Straight-Ray Escape Lemma).** *For every $p \in \mathbb{R}^{d+1} \setminus R(H)$, there exists a straight ray $\gamma(t) = p + t v$ ($t \ge 0$, $v \in \mathbb{R}^{d+1} \setminus \{0\}$) contained in $\mathbb{R}^{d+1} \setminus R(H)$. (Similarly, for every $p \in \mathbb{R}^{d+1} \setminus H$, the ray $p + t\,\operatorname{sign}(n \cdot p - \alpha)\,n$ stays in $\mathbb{R}^{d+1} \setminus H$.)*

*Proof of Lemma.* Fix $p \in \mathbb{R}^{d+1} \setminus R(H)$.
- **Case 1 ($\exists k$ with $p_k \le 0$):** Take $\gamma(t) := p - t e_k$ ($t \ge 0$). At $t = 0$, $\gamma(0) = p \notin R(H)$. For every $t > 0$, $\gamma(t)_k = p_k - t < 0$, so $\gamma(t) \notin \mathbb{R}_{\ge 0}^{d+1} \supseteq R(H)$.
- **Case 2 ($p \in \mathbb{R}_{>0}^{d+1}$):** Since $R(p) = p \notin R(H)$, we have $p \notin H$, so $n \cdot p \neq \alpha$. Replacing $(n, \alpha)$ with $(-n, -\alpha)$ if necessary, assume $n \cdot p < \alpha$.
  - *Subcase 2a ($\exists k$ with $n_k > 0$):* Take $\gamma(t) := p - t e_k$ ($t \ge 0$).
    - For $0 \le t < p_k$: $\gamma(t) \in \mathbb{R}_{>0}^{d+1}$, so its unique $R$-preimage is $u = \gamma(t)$, which satisfies $n \cdot u = n \cdot p - t n_k \le n \cdot p < \alpha$, hence $\gamma(t) \notin R(H)$.
    - For $t = p_k$: $\gamma(p_k)_k = 0$ and $\gamma(p_k)_j = p_j > 0$ for $j \neq k$. Any $u \in \mathbb{R}^{d+1}$ with $R(u) = \gamma(p_k)$ must satisfy $u_j = p_j$ ($j \neq k$) and $u_k \le 0 < p_k$. Since $n_k > 0$,
      $$n \cdot u = n \cdot p + n_k (u_k - p_k) < n \cdot p < \alpha,$$
      so $u \notin H$; hence $\gamma(p_k) \notin R(H)$.
    - For $t > p_k$: $\gamma(t)_k = p_k - t < 0$, so $\gamma(t) \notin \mathbb{R}_{\ge 0}^{d+1} \supseteq R(H)$.
  - *Subcase 2b ($n_i \le 0$ for all $i$):* Since $n \neq 0$, pick $k$ with $n_k < 0$ and take $\gamma(t) := p + t e_k$ ($t \ge 0$). For all $t \ge 0$, $\gamma(t) \in \mathbb{R}_{>0}^{d+1}$, so its unique $R$-preimage is $u = \gamma(t)$ with $n \cdot u = n \cdot p + t n_k \le n \cdot p < \alpha$, hence $\gamma(t) \notin R(H)$. $\square$

**Step 1 (Non-separation of every compact $K \subseteq R(H)$).** Decomposing $\mathbb{R}^{d+1} = \bigcup_{\sigma \in \{-,+\}^{d+1}} Q_\sigma$ into closed orthants, $R|_{Q_\sigma}$ is a linear coordinate projection $\pi_\sigma$, so $R(H) = \bigcup_\sigma \pi_\sigma(H \cap Q_\sigma)$ is a finite union of closed convex polyhedra, each contained in an affine subspace $\pi_\sigma(H)$ of dimension $\le d$. Hence $R(H)$ has empty interior in $\mathbb{R}^{d+1}$.

Let $K \subseteq R(H)$ be compact and pick $R_0 > 0$ with $K \subset B(0, R_0)$. Since $d + 1 \ge 2$, the exterior $E := \mathbb{R}^{d+1} \setminus \overline{B(0, R_0)}$ is path-connected. Given any $x \in \mathbb{R}^{d+1} \setminus K$, choose $\varepsilon > 0$ with $B(x, \varepsilon) \subseteq \mathbb{R}^{d+1} \setminus K$. Since $R(H)$ has empty interior, pick $p \in B(x, \varepsilon) \setminus R(H)$. Walking along the segment $[x, p] \subset B(x, \varepsilon)$ and then along the escape ray $\gamma$ from $p$ connects $x$ to $E$ in $\mathbb{R}^{d+1} \setminus K$. Thus $\mathbb{R}^{d+1} \setminus K$ is path-connected, and hence $S^{d+1} \setminus K$ is path-connected in the one-point compactification $S^{d+1} = \mathbb{R}^{d+1} \cup \{\infty\}$. (The case $K \subseteq H$ is identical.)

**Step 2 (Proof of (1) via Tietze extension and Alexander Duality).** Suppose for contradiction that continuous maps $f : S^d \to \Phi(X)$ and $g : \Phi(X) \to S^d$ satisfy $\deg(g \circ f) \neq 0$. Let $Z := f(S^d) \subseteq \Phi(X) \subseteq R(H)$, which is nonempty and compact. Viewing $S^d \subset \mathbb{R}^{d+1}$, extend $g|_Z : Z \to S^d$ componentwise by the Tietze Extension Theorem to a continuous map $G : \mathbb{R}^{d+1} \to \mathbb{R}^{d+1}$. Let $U := \{y \in \mathbb{R}^{d+1} : G(y) \neq 0\}$ (an open neighborhood of $Z$) and $\tilde{g} : U \to S^d$, $\tilde{g}(y) := G(y)/\|G(y)\|_2$, so $\tilde{g}|_Z = g|_Z$.

Choose $M > 0$ with $Z \subset (-M, M)^{d+1}$, and set $P := R(H) \cap [-M, M]^{d+1}$, a finite union of compact convex polytopes of dimension $\le d$, hence triangulable as a finite simplicial complex of dimension $\le d$ (Rourke–Sanderson, *Introduction to Piecewise-Linear Topology*, Theorem 2.11). Taking iterated barycentric subdivisions so every simplex has diameter $< \operatorname{dist}(Z, \mathbb{R}^{d+1} \setminus U)$ (or taking $N = P$ if $U = \mathbb{R}^{d+1}$), let $N$ be the union of all closed simplices meeting $Z$. Then $N$ is a nonempty compact finite simplicial complex (hence compact and locally contractible) with $Z \subseteq N \subseteq R(H) \cap U$.

By Step 1, $\tilde{H}_0(S^{d+1} \setminus N; \mathbb{Z}) = 0$, so by Alexander Duality (Hatcher, *Algebraic Topology*, Theorem 3.44),
$$H^d(N; \mathbb{Z}) \cong \tilde{H}_0(S^{d+1} \setminus N; \mathbb{Z}) = 0.$$
Yet $S^d \xrightarrow{f} N \xrightarrow{\tilde{g}|_N} S^d$ equals $g \circ f$, so $(g \circ f)^* : H^d(S^d; \mathbb{Z}) \to H^d(S^d; \mathbb{Z})$ is multiplication by $\deg(g \circ f) \neq 0$ while factoring through $H^d(N; \mathbb{Z}) = 0$ — contradiction.

**Step 3 (Proof of (2) — vanishing of $\check{H}^d$ and $H_d(\Phi(X); G)$).** For any compact $K = \Phi(X) \subseteq R(H)$ (with $h$ continuous), Step 1 gives $\tilde{H}_0(S^{d+1} \setminus K; \mathbb{Z}) = 0$, so Čech Alexander duality immediately yields $\check{H}^d(K; \mathbb{Z}) \cong \tilde{H}_0(S^{d+1} \setminus K; \mathbb{Z}) = 0$. When $N := \Phi(X) \subseteq R(H)$ is a compact polyhedron (triangulable by Rourke–Sanderson Thm 2.11) or compact semialgebraic set (triangulable by Hardt/Łojasiewicz semialgebraic triangulation, hence compact and locally contractible of dimension $\le d$), Alexander Duality gives $H^d(N; \mathbb{Z}) = 0$. By the Universal Coefficient Theorem for cohomology, $\operatorname{Hom}_{\mathbb{Z}}(H_d(N; \mathbb{Z}), \mathbb{Z}) = 0$ and $\operatorname{Ext}^1_{\mathbb{Z}}(H_{d-1}(N; \mathbb{Z}), \mathbb{Z}) = 0$. Since $\dim N \le d$, $H_d(N; \mathbb{Z}) = \ker(\partial_d)$ is free abelian, forcing $H_d(N; \mathbb{Z}) = 0$, and $H_{d-1}(N; \mathbb{Z})$ is finitely generated and torsion-free. By the UCT for homology, $H_d(N; G) = 0$ for every abelian group $G$.

If $X = \mathbb{R}^d$ (or any closed polyhedron), write $\Phi(X) = \bigcup_{M=1}^\infty Z_M$ with $Z_M := \Phi(X) \cap [-M, M]^{d+1}$ a compact polyhedron of dimension $\le d$ in $R(H)$. Because singular homology commutes with directed unions of compact subsets covering all compacta, $H_d(\Phi(X); G) \cong \varinjlim_M H_d(Z_M; G) = 0$. $\blacksquare$

---

## 4. Bonus Structural Remarks (Sketches for Arbitrary Output Width $(d, d, m)$)

1. **Full contractibility when $\operatorname{rank}(W_1) = d$ and $X = \mathbb{R}^d$ (sketch / partial):** If $W_1 \in \mathbb{R}^{d \times d}$ is invertible, then $h(\mathbb{R}^d) = \mathbb{R}_{\ge 0}^d$ is convex, so $C = W_2 \mathbb{R}_{\ge 0}^d + b_2 \subset \mathbb{R}^m$ is a closed convex polyhedral cone. Whenever $C$ meets the nonpositive orthant $\mathbb{R}_{\le 0}^m$ at some $z_0 \le 0$, the straight-line homotopy $(z, t) \mapsto R_m((1-t)z + t z_0)$ contracts $R_m(C)$ to $0$ (general case when $C \cap \mathbb{R}_{\le 0}^m = \emptyset$ is left conjectural).
2. **Sharpness of $\operatorname{rank}(W_1) = d$:** If $\operatorname{rank}(W_1) < d$, lower-dimensional spheres can appear: e.g., for $(2, 2, 3)$ with $h(x_1, x_2) = (\max(0, x_1), \max(0, 1 - x_1))$, applying Theorem 1 for $d = 1$ yields $S^1 \subset \mathbb{R}^3$ (so $H_1 \neq 0$ while $H_2 = 0$, confirming Theorem 2 needs no rank hypothesis).
3. **Compact convex $X \subset \mathbb{R}^d$, arbitrary output width $m$ (conjectural sketch):** Splitting $R_d(W_1 X + b_1) = A \cup B$ into its compact convex interior-orthant part $A$ and boundary part $B \subseteq \partial \mathbb{R}_{\ge 0}^d$ ($\dim B \le d - 1$) suggests $\check{H}^d(\Phi(X); G) = 0$ via Vietoris–Begle and strong excision, left unproved here.
