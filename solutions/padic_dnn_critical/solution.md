# Equilibria of Hierarchical and Random $p$-Adic Deep Neural Networks

**Target paper:** W. A. Zúñiga-Galindo, *Critical Organization of $p$-Adic Deep Neural Networks*, [arXiv:2601.19070](https://arxiv.org/abs/2601.19070) (January 2026), Section 10, Conjectures `Conj_tree` and `Conj_random`.  
**Companion files:** [`solutions/padic_dnn_critical/explore.py`](explore.py), [`solutions/padic_dnn_critical/proof.pdf`](proof.pdf).

---

## Verdicts

- **Conjecture `Conj_tree`: PROVED (and sharpened).** For $\phi = \tanh$, zero input, and $T_W = \sum_{j \ge 0} c_j P_j$ with $c_j \ge 0$, $A_l = \sum_{j \ge l} c_j > 1$, and $\beta_l = \sum_{j < l} c_j < M_{A_l} = A_l \tanh s_{A_l} - s_{A_l}$, the level space $D^l(\mathbb{Z}_p)$ contains **exactly $3^{p^l}$ equilibria**—one for each three-branch pattern in $\{-1, 0, +1\}^{G_l}$. Every equilibrium in $D^l(\mathbb{Z}_p)$ is hyperbolic in $D^l(\mathbb{Z}_p)$ with instability index equal to its number of middle-branch coordinates (hence $\binom{p^l}{k} 2^{p^l - k}$ states of index $k$). All $2^{p^l}$ outer-only equilibria are linearly stable in the full infinite-dimensional space $L^2(\mathbb{Z}_p)$. (This makes no assertion about additional equilibria outside $D^l(\mathbb{Z}_p)$.)
- **Conjecture `Conj_random`: FALSE (as stated for general fixed nonnegative profiles).** For $p = 2$, there exists a single deterministic, finite-valued, nonnegative variance profile $v$ (with $v(|\cdot|_2) \in L^1(\mathbb{Z}_2)$) satisfying $0 < \rho(v)^2 \le 1/4096$ (so $\rho(v) \le 1/64 < 1$, and arbitrarily small with a smaller mass budget) such that along an infinite sequence of dyadic levels $N_k = 2^{l_k}$, the random network has at least three equilibria with probability at least $\frac{1}{2} - 2^{-k} \to \frac{1}{2}$. This refutes the subcritical uniqueness claim.
- **Scope notes:** The counterexample profile $v$ satisfies the literal hypotheses of `Conj_random` ($v \ge 0$, finite everywhere, $v(|\cdot|_2) \in L^1(\mathbb{Z}_2)$), but $v(|x|_2)$ is unbounded as $|x|_2 \to 0$. Bounded fixed profiles ($v \in L^\infty$) and the independent supercritical ($\rho(v) > 1$) exponential-growth implication remain open. Note also that Wainrib and Touboul (*Phys. Rev. Lett.* 110, 118101, 2013), cited in the paper for constant $v$, proves exponential growth of the *expected* equilibrium count rather than high-probability behavior.

---

## 1. Proof and Sharpening of `Conj_tree`

Normalize Haar measure on $\mathbb{Z}_p$ to mass $1$. Let $G_l = \mathbb{Z}_p / p^l \mathbb{Z}_p$ index the $n := p^l$ balls of measure $p^{-l}$, let $D^l = D^l(\mathbb{Z}_p) \subset L^2(\mathbb{Z}_p)$ be the $n$-dimensional space of functions constant on these balls, and let $P_j f(x) := p^j \int_{x + p^j \mathbb{Z}_p} f(y)\,dy$ be the orthogonal projection onto $D^j$. For $T_W = \sum_{j \ge 0} c_j P_j$ with $c_j \ge 0$ and $\sum_{j \ge 0} c_j < \infty$, define
$$
A := A_l = \sum_{j \ge l} c_j, \qquad \beta := \beta_l = \sum_{j < l} c_j.
$$
The zero-input dynamics is $\dot{h} = -h + T_W \tanh h$. At an equilibrium $h$, linear stability in $L^2(\mathbb{Z}_p)$ means that the spectrum of $-I + T_W D$ (where $D$ is multiplication by $\operatorname{sech}^2 h$) lies strictly in $\{\operatorname{Re} z < 0\}$.

On $D^l$, we have $P_j|_{D^l} = I$ for all $j \ge l$, so $T_W|_{D^l} = AI + B$, where $B := \sum_{j < l} c_j P_j|_{D^l}$ is symmetric, positive semidefinite, entrywise nonnegative, and has constant row sums $\beta$. In rate coordinates $u_i := \tanh h_i \in (-1, 1)$, the equilibrium equation in $D^l$ is
$$
f(u_i) = (Bu)_i \quad (i = 1, \dots, n), \qquad f(t) := \operatorname{atanh} t - At \quad (-1 < t < 1).
$$
For $A > 1$, let $s_A := \operatorname{arcosh}\sqrt{A} > 0$ and $r := \tanh s_A = \sqrt{1 - 1/A} \in (0, 1)$, so $A = (1 - r^2)^{-1}$. Then $f$ is odd, strictly convex on $(0, 1)$, strictly decreasing on $[-r, r]$, and strictly increasing on $[r, 1)$, with fold height
$$
M := M_A = A\tanh s_A - s_A = Ar - s_A = -f(r) > 0.
$$
The three open branch intervals in rate coordinates are $I_- = (-1, -r)$, $I_0 = (-r, r)$, and $I_+ = (r, 1)$.

**Theorem 1 (Exact tree-indexed count and full-space stability).** *Assume $A > 1$ and $\beta < M_A$. Then:*
1. *For every branch pattern $\sigma \in \{-, 0, +\}^n$, there is a unique equilibrium $u \in D^l$ with $u_i \in I_{\sigma_i}$ for all $i$. Hence $D^l$ contains **exactly $3^{p^l}$ equilibria**.*
2. *Every equilibrium $u \in D^l$ is hyperbolic in $D^l$ with instability index equal to $|\{i : \sigma_i = 0\}|$. In particular, $D^l$ has exactly $\binom{p^l}{k} 2^{p^l - k}$ equilibria of index $k$.*
3. *All $2^{p^l}$ outer-only equilibria ($\sigma \in \{-, +\}^n$) are linearly stable in the full space $L^2(\mathbb{Z}_p)$, whereas every remaining equilibrium in $D^l$ has a positive eigenvalue already in $D^l$.*

*Proof.* **Step 1 (Existence per pattern).** Since $\beta < M$, each of the three branches of $f$ has a continuous inverse $g_-, g_0, g_+$ on $[-\beta, \beta]$. For any pattern $\sigma$, the continuous map $u \mapsto (g_{\sigma_i}((Bu)_i))_{i=1}^n$ maps $[-1, 1]^n$ into itself (because $\|Bu\|_\infty \le \beta$). Brouwer's fixed-point theorem yields an equilibrium in each pattern, giving at least $3^n$ distinct equilibria in $D^l$ (including when $\beta = 0$).

**Step 2 (Weighted Gershgorin bound on the outer block).** Let $u \in D^l$ be any equilibrium. Since $\|Bu\|_\infty \le \beta < M$, no coordinate satisfies $|u_i| = r$, so $O := \{i : |u_i| > r\}$ and $C := \{i : |u_i| < r\}$ partition $\{1, \dots, n\}$. Consider the symmetric matrix
$$
H := \operatorname{diag}(f'(u_i)) - B.
$$
We show that the outer principal submatrix $H_{OO}$ is strictly positive definite. For $i \in O$, put $t_i := |u_i| \in (r, 1)$, $w_i := \min(t_i - r, 1 - t_i) > 0$, $m := (1 + r)/2$, and $d := (1 - r)/2$, so $0 < w_i \le d$. We prove
$$
f'(t_i)\, w_i > \sum_{j \in O} B_{ij} w_j \qquad (\forall i \in O).
\tag{1}
$$
- *Case $t_i \le m$:* Here $w_i = t_i - r$. By strict convexity of $f$ on $(r, 1)$ and oddness of $f$,
  $$
  f'(t_i)(t_i - r) > f(t_i) - f(r) = M + \operatorname{sgn}(u_i)(Bu)_i > \beta + \operatorname{sgn}(u_i)(Bu)_i = \sum_{j=1}^n B_{ij}\bigl(1 + \operatorname{sgn}(u_i) u_j\bigr).
  $$
  Since $B_{ij} \ge 0$ and $|u_j| < 1$, every summand is nonnegative; dropping $j \in C$ and using $1 + \operatorname{sgn}(u_i)u_j \ge 1 - |u_j| \ge w_j$ for $j \in O$ proves (1).
- *Case $t_i \ge m$:* Here $w_i = 1 - t_i$. On $[r, 1)$, $(1 - t)f'(t) = (1 + t)^{-1} - A(1 - t)$ has derivative $A - (1 + t)^{-2} > 1 - 1 = 0$, so it is strictly increasing. Direct algebra with $A = (1 - r^2)^{-1}$ gives
  $$
  f'(m) = \frac{4}{(1 - r)(3 + r)} - \frac{1}{1 - r^2} = \frac{1 + 3r}{(1 - r^2)(3 + r)} = Ar + \frac{1}{3 + r} > Ar - s_A = M > \beta.
  $$
  Therefore $f'(t_i)(1 - t_i) \ge (1 - m)f'(m) = d f'(m) > d\beta \ge \sum_{j \in O} B_{ij} w_j$, proving (1) for all $i \in O$.

Writing $W := \operatorname{diag}(w_i)_{i \in O}$, the similarity $W^{-1} H_{OO} W$ has every Gershgorin disk contained strictly in $(0, \infty)$. Since $H_{OO}$ is real symmetric, $H_{OO} \succ 0$.

**Step 3 (Hyperbolicity and exact instability index).** For $i \in C$, $f'(u_i) < 0$; since $B_{CC} \succeq 0$, we have $H_{CC} \prec 0$. Because $H_{OO} \succ 0$, the Schur complement $H_{CC} - H_{CO} H_{OO}^{-1} H_{OC} \prec 0$ (and the single-block cases $C = \emptyset$ or $O = \emptyset$ are immediate). By Sylvester's law of inertia, $H$ is invertible with exactly $|C|$ negative and $|O|$ positive eigenvalues. In leaf coordinates, $D = \operatorname{diag}(1 - u_i^2)$ satisfies $f'(u_i) = D_{ii}^{-1} - A$, so $H = D^{-1} - (AI + B)$. Hence the Jacobian $J = -I + (AI + B)D$ satisfies
$$
D^{1/2} J D^{-1/2} = -I + D^{1/2}(AI + B)D^{1/2} = -D^{1/2} H D^{1/2}.
$$
By congruence, $J$ is hyperbolic with exactly $|C|$ positive and $|O|$ negative eigenvalues.

**Step 4 (Homotopy uniqueness).** Replace $B$ by $bB$ for $b \in [0, 1]$. Any equilibrium $h = (AI + bB)\tanh h$ obeys $\|h\|_\infty \le A + \beta$, so $u$ remains inside the compact cube $[-\tanh(A + \beta), \tanh(A + \beta)]^n \subset (-1, 1)^n$, and $|(bBu)_i| \le \beta < M$ prevents any coordinate from touching $\pm r$. By Step 3, the derivative $H$ of $u \mapsto f(u) - bBu$ is invertible at every solution for every $b \in [0, 1]$. Thus each fiber over $b$ is finite, each root continues uniquely locally by the implicit function theorem, and compactness rules out roots entering from elsewhere. Consequently, the number of equilibria per branch pattern is locally constant—hence constant—on $[0, 1]$. At $b = 0$, $f(u_i) = 0$ has exactly one root per branch, giving exact cardinality $3^n$ and $\binom{n}{k} 2^{n - k}$ states of index $k$.

**Step 5 (Full $L^2(\mathbb{Z}_p)$ linear stability).** For any equilibrium $h \in D^l$, the multiplication operator $D = \operatorname{sech}^2 h$ is constant on each level-$l$ ball and therefore commutes with $P_l$; every $P_j$ also commutes with $P_l$. Thus $D^l$ and $(D^l)^\perp$ are invariant under $-I + T_W D$. On $(D^l)^\perp$, every $P_j$ with $j < l$ vanishes, so $\|T_W|_{(D^l)^\perp}\|_{\mathrm{op}} \le \sum_{j \ge l} c_j = A$. For an outer-only state ($C = \emptyset$), every $|u_i| > r = \sqrt{1 - 1/A}$, so $\|D\|_{\mathrm{op}} = \max_i (1 - u_i^2) < 1/A$. Hence $\|T_W D|_{(D^l)^\perp}\|_{\mathrm{op}} \le A\|D\|_{\mathrm{op}} < 1$, placing the complementary spectrum in the disk $\{|z + 1| \le A\|D\|_{\mathrm{op}}\} \subset \{\operatorname{Re} z < 0\}$. Combined with Step 3 on $D^l$, all $2^n$ outer-only equilibria are linearly stable in $L^2(\mathbb{Z}_p)$. $\square$

---

## 2. Fixed-Profile Counterexample to `Conj_random`

At level $l$, $M^{(l)} \in \mathbb{R}^{p^l \times p^l}$ has independent centered Gaussian entries $M^{(l)}_{IK} \sim \mathcal{N}(0, p^{-l} v(|I - K|_p))$ for a nonnegative profile $v$, with equilibria $h = M^{(l)} \tanh h$ and $\rho(v)^2 := \lim_{l \to \infty} \sum_{K \in G_l} p^{-l} v(|K|_p)$. Matrix norms below are Euclidean operator norms and logarithms are natural.

**Lemma 2 (Deterministic Brouwer degree test).** *For any real $n \times n$ matrix $M$, if $\det(I - M) < 0$, then $h = M \tanh h$ has at least three solutions in $\mathbb{R}^n$.*

*Proof.* Set $F(h) := h - M\tanh h$ and $R := 1 + \|M\|\sqrt{n}$. On the sphere $\|h\|_2 = R$, $\|b M \tanh h\|_2 \le \|M\|\sqrt{n} < R$ for all $b \in [0, 1]$, so the homotopy $h - b M\tanh h$ is nonvanishing on $\partial B_R(0)$ and $\deg(F, B_R(0), 0) = \deg(I, B_R(0), 0) = 1$, with all zeros interior. Because $DF(0) = I - M$ has negative determinant, the origin is an isolated zero of local index $-1 \neq 1$. By additivity of Brouwer degree, $F$ has a nonzero root $h^*$, and oddness of $F$ yields the distinct root $-h^*$. $\square$

Set $p = 2$. An active shell at $2$-adic distance $2/N_j = 2^{-(l_j - 1)}$ (where $N_j = 2^{l_j}$) is assigned profile value $v(2/N_j) := N_j a_j$ (and $v(0) := 0$). At dyadic width $n = 2^l \ge N_j$, each row $I \in G_l$ has exactly $n/N_j$ entries $K \in G_l$ at distance $2/N_j$ (the opposite level-$l_j$ child inside the same level-$(l_j - 1)$ ball), each of variance $N_j a_j / n$, contributing exact row variance $a_j$. At widths $n < N_j$, no pair in $G_l$ lies at distance $2/N_j$, so the shell is absent.

**Lemma 3 (Finite-extension lemma).** *Fix finitely many earlier active shells at dyadic scales $N_1 < \cdots < N_m$ with $\sum_{j=1}^m \sqrt{a_j} \le 1/64$. For a candidate dyadic width $n = 2^l > N_m$, add a new shell at distance $2/n$ with mass $a := 1/L$, where $L := \log n - \log\log n$, and let $M = B + S$ be the resulting $n \times n$ random matrix (decomposed into earlier shells $B$ and top shell $S$). Then as $n = 2^l \to \infty$ (with earlier shells fixed),*
$$
\Pr\bigl(\det(I - M) < 0\bigr) \longrightarrow \frac{1}{2}.
$$

*Proof.* The matrices $B$ and $S$ are independent, and $S$ decomposes into $n/2$ disjoint $2 \times 2$ blocks $\begin{psmallmatrix} 0 & X_r \\ Y_r & 0 \end{psmallmatrix}$ with all $n$ directed entries independent $\mathcal{N}(0, 1/L)$.

1. *Operator norm of $B$.* Up to separate row and column permutations, shell $j$ consists of $N_j$ disjoint square blocks of side $d_j = n / N_j$ with i.i.d. $\mathcal{N}(0, a_j / d_j)$ entries. By packing disjoint balls of radius $1/8$ inside the ball of radius $9/8$, the unit sphere in $\mathbb{R}^{d_j}$ admits a $1/4$-net $\mathcal{N}$ of size at most $9^{d_j}$. Approximating both unit vectors in a bilinear form by net points loses at most half the operator norm, so a Gaussian tail and union bound over $\mathcal{N} \times \mathcal{N}$ give
   $$
   \Pr\bigl(\|\text{block}\| > 8\sqrt{a_j}\bigr) \le 2 \cdot 9^{2 d_j} e^{-8 d_j}.
   $$
   Since $81 < e^8$ and the number of old blocks is fixed as $d_j \to \infty$, summing over shells yields $\|B\| \le 8 \sum_j \sqrt{a_j} \le 1/8 \le 1/4$ with probability $1 - o(1)$. Moreover, every entry of $B$ has variance at most $K/n$ for the fixed constant $K := \max(\{1\} \cup \{N_j a_j\}_j)$.

2. *Determinant sign and non-singularity of $I - S$.* Put $q := \Pr(XY > 1)$ for independent $X, Y \sim \mathcal{N}(0, 1/L)$. In polar coordinates for $(\sqrt{L} X, \sqrt{L} Y)$, the angle is uniform and the squared radius is $\operatorname{Exp}(1/2)$, so for every $t > 0$,
   $$
   Q(t) := \Pr(XY > t) = \frac{1}{2\pi} \int_{\cos 2\theta > 0} e^{-Lt / \cos 2\theta}\,d\theta.
   \tag{2}
   $$
   Hence $q \le e^{-L} \to 0$. On $|\theta| \le 1/(4\sqrt{L})$, $\cos 2\theta \ge 1 - 2\theta^2 \ge 1 - 1/(8L)$ implies $1/\cos 2\theta \le 1 + 1/L$ for $L \ge 1$, whence $q \ge c\,e^{-L}/\sqrt{L}$ for an absolute constant $c > 0$. Because $e^{-L} = (\log n)/n$, we obtain $nq \ge c\sqrt{\log n} \to \infty$. By independence of the $n/2$ blocks ($\det(I - S) = \prod_{r=1}^{n/2} (1 - X_r Y_r)$),
   $$
   \Pr\bigl(\det(I - S) < 0\bigr) = \frac{1 - (1 - 2q)^{n/2}}{2} \longrightarrow \frac{1}{2}.
   \tag{3}
   $$
   Set $\delta := (\log n)^{-4}$. Differentiating (2) under the integral (justified since $z e^{-Ltz}$ decreases on $z \ge 1$ whenever $Lt \ge 1$) gives $|Q'(t)| \le L e^{-Lt}$ for $t \ge 1 - \delta$ and large $n$. Thus $\Pr(\exists r : |1 - X_r Y_r| \le \delta) \le n \delta L e^{-L(1 - \delta)} = o(1)$. Also $\Pr(\max |S_{ij}| > 3) \le 2n e^{-9L/2} = o(1)$. On the intersection of these high-probability events, the $2 \times 2$ inverse formula yields $\|(I - S_r)^{-1}\| \le (1 + \max(|X_r|, |Y_r|))/\delta \le 4/\delta$ for every pair block $r$.

3. *Schur elimination on ordinary pairs.* Call a pair exceptional if $\max(|X_r|, |Y_r|) > 1/2$; let $E$ be the set of exceptional coordinates, $G$ its complement, and $d := |E|$. Then $\mathbb{E}[d] \le 4n e^{-L/8}$, so $\Pr(d > n^{15/16}) = o(1)$ by Markov's inequality. Notice that $(E, G)$ depends only on $S$. On $\|B\| \le 1/4$, since $\|S_{GG}\| \le 1/2$, we have
   $$
   V := I - S_{GG} - B_{GG}, \qquad \|V^{-1}\| \le 4, \qquad \det V > 0
   $$
   (scaling $S_{GG} + B_{GG}$ from $0$ to $1$ keeps the determinant strictly positive). The Schur complement of $V$ in $I - M$ is
   $$
   U = I - S_{EE} - R, \qquad R := B_{EE} + B_{EG} V^{-1} B_{GE}.
   $$
   Set $\eta := n^{-1/64}$ (so $n\eta^2 = n^{31/32}$). Condition on any $S$ with $d \le n^{15/16}$ (if $E = \emptyset$, the sign equality is immediate). For any unit vectors $x, y \in \mathbb{R}^E$, $x^\top B_{EE} y$ is centered Gaussian of variance at most $K/n$; the two-net bound gives
   $$
   \Pr(\|B_{EE}\| > \eta \mid S) \le 2 \cdot 9^{2d} \exp\bigl(-n\eta^2 / (8K)\bigr) = o(1)
   $$
   uniformly in $S$. Next, condition further on $(B_{GG}, B_{GE})$ on the event that $V$ is invertible and $\|V^{-1} B_{GE}\| \le 1$ (which is implied by $\|B\| \le 1/4$, failing with probability $o(1)$). Putting $H := V^{-1} B_{GE}$ ($\|H\| \le 1$), independence of the directed entries of $B$ makes $B_{EG}$ independent of $(S, B_{GG}, B_{GE})$. Hence for fixed unit $x, y \in \mathbb{R}^E$, $x^\top B_{EG} H y$ is centered Gaussian of variance at most $(K/n)\|x\|_2^2 \|Hy\|_2^2 \le K/n$, and the same two-net bound yields $\Pr(\|B_{EG} H\| > \eta) = o(1)$. Thus $\|R\| \le 2n^{-1/64}$ with probability $1 - o(1)$.

   Combining this with $\|(I - S_{EE})^{-1}\| \le 4/\delta$, we obtain $\|(I - S_{EE})^{-1} R\| \le 8 n^{-1/64} / \delta < 1$ for large $n$ with probability $1 - o(1)$, so scaling $R$ continuously to $0$ preserves $\operatorname{sgn}\det U = \operatorname{sgn}\det(I - S_{EE})$. Since $\det V > 0$ and every ordinary pair contributes $1 - X_r Y_r \ge 3/4 > 0$ to $\det(I - S)$, Schur's determinant formula $\det(I - M) = \det(V)\det(U)$ gives
   $$
   \Pr\bigl(\operatorname{sgn}\det(I - M) = \operatorname{sgn}\det(I - S)\bigr) \longrightarrow 1.
   $$
   Together with (3), this proves the lemma. $\square$

**Theorem 4 (Fixed-profile refutation of `Conj_random`).** *For $p = 2$, there exists a deterministic nonnegative finite-valued profile $v$ with $v(|\cdot|_2) \in L^1(\mathbb{Z}_2)$ and $0 < \rho(v)^2 \le 1/4096$ (so $\rho(v) \le 1/64 < 1$), together with an increasing sequence of dyadic levels $N_k = 2^{l_k}$, such that*
$$
\Pr_{N_k}\bigl(h = M^{(l_k)}\tanh h \text{ has at least } 3 \text{ equilibria}\bigr) \ge \frac{1}{2} - 2^{-k} \qquad (\forall k \ge 1).
$$

*Proof.* Initialize $N_0 := 1$. Recursively at step $k \ge 1$, given already chosen $N_1 < \cdots < N_{k-1}$ satisfying $\sum_{j=1}^{k-1} \sqrt{a_j} < 1/64$, apply Lemma 3 to pick the least dyadic width $N_k = 2^{l_k} > N_{k-1}$ such that
$$
\sqrt{a_k} = (\log N_k - \log\log N_k)^{-1/2} \le \frac{2^{-k}}{64}, \qquad
\Bigl|\Pr_{N_k}\bigl(\det(I - M) < 0\bigr) - \frac{1}{2}\Bigr| \le 2^{-k}.
$$
Define $v(2/N_k) := N_k a_k$ for all $k \ge 1$ and $v(r) := 0$ for all other $r$. Since shells with $N_j > N_k$ are absent at width $N_k$, every probability bound above persists for the single fixed profile $v$. At every dyadic width $n$, the normalized row-sum variance is $\sum_{k : N_k \le n} a_k$, so the limit exists with
$$
0 < \rho(v)^2 = \sum_{k \ge 1} a_k = \int_{\mathbb{Z}_2} v(|x|_2)\,dx \le \Bigl(\sum_{k \ge 1} \sqrt{a_k}\Bigr)^2 \le \frac{1}{4096},
$$
because the $2$-adic shell $\{|x|_2 = 2/N_k\}$ has Haar measure $1/N_k$. Applying Lemma 2 at each width $N_k$ yields at least three equilibria with probability at least $\frac{1}{2} - 2^{-k}$. $\square$

---

## 3. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/padic_dnn_critical/explore.py
```

The script [`explore.py`](explore.py) uses a fixed random seed (`19070`), writes no output files, and verifies:
- Exact SymPy identities for $f'(m) = Ar + (3 + r)^{-1}$, $(1 - t)f'(t) = (1 + t)^{-1} - A(1 - t)$, and $\det\begin{psmallmatrix} 1 & -x \\ -y & 1 \end{psmallmatrix} = 1 - xy$;
- Projection products $P_j P_k = P_{\min(j, k)}$ and branch-pattern equilibria at $A = 2.25$, $\beta = 0.999 M_A$ for $p \in \{2, 3\}$ and $l \in \{1, 2, 3\}$—checking residuals ($< 2 \times 10^{-12}$), branch membership, hyperbolicity, and exact instability index across **26,620 patterns** (complete enumeration for $n \le 9$, plus 259 sampled patterns at $n = 27$);
- Finite-size two-shell illustrations of Lemma 3 at $n \in \{32, 64, 128\}$ (128 samples each), verifying Schur-complement determinant agreement in all 384/384 cases and pair-vs.-full sign agreement in 383/384 cases:

| Width $n$ | Samples | $\det(I - S) < 0$ | $\det(I - M) < 0$ | Pair/full sign agreement | Schur formula match |
|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 25 | 25 | 128 / 128 | 128 / 128 |
| 64 | 128 | 36 | 35 | 127 / 128 | 128 / 128 |
| 128 | 128 | 39 | 39 | 128 / 128 | 128 / 128 |

All assertions pass with exit code `0`.

---

## References

- W. A. Zúñiga-Galindo, *Critical Organization of $p$-Adic Deep Neural Networks*, [arXiv:2601.19070](https://arxiv.org/abs/2601.19070), January 2026.
- G. Wainrib and J. Touboul, *Topological and Dynamical Complexity of Random Neural Networks*, Physical Review Letters 110:118101, 2013.
- H. Sompolinsky, A. Crisanti, and H.-J. Sommers, *Chaos in Random Neural Networks*, Physical Review Letters 61:259–262, 1988.
