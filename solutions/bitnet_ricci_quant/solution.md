# Disproof of Fisher-Volume Scalar Curvature Quantization for Bitnets

**Target papers:** Carlos C. Rodríguez, *The Volume of Bitnets*, AIP Conference Proceedings **735**, 555–564 (2004), Conjecture `conj:main`; and Carlos C. Rodríguez, *Quantization of Ricci Curvature in Information Geometry*, [arXiv:2603.10054v1](https://arxiv.org/abs/2603.10054v1) (March 2026), Conjecture 1.1.  
**Companion files:** [`solutions/bitnet_ricci_quant/explore.py`](explore.py), [`solutions/bitnet_ricci_quant/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:main`: FALSE**, even for **tree-structured** bitnets.
  - **Five-node tree counterexample (Theorem 1):** The 5-node binary Bayesian network with edges $A \to B$, $A \to C$, $B \to D$, $B \to E$ ($d = 9$ free CPT parameters) has exact normalized Fisher-volume average scalar curvature
    $$
    \langle R \rangle = \frac{14}{11} \notin \tfrac{1}{2}\mathbb{Z}.
    $$
    Because $2(14/11) = 28/11 \notin \mathbb{Z}$, this refutes Rodríguez's conjecture under either convention for whether $0 \in \mathbb{Z}^+$ (note that the single-node network $\tilde{K}_1$ has $\langle R \rangle = 0$).
  - **Six-node tree counterexample (Proposition 2):** Attaching two leaves to $A$ and two leaves to $B$ ($d = 11$ parameters) gives a purely polynomial Dirichlet$(1/2,1/2,1/2,1/2)$ integral with $\langle R \rangle = \frac{25}{14} \notin \tfrac{1}{2}\mathbb{Z}$.
  - **Seven-node 1-loop counterexample (Proposition 4):** Adding an independent root $H$ with shared children $C, D$ ($d = 1 + 2 + 1 + 4 + 4 + 2 + 2 = 16$ parameters, unique chordless four-cycle $A - C - H - D - A$) yields $\langle R \rangle = \frac{24}{5} \notin \tfrac{1}{2}\mathbb{Z}$.
- **Correction of arXiv:2603.10054v1 claims (Proposition 3):** Rodríguez (2026) claimed to resolve his 2004 conjecture by proving half-integer quantization on all trees (Theorem 4.3) and disproving it via the double-collider loop $D_4$ (Proposition 5.2, reporting $36/5$). Both claims are mathematically erroneous:
  - Theorem 4.3 fails already on the 3-node V-structure (as noted in Remark 4.4 of arXiv:2603.10054v1) and is refuted by our 5-node and 6-node trees.
  - For $n$ independent roots and $m$ common sinks ($K = m 2^{n-1}$), exact warped-product integration gives $\langle R \rangle = nK/2 \in \tfrac{1}{2}\mathbb{Z}^+$. In particular, the double collider $D_4$ ($n=2, m=2$) has root volume exponent $3/2$ (not $5/2$) and true average $\langle R \rangle_{D_4} = 4 \in \tfrac{1}{2}\mathbb{Z}^+$; the 3-node fork $\tilde{E}_3$ and chain $\tilde{L}_3$ both have $\langle R \rangle = 1$ (confirming Rodríguez 2004's $m/2$ for $\tilde{E}_{m+1}$ and refuting Theorem 1.2/3.2's $(2n-1)/2$); and the 5-parent collapsing star $\tilde{C}_6$ has $\langle R \rangle = 40 > 0$ (refuting Theorem 3.4's claimed sign inversion to $-272$).
  - Consequently, the present calculations provide the first valid disproof of Conjecture `conj:main`.
- **Directed diamond:** For the 4-node directed diamond ($d = 9$), we also derive an exact, provably absolutely convergent binomial-series representation of both curvature and volume integrals.

---

## 1. Model, Fisher Metric, and Multi-Warped Product Curvature

A *bitnet* is a finite DAG of observed binary variables whose parameter space is the open unit cube with one free coordinate $t = P(X_i = 1 \mid X_{\mathrm{pa}(i)} = s) \in (0, 1)$ per node $i$ and parent configuration $s$ (including one marginal parameter per root). Let $q = P(X_{\mathrm{pa}(i)} = s)$. The score function for $t$ is the configuration indicator multiplied by $(X_i - t)/[t(1-t)]$, which has conditional mean zero given all predecessors in a topological ordering. Scores at distinct nodes or disjoint parent configurations are therefore orthogonal, so the one-observation Fisher metric is diagonal with entries $q/[t(1-t)]$.

Setting $z = 2\arcsin\sqrt{t} \in (0, \pi)$ transforms $dt^2/[t(1-t)] = dz^2$. Thus every sink CPT entry contributes a flat interval of length $\pi$ with squared length element $q\,dz^2$. We study the normalized Jeffreys-volume average of the scalar curvature,
$$
\langle R \rangle = \frac{\int R\,dV_g}{\int dV_g}, \qquad dV_g = \sqrt{\det g}\,d\theta,
$$
under the paper's sign/scale convention assigning $R(\tilde{K}_n) = d(d-1)/4$ to a radius-$2$ sphere of dimension $d$. Integrating out a sink interval contributes a factor $\pi\sqrt{q}$ to the marginal base volume density.

**Lemma 1 (Multi-warped product scalar curvature).** *Let $(B, g_0)$ be a Riemannian manifold of scalar curvature $R_0$, and equip $B \times \prod_j (0, \pi)$ with the metric $g = g_0 + \sum_j f_j^2\,dz_j^2$, where each $f_j > 0$ is smooth on $B$. Put $F := \prod_j f_j$, $\Delta := \operatorname{div}_{g_0}\nabla$, and $\Gamma(h, k) := \langle \nabla h, \nabla k \rangle_{g_0}$. Then*
$$
R = R_0 - 2\Delta \log F - \Gamma(\log F, \log F) - \sum_j \Gamma(\log f_j, \log f_j),
\tag{1}
$$
*and the volume density relative to $dV_{g_0}\prod_j dz_j$ is $F$.*

*Proof.* Let $X$ be a unit horizontal vector and $e_j = f_j^{-1}\partial_{z_j}$ be the vertical orthonormal frame. From metric compatibility and the Lie brackets $[X, e_j] = -X(\log f_j)e_j$, $[e_i, e_j] = 0$, Koszul's formula gives $\nabla_X e_j = 0$, $\nabla_{e_j} X = X(\log f_j)e_j$, $\nabla_{e_j} e_j = -\nabla \log f_j$, and $\nabla_{e_i} e_j = 0$ ($i \ne j$), while the horizontal connection coincides with that of $g_0$. Evaluating curvature gives mixed sectional curvature $-\operatorname{Hess}(f_j)(X, X)/f_j$ and vertical sectional curvature $-\Gamma(f_i, f_j)/(f_i f_j)$ ($i \ne j$). Writing $\Delta f_j / f_j = \Delta \log f_j + \Gamma(\log f_j, \log f_j)$ and summing twice over all orthonormal coordinate 2-planes yields (1):
$$
R = R_0 - 2\sum_j \frac{\Delta f_j}{f_j} - \sum_{i \ne j} \Gamma(\log f_i, \log f_j)
  = R_0 - 2\Delta \log F - \Gamma(\log F, \log F) - \sum_j \Gamma(\log f_j, \log f_j). \qquad \square
$$

---

## 2. Two Adjacent Internal Nodes and Tree Counterexamples

Consider the single-edge base model $A \to B$, which parametrizes all strictly positive four-cell joint laws $p_{ij} = P(A=i, B=j)$ ($\sum p_{ij} = 1$). Put
$$
a = P(A = 1), \quad b = P(B = 1), \quad u = a(1-a), \quad v = b(1-b), \quad w = p_{11} - ab.
$$
The Fisher metric $\sum dp_{ij}^2 / p_{ij}$ is pulled back by $p_{ij} \mapsto x_{ij} = 2\sqrt{p_{ij}}$ to the positive orthant of the radius-$2$ sphere $S^3(2)$, which has constant scalar curvature $R_0 = 3(3-1)/4 = 3/2$. In three free cell coordinates, $\det g = 1/\prod_{i,j} p_{ij}$, so the normalized Fisher volume law $E_0$ is $\mathrm{Dirichlet}(1/2, 1/2, 1/2, 1/2)$.

**Lemma 2 (Edge base operators and $(k, l)$-leaf curvature).** *On the base $A \to B$,*
$$
\Gamma(a, a) = u, \quad \Gamma(b, b) = v, \quad \Gamma(a, b) = w, \qquad \Delta a = 1 - 2a, \quad \Delta b = 1 - 2b.
$$
*Attaching $k$ single-parent leaves to $A$ and $l$ single-parent leaves to $B$ yields $F = u^{k/2} v^{l/2}$ and scalar curvature*
$$
R = \frac{3}{2} + C_k(u) + C_l(v) - \frac{kl}{2}\frac{w(1 - 2a)(1 - 2b)}{uv},
\qquad C_k(u) := k^2 + \frac{5k}{2} - \frac{k(k+1)}{4u}.
\tag{2}
$$

*Proof.* On $S^3(2)$, each ambient coordinate $x_\alpha = 2\sqrt{p_\alpha}$ satisfies $\Delta x_\alpha = -\frac{3}{4} x_\alpha$ and $\langle \nabla x_\alpha, \nabla x_\beta \rangle = \delta_{\alpha\beta} - \frac{1}{4} x_\alpha x_\beta$. For any event probability $P(E) = \frac{1}{4}\sum_{\alpha \in E} x_\alpha^2$, the product rule gives $\Gamma(P(E_1), P(E_2)) = \operatorname{Cov}(\mathbf{1}_{E_1}, \mathbf{1}_{E_2})$ and
$$
\Delta P(E) = \frac{1}{2}\sum_{\alpha \in E}\bigl(x_\alpha \Delta x_\alpha + |\nabla x_\alpha|^2\bigr)
= \frac{1}{2}\sum_{\alpha \in E}\bigl(1 - x_\alpha^2\bigr) = \frac{|E|}{2} - 2P(E).
$$
Since $a = p_{10} + p_{11}$ and $b = p_{01} + p_{11}$ each sum $|E| = 2$ cells, $\Delta a = 1 - 2a$, $\Delta b = 1 - 2b$, $\Gamma(a, a) = u$, $\Gamma(b, b) = v$, and $\Gamma(a, b) = w$.

With $k$ leaves at $A$ and $l$ leaves at $B$, the warps consist of $k$ copies each of $\sqrt{a}, \sqrt{1-a}$ and $l$ copies each of $\sqrt{b}, \sqrt{1-b}$, so $F = u^{k/2} v^{l/2}$. From $\nabla u = (1-2a)\nabla a$, we compute $\Delta \log u = -2$ and $\Gamma(\log u, \log u) = (1-2a)^2 / u = 1/u - 4$. Also,
$$
\sum_{j \in A\text{-leaves}} \Gamma(\log f_j, \log f_j) = \frac{k}{4}\Bigl(\frac{u}{a^2} + \frac{u}{(1-a)^2}\Bigr) = \frac{k}{4}\Bigl(\frac{1}{u} - 2\Bigr).
$$
Collecting terms for $A$ in (1) gives $-2\cdot \frac{k}{2}(-2) - \frac{k^2}{4}(1/u - 4) - \frac{k}{4}(1/u - 2) = C_k(u)$, and symmetrically $C_l(v)$ for $B$. The sole cross term comes from $-\Gamma(\log F, \log F)$:
$$
-2 \cdot \frac{k}{2}\cdot\frac{l}{2}\,\Gamma(\log u, \log v) = -\frac{kl}{2}\frac{w(1-2a)(1-2b)}{uv}.
$$
Adding $R_0 = 3/2$ gives (2). $\square$

**Theorem 1 (Five-node tree counterexample).** *For the 5-node tree bitnet $A \to B$, $A \to C$, $B \to D$, $B \to E$ ($(k, l) = (1, 2)$, $d = 9$ parameters),*
$$
\langle R \rangle = \frac{14}{11} \notin \frac{1}{2}\mathbb{Z}.
$$

*Proof.* Under $E_0 = \mathrm{Dirichlet}(1/2, 1/2, 1/2, 1/2)$, transforming from $(p_{00}, p_{01}, p_{10}, p_{11})$ to $(a, t_0, t_1)$ where $t_i = P(B=1 \mid A=i)$ has Jacobian $u = a(1-a)$, which cancels the $u^{-1/2}\cdot u^{-1/2}$ factor in $(p_{00}p_{01}p_{10}p_{11})^{-1/2}$. Thus $a \sim \mathrm{Unif}(0, 1)$ and $t_0, t_1 \sim \mathrm{Beta}(1/2, 1/2)$ are mutually independent, with $\mathbb{E}[t_i] = 1/2$ and $\operatorname{Var}(t_i) = 1/8$. Writing $b = (1-a)t_0 + a t_1$ and $w = u(t_1 - t_0)$, direct conditional expectation gives
$$
E_0[v \mid a] = \frac{1 + 2u}{8}, \qquad E_0[w(1 - 2b) \mid a] = \frac{u(1 - 2a)}{4}.
$$
For $(k, l) = (1, 2)$, Eq. (2) and $F = \sqrt{u}\,v$ give
$$
R = 14 - \frac{1}{2u} - \frac{3}{2v} - \frac{w(1 - 2a)(1 - 2b)}{uv}.
$$
Let $J_s := \int_0^1 [a(1-a)]^s\,da = B(s+1, s+1) = \Gamma(s+1)^2 / \Gamma(2s+2)$, so $J_{-1/2} = \pi$, $J_{1/2} = \pi/8$, and $J_{3/2} = 3\pi/128$. Because $|w| \le u$ bounds $|F w(1-2a)(1-2b)/(uv)| \le \sqrt{u}$, all four weighted integrals are absolutely convergent and evaluate to:
$$
\begin{aligned}
E_0[F] &= \frac{J_{1/2} + 2J_{3/2}}{8} = \frac{11\pi}{512}, &
E_0[F/u] &= \frac{J_{-1/2} + 2J_{1/2}}{8} = \frac{5\pi}{32}, \\
E_0[F/v] &= J_{1/2} = \frac{\pi}{8}, &
E_0\!\left[\frac{F w(1-2a)(1-2b)}{uv}\right] &= \frac{J_{1/2} - 4J_{3/2}}{4} = \frac{\pi}{128}.
\end{aligned}
$$
Therefore,
$$
\langle R \rangle = 14 - \frac{\frac{1}{2}\cdot\frac{5\pi}{32} + \frac{3}{2}\cdot\frac{\pi}{8} + \frac{\pi}{128}}{11\pi/512}
= 14 - \frac{140}{11} = \frac{14}{11}. \qquad \square
$$

**Proposition 2 (Six-node tree via purely polynomial Dirichlet integration).** *For $(k, l) = (2, 2)$ ($d = 11$ parameters), $F = uv$ and all volume-weighted terms in (2) are polynomials in $(p_{ij})$. By the exact $\mathrm{Dirichlet}(1/2, 1/2, 1/2, 1/2)$ moment formula $E_0[\prod p_\alpha^{n_\alpha}] = \prod (1/2)_{n_\alpha} / (2)_{\sum n_\alpha}$,*
$$
E_0[uv] = \frac{7}{240}, \qquad E_0[u] = E_0[v] = \frac{1}{6}, \qquad E_0[w(1 - 2a)(1 - 2b)] = \frac{1}{120},
$$
*and Eq. (2) gives*
$$
\langle R \rangle = \frac{39}{2} - \frac{3(1/6) + 2(1/120)}{7/240} = \frac{25}{14} \notin \frac{1}{2}\mathbb{Z}.
$$

---

## 3. Shared-Root Colliders, Refutation of arXiv:2603.10054v1 Claims, and a 1-Loop Counterexample

**Proposition 3 (Shared-root colliders).** *Consider a bitnet with $n \ge 1$ independent binary roots and $m \ge 1$ sinks, each having all $n$ roots as parents. Put $K := m 2^{n-1}$. In root coordinates $r_1, \dots, r_n \in (0, 1)$ with $U_i := r_i(1 - r_i)$,*
$$
R = \sum_{i=1}^n S_K(r_i), \qquad S_K(r) := \frac{K(2K + 1)}{2} - \frac{K(K - 1)}{4U},
\tag{3}
$$
*and the marginal Fisher volume density is proportional to $\prod_{i=1}^n U_i^{(K-1)/2}\,dr_i$. Consequently,*
$$
\langle R \rangle = \frac{nK}{2} = m n 2^{n-2} \in \frac{1}{2}\mathbb{Z}^+.
$$

*Proof.* In root angular coordinates, the base is flat ($R_0 = 0$) and separates across the $n$ roots. Each of the $m 2^n$ sink warps is a product over roots of $\sqrt{r_i}$ or $\sqrt{1 - r_i}$, with each root state appearing $K = m 2^{n-1}$ times. Hence $\log F = \frac{K}{2}\sum_{i=1}^n \log U_i$. On one root coordinate, $\Gamma(h, h) = U(h')^2$ and $\Delta h = U h'' + \frac{1}{2}(1-2r)h'$. Substituting into (1) gives
$$
S_K(r) = -2\cdot\frac{K}{2}\Bigl(-\frac{1}{2U} - 1\Bigr) - \frac{K^2}{4}\Bigl(\frac{1}{U} - 4\Bigr) - \frac{K}{4}\Bigl(\frac{1}{U} - 2\Bigr)
= \frac{K(2K+1)}{2} - \frac{K(K-1)}{4U}.
$$
The root's own Fisher density contributes $U^{-1/2}\,dr$ and the $2K$ sink warps contribute $U^{K/2}$, giving marginal density $\propto U^{(K-1)/2}\,dr$. For $K = 1$, $S_1 \equiv 3/2$. For $K > 1$, the Beta integral ratio gives $\mathbb{E}[1/U] = \frac{B((K-1)/2, (K-1)/2)}{B((K+1)/2, (K+1)/2)} = \frac{4K}{K-1}$, whence $\mathbb{E}[S_K(r)] = K(2K+1)/2 - K^2 = K/2$, and summing over $n$ roots gives $\langle R \rangle = nK/2$. $\square$

Proposition 3 corrects three erroneous calculations in arXiv:2603.10054v1:
1. **Double collider $D_4$ ($n = 2, m = 2, K = 4$):** Eq. (3) reproduces the pointwise scalar curvature $R_{D_4} = 36 - 3/U_1 - 3/U_2$, but the true marginal volume exponent is $(K-1)/2 = 3/2$ (not $5/2$). Hence $\mathbb{E}[1/U_i] = 16/3$ and $\langle R \rangle_{D_4} = 36 - 6(16/3) = 4 \in \tfrac{1}{2}\mathbb{Z}^+$, refuting Proposition 5.2's claimed $36/5$.
2. **Three-node fork $\tilde{E}_3$ ($n = 1, m = 2, K = 2$) and chain $\tilde{L}_3$:** Both parametrize the exact same statistical model of conditional independence given the middle node (reversing the first arrow is a smooth reparametrization). Proposition 3 gives $R = 5 - 1/(2U)$ and $\langle R \rangle = 1$, confirming Rodríguez (2004)'s $\langle R \rangle(\tilde{E}_{m+1}) = m/2$ and refuting Theorems 1.2 and 3.2 of arXiv:2603.10054v1, which claim $(2n-1)/2$.
3. **Five-parent collapsing star $\tilde{C}_6$ ($n = 5, m = 1, K = 16$):** Eq. (3) yields constant term $5 \cdot 16(33)/2 = 1320$ (not $1008$), singular coefficient $60$ per root, and $\langle R \rangle(\tilde{C}_6) = 40 > 0$, refuting Theorem 3.4's claimed "curvature sign inversion" to $-272$.

**Proposition 4 (Seven-node 1-loop non-half-integer).** *Take $7$ nodes with edge $A \to B$, an independent root $H$, leaves $C, D$ each with parents $\{A, H\}$, and leaves $E, G$ each with parent $\{B\}$. This bitnet has $d = 1 + 2 + 1 + 4 + 4 + 2 + 2 = 16$ parameters and unique chordless four-cycle $A - C - H - D - A$. Its exact average scalar curvature is*
$$
\langle R \rangle = \frac{24}{5} \notin \frac{1}{2}\mathbb{Z}.
$$

*Proof.* The base is the orthogonal product of the edge $A \to B$ and the independent root $H$. Each warp for $C, D$ factors as an $A$-factor times an $H$-factor, so along $A \to B$ we have $(k, l) = (4, 2)$ in Eq. (2) with volume weight $u^2 v$, while along $H$ we have an independent additive summand $S_4(r_H)$ from Eq. (3) of average $2$. Using $\mathrm{Dirichlet}(1/2, 1/2, 1/2, 1/2)$ moments on $A \to B$, $E_0[u^2 v] = 1/168$ and
$$
E_0\!\Bigl[\Bigl(\frac{3}{2} + 26 + 9\Bigr)u^2 v - 5uv - \frac{3}{2}u^2 - 4uw(1 - 2a)(1 - 2b)\Bigr] = \frac{1}{60}.
$$
Hence $\langle R \rangle = 2 + (1/60)/(1/168) = 24/5$. $\square$

---

## 4. Exact Convergent-Series Integral for the Directed Diamond

For the 4-node directed diamond ($A \to B, A \to C, B \to D, C \to D$; $d = 9$ parameters), write $r = P(A=1)$, $b_h = P(B=1 \mid A=h)$, $c_h = P(C=1 \mid A=h)$, $u = r(1-r)$, $p_0 = 1-r$, $p_1 = r$, and $b_{h,1} = b_h$, $b_{h,0} = 1 - b_h$ (similarly for $c$). The four parent-configuration probabilities of $D$ are $q_{ij} = \sum_{h=0}^1 p_h b_{h,i} c_{h,j}$, with product $Q := \prod_{i,j} q_{ij}$. Let $L$ denote expectation over five independent $\mathrm{Beta}(1/2, 1/2)$ coordinates $(r, b_0, b_1, c_0, c_1)$. By Proposition 3 ($n = 1, m = 2$), the fork base $\{A, B, C\}$ has normalized Fisher density $8u\,dL$ and scalar curvature $R_0 = 5 - 1/(2u)$. Applying Lemma 1 with warps $f_s = \sqrt{q_s}$ ($s \in \{0,1\}^2$) yields $\langle R \rangle = N / V$, where
$$
V = L[8u\sqrt{Q}], \qquad
N = L\!\left[8u\sqrt{Q}\left(R_0 - \sum_s \frac{\Delta q_s}{q_s} + \frac{1}{2}\sum_s \frac{\Gamma(q_s, q_s)}{q_s^2} - \frac{1}{2}\sum_{s < t}\frac{\Gamma(q_s, q_t)}{q_s q_t}\right)\right].
$$
Because each $q_s$ is affine in every coordinate and each child derivative contains $p_h$, both $\Gamma(q_s, q_t)$ and
$$
\Delta q_s = \frac{3}{2}(1 - 2r)\partial_r q_s + \sum_{h=0}^1 \sum_{t \in \{b_h, c_h\}} \frac{1 - 2t}{2p_h}\partial_t q_s
$$
are polynomials. Writing $\sqrt{Q} = Q \prod_s q_s^{-1/2}$ and cancelling denominators against $Q$ expresses both integrands as finite sums of polynomials times $\prod_s q_s^{-\alpha_s}$ with $\alpha_s \in \{1/2, 3/2\}$. Expanding $q_s^{-\alpha_s} = \sum_{k \ge 0} \frac{(\alpha_s)_k}{k!}(1 - q_s)^k$ and integrating termwise via $L[t^j] = \binom{2j}{j}/4^j$ yields exact rational multi-index series for $N$ and $V$.

*Absolute convergence.* In any unit Fisher tangent direction, the derivative of $q_s$ is the covariance of $\mathbf{1}_{s}$ with a unit-variance score, so Cauchy–Schwarz gives $\Gamma(q_s, q_s) \le q_s(1 - q_s)$. Thus after multiplying by $\sqrt{Q}$, every diagonal derivative and Laplacian term is bounded by $C q_s^{-1/2}$, every cross term is bounded by a constant, and $8u R_0 = 40u - 4$ is bounded. By AM–GM, $q_{ij} \ge 2\sqrt{u\,b_{0,i} b_{1,i} c_{0,j} c_{1,j}}$, so $q_{ij}^{-1/2}$ introduces at worst a $-1/4$ power at each coordinate endpoint and is $L$-integrable. Because each binomial expansion has nonnegative terms before multiplication by its signed polynomial, Fubini–Tonelli justifies exchanging sums and integrals.

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/bitnet_ricci_quant/explore.py
```

The script [`explore.py`](explore.py) performs exact symbolic verification in SymPy: it checks the Beta and Dirichlet moments for the 5-node tree ($14/11$), 6-node tree ($25/14$), and 7-node 1-loop graph ($24/5$); independently computes scalar curvature from the full 9D and 11D diagonal Fisher metrics at rational interior points to verify Eq. (2); verifies Eq. (3) and exact averages across all seven shared-root colliders; and evaluates the leading and first-order exact rational coefficients of the directed diamond series.

| Bitnet topology | Parameters $d$ | Exact $\langle R \rangle$ | In $\tfrac{1}{2}\mathbb{Z}^+$? |
|---|---:|---:|:---:|
| Five-node tree $(k, l) = (1, 2)$ | 9 | $14/11$ | **No** |
| Six-node tree $(k, l) = (2, 2)$ | 11 | $25/14$ | **No** |
| Seven-node 1-loop graph | 16 | $24/5$ | **No** |
| Two-node chain $\tilde{K}_2$ ($n=1, m=1$) | 3 | $3/2$ | Yes |
| Three-node fork $\tilde{E}_3$ / chain $\tilde{L}_3$ | 5 | $1$ | Yes |
| Two-parent collider $\tilde{C}_3$ ($n=2, m=1$) | 6 | $2$ | Yes |
| Double collider $D_4$ ($n=2, m=2$) | 10 | $4$ | Yes |
| Collapsing stars $\tilde{C}_4, \tilde{C}_5, \tilde{C}_6$ | 11, 20, 37 | $6, 16, 40$ | Yes |

For the directed diamond series, the exact $(0,0,0,0)$ numerator/volume coefficients are $(-5347/2097152,\, 1017/1048576)$, and each of the four unit-index coefficients is $(-394549/536870912,\, 3051/8388608)$. All assertions passed.

---

## References

- C. C. Rodríguez, *The Volume of Bitnets*, AIP Conference Proceedings **735**, 555–564, 2004.
- C. C. Rodríguez, *Quantization of Ricci Curvature in Information Geometry*, [arXiv:2603.10054v1](https://arxiv.org/abs/2603.10054v1), March 2026.
