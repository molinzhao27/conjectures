# Proof of the Exact Recovery Threshold for Multi-View Gaussian Planted Matching

**Target paper:** Ruiqi Wang and Jingtong Hu, *Shared information governs multi-view planted matching*, [arXiv:2611.32892](https://arxiv.org/abs/2611.32892), Conjecture `conj:c1prime` (Exact recovery for Gaussian coordinate laws).  
**Companion files:** [`solutions/multiview_planted_align/explore.py`](explore.py), [`solutions/multiview_planted_align/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:c1prime`: PROVED** for every fixed number of views $K \ge 2$ and every fixed positive-definite coordinate covariance matrix $R \succ 0$.
- **Achievability (Theorem 1 — new proof):** Whenever $\liminf_{n \to \infty} d L(R) / \log n > 1$, ordinary joint maximum likelihood recovers all $K - 1$ relative permutations exactly with probability $1 - o(1)$. The proof proceeds in three self-contained steps:
  1. **Cramér LDP admissibility:** Check the empirical uncentered second-moment KL divergence $D(G_U \| p^{\otimes s}) \le sc$ on every subset $U$ of $s \le M$ rows; the planted alignment passes with probability $1 - o(1)$.
  2. **Deterministic component-size cap:** Superimposing any two admissible alignments forms a $K$-regular bipartite multigraph; greedy weight activation, sequential cell merging with data processing, a strict Hall prefix inequality, Abel summation, and two-sided cancellation force every connected component to have size at most $H = 1 + KT$ rows per side.
  3. **Bounded-component Gaussian swap bound & Bayes transfer:** A Cauchy–Schwarz reduction over active view kernels bounds the error probability of each $m$-row alternative ($2 \le m \le H$) by $e^{-dm I_*/2} \le e^{-dm L(R)}$, so a union bound shows the admissibility-constrained ML decoder succeeds with probability $1 - o(1)$; Bayes optimality under the uniform prior transfers the guarantee directly to unconstrained joint ML.
- **Converse (Wang & Hu, summarized in Section 3):** Whenever $\limsup_{n \to \infty} d L(R) / \log n < 1$, every estimator has exact recovery probability tending to $0$, combining the two-block swap second-moment bound (for $I_*/2$) and the rare-subpopulation tilting argument (for $R_{\mathcal{P}}$ with $|\mathcal{P}| \ge 3$).

---

## 1. Model, Rate Functional, and Universal Partition Inequality

There are $n$ independent objects; object $i \in [n]$ has $d$ i.i.d. coordinate vectors drawn from $p = \mathcal{N}(0, R)$ on $K \ge 2$ scalar views, where $R \in \mathbb{R}^{K \times K}$ is fixed and positive definite ($R \succ 0$). Each view $a \in \{1, \dots, K\}$ permutes the $n$ objects independently and uniformly. Anchoring view $1$ ($\pi_1 = \mathrm{id}$) leaves $K - 1$ independent uniform relative permutations $(\pi_2, \dots, \pi_K)$. Exact recovery requires recovering all $K - 1$ permutations with zero errors. All logarithms are natural.

For any nonempty proper cut $S \subsetneq [K]$, write $I_p(X^S; X^{S^c})$ for the mutual information under $p$, and set $I_* := \min_{\varnothing \neq S \subsetneq [K]} I_p(X^S; X^{S^c})$. For a probability law $G$ on $\mathbb{R}^K$ and a partition $\mathcal{P}$ of $[K]$ into $q = |\mathcal{P}| \ge 2$ blocks, define the multi-information (total correlation) $T_{\mathcal{P}}(G) := D(G \| \bigotimes_{B \in \mathcal{P}} G_B)$ and the partition rate
$$
R_{\mathcal{P}} := \inf_{G} \left[ D(G \| p) + \frac{T_{\mathcal{P}}(G)}{q - 1} \right]
= \frac{1}{q - 1} \inf_{\nu = \bigotimes_{B \in \mathcal{P}} \nu_B} D_{(q-1)/q}(p \| \nu).
$$
The conjectured exact-recovery threshold rate is
$$
L(R) := \min\left\{\frac{I_*}{2},\ \min_{|\mathcal{P}| \ge 3} R_{\mathcal{P}}\right\}
$$
(with an empty minimum equal to $+\infty$ when $K = 2$).

**Conjecture `conj:c1prime` ([arXiv:2611.32892](https://arxiv.org/abs/2611.32892)).** *Let $K \ge 2$ be fixed and $p = \mathcal{N}(0, R)$ with $R \succ 0$. With fixed margins, maximum likelihood recovers all relative permutations exactly with probability $1 - o(1)$ if $\liminf d L(R)/\log n > 1$, and no estimator succeeds with non-vanishing probability if $\limsup d L(R)/\log n < 1$.*

**Lemma 1 (Universal partition inequality).** *For **every** partition $\mathcal{P}$ of $[K]$ with $q = |\mathcal{P}| \ge 2$ and every nonsingular centered Gaussian $G$ on $\mathbb{R}^K$,*
$$
T_{\mathcal{P}}(G) \ge (q - 1)\bigl[L(R) - D(G \| p)\bigr]. \tag{1}
$$

*Proof.* For $q \ge 3$, $R_{\mathcal{P}} \ge L(R)$ by definition, so $D(G\|p) + T_{\mathcal{P}}(G)/(q-1) \ge R_{\mathcal{P}} \ge L(R)$. For $q = 2$, let $\mathcal{P} = \{S, S^c\}$ and view $Z := \sqrt{p}$ as a Hilbert–Schmidt kernel across $S \mid S^c$. By the two-block affinity formula, $R_{\mathcal{P}} = -\log \|Z\|_{\mathrm{op}}^2$ (since $\sup_{\|f\|_2=\|g\|_2=1} \int \sqrt{p}\, f g = \|Z\|_{\mathrm{op}}$ gives $\inf_{\nu_S \otimes \nu_{S^c}} D_{1/2}(p \| \nu_S \otimes \nu_{S^c}) = -2\log \|Z\|_{\mathrm{op}}$). Since $\|Z\|_{\mathrm{op}}^4 \le \operatorname{tr}[(Z^* Z)^2]$ and the Gaussian swap identity (evaluated explicitly in Section 2.3) gives $\operatorname{tr}[(Z^* Z)^2] = e^{-I_p(X^S; X^{S^c})}$, we obtain $R_{\mathcal{P}} \ge I_p(X^S; X^{S^c})/2 \ge I_*/2 \ge L(R)$. Rearranging gives (1) for all $q \ge 2$. $\square$

*(Handling $q = 2$ uniformly with $q \ge 3$ is essential in Section 2.2 because intermediate mergers can coarsen a partition down to two blocks.)*

---

## 2. Achievability Proof ($\liminf d L(R) / \log n > 1$)

Assume $\liminf_{n \to \infty} d L(R) / \log n > 1$ (so $L := L(R) > 0$). Choose a constant $c$ satisfying
$$
\limsup_{n \to \infty} \frac{\log n}{d} < c < L, \qquad \eta := L - c > 0.
$$
Fix an integer $T > 2 + 2c/\eta$, and set the constants $H := 1 + KT$ and $M := KH$.

### 2.1 Step 1: Empirical Second-Moment Admissibility (Cramér LDP)

For any candidate alignment and any subset $U$ of $s \le M$ proposed rows (in a fixed order), stack their $K$ views at each of the $d$ coordinates to obtain vectors $y_1, \dots, y_d \in \mathbb{R}^{sK}$, and let $\widehat{\Sigma}_U := \frac{1}{d}\sum_{\ell=1}^d y_\ell y_\ell^\top$ be the **uncentered** sample second-moment matrix. Let $G_U := \mathcal{N}(0, \widehat{\Sigma}_U)$ whenever $\widehat{\Sigma}_U \succ 0$. Restricting to a subcollection of variables commutes with forming $G_U$ because it extracts the corresponding principal submatrix of $\widehat{\Sigma}_U$. Call a candidate alignment **admissible** if for every set $U$ of $1 \le s \le M$ rows, $\widehat{\Sigma}_U \succ 0$ and
$$
D(G_U \| p^{\otimes s}) \le s c. \tag{2}
$$

**Lemma 2 (Planted admissibility).** *With probability tending to $1$ as $n \to \infty$, the planted alignment is admissible.*

*Proof.* Eventually $d \ge KM$. Fix $1 \le s \le M$, set $h := sK$, and whiten each stacked planted vector by $(I_s \otimes R)^{-1/2}$ so that $z_1, \dots, z_d \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, I_h)$ and $V := \frac{1}{d}\sum_{\ell=1}^d z_\ell z_\ell^\top$ satisfies $D(G_U \| p^{\otimes s}) = I(V) := \frac{1}{2}(\operatorname{tr} V - \log\det V - h)$ for $V \succ 0$ (and $+\infty$ if $V$ is singular). Apply finite-dimensional Cramér's theorem in the space of symmetric $h \times h$ matrices to the i.i.d. outer products $z_\ell z_\ell^\top$. Their log moment generating function against a symmetric matrix $A$ is
$$
\Lambda(A) = \log \mathbb{E}\bigl[\exp(\operatorname{tr}(A z z^\top))\bigr] = -\frac{1}{2}\log\det(I_h - 2A)
\qquad (I_h - 2A \succ 0),
$$
and $+\infty$ otherwise. For $V \succ 0$, the Fenchel–Legendre conjugate $\Lambda^*(V) = \sup_A \{\operatorname{tr}(AV) - \Lambda(A)\}$ is uniquely attained at $2A = I_h - V^{-1}$, giving $\Lambda^*(V) = I(V)$. If $V \succeq 0$ is singular with unit null vector $u$, taking $A = -t u u^\top$ with $t \to +\infty$ yields $\Lambda^*(V) = +\infty$. Thus the failure region $\{V \succeq 0 : I(V) \ge sc\}$ (with $I(V) = +\infty$ on singular matrices) is closed in the positive semidefinite cone and has infimal rate $sc$. By the Cramér upper bound, $\mathbb{P}(I(V) \ge sc) \le \exp(-d(sc - o(1)))$ for each $U$. Union bounding over $\binom{n}{s} \le n^s = \exp(s\log n)$ row sets and $s = 1, \dots, M$ gives total failure probability at most $\sum_{s=1}^M \exp\bigl(-sd(c - \frac{\log n}{d} - o(1))\bigr) \to 0$. $\square$

### 2.2 Step 2: Deterministic Component-Size Cap

**Lemma 3 (Deterministic component bound).** *Let $\pi^{(1)}$ and $\pi^{(2)}$ be any two admissible alignments on the same observed data. In the bipartite multigraph whose left vertices are the rows of $\pi^{(1)}$, whose right vertices are the rows of $\pi^{(2)}$, and whose edges correspond to the $nK$ observed view-rows joining their left and right owners, every connected component has at most $H = 1 + KT$ vertices on each side.*

*Proof.* Every vertex in the bipartite multigraph has degree $K$ (counting multiplicity), and each view $a \in [K]$ forms a perfect matching. Hence every connected component has equal left and right sizes $m$. For any row vertex $v$ on either side, set
$$
D_v := D(G_v \| p), \qquad b_v := c - D_v \in [0, c],
$$
where $b_v \ge 0$ by admissibility (2) for $s = 1$. Notice that for any set of physical view-rows, the empirical second-moment law depends only on the variables themselves, not on which alignment supplies them.

Suppose for contradiction that some connected component has $m > H$ rows on each side. Activate its $2m$ vertices one by one in decreasing order of $b_v$ (breaking ties arbitrarily), stopping at the first step when some connected component $C$ of the induced active subgraph reaches $|C| \ge T$. Since the last activated vertex has degree $K$, it merges at most $K$ earlier active components each of size at most $T - 1$, so
$$
T \le |C| \le 1 + K(T - 1) \le H.
$$
Let $A$ and $B$ be the left and right vertex sets of $C$, and let $\lambda \in [0, c]$ be the weight of the last activated vertex. Because $T \ge 2$ and the graph is bipartite, both $A$ and $B$ are nonempty. Every vertex in $C$ has weight $b_v \ge \lambda$, while every neighbor of $C$ outside $C$ is inactive and thus has weight $b_w \le \lambda$. Let $N(B)$ denote the full left neighborhood of $B$ in the component, and put $s := |N(B)| \le K|B| \le KH = M$. We have $A \subseteq N(B)$, and viewing $B$ as hyperedges on $N(B)$ connects $N(B)$ because $C$ is connected.

By admissibility (2) applied to the left row set $N(B)$ ($s \le M$), the total correlation of the $s$ left rows under $G_{N(B)}$ satisfies
$$
\mathcal{T} := D\!\left(G_{N(B)} \,\middle\|\, \bigotimes_{u \in N(B)} G_u\right)
= D(G_{N(B)} \| p^{\otimes s}) - \sum_{u \in N(B)} D_u
\le \sum_{u \in N(B)} (c - D_u) = \sum_{u \in N(B)} b_u. \tag{3}
$$
Now start with the $s$ singleton cells $\{\{u\} : u \in N(B)\}$ and process the right vertices $v \in B$ in decreasing order of $b_v$. At vertex $v$, merge all current cells that intersect $N(v)$, and let $k_v \ge 0$ be the decrease in the number of cells (so $k_v + 1$ cells merge into one). Since the hyperedges $B$ connect $N(B)$, all $s$ singletons merge into one final cell $N(B)$, so $\sum_{v \in B} k_v = s - 1$. By telescoping the differential entropies of the intermediate merged cells, $\mathcal{T}$ equals the sum over $v \in B$ of the total correlation among the $k_v + 1$ full-row cells merged at $v$. Projecting each of those $k_v + 1$ cells onto the subset of views it supplies to $v$ induces a partition $\mathcal{P}_v$ of $[K]$ into $k_v + 1$ nonempty blocks. By the data-processing inequality for KL divergence and Lemma 1 (when $k_v \ge 1$; trivially $0 = 0$ when $k_v = 0$), the merger at $v$ contributes at least
$$
T_{\mathcal{P}_v}(G_v) \ge k_v\bigl(L - D_v\bigr) = k_v(b_v + \eta).
$$
Summing over $v \in B$ yields
$$
\mathcal{T} \ge \sum_{v \in B} k_v(b_v + \eta), \qquad \sum_{v \in B} k_v = s - 1. \tag{4}
$$

*Strict Hall prefix inequality.* Let $F \subseteq B$ be any nonempty prefix of the processing order. After processing $F$, the cells are the connected components of the bipartite incidence graph on $F \cup N(F)$, say $j(F)$ of them, so $\sum_{v \in F} k_v = |N(F)| - j(F)$. For any nonempty proper subset $E$ of right vertices of the original $m$-by-$m$ connected component, its $K|E|$ incident edges enter $N(E)$, which has $K|N(E)|$ edge slots; equality $|N(E)| = |E|$ would fill all slots and disconnect $E \cup N(E)$ from the rest of the component. Thus $|N(E)| \ge |E| + 1$. Because $|B| \le H < m$, each of the $j(F)$ connected components of $F \cup N(F)$ has a nonempty proper right vertex set $E_r$, giving $|N(E_r)| \ge |E_r| + 1$. Summing over $r = 1, \dots, j(F)$ yields
$$
\sum_{v \in F} k_v = |N(F)| - j(F) \ge |F| \qquad (\forall\, \text{prefix } F \subseteq B). \tag{5}
$$

*Abel summation.* Write $b_v = \lambda + \int_\lambda^\infty \mathbf{1}_{b_v > t}\,dt$ for every $v \in B$ (as $b_v \ge \lambda$). Applying (4) and (5) to each threshold prefix $F_t := \{v \in B : b_v > t\}$ gives
$$
\mathcal{T}
\ge (s - 1)(\lambda + \eta) + \int_\lambda^\infty \sum_{v \in F_t} k_v\,dt
\ge (s - 1)(\lambda + \eta) + \int_\lambda^\infty |F_t|\,dt
= (s - 1)(\lambda + \eta) + \sum_{v \in B} (b_v - \lambda). \tag{6}
$$
On the other hand, every $u \in N(B) \setminus A$ lies outside $C$ and therefore has $b_u \le \lambda$, so the right-hand side of (3) is at most $s\lambda + \sum_{u \in A} (b_u - \lambda)$. Combining (3) and (6) yields
$$
(s - 1)\eta \le \lambda + \sum_{u \in A} (b_u - \lambda) - \sum_{v \in B} (b_v - \lambda).
$$
Running the identical argument from the right alignment (which is also admissible) with $r := |N(A)| \le M$ swaps the roles of $A$ and $B$, reversing the signs of the two sums. Adding the two inequalities cancels the sums completely:
$$
(s + r - 2)\eta \le 2\lambda \le 2c.
$$
Yet $s + r = |N(B)| + |N(A)| \ge |A| + |B| = |C| \ge T > 2 + 2c/\eta$, a contradiction. Therefore $m \le H$. $\square$

### 2.3 Step 3: Bounded-Component Gaussian Swap Bound and Bayes Transfer

**Theorem 1 (Achievability for `conj:c1prime`).** *If $\liminf_{n \to \infty} d L(R) / \log n > 1$, then ordinary joint maximum likelihood recovers all $K - 1$ relative permutations exactly with probability $1 - o(1)$.*

*Proof.* Consider the constrained decoder $\widehat{\pi}_{\mathrm{adm}}$ that maximizes the joint Gaussian log-likelihood over admissible alignments (falling back arbitrarily if none exist). On the high-probability event that the planted alignment $\pi^*$ is admissible (Lemma 2), an error of $\widehat{\pi}_{\mathrm{adm}}$ implies the existence of an admissible alignment $\pi \neq \pi^*$ with likelihood ratio $\ge 1$. Because the joint likelihood ratio factors across the connected components of the superposition multigraph of $(\pi^*, \pi)$, at least one nontrivial component ($m \ge 2$) has likelihood ratio $\ge 1$. By Lemma 3, its size satisfies $2 \le m \le H$. Replacing $\pi^*$ by $\pi$ on that single component yields an alternative $\tilde{\pi}$ differing from $\pi^*$ on $m \in \{2, \dots, H\}$ rows whose likelihood ratio is $\ge 1$.

Fix one such alternative on $m$ moved rows; relabel so the planted alignment is the identity and the moved rows are $1, \dots, m$, with candidate row $i$ equal to $y_i = (x_{\pi_a(i), a})_{a=1}^K$ ($\pi_1 = \mathrm{id}$). For each moved row $i$, let $S_i := \{a \in [K] : \pi_a(i) = i\}$; since view 1 is anchored and row $i$ is moved, $\varnothing \neq S_i \subsetneq [K]$. For a single coordinate ($d = 1$), the Chernoff midpoint ($t = 1/2$) affinity is
$$
\rho(\tilde{\pi}) := \int \prod_{i=1}^m \sqrt{p(x_i)\, p(y_i)}\, dx.
$$
Integrate out the fixed variables $x_{i, S_i}$ inside each row $i$: writing $Z_i(x_{S_i}, x_{S_i^c}) := \sqrt{p(x_{S_i}, x_{S_i^c})}$, the marginal over $x_{i, S_i}$ produces the positive semidefinite kernel $H_i(x_{i, S_i^c}, y_{i, S_i^c}) := (Z_i^* Z_i)(x_{i, S_i^c}, y_{i, S_i^c})$. Each remaining variable $x_{i, a}$ ($a \notin S_i$) appears in exactly two distinct kernels ($H_i$ and $H_{\pi_a^{-1}(i)}$). Integrating out those variables sequentially via Cauchy–Schwarz bounds $\rho(\tilde{\pi}) \le \prod_{i=1}^m \|H_i\|_2$ (where $\|H_i\|_2 = (\operatorname{tr}(H_i^2))^{1/2}$ is the Hilbert–Schmidt norm, preserved at every step by marginalizing squared norms).

Now $\|H_i\|_2^2 = \operatorname{tr}[(Z_i^* Z_i)^2]$ is the two-row swap affinity across the cut $S_i \mid S_i^c$, i.e., the $t = 1/2$ affinity between $(X_1, X_2) \sim \mathcal{N}(0, R \oplus R)$ and $(X_1', X_2')$ obtained by swapping $S_i^c$ between the two rows. Applying the orthogonal change of coordinates $(X_1 \pm X_2)/\sqrt{2}$, the sum block has covariance $R$ under both laws, while in the difference block swapping $S_i^c$ negates the cross-covariance $R_{S_i, S_i^c}$. The average of the two difference covariances is therefore $R_{S_i S_i} \oplus R_{S_i^c S_i^c}$, so the Gaussian Bhattacharyya formula gives
$$
-\log \|H_i\|_2^2
= \frac{1}{2}\log \frac{\det(R_{S_i S_i})\det(R_{S_i^c S_i^c})}{\det R}
= I_p(X^{S_i}; X^{S_i^c}) \ge I_*.
$$
Hence $\rho(\tilde{\pi}) \le e^{-m I_*/2}$. Over $d$ independent coordinates, Markov's inequality on the square-root likelihood ratio gives
$$
\mathbb{P}\!\left(\frac{\mathcal{L}(\tilde{\pi})}{\mathcal{L}(\pi^*)} \ge 1\right) \le \rho(\tilde{\pi})^d \le e^{-d m I_*/2} \le e^{-d m L}.
$$
For each $2 \le m \le H$, there are at most $\binom{n}{m}(m!)^{K-1} \le n^m (m!)^{K-1}$ alternatives on $m$ moved rows. Because $H$ is a constant and $d L - \log n \ge \eta d$ grows linearly in $d$,
$$
\mathbb{P}(\widehat{\pi}_{\mathrm{adm}} \neq \pi^*)
\le \mathbb{P}(\pi^* \text{ inadmissible}) + \sum_{m=2}^H (m!)^{K-1} \exp\bigl(-m(d L - \log n)\bigr) \longrightarrow 0.
$$
Finally, because the prior over all $(n!)^{K-1}$ relative alignments is uniform, ordinary unconstrained joint maximum likelihood is the Bayes-optimal estimator for 0–1 exact-recovery loss and therefore achieves $\mathbb{P}(\widehat{\pi}_{\mathrm{ML}} = \pi^*) \ge \mathbb{P}(\widehat{\pi}_{\mathrm{adm}} = \pi^*) \to 1$. $\square$

---

## 3. Converse (Wang & Hu)

For completeness, we recall the matching strong converse from [arXiv:2611.32892](https://arxiv.org/abs/2611.32892) when $\limsup_{n \to \infty} d L(R) / \log n < 1$:

- **Bipartition regime ($d I_* \le (2 - \epsilon)\log n$):** Reveal all within-side correspondences across a minimizing cut $S \mid S^c$, reducing the problem to aligning two Gaussian databases with total mutual information $I = d I_*$. For $I > 0$, a single transposition strictly improves the likelihood with probability at least $\frac{3}{8} e^{-I - \sqrt{8I}}$, while two transpositions sharing a row both improve it with probability at most $e^{-3I/2}$; disjoint swaps are independent, so the count $W$ of winning swaps satisfies $\operatorname{Var}(W)/(\mathbb{E} W)^2 = O(n^{-2} e^{I + O(\sqrt{I})} + n^{-1} e^{I/2 + O(\sqrt{I})}) = o(1)$, driving exact recovery probability to $0$. (If $I = 0$, the remaining permutation is independent of the data.)
- **Multi-block regime ($d R_{\mathcal{P}} \le (1 - \epsilon)\log n$ for some $|\mathcal{P}| = q \ge 3$):** Take an optimizing tilt $Q$ with $D = D(Q\|p)$, $V = \bigotimes_{B \in \mathcal{P}} Q_B$, $T' = D(Q\|V)$, and $R_{\mathcal{P}} = D + T'/(q-1)$. Subsample rows in the $Q^{\otimes d}$-typical event (where log density ratios against $p^{\otimes d}, V^{\otimes d}$ are within $du$ of $dD, dT'$) with rejection probability $e^{-d(D+u)} dQ^{\otimes d}/dp^{\otimes d}$. With high probability the marked subpopulation size $m$ satisfies $\log m = \log n - d(D+u) + O(1)$ and grows polynomially. Revealing the marked indices, all unmarked correspondences, and all within-block permutations leaves $q - 1$ uniform bijections on $m$ items. Under the product reference $V^{\otimes d}$, exact guessing has probability $(m!)^{-(q-1)}$; changing measure back to $Q^{\otimes d}$ costs at most $\exp(m(d(T'+u)+o(1)))$, yielding log success probability at most $m[(q-1)(dR_{\mathcal{P}} - \log n) + qdu + O(1)] \to -\infty$ for small $u > 0$. (For $R_{\mathcal{P}} > 0$, reveal extra coordinates up to $\lfloor\sqrt{\log n}\rfloor$ if needed for typicality; at $R_{\mathcal{P}} = 0$ the views factor across $\mathcal{P}$ and are unobservable.)

---

## 4. Verification and Reproduction

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -B solutions/multiview_planted_align/explore.py
```

With default seed `261132892` and `--max-support 4`, all assertions passed:

- **Finite Chernoff census:** Enumerated **15,212** simultaneous-conjugacy classes representing **341,638** labeled full-support alternatives for $K \in \{3, 4, 5\}$ and $m \in \{2, 3, 4\}$ (matching inclusion–exclusion counts exactly), across 6 covariance models per $K$ (chain, positive equicorrelation, negative equicorrelation, near-boundary negative equicorrelation, and two seeded random correlation matrices): **91,272** Chernoff/midpoint evaluations plus 18 fully split cyclic checks. In every model, the minimum Chernoff exponent per moved row matched $I_*/2$ to within $10^{-9}$ (largest deviation of the optimal tilt from $1/2$: `0.01029218` at $K = 5$).
- **Partition-rate consistency:** All **414** nontrivial partition-rate fixed points matched the KL variational objective to $10^{-8}$. For $K = 3$ with $r = -1/3$, $L = 0.098353188$ vs. $I_*/2 = 0.101366277$ (matching the paper to $10^{-10}$); for $K = 4$ with $r = -1/4$, $L = 0.101510479$ vs. $I_*/2 = 0.117500907$ ($1/L \in (9.8511971, 9.8512024)$).
- **Combinatorial Lemma 3 checks:** **5,728** deduplicated proper subsets of small connected incidence graphs satisfied the prefix-rank bound (5), and **3,000** seeded weighted graphs ($m = 90$ per side) passed both the Abel-summed merger inequality (6) and the two-sided budget contradiction.
- **Exact joint ML simulations** (80 trials per dimension at $d \approx 0.6, 1.0, 1.4 \times \log n / L$, nested coordinates; exhaustive ML on dense models and exact edge-wise linear sum assignment on tridiagonal-precision chains):

| Views $K$ | Coordinate law | Objects $n$ | Dimensions $d$ | Exact ML successes (of 80) |
|---:|---|---:|---|---|
| 3 | Negative equicorrelation | 6 | 11, 18, 26 | 28, 63, 75 |
| 3 | Random 1 | 6 | 51, 84, 118 | 53, 74, 79 |
| 3 | Chain | 128 | 25, 42, 58 | 0, 75, 80 |
| 4 | Negative equicorrelation | 4 | 8, 14, 19 | 37, 67, 80 |
| 4 | Random 1 | 4 | 12, 20, 27 | 60, 77, 79 |
| 4 | Chain | 128 | 37, 62, 86 | 2, 79, 80 |
| 5 | Negative equicorrelation | 4 | 8, 14, 20 | 22, 63, 77 |
| 5 | Random 1 | 4 | 5, 9, 12 | 48, 74, 79 |
| 5 | Chain | 128 | 57, 95, 133 | 3, 75, 80 |
