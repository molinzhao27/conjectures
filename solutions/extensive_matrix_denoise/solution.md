# Resolution of Gaussian Equivalence for Extensive-Rank Matrix Denoising

**Target paper:** Zhuohang He, Qun Chen, Junjie Ma, Haochuan Zhang, and Xiaojun Yuan, *Extensive-Rank Matrix Denoising Under i.i.d. Non-Gaussian Noise*, [arXiv:2611.29179](https://arxiv.org/abs/2611.29179), Section III-C, Conjecture 1 (`Conj:Effective`).  
**Companion files:** [`solutions/extensive_matrix_denoise/explore.py`](explore.py), [`solutions/extensive_matrix_denoise/proof.pdf`](proof.pdf).

---

## Verdict

1. **Conjecture 1 (`Conj:Effective`) under spectral-integral SE and Appendix A bounded RIE: PROVED.**
   With the bounded continuous extension of the rotationally invariant estimator (RIE) from Appendix A of arXiv:2611.29179 and the **spectral-integral** state evolution (SE) recursion, both conclusions (i) and (ii) of Conjecture 1 hold in probability at every **fixed iteration $t \ge 1$** with positive population input variances ($\tau_{\mathsf{L}}^t, \tau_{\mathsf{S}}^t > 0$). In the independent bi-orthogonally invariant model with a compactly supported limiting spectral law $\nu_L$ of positive first moment $p_{\mathsf{L}} > 0$, the sharp signal condition is convergence of normalized signal power $\|L\|_F^2/(mn) \to p_{\mathsf{L}}$ in probability (implied by Section II's almost-sure operator-norm bound, and necessary at $t=1$). Only a finite second moment $0 < p_{\mathsf{S}} < \infty$ of the entrywise noise prior is required (including heavy-tailed priors with infinite fourth moment).

2. **Unconditional expected-risk calibration without uniform integrability: DISPROVED (sharp equivalence proved).**
   The paper's defining equation for $\mathsf{m}_{\mathsf{L}}(\tau)$ identifies the spectral integral with $\lim_{n\to\infty}(mn)^{-1}\mathbb{E}\|L - \operatorname{RIE}_{\beta,\tau}(L+\sqrt{\tau}G)\|_F^2$. Under the stated almost-sure assumptions alone, these two quantities need not coincide. For any bounded RIE extension, **uniform integrability of normalized signal power** $Q_n := (mn)^{-1}\|L\|_F^2$ is both necessary and sufficient for the expected-risk limit to match the spectral integral. Without uniform integrability, an explicit full-rank bi-orthogonally invariant mixture counterexample ($\delta = 0.05$) violates part (ii)'s expected-risk target at iteration 1 (empirical risk $0.449600$ vs. expected-risk target $0.499600$) and violates part (i)'s entrywise $W_2$ Gaussian limit at iteration 2 (normalized cross moment $\langle S, R_{\mathsf{S}}^2 - S\rangle_n \to -kw \approx 0.128737 \neq 0$).

3. **Off-support literal Hilbert-pole criterion ($\beta < 1$):**
   For $\beta < 1$, continuing the main-text unregularized Hilbert formula literally below the spectral gap introduces a $-\tau(1-\beta)/s$ pole (in native $\sqrt{n}$ units). On a given spectral input, vanishing empirical inverse-square mass near zero, $m^{-1}\sum_{s_i < \eta} s_i^{-2} \to 0$ in probability (for fixed $\eta > 0$ below the lower support edge, counting $s_i = 0$ as $+\infty$), is both necessary and sufficient for per-entry norm equivalence to the Appendix A bounded extension, and necessary for matching the finite spectral risk limit.

4. **Scope:**
   The theorem governs **fixed iterations $t$**, fixed limiting priors, and **population SE parameters**. Letting $t \to \infty$ jointly with $n$ or substituting data-estimated (e.g., SURE) parameters lies outside this proof and requires separate analysis.

---

## 1. Model, Calibration, and Statement of Conjecture 1

Let $Y = L + S \in \mathbb{R}^{m \times n}$ with $m \le n$ and $m/n \to \beta \in (0, 1]$. The matrices $L$ and $S$ are independent. The signal $L$ is bi-orthogonally invariant (its law is invariant under left and right multiplication by deterministic orthogonal matrices), and the entries of $S$ are i.i.d. from a fixed centered prior $P_{\mathsf{S}}$ with variance $0 < p_{\mathsf{S}} < \infty$. Section II of arXiv:2611.29179 assumes almost-sure weak convergence of $\nu_{L,n} := m^{-1}\sum_{i=1}^m \delta_{\lambda_i(LL^\top/n)}$ to a deterministic measure $\nu_L$ on $\mathbb{R}_+$ and almost-sure finiteness of $\limsup_{n\to\infty}\|L\|_{\mathrm{op}}/\sqrt{n}$, with $p_{\mathsf{L}} := \int x\,\nu_L(dx) > 0$.

For any variance $v > 0$:
- **Entrywise module:** Let $\mathsf{R}_v := \mathsf{S} + \sqrt{v}\,\mathsf{Z}$ with independent $\mathsf{S} \sim P_{\mathsf{S}}$ and $\mathsf{Z} \sim \mathcal{N}(0,1)$. Take the Gaussian-convolution version of the posterior mean $F_{\mathsf{S}}(r, v) := \mathbb{E}[\mathsf{S} \mid \mathsf{R}_v = r]$, with risk $m_{\mathsf{S}}(v) := \mathbb{E}[(\mathsf{S} - F_{\mathsf{S}}(\mathsf{R}_v, v))^2]$.
- **Spectral module:** Let $\bm{\mathsf{R}}_v := L + \sqrt{v}\,G$ with $G \in \mathbb{R}^{m\times n}$ having i.i.d. $\mathcal{N}(0,1)$ entries independent of $L$. In the paper's $p_{\mathsf{L}}$-normalized units $s = \sigma / \sqrt{np_{\mathsf{L}}}$, let $\mu_v$ be the limiting singular-value distribution of $\bm{\mathsf{R}}_v/\sqrt{np_{\mathsf{L}}}$ with density $\rho_v$, symmetrization $\widehat{\mu}_v$, and principal-value Hilbert transform $\mathcal{H}_v(s) = \operatorname{p.v.}\int_{\mathbb{R}} (s-x)^{-1}\,d\widehat{\mu}_v(x)$. On the positive support,
  $$
  f_v(s) = s - \frac{v(1-\beta)}{p_{\mathsf{L}} s} - \frac{2\beta v}{p_{\mathsf{L}}}\mathcal{H}_v(s).
  $$
  Following Appendix A of arXiv:2611.29179, extend $f_v$ to a fixed bounded continuous function on $[0,\infty)$ with $f_v(0) = 0$, and apply $F_{\mathsf{L}}(R, v) := \sqrt{np_{\mathsf{L}}}\,U\operatorname{diag}(f_v(s_1),\dots,f_v(s_m))V^\top$ to the thin SVD $R = U\operatorname{diag}(\sigma_1,\dots,\sigma_m)V^\top$. Define the **spectral-integral** risk calibration
  $$
  m_{\mathsf{L}}^{\mathrm{int}}(v)
  := v - \frac{v^2}{p_{\mathsf{L}}} \int_{\mathbb{R}_+}
     \left[\frac{(1-\beta)^2}{s^2} + \frac{\pi^2\beta^2}{3}\rho_v(s)^2\right]\mu_v(ds).
  $$
- **OAMP recursion and population SE:** For $D \in \{\mathsf{S}, \mathsf{L}\}$, set $a_D(v) := m_D(v)/v$ and $\Phi_D(v) := m_D(v)/(1 - a_D(v))$. Starting from $L^1 := 0$ and $v_1 := p_{\mathsf{L}}$, define recursively $w_t := \Phi_{\mathsf{S}}(v_t)$, $v_{t+1} := \Phi_{\mathsf{L}}(w_t)$, and
  $$
  R_{\mathsf{S}}^t = Y - L^t,\quad S^t = \frac{F_{\mathsf{S}}(R_{\mathsf{S}}^t, v_t) - a_{\mathsf{S}}(v_t)R_{\mathsf{S}}^t}{1 - a_{\mathsf{S}}(v_t)},
  \qquad
  R_{\mathsf{L}}^t = Y - S^t,\quad L^{t+1} = \frac{F_{\mathsf{L}}(R_{\mathsf{L}}^t, w_t) - a_{\mathsf{L}}(w_t)R_{\mathsf{L}}^t}{1 - a_{\mathsf{L}}(w_t)}.
  $$

**Conjecture 1 (`Conj:Effective`, arXiv:2611.29179).** For each fixed $t \ge 1$ with $v_t, w_t > 0$, as $m, n \to \infty$ with $m/n \to \beta$, in probability:
- **(i)** The three-dimensional empirical distribution $(mn)^{-1}\sum_{i,j}\delta_{(S_{ij}, R_{\mathsf{S},ij}^t, S_{ij}^t)}$ converges in $W_2$ to $\mathcal{L}(\mathsf{S}, \mathsf{R}_{v_t}, \mathsf{S}^t)$, where $\mathsf{S}^t = (F_{\mathsf{S}}(\mathsf{R}_{v_t}, v_t) - a_{\mathsf{S}}(v_t)\mathsf{R}_{v_t})/(1 - a_{\mathsf{S}}(v_t))$.
- **(ii)** With $\mu_n(M) := m^{-1}\sum_{i=1}^m \delta_{\sigma_i(M)/\sqrt{n}}$, we have $W_2(\mu_n(R_{\mathsf{L}}^t), \mu_n(\bm{\mathsf{R}}_{w_t})) \to 0$ and $(mn)^{-1}\|L - F_{\mathsf{L}}(R_{\mathsf{L}}^t, w_t)\|_F^2 \to m_{\mathsf{L}}(w_t)$.

Throughout, write $|X|_n := \|X\|_F / \sqrt{mn}$ and $\langle X, Z\rangle_n := \operatorname{tr}(X^\top Z)/(mn)$. All convergence of random empirical quantities below is in probability unless stated otherwise.

---

## 2. Proof of Conjecture 1 Under Spectral-Integral Calibration

**Theorem 1 (Gaussian equivalence at fixed iterations).** *Assume $L$ and $S$ are independent, $L$ is bi-orthogonally invariant, $S$ has i.i.d. centered entries with $0 < p_{\mathsf{S}} < \infty$, and $\mu_n(L)$ converges in $W_2$ in probability to the square-root pushforward of a compactly supported deterministic probability measure $\nu_L$ with $p_{\mathsf{L}} = \int x\,\nu_L(dx) > 0$ (equivalently, $\nu_{L,n} \Rightarrow \nu_L$ and $|L|_n^2 \to p_{\mathsf{L}}$ in probability). With the Appendix A bounded continuous RIE extension and population spectral-integral SE calibration $m_{\mathsf{L}} = m_{\mathsf{L}}^{\mathrm{int}}$, both conclusions (i) and (ii) of Conjecture 1 hold in probability for every fixed $t \ge 1$ with $v_t, w_t > 0$. Conversely, $|L|_n^2 \to p_{\mathsf{L}}$ in probability is necessary for (i) at $t = 1$ (by applying $(s,r,e)\mapsto (r-s)^2$).*

The proof constructs a graph-indexed matrix Hilbert space with simultaneous entrywise and spectral descriptions, related in spirit to traffic-freeness calculus (Male, 2020) and tensor-network moment methods for AMP (Wang–Zhong–Fan, 2024), while keeping all graph sizes and polynomial degrees fixed before taking $n \to \infty$.

### 2.1 Graph-indexed matrix Hilbert space

Start with an i.i.d. array $(A_{ij}, S_{ij})_{1\le i\le m,\, 1\le j\le n}$, where $A_{ij} \sim \mathcal{N}(0,1)$ is independent of $S_{ij}$. Fix a real orthonormal basis of centered functions in $L^2(A, S)$ whose elements have all moments finite and which includes the coordinate function $A$ (for instance, products of normalized Hermite polynomials in $A$ with a bounded orthonormal basis of $L^2(P_{\mathsf{S}})$ containing $1$, omitting the constant product; the bounded basis in $S$ is obtained by Gram–Schmidt on simple functions from a countable generating algebra).

Define finite simple bipartite graphs with two distinguished terminals (a left terminal and a right terminal) and edges labeled by basis functions, generated recursively from a single labeled edge:
- A **string** joins an odd number $k \ge 3$ of blocks end-to-end, reversing orientation on every second block, with all other vertices distinct.
- A **block** is either a single labeled edge, or at least two parallel branches joining the two terminals with pairwise disjoint interiors, where each branch is a string except that at most one branch may be a single edge.

Every string has no direct terminal edge and has a connected interior (after removing both terminals). Every multi-branch block is 2-vertex-connected, and deleting either terminal leaves a connected graph. Hence both decompositions are unique: cutting a string at its articulation vertices recovers its ordered blocks, and removing both terminals of a parallel block recovers its string branches together with any direct edge.

For an isomorphism class $H$ (preserving terminals, vertex bipartition, and edge labels) with $l$ internal left vertices and $r$ internal right vertices, define
$$
X_H(i, j) := \frac{1}{m^{l/2} n^{r/2} \sqrt{\operatorname{aut}(H)}} \sum_{\phi} \prod_{e \in E(H)} \psi_e\bigl(A_{\phi(e)}, S_{\phi(e)}\bigr),
$$
where $\phi$ ranges over color-preserving embeddings fixing the two terminals at $(i, j)$ that are injective within each color class. Distinct classes $H$ are orthogonal at fixed $(i,j)$ and satisfy $\mathbb{E}[X_H(i,j)^2] \to 1$, since nonzero second-moment terms require identical embedded edge sets with matching labels, yielding $\operatorname{aut}(H)$ embeddings.

Let $\mathcal{H}_{\mathrm{gr}}$ be the abstract Hilbert space with orthonormal basis $\{H\}$. A sequence of $m \times n$ matrices $M_n$ **represents** $h \in \mathcal{H}_{\mathrm{gr}}$ if for finite basis sums $h_j \to h$ in $\mathcal{H}_{\mathrm{gr}}$, $\lim_{j\to\infty}\limsup_{n\to\infty}\mathbb{P}(|M_n - X_{h_j}|_n > \varepsilon) = 0$ for every $\varepsilon > 0$. By the empirical isometry proved next, this depends only on $h$, and no graph size or polynomial degree grows with $n$.

### 2.2 Entrywise Gaussian and Wick–Hermite limit

**Lemma 1 (Entrywise description).** *At any fixed entry and empirically in probability, the string matrices $(X_H)_{H \text{ string}}$ converge to jointly independent standard normals $(Z_H)$ independent of $(A,S)$. A parallel block with direct edge label $\psi$ (or $1$ if absent) and $k_H$ parallel copies of each string type $H$ represents*
$$
\psi(A, S) \prod_H \frac{\operatorname{He}_{k_H}(Z_H)}{\sqrt{k_H!}},
$$
*where $\operatorname{He}_k$ are the probabilists' Hermite polynomials. Consequently, $\mathcal{H}_{\mathrm{gr}}$ is empirically isometric to the centered subspace of $L^2(A, S, (Z_H)_H)$, and the closed subspace $\mathcal{G} \subset \mathcal{H}_{\mathrm{gr}}$ spanned by the single edge $A$ and all strings consists of centered Gaussian coordinates jointly Gaussian with $A$ and independent of $S$.*

*Proof sketch.* Expand a mixed moment of strings with $d$ total internal vertex occurrences (normalization order $n^{-d/2}$). Each distinct internal vertex in a nonzero term must occur at least twice (since incident edges must repeat and each embedding is injective), so terms with $< d/2$ distinct internal vertices vanish. At equality, every internal vertex has a unique partner in another factor; across an internal edge, both endpoints must match into the same partner factor. Connectedness of each string interior propagates this matching to the entire string, producing pairings of isomorphic strings with matching edge labels and no other identifications—precisely the Gaussian moment formula. Since strings contain no direct terminal edge, they are asymptotically independent of $(A,S)$. For two entries $(i,j)$ and $(i',j')$ with disjoint terminals, excluding the other pair's terminals from each sum incurs $O(n^{-1})$ squared $L^2$ error, decoupling the two entries; row- or column-sharing pairs form a vanishing $O(n^{-1})$ fraction, giving empirical convergence in probability and uniform moment bounds at higher even orders. For a parallel block, injectivity forbids within-block contractions, leaving precisely the Wick pairings (with the $k_H!$ permutation automorphisms matching the Hermite normalization). Extending from finite sums to all of $\mathcal{H}_{\mathrm{gr}}$ by $|\cdot|_n$-approximation yields empirical isometry and weak/second-moment convergence. $\square$

### 2.3 Spectral trace moments and odd open-word expansion

**Lemma 2 (Spectral description).** *For any finite list of distinct blocks $B_1, \dots, B_p$, the normalized matrices $X_{B_1}/\sqrt{n}, \dots, X_{B_p}/\sqrt{n}$ have the same limiting normalized mixed trace moments (in expectation and in probability) as independent rectangular standard Gaussian matrices divided by $\sqrt{n}$. Moreover, every odd-length open word $W_n$ in these blocks (scaled by $n^{-(k-1)/2}$) is represented in $\mathcal{H}_{\mathrm{gr}}$ by a finite linear combination of length-1 blocks and strings built from these blocks, whose coefficients on length-1 blocks equal the corresponding Gaussian mixed inner products.*

*Proof sketch.* Expand a normalized cyclic trace of $q$ block factors. For a fixed vertex-identification pattern whose union has $V$ vertices, where factor $i$ has $v_i$ vertices (terminals included), the leading power of $n$ is $n^{V - 1 - \frac{1}{2}\sum_{i=1}^q (v_i - 1)}$. Every edge of the union must be covered at least twice for a nonzero expectation. For any spanning tree $T$ of the union, factor $i$ covers at most $v_i - 1$ edges of $T$ (since its restriction is a forest), so $2(V - 1) \le \sum_i (v_i - 1)$. Equality holds only if every edge of every spanning tree—hence every edge of the union—is covered exactly twice, and no external path connects two distinct vertices of a single factor (otherwise extending that path to a spanning tree would disconnect the restriction to that factor). Thus each factor is a whole biconnected component (or bridge in the single-edge case), traversed twice in opposite directions between the same terminals, with all other components attaching only at those terminals. Collapsing each paired block to one edge yields a double-walked tree—identical to the leading diagrams for independent rectangular Gaussians, with internal vertex dimensions canceling the normalization and root-preserving isomorphisms canceling $\operatorname{aut}(B)$. For the trace variance, diagrams joining the two cycles lose at least one additional power of $n$, giving concentration in probability.

For an odd open word of length $k$ scaled by $n^{-(k-1)/2}$, define a *restricted word* by enforcing global injectivity across all internal vertices of the string (yielding the block itself for $k=1$, and $(m/n)^{(k-1)/4}$ times the string matrix for $k \ge 3$). Approximate the unrestricted word by summing over all noncrossing partial pairings of positions with matching block labels, opposite traversal directions, and no unpaired position inside a pair, weighting each deleted pair by $d/n$ where $d \in \{m, n\}$ is the dimension at the end of its first traversal. Closing one open word against the reverse of another and applying the spanning-tree count decomposes each leading closed tree into a shared simple path (the restricted word) plus noncrossing closed excursions inside each half, matching the three terms of the expanded squared $|\cdot|_n$-difference with coefficients $1, -2, 1$. $\square$

### 2.4 Signal coupling, spectral transfer, and entrywise continuity

1. **Signal coupling:** Couple $L$ to the thin SVD of $A$ by assigning $L$ the same left and right singular vectors as $A$ (using independent Haar factors if needed for redundant signs) and inserting an independent copy of the ordered singular values of $L$, keeping $S$ independent. By bi-orthogonality, this has the exact joint law of $(L, S)$. Let $h$ be the quantile map from the continuous limiting singular law of $A/\sqrt{n}$ to $\nu_L^{1/2}$ (bounded at endpoints). Because the singular values are ordered with matching weak limit and convergent second moments, $|L - h(A)|_n^2 = W_2^2(\mu_n(L), \mu_n(h(A))) \to 0$ in probability. Approximating $h$ by odd polynomials in $L^2$ of the Gaussian singular law and applying Lemma 2 places $L \in \mathcal{G}$ with $|L|_n^2 \to p_{\mathsf{L}}$.

2. **Spectral transfer:** Suppose $N$ lies in the closed block subspace orthogonal to $A$, with $|N|_n^2 \to w > 0$. In a comparison model replacing each block by an independent i.i.d. $\mathcal{N}(0,1)$ matrix (with the $A$-matrix as signal seed $A_{\mathrm{c}}$), the comparison noise $N_{\mathrm{c}}$ is i.i.d. $\mathcal{N}(0, w)$ independent of $L_{\mathrm{c}} = h(A_{\mathrm{c}})$. Approximating $L$ by an odd polynomial in $A$ and $N$ by a finite sum of blocks orthogonal to $A$, Lemma 2 matches all mixed trace moments with the compactly supported Gaussian comparison; controlling tails via higher moments and returning to $(L, N)$ via the singular-value $W_2$ perturbation bound gives $W_2(\mu_n(L+N), \mu_n(L+\sqrt{w}G)) \to 0$ in probability. Next, uniformly approximate the bounded continuous odd singular profile (from Appendix A, extended oddly to $\mathbb{R}$) by Lipschitz profiles. An odd $K$-Lipschitz singular map is $K$-Lipschitz in Frobenius norm (apply the scalar function to the symmetric dilation and bound divided differences in the two eigenbases), so input approximations pass through $F_{\mathsf{L}}$ in $|\cdot|_n$. Approximating each Lipschitz profile on the compact limiting support by odd polynomials and invoking Lemma 2 shows that $F_{\mathsf{L}}(L+N) \in \mathcal{H}_{\mathrm{gr}}$ and that its limiting norm and inner products with $L$ and every block agree with the Gaussian comparison.

3. **Entrywise continuity for arbitrary finite-variance priors:** Continuous maps of at most linear growth preserve joint empirical $W_2$ convergence. For any $P_{\mathsf{S}}$ with finite variance and any $v > 0$, choose $K > 0$ with $c := P_{\mathsf{S}}([-K, K]) > 0$. Comparing the Gaussian likelihood against $[-K, K]$ yields
   $$
   \mathbb{P}\bigl(|\mathsf{S}| > 2|r| + K + u \mid \mathsf{S} + \sqrt{v}\,\mathsf{Z} = r\bigr)
   \le c^{-1} e^{-u^2/(2v)} \qquad (\forall u \ge 0),
   $$
   which gives a uniform linear growth bound $|F_{\mathsf{S}}(r, v)| \le 2|r| + K + C(c, v)$, while Gaussian convolution gives continuity. Approximating the centered output in $L^2$ of the limiting coordinate law by finite product-basis sums therefore places $F_{\mathsf{S}}(R_{\mathsf{S}}, v)$ in $\mathcal{H}_{\mathrm{gr}}$ without assuming a bounded derivative of $F_{\mathsf{S}}$.

### 2.5 Induction and smooth odd-polynomial Gaussian-IBP calibration

We prove by induction on $t \ge 1$ that $E^t := L - L^t \in \mathcal{G}$ with $|E^t|_n^2 \to v_t$. At $t=1$, $E^1 = L \in \mathcal{G}$ and $|E^1|_n^2 \to p_{\mathsf{L}} = v_1$.

- **Entrywise step:** Given $E^t \in \mathcal{G}$ with variance $v_t > 0$, Lemma 1 shows that $(S, E^t)$ has empirical $W_2$ limit $(\mathsf{S}, \sqrt{v_t}\mathsf{Z})$ with $\mathsf{Z} \perp \mathsf{S}$. Entrywise continuity proves part (i) of Conjecture 1. Writing $M = m_{\mathsf{S}}(v_t)$ and $a = M/v_t$, posterior orthogonality yields $\mathbb{E}[F_{\mathsf{S}}(\mathsf{R}_{v_t}, v_t)\,\sqrt{v_t}\mathsf{Z}] = M$, so $N^t := S - S^t$ is centered, orthogonal to $E^t$, and has squared norm $M/(1-a) = w_t$. Moreover, $N^t$ is orthogonal to $A$ and every string: regressing any $Z \in \mathcal{G}$ on $E^t$ leaves an independent Gaussian remainder jointly independent of $S$. By Lemma 1, $N^t$ lies in the closed block subspace orthogonal to $A$.

- **Spectral calibration via smooth odd-polynomial Stein/IBP:** On the canonical bounded Gaussian comparison $X = L_{\mathrm{c}} + N_{\mathrm{c}}$ ($N_{\mathrm{c}} = \sqrt{w}\,G$, $w = w_t$), work in native $\sqrt{n}$ units with limiting singular law $\mu$ and optimal on-support profile $f(s) = s - w(1-\beta)/s - 2\beta w\,\mathcal{H}(s)$. For any odd polynomial $g(s) = s\tilde{g}(s^2)$, the rescaled singular-value map
  $$
  g(X) := X\,\tilde{g}\!\left(\frac{X^\top X}{n}\right)
  $$
  is a **polynomial—hence smooth ($C^\infty$)—function of the matrix entries of $X$**, completely avoiding SVD differentiability issues at repeated or zero singular values. By Gaussian integration by parts (Stein's lemma) and the SVD divergence identity on the dense open set of matrices with simple positive singular values,
  $$
  \frac{1}{mn}\operatorname{div}_X g(X)
  = \frac{n-m}{n}\cdot\frac{1}{m}\sum_{i=1}^m \frac{g(s_i)}{s_i}
    + \frac{m}{n}\cdot\frac{1}{m^2}\sum_{i\neq j}\frac{s_i g(s_i) - s_j g(s_j)}{s_i^2 - s_j^2}
    + \frac{1}{n}\cdot\frac{1}{m}\sum_{i=1}^m g'(s_i),
  $$
  where $s_i = \sigma_i(X)/\sqrt{n}$. For an odd polynomial, the diagonal divided difference is removable and the derivative term is $O(1/n) \to 0$. Combining expectation via Stein's lemma with trace concentration (using the bounded comparison signal $L_{\mathrm{c}}$ and Gaussian operator-norm tails) gives in probability:
  $$
  \langle g(X), N_{\mathrm{c}}\rangle_n
  \longrightarrow w\left[(1-\beta)\int \frac{g(s)}{s}\,d\mu(s)
    + \beta\iint \frac{s g(s) - u g(u)}{s^2 - u^2}\,d\mu(s)\,d\mu(u)\right]
  = \int g(s)\bigl(s - f(s)\bigr)\,d\mu(s),
  $$
  where the last equality symmetrizes the principal value via $|s-u| > \varepsilon$ truncation and $L^2$ boundedness of the Hilbert transform on the compactly supported bounded density. Subtracting from $\langle g(X), X\rangle_n \to \int s g(s)\,d\mu(s)$ yields $\langle g(X), L_{\mathrm{c}}\rangle_n \to \int g f\,d\mu$. Taking $g(s)=s$ and approximating $g \to f$ establishes the bulk Gram limits
  $$
  \langle F_{\mathsf{L}}(X), X\rangle_n \to p_{\mathsf{L}},
  \qquad
  \langle F_{\mathsf{L}}(X), L_{\mathrm{c}}\rangle_n \to p_{\mathsf{L}} - m_{\mathsf{L}}^{\mathrm{int}}(w),
  \qquad
  |F_{\mathsf{L}}(X)|_n^2 \to p_{\mathsf{L}} - m_{\mathsf{L}}^{\mathrm{int}}(w),
  $$
  using $m_{\mathsf{L}}^{\mathrm{int}}(w) = p_{\mathsf{L}} - \int f^2\,d\mu$ from Appendix A of arXiv:2611.29179.

- **Closing the induction:** By spectral transfer, the same Gram identities hold for $F_{\mathsf{L}}(R_{\mathsf{L}}^t, w_t)$ with $R_{\mathsf{L}}^t = L + N^t$, proving part (ii) of Conjecture 1. Consequently $L^{t+1}$ is orthogonal to $N^t$ with $|L - L^{t+1}|_n^2 \to \Phi_{\mathsf{L}}(w_t) = v_{t+1}$. In the Gaussian comparison, any finite linear combination of blocks orthogonal to both $A$ and $N^t$ is an independent Gaussian matrix relative to $(A_{\mathrm{c}}, N_{\mathrm{c}})$, so its inner product with $L^{t+1}$ has conditional variance $O(|L^{t+1}|_n^2/(mn)) \to 0$. Transferring these orthogonalities to all blocks perpendicular to $A$ and noting that $L \in \mathcal{G}$ already has this orthogonality places $E^{t+1} = L - L^{t+1} \in \mathcal{G}$ at variance $v_{t+1}$, closing the induction. $\square$

---

## 3. Sharp Uniform-Integrability Condition and Counterexample to Unconditional Expected-Risk Calibration

**Theorem 2 (Uniform-integrability equivalence and iteration-1/2 refutation).**
1. *Let $Q_n := |L|_n^2$ and $D_n := |L - F_{\mathsf{L}}(L + \sqrt{w}\,G, w)|_n^2$ for fixed $w > 0$, using any bounded normalized singular profile ($|F_{\mathsf{L}}|_n \le C_w$ deterministically). Assume $D_n \to m_{\mathsf{L}}^{\mathrm{int}}(w)$ in probability (e.g., under the hypotheses of Theorem 1). Then:*
   $$
   \mathbb{E}[D_n] \to m_{\mathsf{L}}^{\mathrm{int}}(w)
   \iff (D_n)_{n\ge 1} \text{ is uniformly integrable}
   \iff (Q_n)_{n\ge 1} \text{ is uniformly integrable}.
   $$
   *Given $Q_n \to p_{\mathsf{L}}$ in probability, this is equivalent to $\mathbb{E}[Q_n] \to p_{\mathsf{L}}$.*
2. *Under Section II's almost-sure hypotheses alone, uniform integrability can fail even for full-rank bi-orthogonally invariant signals with uniformly bounded entrywise second moments, causing the expected-risk calibration to fail at iterations 1 and 2.*

*Proof of (1).* Deterministically, $D_n \le 2Q_n + 2C_w^2$ and $Q_n \le 2D_n + 2C_w^2$, so uniform integrability of $(Q_n)$ is equivalent to that of $(D_n)$. For nonnegative $D_n \xrightarrow{\mathbb{P}} m_{\mathsf{L}}^{\mathrm{int}}(w) < \infty$, Vitali's convergence theorem gives $\mathbb{E}[D_n] \to m_{\mathsf{L}}^{\mathrm{int}}(w) \iff (D_n)$ is uniformly integrable. $\square$

### Explicit full-rank bi-invariant counterexample ($\delta = 0.05$)

Fix $m = n$, $p_{\mathsf{L}} = p_{\mathsf{S}} = 1$, and i.i.d. Rademacher entries $S_{ij} \in \{\pm 1\}$. Independently of $S$, draw $L$ from the two-component mixture:
- with probability $1 - n^{-2}$, $L = A$ (i.i.d. $\mathcal{N}(0, 1)$);
- with probability $n^{-2}$, $L = n\sqrt{\delta}\,\sqrt{n}\,O_n$, where $O_n \sim \operatorname{Haar}(O_n)$ and $\delta = 0.05$.

Both components are full rank and bi-orthogonally invariant. By Borel–Cantelli, only finitely many exceptional draws occur almost surely, so Section II's almost-sure weak convergence and operator-norm bound hold with the standard square Gaussian signal law ($\nu_L$ Marchenko–Pastur with $\beta=1$, $p_{\mathsf{L}} = 1$). Every entry has uniformly bounded second moment $\mathbb{E}[L_{ij}^2] = 1 - n^{-2} + \delta \to 1.05$, yet $\mathbb{E}[Q_n] \to 1 + \delta = 1.05 \neq 1$.

Set $M := m_{\mathsf{S}}(1) = \mathbb{E}[\operatorname{sech}^2(1 + \mathsf{Z})]$, $w := M/(1-M)$, $b := w/(1+w) \approx 0.449600$, and $c := 1/(1+w)$. On the Gaussian spectral reference, the on-support optimal shrinker is $f(s) = cs$; use the Appendix A bounded continuous extension $f(s) = c\min(s, 2\sqrt{1+w})$, so $|F_{\mathsf{L}}|_n \le 2/\sqrt{1+w}$ deterministically.
- **Expected-risk vs. spectral-integral mismatch:** On the ordinary branch, the reference risk converges to $b$ in probability and in expectation. On the exceptional branch, $|L|_n = n\sqrt{\delta}$ while $|F_{\mathsf{L}}|_n = O(1)$, so the rare branch contributes $n^{-2}(n\sqrt{\delta} + O(1))^2 \to \delta$ to $\mathbb{E}[D_n]$. Hence the expected-risk limit is $m_{\mathsf{L}}^{\mathrm{exp}}(w) = b + \delta \approx 0.499600$, while $m_{\mathsf{L}}^{\mathrm{int}}(w) = b \approx 0.449600$.
- **Iteration 1 failure (part (ii)):** On the ordinary branch (probability $1 - n^{-2} \to 1$), the entries of $R_{\mathsf{L}}^1$ are i.i.d. copies of $H = (S + A - F_{\mathsf{S}}(S+A))/(1-M)$, which are centered with $\mathbb{E}[AH] = 1$, $\mathbb{E}[SH] = w$, and $\mathbb{E}[H^2] = 1 + w$. By the finite-variance Marchenko–Pastur law and the law of large numbers for second moments, $\mu_n(R_{\mathsf{L}}^1)$ converges in $W_2$ to the scaled quarter-circle law supported on $[0, 2\sqrt{1+w}]$. Thus $|F_{\mathsf{L}}(R_{\mathsf{L}}^1) - c R_{\mathsf{L}}^1|_n \to 0$ and
  $$
  |L - F_{\mathsf{L}}(R_{\mathsf{L}}^1)|_n^2 \xrightarrow{\mathbb{P}} 1 - 2c + c^2(1+w) = b \approx 0.449600 \neq 0.499600 = m_{\mathsf{L}}^{\mathrm{exp}}(w).
  $$
- **Iteration 2 failure (part (i)):** Using the expected-risk correction $a' = (b+\delta)/w$ satisfies $a' < 1 \iff \delta < M^2/(1-M)$; since $M = \mathbb{E}[\operatorname{sech}^2(1+\mathsf{Z})] > 0.21$ (e.g., restricting to $|1+\mathsf{Z}| \le 1/2$ and $1/2 < |1+\mathsf{Z}| \le 1$), $M^2/(1-M) > 0.0558 > 0.05 = \delta$, so the next SE variance is strictly positive. The iterate satisfies $L^2 = k R_{\mathsf{L}}^1 + o_P(1)$ in $|\cdot|_n$ with $k = (c - a')/(1 - a')$, whence the continuous quadratic-growth $W_2$ test obeys
  $$
  \langle S, R_{\mathsf{S}}^2 - S\rangle_n \xrightarrow{\mathbb{P}} -kw \approx 0.128737 \neq 0,
  $$
  contradicting part (i)'s independent Gaussian-channel prediction of $0$ for every positive channel variance. Under the spectral-integral calibration $a_{\mathsf{L}} = c$, $k = 0$ and the contradiction disappears.

---

## 4. Off-Support Extensions and the $\beta < 1$ Literal Hilbert-Pole Criterion

Any bounded continuous extension of $f_v$ on $[0, \infty)$ agreeing on the positive support gives the same in-probability Theorem 1 (and for $\beta = 1$, any continuous extension of at most linear growth suffices by the $W_2$ tail bound, though the uniform-integrability equivalence in Theorem 2 uses boundedness).

**Proposition 3 (Literal unregularized Hilbert pole for $\beta < 1$).** *Suppose $\beta < 1$ and $R = R_{\mathsf{L}}^t$ satisfies $\mu_n(R) \xrightarrow{W_2} \mu$ in probability, where $\mu$ is the Gaussian-reference singular law in native $\sqrt{n}$ units, supported away from $0$. Fix $\eta > 0$ strictly below the lower support edge of $\mu$. Denote by $F_{\mathsf{L}}^{\mathrm{lit}}(R)$ the unregularized main-text Hilbert formula continued literally to all $s > 0$ (which has a $-w(1-\beta)/s$ pole as $s \downarrow 0$), and by $F_{\mathsf{L}}^{\mathrm{bdd}}(R)$ the Appendix A bounded continuous extension. Then:*
1. *$|F_{\mathsf{L}}^{\mathrm{lit}}(R) - F_{\mathsf{L}}^{\mathrm{bdd}}(R)|_n \to 0$ in probability if and only if*
   $$
   \frac{1}{m}\sum_{s_i < \eta} s_i^{-2} \xrightarrow{\mathbb{P}} 0
   $$
   *(where $s_i = \sigma_i(R)/\sqrt{n}$ and any $s_i = 0$ contributes $+\infty$).*
2. *The same condition is necessary for $|L - F_{\mathsf{L}}^{\mathrm{lit}}(R)|_n^2 \to m_{\mathsf{L}}^{\mathrm{int}}(w)$ in probability.*

*Proof.* On $[\eta, \infty)$, the literal Hilbert formula is continuous, agrees with $F_{\mathsf{L}}^{\mathrm{bdd}}$ on $\operatorname{supp}(\mu)$, and grows at most linearly at infinity, so $W_2$ convergence of $\mu_n(R)$ forces the squared discrepancy from $s_i \ge \eta$ to vanish in probability. On $(0, \eta)$, the discrepancy is asymptotic to $w(1-\beta)/s$, giving (1). For (2), let $P_\eta$ be the orthogonal projection onto the subspace spanned by singular vectors with $s_i < \eta$. Since $\mu([0,\eta]) = 0$, $\operatorname{rank}(P_\eta) = o_P(m)$. By $W_2$ convergence of $\mu_n(L)$ to the compactly supported law $\nu_L^{1/2}$ and the singular-value variational bound, $|P_\eta L|_n^2 \to 0$ in probability. Hence excess energy $|P_\eta(F_{\mathsf{L}}^{\mathrm{lit}}(R) - F_{\mathsf{L}}^{\mathrm{bdd}}(R))|_n^2$ cannot cancel against $L$ in $|L - F_{\mathsf{L}}^{\mathrm{lit}}(R)|_n^2$. $\square$

---

## 5. Reproducible Numerical Verification and Literature Context

Running from the project root:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B solutions/extensive_matrix_denoise/explore.py
```

with default seed `261129179` checks three replicates at $n \in \{64, 128, 256\}$, three OAMP iterations, two unit-power signal ensembles (nonlinear Haar orthogonal $L = \sqrt{n}\,O_n$ and i.i.d. Gaussian $L$), and four unit-variance entrywise priors (Rademacher, Laplace, Student $t_3/\sqrt{3}$ with infinite fourth moment, and Gaussian). For the nonlinear Haar orthogonal signal at $t = 3$, replicate means are:

| Entry prior | Spectral $W_2$ ($n=64$) | Spectral $W_2$ ($n=256$) | Triple coupling bound ($n=256$) | Spectral risk gap ($n=256$) |
|---|---:|---:|---:|---:|
| Rademacher | 0.04297 | 0.01302 | 0.21935 | +0.00122 |
| Laplace | 0.04525 | 0.00995 | 0.21066 | +0.00025 |
| Student $t_3$ | 0.03465 | 0.00895 | 0.66826 | +0.00065 |
| Gaussian | 0.03515 | 0.01094 | 0.14203 | −0.00082 |

Across all 24 ordinary model/prior/iteration combinations at $n = 256$, the largest absolute replicate-mean spectral risk gap is $0.002152$. For the rare-event mixture counterexample ($\delta = 0.05$, stratified over both branches with exact mixture weights $1 - n^{-2}$ and $n^{-2}$):

| $n$ | Ordinary reference risk | Weighted reference risk | Iteration-2 cross moment |
|---:|---:|---:|---:|
| 64 | 0.444842 | 0.484913 | 0.137741 |
| 128 | 0.448012 | 0.492936 | 0.123486 |
| 256 | 0.450177 | 0.497612 | 0.132802 |
| Analytic limit | 0.449600 | 0.499600 | 0.128737 |

**Related literature:**
- **Traffic probability and graph-indexed random matrices:** C. Male, *Traffic Distributions and Independence: Permutation Invariant Random Matrices and the Three Notions of Independence*, *Memoirs of the AMS* 267(1300), 2020 ([arXiv:1111.4662](https://arxiv.org/abs/1111.4662)).
- **Tensor-network / graph-matrix AMP universality and orthogonally invariant AMP:** T. Wang, X. Zhong, and Z. Fan, *Universality of Approximate Message Passing Algorithms and Tensor Networks*, *Annals of Applied Probability* 34(4), 2024 ([arXiv:2206.13037](https://arxiv.org/abs/2206.13037)); and *Approximate Message Passing for Orthogonally Invariant Ensembles: Multivariate Non-Linearities and Spectral Initialization*, *Information and Inference* 13(3), 2024 ([arXiv:2110.02318](https://arxiv.org/abs/2110.02318)).
- **Optimal rectangular RIE under Gaussian noise:** E. Troiani, V. Erba, F. Krzakala, A. Maillard, and L. Zdeborová, *Optimal Denoising of Rotationally Invariant Rectangular Matrices*, PMLR 190, 2022 ([arXiv:2203.07752](https://arxiv.org/abs/2203.07752)), cited in arXiv:2611.29179 for the Gaussian-channel shrinkage formula and risk integral.
