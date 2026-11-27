# Proof of Rank-Constrained Capacity Equivalence for Gaussian Linear Associative Memory

**Target paper:** Alessio Giorlandino, Sebastian Goldt, and Antoine Maillard, *Factual recall in linear associative memories: sharp asymptotics and mechanistic insights*, [arXiv:2605.10795](https://arxiv.org/abs/2605.10795), Section 2, Conjecture `conjecture-capacity` (Capacity Equivalence).  
**Companion files:** [`solutions/linear_assoc_memory/explore.py`](explore.py), [`solutions/linear_assoc_memory/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conjecture-capacity`: PROVED for every fixed rank fraction $0 < \kappa \le 1$.**
- In the proportional-rank high-dimensional limit $m/d \to \kappa \in (0, 1]$ and $p \log p / d^2 \to \alpha \in (0, \infty)$, the original shared-output problem (OP) and the decoupled problem (DP) share the identical sharp storage threshold:
  $$
  \alpha_c^{\mathrm{OP}}(\kappa) = \alpha_c^{\mathrm{DP}}(\kappa) = \frac{1}{2} E_\kappa,
  \qquad
  E_\kappa := \int_{q_\kappa}^{2} s^2 \frac{\sqrt{4 - s^2}}{\pi}\,ds,
  \qquad
  \int_{q_\kappa}^{2} \frac{\sqrt{4 - s^2}}{\pi}\,ds = \kappa.
  $$
  Specifically, the probability that a matrix $W \in \mathbb{R}^{d \times d}$ of rank at most $m$ stores all $p$ associations tends to $1$ for $\alpha < E_\kappa / 2$ and to $0$ for $\alpha > E_\kappa / 2$. At full rank ($\kappa = 1$), $q_1 = 0$ and $\alpha_c^{\mathrm{OP}}(1) = \alpha_c^{\mathrm{DP}}(1) = 1/2$.
- **What was open and what is proved here:** Giorlandino, Goldt, and Maillard ([arXiv:2605.10795](https://arxiv.org/abs/2605.10795), Theorem "Rank-constrained capacity threshold") established achievability below $E_\kappa / 2$ for both OP and DP and impossibility above $E_\kappa / 2$ for DP, while Barnfield et al. ([arXiv:2605.05189](https://arxiv.org/abs/2605.05189), Theorem 1) established the full-rank OP converse at $\alpha > 1/2$. The remaining open regime was the OP converse for $\kappa \in (0, 1)$ and $E_\kappa / 2 < \alpha \le 1/2$. Below we prove the sharp OP converse for all $0 < \kappa \le 1$ uniformly across all matrices of rank at most $m$, completing the proof of Conjecture `conjecture-capacity`.

---

## 1. Model and Setup

Using the standard-normal scaling of the paper's main text (multiplying all embeddings by $1/\sqrt{d}$ yields the appendix scaling and preserves feasibility), let the $p$ keys $e_1, \dots, e_p \in \mathbb{R}^d$ and $p$ matched outputs $u_1, \dots, u_p \in \mathbb{R}^d$ be mutually independent $\mathcal{N}(0, I_d)$ vectors, with the injective target rule relabeled to the identity $f^*(i) = i$.
- **Original Problem (OP):** There exists $W \in \mathbb{R}^{d \times d}$ of rank at most $m$ such that $u_i^\top W e_i > u_j^\top W e_i$ for all $i \in [p]$ and all $j \neq i$ (ties fail).
- **Decoupled Problem (DP):** For each $i \in [p]$, the $p - 1$ competitors are fresh $\mathcal{N}(0, I_d)$ vectors independent across queries and independent of all keys and targets.

Write $L := \log p$, $C := \sum_{i=1}^p u_i e_i^\top$, and
$$
S(W) := \sum_{i=1}^p \max_{j \neq i} u_j^\top W e_i.
$$
Feasibility of $W$ in OP implies $\langle C, W \rangle_F = \sum_{i=1}^p u_i^\top W e_i > S(W)$, and by positive homogeneity we may normalize $\|W\|_F = 1$. Throughout, $d^2 \asymp p L$, so $p \to \infty$, $d \asymp \sqrt{p L}$, and every fixed power of $L$ is $o(d)$.

---

## 2. Proof of the Shared-Output Rank-Constrained Converse

Set $k := \lceil L^{24} \rceil$. For any unit-Frobenius matrix $W \in \mathbb{R}^{d \times d}$, split its singular value decomposition as $W = H + T$, where $H$ retains the top $k$ singular values and $T$ the remainder, and define $b(W) := \|T\|_F \in [0, 1]$. Note that:
- $b(W) = \operatorname{dist}_F(W, \{X : \operatorname{rank}(X) \le k\})$ is $1$-Lipschitz in Frobenius norm;
- $\|T\|_{\mathrm{op}} \le 1/\sqrt{k} \le L^{-12}$ (since the $k$-th singular value of $W$ is at most $\|W\|_F/\sqrt{k} = 1/\sqrt{k}$).

Our goal is to establish the following two uniform lower bounds on the unit Frobenius sphere: for each fixed $\varepsilon \in (0, 1)$, with probability tending to $1$, uniformly over all $\|W\|_F = 1$,
$$
\frac{S(W)}{p\sqrt{2L}} \ge c_0 := \frac{1}{4\sqrt{\pi}},
\qquad
\frac{S(W)}{p\sqrt{2L}} \ge (1 - \varepsilon)\, b(W) - o(1). \tag{1}
$$

### Step 1: Two elementary properties of the output cloud

Let $U = [u_1, \dots, u_p] \in \mathbb{R}^{d \times p}$. We claim that with probability tending to $1$, simultaneously for every unit vector $v \in S^{d-1}$:
1. at least $p/3$ outputs satisfy $v^\top u_j \ge 0$; and
2. at least two outputs satisfy $v^\top u_j \ge \sqrt{L}/2$.

*Proof of Step 1.* Take a $d^{-2}$-net of $S^{d-1}$ of cardinality $\exp(O(d \log d))$. For a fixed net point, the number of $j$ with $v^\top u_j \ge 0.1$ is $\operatorname{Bin}(p, q)$ with $q > 0.45$, so having fewer than $p/3$ fails with probability $e^{-\Omega(p)}$. For the second property, at cutoff $\sqrt{L}/2 + 1$, standard Gaussian tail bounds give $\Pr(v^\top u_j \ge \sqrt{L}/2 + 1) = p^{-1/8 + o(1)}$, so the expected count is $p^{7/8 + o(1)}$ and the probability of fewer than two successes is at most $\exp(-p^{7/8 + o(1)})$. Since $d \log d \asymp \sqrt{p} L^{3/2}$, both exponents strictly dominate $O(d \log d)$. On the high-probability event $\max_j \|u_j\| \le 2\sqrt{d}$, the $d^{-2}$-net approximation error is at most $2 d^{-3/2} = o(1)$, yielding both uniform bounds.

In particular, even after removing $j = i$, $\max_{j \neq i} u_j^\top W e_i \ge \|W e_i\| \sqrt{L}/2 \ge 0$, so every summand of $S(W)$ is nonnegative. By the "Uniform Gaussian input-frame bounds" lemma of [arXiv:2605.10795](https://arxiv.org/abs/2605.10795), $p^{-1}\sum_{i=1}^p \|W e_i\| \ge \|W\|_F / \sqrt{2\pi}$ simultaneously for all $W$ with high probability, proving the first inequality of (1) with $c_0 = \frac{\sqrt{L}/2}{\sqrt{2L}} \cdot \frac{1}{\sqrt{2\pi}} = \frac{1}{4\sqrt{\pi}}$.

### Step 2: Many weakly correlated competitors in every large subset

Fix a **deterministic** matrix $B \in \mathbb{R}^{d \times d}$ with $\|B\|_F = 1$ and $\|B\|_{\mathrm{op}} \le L^{-10}$. Set $z_j := B^\top u_j$, $\eta := L^{-1}$, $\rho := L^{-4}$, and $n := \lfloor p/16 \rfloor$. We claim that there is a constant $c > 0$ such that, except on an event of probability at most $\exp(-c p L^{12})$, every subset of $[p]$ of size at least $p/4$ contains a subset of $n$ indices satisfying
$$
1 - \eta \le \|z_j\| \le 1 + \eta,
\qquad
\biggl|\Bigl\langle \frac{z_j}{\|z_j\|}, \frac{z_l}{\|z_l\|} \Bigr\rangle\biggr| \le \rho
\qquad (j \neq l). \tag{2}
$$

*Proof of Step 2.* For a fresh $z = B^\top u$ ($u \sim \mathcal{N}(0, I_d)$), $\mathbb{E}\|z\|^2 = \|B\|_F^2 = 1$. Since $u \mapsto \|B^\top u\|$ is $\|B\|_{\mathrm{op}}$-Lipschitz, the Gaussian Poincaré inequality gives $\operatorname{Var}(\|z\|) \le \|B\|_{\mathrm{op}}^2 \le L^{-20}$. Therefore
$$
1 - L^{-20} \le (\mathbb{E}\|z\|)^2 \le 1 \implies \mathbb{E}\|z\| \ge 1 - L^{-20}.
$$
By Gaussian Lipschitz concentration around $\mathbb{E}\|z\|$, for large $p$ the event $|\|z\| - 1| > \eta = L^{-1}$ implies $|\|z\| - \mathbb{E}\|z\|| > \eta/2$, which has probability at most $2\exp(-c' L^{18})$. Moreover, for any fixed unit vector $v$, $\langle v, z \rangle = \langle Bv, u \rangle$ is centered Gaussian with variance $\|Bv\|^2 \le L^{-20}$. Hence the probability that a norm-good fresh point $z$ has $|\langle v, z/\|z\| \rangle| > \rho$ for at least one of at most $p$ fixed unit vectors is at most
$$
2p \exp\!\Bigl(-\frac{(1 - \eta)^2 \rho^2 L^{20}}{2}\Bigr) = 2p \exp\!\Bigl(-\frac{(1 - o(1)) L^{12}}{2}\Bigr).
$$
Combining both bounds, the fresh-point failure probability is at most $\exp(-c_1 L^{12})$ for large $p$.

Now suppose (2) fails on some subset $A \subseteq [p]$ with $|A| \ge p/4$. Let $J \subseteq A$ be an inclusion-maximal subset of norm-good indices in $A$ satisfying the pairwise inner-product condition in (2) (possibly $J = \varnothing$). Then $|J| < n \le p/16$, so at least $p/4 - p/16 \ge p/8$ indices in $A \setminus J$ are either norm-bad or incompatible with some vector in $J$. For each fixed candidate pair $(J, F)$ with $F \subseteq [p] \setminus J$ and $|F| = \lceil p/8 \rceil$, condition on $\{u_l : l \in J\}$: the remaining outputs $\{u_j : j \in F\}$ are independent, and each fails the fresh-point test relative to $\{z_l / \|z_l\| : l \in J\}$ with probability at most $\exp(-c_1 L^{12})$. A union bound over the at most $4^p$ choices of $(J, F)$ gives total failure probability at most $4^p \exp(-c_1 (p/8) L^{12}) \le \exp(-c p L^{12})$.

### Step 3: Conditional lower-tail bound via Slepian's comparison and net transfer

Let $\mathcal{N}$ be a deterministic $d^{-3}$-net of the unit Frobenius sphere in $\mathbb{R}^{d \times d}$, of log cardinality $O(d^2 \log d) = O(p L^2)$. For each $V \in \mathcal{N}$ with $b = b(V) \ge L^{-1}$, fix an SVD split $V = H + T$ and apply Step 2 to $B := T / b$, which satisfies $\|B\|_F = 1$ and $\|B\|_{\mathrm{op}} \le L^{-11} \le L^{-10}$. Because $p L^{12} \gg p L^2$, a union bound ensures that Step 1 and Step 2 hold simultaneously for all such $V \in \mathcal{N}$ with probability $1 - o(1)$. Condition on a realization of the outputs $U$ in this good event.

Fix one such $V \in \mathcal{N}$ and one query $i \in [p]$. Split the standard Gaussian key $e_i$ into its independent projections onto the top-$k$ right singular subspace of $V$ and its orthogonal complement, and condition on the first projection. This fixes $H e_i$ while leaving the orthogonal key coordinates seen by $T$ standard Gaussian and independent. By Step 1, at least $p/3 - 1 \ge p/4$ indices $j \neq i$ satisfy $u_j^\top H e_i \ge 0$ (including when $H e_i = 0$). Using only the conditioned data (e.g., lexicographically first), select $n$ such indices $I_i$ satisfying (2). For $j \in I_i$, the variables
$$
X_j := \frac{u_j^\top T e_i}{b \|z_j\|} = \Bigl\langle \frac{z_j}{\|z_j\|}, e_i \Bigr\rangle
$$
are jointly centered Gaussian with $\operatorname{Var}(X_j) = 1$ and $\operatorname{Cov}(X_j, X_l) = \langle z_j/\|z_j\|, z_l/\|z_l\| \rangle \le \rho$ for $j \neq l$, because every $z_j = B^\top u_j$ lies in the complementary right singular subspace and $I_i$ is independent of the complementary key coordinates.

Fix $\delta \in (0, 1/4)$ and set $t := (1 - \delta)\sqrt{2L}$. Let $Y_1, \dots, Y_n$ be equicorrelated standard Gaussians $Y_r := \sqrt{\rho}\, G_0 + \sqrt{1 - \rho}\, G_r$ with $G_0, G_1, \dots, G_n \stackrel{\mathrm{i.i.d.}}{\sim} \mathcal{N}(0, 1)$, so $\operatorname{Cov}(Y_r, Y_s) = \rho \ge \operatorname{Cov}(X_j, X_l)$ for $j \neq l$. By Slepian's comparison lemma (Slepian, 1962),
$$
\Pr\Bigl(\max_{j \in I_i} X_j < t \;\Big|\; U, H e_i\Bigr)
\le \Pr\Bigl(\max_{1 \le r \le n} Y_r < t\Bigr).
$$
On the event $\sqrt{\rho}\, G_0 \ge -\frac{\delta}{4}\sqrt{2L}$, having $\max_{r \le n} Y_r < t$ implies $\max_{r \le n} G_r < (1 - \delta/2)\sqrt{2L}$ for all sufficiently large $p$. The excluded shift event has probability at most $\exp(-c_\delta L / \rho) = \exp(-c_\delta L^5)$, while the independent maximum event has probability at most $\exp(-p^{1 - (1 - \delta/2)^2 + o(1)}) \ll \exp(-c_\delta L^5)$ by standard Gaussian tail bounds. Since $u_j^\top H e_i \ge 0$ and $\|z_j\| \ge 1 - \eta$ on $I_i$, any positive maximum of $X_j$ converts to a score lower bound $u_j^\top V e_i \ge b (1 - \eta) X_j$; integrating over $H e_i$ gives, for all large $p$,
$$
\Pr\!\left(\max_{j \neq i} u_j^\top V e_i < b(1 - \eta)(1 - \delta)\sqrt{2L} \;\middle|\; U\right)
\le \exp(-c_\delta L^5). \tag{3}
$$

Conditional on $U$, the $p$ failure events in (3) depend on distinct independent keys $e_1, \dots, e_p$, so they are conditionally independent. The probability that more than $\delta p$ queries fail for a fixed $V$ is at most $2^p \exp(-c_\delta \delta p L^5)$, which pays for a second union bound over $|\mathcal{N}| = \exp(O(p L^2))$. Because every summand of $S(V)$ is nonnegative by Step 1, summing over the at least $(1 - \delta)p$ successful queries gives $S(V)/(p\sqrt{2L}) \ge (1 - \delta)^2(1 - \eta) b(V)$ simultaneously over all $V \in \mathcal{N}$ with $b(V) \ge L^{-1}$ (and trivially with an additive $-L^{-1}$ when $b(V) < L^{-1}$).

Finally, with high probability $\max_i \|e_i\| \le 2\sqrt{d}$ as well as $\max_j \|u_j\| \le 2\sqrt{d}$, on which $|S(W) - S(V)| \le 4 p d \|W - V\|_F$. Approximating any unit-Frobenius $W$ by its nearest net point $V \in \mathcal{N}$ incurs error $O(d^{-2} / \sqrt{L}) = o(1)$ in $S(W)/(p\sqrt{2L})$. Using the $1$-Lipschitzness of $b$ (no singular-subspace continuity is needed) and taking $\delta > 0$ small enough for the prescribed $\varepsilon$ establishes both inequalities of (1).

### Step 4: Comparing target energy to competitor scores via von Neumann's inequality

In standard-normal units, the correlation-matrix estimates of [arXiv:2605.10795](https://arxiv.org/abs/2605.10795) give
$$
\frac{\|C\|_{\mathrm{op}}}{\sqrt{p d}} = O_{\Pr}(1),
\qquad
\frac{\|C_m\|_F}{d \sqrt{p}} \xrightarrow{\Pr} \sqrt{E_\kappa},
$$
where $C_m$ denotes the best rank-$m$ approximation (truncated SVD) of $C$. (Conditional on $U$, $C/\sqrt{pd}$ has the law of $(U U^\top / p)^{1/2} G / \sqrt{d}$ with $G$ i.i.d. standard Gaussian; since $p/d \to \infty$, $U U^\top / p \to I_d$ in operator norm, and the quarter-circle law yields the top-$\kappa$ singular-value energy limit $E_\kappa$.) Crucially, since $\kappa \in (0, 1]$, we have $q_\kappa < 2$ and hence
$$
E_\kappa = \int_{q_\kappa}^{2} s^2 \frac{\sqrt{4 - s^2}}{\pi}\,ds > 0.
$$

For every unit-Frobenius matrix $W = H + T$ of rank at most $m$, the head $H$ has nuclear norm $\|H\|_* \le \sqrt{k}\|H\|_F \le \sqrt{k}$, and the tail $T$ has rank at most $m$ and Frobenius norm $b(W)$. Applying von Neumann's trace inequality (von Neumann, 1937) to both parts gives the simultaneous upper bound
$$
\frac{\langle C, W \rangle_F}{d \sqrt{p}}
\le \frac{\|H\|_* \|C\|_{\mathrm{op}}}{d \sqrt{p}} + \frac{\|C_m\|_F}{d \sqrt{p}}\, b(W)
\le \frac{\sqrt{k}\|C\|_{\mathrm{op}}}{d \sqrt{p}} + \frac{\|C_m\|_F}{d \sqrt{p}}\, b(W)
\le \sqrt{E_\kappa}\, b(W) + o_{\Pr}(1), \tag{4}
$$
where the head error vanishes because $\sqrt{k/d} = O(L^{12}/\sqrt{d}) \to 0$. No independence between (1) and (4) is required.

Now suppose $\alpha > E_\kappa / 2$, and choose $\varepsilon \in (0, 1)$ such that $(1 - \varepsilon)\sqrt{2\alpha} > \sqrt{E_\kappa}$. Rescaling (1) from denominator $p\sqrt{2L}$ to $d\sqrt{p}$ multiplies by $\sqrt{2pL}/d \to \sqrt{2\alpha}$. Suppose a feasible unit-Frobenius matrix $W$ of rank at most $m$ exists, so $\langle C, W \rangle_F > S(W)$. Combining the coarse lower bound in (1) with (4) gives
$$
c_0 \sqrt{2\alpha} - o_{\Pr}(1) \le \frac{S(W)}{d \sqrt{p}} < \frac{\langle C, W \rangle_F}{d \sqrt{p}} \le \sqrt{E_\kappa}\, b(W) + o_{\Pr}(1).
$$
Because $E_\kappa > 0$, this forces $b(W) \ge b_0 := \frac{c_0 \sqrt{2\alpha}}{2\sqrt{E_\kappa}} > 0$ with probability tending to $1$. Substituting $b(W) \ge b_0$ into the sharp lower bound in (1) and comparing against (4) then yields
$$
0 < \frac{\langle C, W \rangle_F - S(W)}{d \sqrt{p}}
\le -\bigl((1 - \varepsilon)\sqrt{2\alpha} - \sqrt{E_\kappa}\bigr) b_0 + o_{\Pr}(1),
$$
whose right-hand side is strictly negative with probability tending to $1$, a contradiction. Since all bounds hold simultaneously over all $W$ (including data-dependent feasible solutions), $\Pr(\text{OP feasible}) \to 0$. Combining this with the achievability and DP threshold theorem of [arXiv:2605.10795](https://arxiv.org/abs/2605.10795) completes the proof of Conjecture `conjecture-capacity`. $\square$

---

## 3. Scope Remarks

- **Boundary $\alpha = E_\kappa / 2$:** The conjecture makes strict-side assertions ($\alpha < \alpha_c$ and $\alpha > \alpha_c$). At the critical limiting ratio $\alpha = E_\kappa / 2$, a diagonal subsequence approaching $E_\kappa / 2$ from below (resp. above) achieves limiting success probability $1$ (resp. $0$); finer critical-window asymptotics require specifying the rate of convergence of $p \log p / d^2$.
- **Why standard tools do not directly apply:** Cover's independent-label count fails because the $p(p - 1)$ inequalities reuse keys and shared outputs; vectorizing $W$ produces rows $e_i \otimes (u_i - u_j)$ with dependent non-Gaussian entries, precluding a direct i.i.d.-Gaussian Gordon min–max theorem; and ordinary LP dual infeasibility rules out all ranks simultaneously. Here rank dependence enters through the support function of the rank-$m$ Frobenius unit ball (giving $\|C_m\|_F$), while conditioning on $U$ and the top-$k$ head projection produces genuinely Gaussian vectors with controlled covariance for Slepian's lemma.

---

## 4. Finite-Dimensional Verification

Run `.venv/bin/python -B solutions/linear_assoc_memory/explore.py` from the repository root (NumPy + SciPy; default seed `260510795`). Across $d \in \{12, 16\}$ and four target loads $\alpha \in \{0.3, 0.6, 1.0, 1.4\}$ (4 independent trials per load; OP and DP share keys and targets within each trial; witnesses pass all $p(p-1)$ comparisons with normalized margin $> 10^{-8}$; full-rank failures are checked by HiGHS constraint-generation LP):

| $d$ | $p$ | $p\log(p)/d^2$ | OP witnesses ($\kappa = 0.25, 0.50, 0.75, 1.00$) | DP witnesses ($\kappa = 0.25, 0.50, 0.75, 1.00$) | Full-rank LP infeasible (OP / DP) |
|---:|---:|---:|:---|:---|:---|
| 12 | 16 | 0.3081 | 4, 4, 4, 4 | 4, 4, 4, 4 | 0 / 0 |
| 12 | 26 | 0.5883 | 4, 4, 4, 4 | 3, 4, 4, 4 | 0 / 0 |
| 12 | 39 | 0.9922 | 0, 2, 4, 4 | 0, 1, 3, 3 | 0 / 1 |
| 12 | 51 | 1.3925 | 0, 0, 0, 0 | 0, 0, 0, 0 | 4 / 4 |
| 16 | 24 | 0.2979 | 4, 4, 4, 4 | 4, 4, 4, 4 | 0 / 0 |
| 16 | 41 | 0.5948 | 4, 4, 4, 4 | 3, 4, 4, 4 | 0 / 0 |
| 16 | 62 | 0.9995 | 0, 0, 2, 2 | 0, 0, 1, 2 | 2 / 2 |
| 16 | 81 | 1.3904 | 0, 0, 0, 0 | 0, 0, 0, 0 | 4 / 4 |

For reference, the limiting asymptotic thresholds $E_\kappa / 2$ at $\kappa = 0.25, 0.50, 0.75, 1.00$ are $0.31145628$, $0.44686540$, $0.49352390$, and $0.50000000$. Across all 64 instances, 43 yielded verified full-rank witnesses and 21 were LP-infeasible (all 16 at the largest load).

---

## References

- D. Slepian, *The One-Sided Barrier Problem for Gaussian Noise*, Bell System Technical Journal **41**(2):463–501, 1962.
- J. von Neumann, *Some matrix-inequalities and metrization of matric-space*, Tomsk Univ. Rev. **1**:286–300, 1937.
- A. Giorlandino, S. Goldt, and A. Maillard, *Factual recall in linear associative memories: sharp asymptotics and mechanistic insights*, [arXiv:2605.10795](https://arxiv.org/abs/2605.10795), 2026.
- M. Barnfield et al., *Sharp capacity thresholds for linear associative memory*, [arXiv:2605.05189](https://arxiv.org/abs/2605.05189), 2026.
