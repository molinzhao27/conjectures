# Refutation of Riemannian Attention Rank-Preservation Conjectures and Exact Two-Token Criterion

**Target paper:** Sen Song, *Riemannian Attention Mechanisms for Transformers: A Theoretical Framework and Architecture Design*, [arXiv:2608.01283](https://arxiv.org/abs/2608.01283), Section 3, Conjectures `conj:rankpreserve` and `conj:rankstrong`.  
**Companion files:** [`solutions/riemannian_attn_rank/explore.py`](explore.py), [`solutions/riemannian_attn_rank/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjectures `conj:rankpreserve` and `conj:rankstrong`: FALSE** (both the single-layer lower bound $\sigma_2(A^{\mathrm{Riem}}) > 0$ and the depth-uniform effective-rank bound $\operatorname{EffRank}(H_n) \ge f(r, \sigma_U^2) > 1$ fail under all stated directional, rank, diversity, and contrast hypotheses).
- **Single-layer exact rank-one collapse with no depth-dependent weights (Theorem 1):** At $L = d = r = 2$ (and lifting to every $2 \le r \le d$ and $L \ge 2$), two full-rank metrics spanning the entire active subspace $D = \mathbb{R}^2$, with arbitrarily large strength $\bar U^2 = 6s$, positive diversity $\sigma_U^2 = 9s^2/4$, and nonconstant within-row metric contrast $3s$ in every row, produce two **identical** attention rows $(p, 1-p)$. Thus $\sigma_2(A) = 0$ and $\operatorname{EffRank}(H_1) = 1$ at layer $1$, with no depth-dependent weights required.
- **Deep collapse despite distinct preferred keys and arbitrarily large fixed margins (Theorem 2):** For any margin schedule $b_n > 1$, a two-token pure Riemannian self-attention + SiLU MetricNet stack maintains distinct preferred keys in every row, full metric rank $r = 2$, positive diversity $\sigma_U^2 = s_n^2$, and $\sigma_2(A_n) = \tanh(b_n/2)$ arbitrarily close to $1$, yet $\operatorname{EffRank}(H_n) = 1 + \prod_{j=1}^n \tanh^2(b_j/2)$. Taking a fixed margin $b_n = b$ gives $\operatorname{EffRank}(H_n) = 1 + \tanh(b/2)^{2n} \to 1$. *Weight-norm note:* realizing fixed margins as states coalesce requires exponentially growing decoder norms $\|B_n\|_{\mathrm{op}} = \tanh(b/2)^{-(n-1)}$ (which the conjectures do not restrict), whereas Theorem 1 needs no depth-dependent weights and Theorem 3 shows that uniformly bounded weights actually force *doubly exponential* collapse whenever the consensus limit is nonzero.
- **General Dobrushin consensus and zero-limit exception (Theorem 3 & Remark):** For any pure Riemannian stack with uniformly bounded score range $R_n \le (1+\bar U_n^2)M_n/\tau_n$, rows converge geometrically to consensus on *either* side of the proposed critical scale $\bar U_{\mathrm{crit}}^2 = \Theta(\tau/L)$; under uniformly bounded weights $(W_Q, W_K, U)$, diameter decay is doubly exponential (as in Dong et al., 2021). However, an unconditional effective-rank collapse claim has a genuine zero-limit exception: a centered equilateral triangle in $\mathbb{R}^2$ (and a heterogeneous, arbitrarily subcritical biased-SiLU extension in $\mathbb{R}^3$) collapses to $H_n \to 0$ while keeping $\operatorname{EffRank}(H_n) = 2$ for every finite $n$.
- **Exact two-token preservation criterion (Theorem 4):** For any two-token pure stochastic stack $A_j = \begin{pmatrix}1-a_j & a_j \\ b_j & 1-b_j\end{pmatrix}$ with $u_j := |1 - a_j - b_j| = \delta(A_j) = |\det A_j|$ and any rank-$2$ input $H_0$, a depth-uniform bound $\inf_n \operatorname{EffRank}(H_n) > 1$ holds **if and only if** $u_j > 0$ for all $j$ and $\sum_{j=1}^\infty (1 - u_j) < \infty$. The paper's separate permutation-margin sufficient condition (Proposition `prop:margin`) survives.

---

## 1. Model and Conjectures

Write layers as $n = 1, 2, \dots$ and consider the pure Riemannian self-attention recursion (identity values, no skip connections, no FFN) $H_n = A_n H_{n-1} = P_n H_0$, $P_n = A_n \cdots A_1$. For row vectors of queries $q_i$ and keys $k_j$, set $v_{ij} := q_i - k_j$,
$$
E_{ij} := \|v_{ij}\|^2, \qquad
G_{ij} := \frac{1}{2}\bigl(\|U_i^\top v_{ij}\|^2 + \|U_j^\top v_{ij}\|^2\bigr), \qquad
A_{ij} := \frac{\exp[-(E_{ij} + G_{ij})/\tau]}{\sum_k \exp[-(E_{ik} + G_{ik})/\tau]}.
$$
The active difference subspace is $D := \operatorname{span}\{v_{ij}\}$, metric strength is $\bar U^2 := \max_i \|U_i\|_F^2$, diversity is the population variance $\sigma_U^2 := \operatorname{Var}_i(\|U_i\|_F^2)$, and effective rank (stable rank) is $\operatorname{EffRank}(X) := \|X\|_F^2 / \sigma_1(X)^2$ for $X \neq 0$.

- `conj:rankpreserve` proposes a depth-uniform bound $\operatorname{EffRank}(H_n) \ge f(r, \sigma_U^2) > 1$ for $\sigma_U^2 > 0$, $r \ge r_{\min}(d, L)$, and metric factors overlapping $D$ (non-row-constant $G$).
- `conj:rankstrong` splits into a subcritical regime ($\bar U^2 < \Theta(\tau/L)$, collapse as in Dong et al., 2021) and a supercritical regime ($\bar U^2 > \Theta(\tau/L)$, $\sigma_U^2 > 0$, overlap with $D$, non-row-constant $G$, and large within-row differences $G_{ij} - G_{ik}$ relative to $\tau$), claiming both a single-layer lower bound on $\sigma_2(A)$ and a depth-uniform bound $\operatorname{EffRank}(H_n) \ge f(r, \sigma_U^2) > 1$.

---

## 2. Single-Layer Active Full-Rank Collapse

**Theorem 1 (Exact rank-one collapse at layer $1$).** *For $L = d = r = 2$, $H_0 = I_2$, and any $s, \tau > 0$, define linear projections giving $q_1 = k_1 = 0$, $q_2 = 2e_2$, $k_2 = e_1$, and full-rank metric factors (realizable by a shared SiLU MetricNet on the orthogonal rows of $H_0$)*
$$
U_1 = \sqrt{s}\,\operatorname{diag}(\sqrt{5}, 1), \qquad U_2 = \sqrt{s}\,\operatorname{diag}(1, \sqrt{2}).
$$
*Then $D = \mathbb{R}^2$, $\bar U^2 = 6s$, $\sigma_U^2 = \frac{9}{4}s^2 > 0$, every row of $G$ is nonconstant with metric contrast $3s$, and $G$ strictly alters the Euclidean attention probabilities; yet both rows of $A$ equal $(p, 1-p)$ with $p = (1 + e^{-(1+3s)/\tau})^{-1}$, so $\sigma_2(A) = 0$ and $\operatorname{EffRank}(H_1) = 1$.*

*Proof.* Since $H_0 = I_2$, linear maps $W_Q, W_K$ realize $(q_1, q_2)$ and $(k_1, k_2)$ directly; for MetricNet, choosing first weight $I_2$ yields $\operatorname{SiLU}(e_i) = \operatorname{SiLU}(1) e_i$, so the output linear map assigns any prescribed pair $(U_1, U_2)$ exactly. Direct computation gives $v_{11} = 0$, $v_{12} = -e_1$, $v_{21} = 2e_2$, $v_{22} = -e_1 + 2e_2$ (spanning $D = \mathbb{R}^2$), and
$$
E = \begin{pmatrix} 0 & 1 \\ 4 & 5 \end{pmatrix}, \qquad
G = s\begin{pmatrix} 0 & 3 \\ 6 & 9 \end{pmatrix}.
$$
Hence $E + G = \begin{pmatrix} 0 & 1 + 3s \\ 4 + 6s & 5 + 9s \end{pmatrix}$. Row $2$ equals row $1$ plus the constant $4 + 6s$, which row-wise softmax cancels. Therefore both rows of $A$ equal $(p, 1-p)$, giving $\operatorname{rank}(A) = 1$, $\sigma_2(A) = 0$, and $\operatorname{EffRank}(H_1) = \operatorname{EffRank}(A) = 1$ for every $s, \tau > 0$. Taking $s/\tau$ arbitrarily large defeats any finite strength or within-row contrast threshold, and all subsequent stochastic multiplications preserve the identical rows. $\square$

*Extensions.* For $r = 1$, the same queries and keys with $U_1 = 2\sqrt{s}\,e_1$ and $U_2 = \sqrt{s}\,e_2$ give $G = s\begin{pmatrix}0 & 2 \\ 2 & 4\end{pmatrix}$ and identical attention rows. Padding the full-rank factors with $r - 2$ common orthogonal columns along extra feature axes and padding $H_0, W_Q, W_K$ with zeros covers every $2 \le r \le d$; duplicating either token type covers every $L \ge 2$. No depth-dependent weights are used anywhere in Theorem 1.

---

## 3. Deep Collapse with Persistent Distinct Preferred Keys

**Theorem 2 (Deep collapse despite persistent margins and $\sigma_2(A_n) \to 1$).** *Fix $H_0 = I_2$, $\tau = 1$, and any margin sequence $b_n > 1$. There exists a pure two-token Riemannian self-attention + shared SiLU MetricNet stack of arbitrary depth in which, at every layer $n \ge 1$, the metric factors have full rank $r = 2$, $\bar U_n^2 = 1 + 3s_n$, $\sigma_{U,n}^2 = s_n^2 > 0$ ($s_n := (b_n - 1)/2 > 0$), each token strictly prefers itself by score margin $b_n$ (with metric contrast $2s_n$), and*
$$
A_n = J + \rho_n T, \qquad \rho_n := \tanh(b_n/2) = \sigma_2(A_n), \qquad
H_n = P_n = J + t_n T, \qquad t_n := \prod_{j=1}^n \rho_j,
$$
*where $J := \frac{1}{2}\mathbf{1}\mathbf{1}^\top$ and $T := I_2 - J$, so $\operatorname{EffRank}(H_n) = 1 + t_n^2$. In particular, for any fixed $b_n = b > 1$, $\sigma_2(A_n) = \tanh(b/2)$ can be arbitrarily close to $1$ while $\operatorname{EffRank}(H_n) = 1 + \tanh(b/2)^{2n} \to 1$.*

*Proof.* At layer $n$, prescribe $q_1 = k_1 = 0$, $q_2 = k_2 = e_1$, and
$$
U_1 = \operatorname{diag}(\sqrt{s_n}, 1), \qquad U_2 = \operatorname{diag}(\sqrt{3s_n}, 1).
$$
Then $E = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ and $G = \begin{pmatrix} 0 & 2s_n \\ 2s_n & 0 \end{pmatrix}$, so $E + G = \begin{pmatrix} 0 & b_n \\ b_n & 0 \end{pmatrix}$, giving $A_n = \frac{1}{1+e^{-b_n}}\begin{pmatrix}1 & e^{-b_n} \\ e^{-b_n} & 1\end{pmatrix} = J + \rho_n T$ with $\rho_n = \tanh(b_n/2)$. Since $J$ and $T$ are complementary orthogonal projections ($J^2 = J$, $T^2 = T$, $JT = TJ = 0$), induction gives $H_n = P_n = J + t_n T$ with singular values $1$ and $t_n$, hence $\operatorname{EffRank}(H_n) = 1 + t_n^2$.

To realize these layers as a self-attention + MetricNet stack on the evolving state $H_{n-1}$, set $t_0 := 1$ and define the predetermined decoder matrix $B_n := J + t_{n-1}^{-1} T$, which satisfies $H_{n-1} B_n = I_2$. Setting $W_Q^{(n)} = W_K^{(n)} = B_n Q_*$ (with rows of $Q_*$ equal to $0$ and $e_1^\top$) reproduces $q_1, q_2, k_1, k_2$, and using $B_n$ as the first linear map of MetricNet produces $\operatorname{SiLU}(H_{n-1} B_n) = \operatorname{SiLU}(1) I_2$, so its second linear map assigns $(U_1, U_2)$ exactly. $\square$

*Explicit weight-norm caveat.* For constant margin $b_n = b$, the decoder norm $\|B_n\|_{\mathrm{op}} = t_{n-1}^{-1} = \tanh(b/2)^{-(n-1)}$ grows exponentially with depth in order to keep the pre-softmax scores and metric diversity constant while $H_{n-1}$ coalesces. Neither `conj:rankpreserve` nor `conj:rankstrong` imposes a uniform bound on the layer weights; moreover, as Theorem 3 shows next, imposing uniformly bounded weights causes collapse even *faster* (doubly exponentially), while Theorem 1 already refutes both conjectures at layer $1$ with no depth-dependent weights at all.

---

## 4. General Dobrushin Consensus and Zero-Limit Exception

For any row-stochastic matrix $A$, the Dobrushin ergodicity coefficient (Dobrushin, 1956) is
$$
\delta(A) := \frac{1}{2}\max_{i,k} \sum_j |A_{ij} - A_{kj}| = 1 - \min_{i,k} \sum_j \min(A_{ij}, A_{kj}).
$$
For any row matrix $X$ with row diameter $d(X) := \max_{i,k}\|X_i - X_k\|$ (in any fixed norm), splitting each pair of rows of $A$ into their common mass and equal remainders of mass $\le \delta(A)$ gives $d(AX) \le \delta(A) d(X)$ and submultiplicativity $\delta(AB) \le \delta(A)\delta(B)$.

**Theorem 3 (General consensus and doubly exponential bounded-weight collapse).**
1. *Let $R_n$ be the maximum within-row score range at layer $n$. Then every entry of $A_n$ is at least $e^{-R_n}/L$, so $\delta(A_n) \le 1 - e^{-R_n}$. For Riemannian scores, $R_n \le (1 + \bar U_n^2)M_n / \tau_n$ where $M_n := \max_{i,j}\|q_i - k_j\|^2$. Whenever $\sum_{n=1}^\infty e^{-R_n} = \infty$ (in particular, whenever $R_n$ is uniformly bounded—on **either** side of $\bar U_{\mathrm{crit}}^2 = \Theta(\tau/L)$), we have $\delta(P_n) \le \prod_{j=1}^n \delta(A_j) \to 0$, so $P_n \to \mathbf{1}\pi^\top$ for a probability vector $\pi$ (cf. Wolfowitz, 1963; Leizarowitz, 1992). Whenever $\pi^\top H_0 \neq 0$ (e.g., $0$ lies outside the convex hull of the rows of $H_0$), $\operatorname{EffRank}(H_n) \to 1$.*
2. *If the layer weights $W_Q^{(n)}, W_K^{(n)}, U_t^{(n)}$ are uniformly bounded and $\tau_n \ge \tau > 0$, then $d(H_n) \le C' d(H_{n-1})^2$ for a constant $C'$ independent of $n$. Since part (1) first drives $d(H_n) \to 0$, once $C' d(H_n) < 1$ the diameter decays **doubly exponentially**; if $\pi^\top H_0 \neq 0$, then $\operatorname{EffRank}(H_n) - 1$ decays doubly exponentially (matching Dong et al., 2021).*

*Proof.* Part (1): Row-wise softmax with score range $\le R_n$ satisfies $A_{ij} \ge e^{-R_n}/L$, so $\sum_j \min(A_{ij}, A_{kj}) \ge e^{-R_n}$ and $\delta(A_n) \le 1 - e^{-R_n} \le \exp(-e^{-R_n})$. Divergence of $\sum e^{-R_n}$ forces $\delta(P_n) \to 0$. Because the convex hulls of the rows of $P_n$ are nested compact sets whose diameter tends to $0$, $P_n \to \mathbf{1}\pi^\top$ and $H_n = P_n H_0 \to \mathbf{1}(\pi^\top H_0)$. Continuity of $\operatorname{EffRank}$ at any nonzero rank-one matrix gives $\operatorname{EffRank}(H_n) \to 1$.

Part (2): Convex averaging keeps $\|H_n\|$ uniformly bounded, so $R_n$ is uniformly bounded and part (1) gives $d(H_n) \to 0$. At layer $n$, replacing every $q_i, k_j$ in $E_{ij} + G_{ij}$ by $q_1, k_1$ (while keeping $U_i, U_j$ fixed) yields scores of the form $c + a_i + a_j$, whose row-wise softmax has identical rows ($\delta = 0$). By $|v^\top g v - w^\top g w| \le \|g\|_{\mathrm{op}}\|v - w\|(\|v\| + \|w\|)$ and uniform boundedness of weights and $H_{n-1}$, this replacement perturbs each score by at most $C d(H_{n-1})$. Since row-wise softmax is Lipschitz, $\delta(A_n) \le C' d(H_{n-1})$ and hence $d(H_n) \le \delta(A_n) d(H_{n-1}) \le C' d(H_{n-1})^2$. For $\pi^\top H_0 \neq 0$, $\sigma_1(H_n)$ stays bounded below while the distance from $H_n$ to a rank-one matrix is at most $\sqrt{L}\,d(H_n)$, giving doubly exponential decay of $\operatorname{EffRank}(H_n) - 1$. $\square$

**Remark (Equilateral-triangle zero-limit exception).** The condition $\pi^\top H_0 \neq 0$ in Theorem 3 cannot be dropped. Take three unit vectors forming a centered equilateral triangle in $\mathbb{R}^2$ as the rows of $H_0$, with $Q = K = H$ and $U = 0$. If $H = x H_0$, every off-diagonal squared distance is $3x^2$, and with $z = e^{-3x^2/\tau}$ attention multiplies the centered subspace by $(1-z)/(1+2z) > 0$. Thus $H_n = x_n H_0$ with $x_n > 0$ for all finite $n$ and $x_n \to 0$, yet both nonzero singular values of $H_n$ are equal, so $\operatorname{EffRank}(H_n) = 2$ for every $n$. A heterogeneous, arbitrarily subcritical version embeds the triangle into the first two coordinates of $\mathbb{R}^3$ (with one initial vertex at $(1, 0, 0)$) and sets $U(h) = \eta \operatorname{diag}(1, 1, 2 + h_1)$ (realizable by a biased SiLU MLP using $\operatorname{SiLU}(x) - \operatorname{SiLU}(-x) = x$). Every active distance scales uniformly by $1 + \eta^2$, $\sigma_U^2 > 0$ at every finite depth, and $\bar U^2 \le 11\eta^2$ can be arbitrarily subcritical, yet $\operatorname{EffRank}(H_n) = 2$ for all $n$ while $H_n \to 0$.

---

## 5. Exact Two-Token Preservation Criterion

**Theorem 4 (Necessary and sufficient condition for $L = 2$).** *Let $A_j = \begin{pmatrix}1 - a_j & a_j \\ b_j & 1 - b_j\end{pmatrix}$ ($j \ge 1$) be any sequence of $2 \times 2$ row-stochastic matrices, let $u_j := |1 - a_j - b_j|$, and let $H_0 \in \mathbb{R}^{2 \times d}$ have rank $2$. Then $\inf_n \operatorname{EffRank}(H_n) > 1$ **if and only if***
$$
u_j > 0 \quad (\forall j \ge 1) \qquad \text{and} \qquad \sum_{j=1}^\infty (1 - u_j) < \infty.
$$

*Proof.* For any $2 \times 2$ row-stochastic matrix $A = \begin{pmatrix}1-a & a \\ b & 1-b\end{pmatrix}$, direct calculation gives $\delta(A) = |1 - a - b| = |\det A|$. Multiplicativity of the determinant therefore gives the exact identity $\delta(P_n) = |\det P_n| = \prod_{j=1}^n u_j$. By the infinite-product criterion ($-\log u \sim 1 - u$ as $u \to 1^-$), $\inf_n |\det P_n| > 0$ holds if and only if $u_j > 0$ for all $j$ and $\sum_{j=1}^\infty (1 - u_j) < \infty$.

Since $P_n$ is $2 \times 2$ row-stochastic, $1 \le \sigma_1(P_n) \le \sqrt{2}$, so $\sigma_2(P_n) / \sigma_1(P_n) = |\det P_n| / \sigma_1(P_n)^2 \in [\frac{1}{2}|\det P_n|, |\det P_n|]$. Because $H_0$ has rank $2$, $H_0 H_0^\top$ is positive definite with $\sigma_2(H_0)^2 P_n P_n^\top \preceq H_n H_n^\top \preceq \sigma_1(H_0)^2 P_n P_n^\top$, which bounds $\sigma_2(H_n)/\sigma_1(H_n)$ above and below by constant multiples of $\sigma_2(P_n)/\sigma_1(P_n)$ (and hence of $|\det P_n|$). Since $\operatorname{EffRank}(H_n) = 1 + (\sigma_2(H_n)/\sigma_1(H_n))^2$, we have $\inf_n \operatorname{EffRank}(H_n) > 1 \iff \inf_n |\det P_n| > 0$. $\square$

For the symmetric margin family of Theorem 2, $1 - u_j = 2/(1 + e^{b_j})$, so preservation holds iff $\sum_j e^{-b_j} < \infty$: even divergent margins such as $b_j = \log(2j + 1)$ collapse, whereas $b_j = \log(2(j+1)^2 - 1)$ preserves effective rank strictly above $1$. In general dimension $L \ge 2$, the paper's permutation-margin result (Proposition `prop:margin`, assuming $\|A_j - \Pi_j\|_{\mathrm{op}} \le \varepsilon_j < 1$ with $\sum \varepsilon_j < \infty$) remains a valid sufficient condition, showing why summable leakage—rather than a fixed large margin—is the sharp threshold.

---

## 6. Verification and References

Running `.venv/bin/python -B solutions/riemannian_attn_rank/explore.py` (NumPy + `mpmath` at 90 decimal digits to control open-loop decoder roundoff) verifies the three single-layer counterexamples ($s = 1, 4, 16$) and all $600$ stack-depth assertions across three margin schedules up to depth $200$:

| Margin $b_j$ at layer $j$ | Exact $\sigma_2(H_n) = \delta(P_n)$ | $\sigma_2(A_{200})$ | $\operatorname{EffRank}(H_{200})$ | $\lim_{n\to\infty} \operatorname{EffRank}(H_n)$ |
|---|---|---:|---:|---:|
| $4$ | $\tanh(2)^n$ | $0.9640275801$ | $1.0000004323$ | $1$ |
| $\log(2j + 1)$ | $1/(n + 1)$ | $0.9950248756$ | $1.0000247519$ | $1$ |
| $\log(2(j+1)^2 - 1)$ | $(n + 2)/(2(n + 1))$ | $0.9999752481$ | $1.2524937502$ | $1.25$ |

- **References:**
  - R. L. Dobrushin (1956), *Central limit theorem for nonstationary Markov chains. I*, Theory of Probability and Its Applications 1(1):65–80.
  - J. Wolfowitz (1963), *Products of indecomposable, aperiodic, stochastic matrices*, Proceedings of the American Mathematical Society 14(5):733–737.
  - A. Leizarowitz (1992), *On infinite products of stochastic matrices*, Linear Algebra and Its Applications 168:189–219.
  - Y. Dong, J.-B. Cordonnier, and A. Loukas (2021), *Attention is not all you need: Pure attention loses rank doubly exponentially with depth*, ICML 2021, PMLR 139:2793–2803.
