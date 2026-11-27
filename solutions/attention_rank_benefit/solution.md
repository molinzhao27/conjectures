# Refutation of the All-Length Low-Rank Attention Conjecture

**Target paper:** Noah Amsel, Gilad Yehudai, and Joan Bruna, *Quality over Quantity in Attention Layers* (ICLR 2025; arXiv title: *On the Benefits of Rank in Attention Layers*, [arXiv:2407.16153](https://arxiv.org/abs/2407.16153)), Section 6, Conjecture 8 (`conjecture:N`).  
**Companion files:** [`solutions/attention_rank_benefit/explore.py`](explore.py), [`solutions/attention_rank_benefit/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 8 (`conjecture:N`): DISPROVED** under the mean-square error criterion of the paper's adjacent theorems (Fact 1 and Section 6).
- **Uniform three-layer rank-1 construction (Theorem 1):** For every fixed input dimension $d \ge 2$ and every $\epsilon > 0$, there exists a fixed-size, fixed-weight **three-layer rank-1 softmax Transformer** $T$ (with rank-1 query/key and value/output matrices in every head, attention and residuals alone, and no MLP or normalization) such that
  $$
  \sup_{N \ge 1} \mathbb{E}\,\bigl\|T(x_1, \dots, x_N, y) - f(X; y)\bigr\|_2^2 < \epsilon,
  $$
  where $x_1, \dots, x_N, y \stackrel{\text{i.i.d.}}{\sim} \operatorname{Unif}(\mathbb{S}^{d-1})$ and $f(X; y) = \arg\max_{x_i} x_i^\top y$. All hyperparameters (grid of search directions, head counts, hidden width, period-two parity positional encoding, and weights) depend only on $(d, \epsilon)$ and are completely independent of the context length $N$.
- **Single-head-per-layer and MLP variants (Corollaries 2–3):** Serializing the three stages yields a fixed-depth Transformer with **one rank-1 head per layer** ($K(K-1) + K + d(K+1)$ layers, independent of $N$). Adding a pointwise ReLU MLP removes even the parity positional bits.
- **Explicit scope boundaries:**
  1. *Dimension scaling:* While the architecture is completely independent of $N$, its width ($2d + 2K + 2$) and head count grow exponentially in $d$ through the $\rho$-net size $K$. Whether a $\operatorname{poly}(d)$-size, $N$-independent low-rank Transformer exists remains open.
  2. *Worst-case-over-inputs error:* Replacing expected squared error by a worst-case supremum over individual inputs $(X, y)$ changes the problem: at $N = 2$ with antipodal candidates $\pm e_1$, the target jumps between $+e_1$ and $-e_1$ across the hyperplane $y^\top e_1 = 0$, so any continuous network (of any attention rank) has worst-case Euclidean error at least $1$ (squared error $\ge 1$).
  3. *Exact hardmax:* Exact hardmax retrieval across all $N$ by a fixed-size low-rank Transformer remains open.

---

## 1. Exact Problem Statement and Architecture Model

Given candidate vectors $X = (x_1, \dots, x_N)$ and a query $y$ drawn independently and uniformly from the unit sphere $\mathbb{S}^{d-1} \subset \mathbb{R}^d$ ($d \ge 2$), the nearest-neighbor target is
$$
f(X; y) := \arg\max_{x_i \in \{x_1, \dots, x_N\}} x_i^\top y = \arg\min_{x_i \in \{x_1, \dots, x_N\}} \|x_i - y\|_2,
$$
which is almost surely unique. Section 3 of Amsel, Yehudai, and Bruna ([arXiv:2407.16153](https://arxiv.org/abs/2407.16153)) defines both softmax and hardmax attention; Fact 1 and the Section 6 depth theorem evaluate approximation by expected squared Euclidean error $\mathbb{E}\|T(X, y) - f(X; y)\|_2^2$ under the uniform product law (the orthogonal-candidate law used in shallow lower bounds is only defined for $N \le d$).

**Conjecture 8 (`conjecture:N`, arXiv:2407.16153 / ICLR 2025, §6).** *There is no multi-layer transformer (with fixed size and weight matrices) of rank $r < d$ that approximates the target $f(X; y)$ for all $N$.*

We adopt the exact theoretical architecture of Section 6 of the paper: each layer applies diagonally-masked multi-head self-attention, sums the head outputs, adds the input as a residual, and the network reads out the final token via a linear map (permitting concatenated constant, positional, and scratch coordinates). In our construction, every head has both query/key matrix and value/output matrix of rank $1$.

---

## 2. Uniform Finite Softmax Temperature Across All Lengths

For any candidate tuple $X = (x_1, \dots, x_N) \in (\mathbb{S}^{d-1})^N$ and any vector $v \in \mathbb{R}^d$, define the softmax-weighted average and candidate mean
$$
B_X(v) := \frac{\sum_{i=1}^N e^{v^\top x_i} x_i}{\sum_{i=1}^N e^{v^\top x_i}},
\qquad
\mu_X := B_X(0) = \frac{1}{N} \sum_{i=1}^N x_i.
$$

**Lemma 1 (Uniformity in length and directional stability).**
1. *Uniform temperature:* $\displaystyle \lim_{\beta \to \infty} \sup_{N \ge 1} \mathbb{E}\bigl\|B_X(\beta y) - f(X; y)\bigr\|_2^2 = 0$.
2. *Lipschitz stability:* For every $N \ge 1$, $X \in (\mathbb{S}^{d-1})^N$, and $v, w \in \mathbb{R}^d$, $\|B_X(v) - B_X(w)\|_2 \le \|v - w\|_2$.

*Proof.* For (1), fix $\delta \in (0, 1)$. Conditional on $y \in \mathbb{S}^{d-1}$, let $p := \Pr(x_1^\top y \ge 1 - \delta) > 0$, which is independent of $y$ by rotational invariance. Let $M_N := |\{i \in [N] : x_i^\top y \ge 1 - \delta\}| \sim \operatorname{Bin}(N, p)$. By Chebyshev's inequality,
$$
\Pr(M_N < Np/2) \le \frac{Np(1-p)}{(Np/2)^2} \le \frac{4}{Np}.
$$
On the event $\{M_N \ge Np/2\}$, the best candidate $f(X; y)$ satisfies $\|f(X; y) - y\|_2^2 \le 2\delta$. Moreover, the total softmax weight $w$ assigned to candidates with $x_i^\top y < 1 - 2\delta$ satisfies
$$
w \le \frac{N e^{\beta(1 - 2\delta)}}{(Np/2) e^{\beta(1 - \delta)}} = \frac{2}{p} e^{-\beta\delta}.
$$
Every candidate with $x_i^\top y \ge 1 - 2\delta$ has $\|x_i - y\|_2^2 \le 4\delta$, and every candidate satisfies $\|x_i - y\|_2^2 \le 4$. By convexity of $z \mapsto \|z - y\|_2^2$, $\|B_X(\beta y) - y\|_2^2 \le 4\delta + 4w$, whence
$$
\|B_X(\beta y) - f(X; y)\|_2^2
\le 2\|B_X(\beta y) - y\|_2^2 + 2\|y - f(X; y)\|_2^2
\le 12\delta + 8w
\le 12\delta + \frac{16}{p} e^{-\beta\delta}.
$$
Since $\|B_X(\beta y) - f(X; y)\|_2^2 \le 4$ everywhere, taking expectations gives
$$
\mathbb{E}\bigl\|B_X(\beta y) - f(X; y)\bigr\|_2^2
\le 12\delta + \frac{16}{p} e^{-\beta\delta} + \frac{16}{Np}.
$$
Given any $\eta > 0$, choose $\delta > 0$ with $12\delta < \eta/3$, then a finite cutoff $N_0$ with $16/(N_0 p) < \eta/3$, and then $\beta_1$ large enough that $\frac{16}{p}e^{-\beta\delta} < \eta/3$ for all $\beta \ge \beta_1$; this makes the expectation $< \eta$ uniformly for all $N \ge N_0$. For each of the finitely many lengths $N \in \{1, \dots, N_0 - 1\}$, the unique maximizing candidate receives softmax weight tending to $1$ almost surely as $\beta \to \infty$, so dominated convergence yields $\beta_2 \ge \beta_1$ controlling all $N < N_0$ simultaneously. (The cutoff $N_0$ is purely analytical and does not enter the network.)

For (2), differentiating $B_X(v)$ shows that the Jacobian $D B_X(v)$ is the covariance matrix of a random vector drawn from $\{x_1, \dots, x_N\}$ with the softmax weights. Since $\|x_i\|_2 = 1$, for every unit vector $u$ we have $u^\top D B_X(v) u \le \mathbb{E}[(u^\top x)^2] \le 1$, so $\|D B_X(v)\|_{\mathrm{op}} \le 1$, giving $\|B_X(v) - B_X(w)\|_2 \le \|v - w\|_2$. $\square$

---

## 3. Three-Layer Rank-1 Routing and Retrieval Construction

Choose a finite Euclidean $\rho$-net of distinct unit directions $u_1, \dots, u_K \in \mathbb{S}^{d-1}$ with $K \ge 2$. For each $y \in \mathbb{S}^{d-1}$, let $h_*(y) := \arg\max_{h \in [K]} u_h^\top y$ (unique away from a finite union of great subspheres of measure zero), so $\|u_{h_*(y)} - y\|_2 \le \rho$.

Embed the input sequence $(x_1, \dots, x_N, y)$ of length $N + 1$ into $\mathbb{R}^{2d + 2K + 2}$ by appending to token $i \in \{1, \dots, N+1\}$ a constant coordinate $1$, the period-two parity bit $b_i := \mathbf{1}_{i \text{ odd}}$, and $2K + d$ zero scratch coordinates (blocks `wins` $\in \mathbb{R}^K$, `gates` $\in \mathbb{R}^K$, and `out` $\in \mathbb{R}^d$). Each layer reads only previously written coordinates and writes only into its fresh block, so the original data, constants, and parity bits pass unchanged through every residual connection.

**Parity-bit comparator primitive.** For $N \ge 2$, consider a rank-1 head evaluated at the final query token $N + 1$ with scalar query $t$, key $b_i$, and value $b_i$. Diagonal masking excludes token $N + 1$, leaving the $N$ candidates, of which $n_1 := \lceil N/2 \rceil$ have $b_i = 1$ and $n_0 := \lfloor N/2 \rfloor$ have $b_i = 0$. The head's scalar output is
$$
S_N(t) = \frac{n_1 e^t}{n_0 + n_1 e^t} = \sigma(t + \ell_N),
\qquad
\ell_N := \log\frac{n_1}{n_0} \in [0, \log 2],
$$
where $\sigma(z) = (1 + e^{-z})^{-1}$. Because the constant coordinate $1$ is present, any affine functional of earlier coordinates is a valid scalar query; hence $S_N(t) \to 1$ as $t \to +\infty$ and $S_N(t) \to 0$ as $t \to -\infty$, uniformly over all $N \ge 2$, without ever computing $N$ or $\ell_N$.

The three diagonally-masked self-attention layers are defined identically at all token positions (using gains $a_1, a_2 > 0$, or a single gain $a = a_1 = a_2$, and inverse temperature $\beta > 0$):
1. **Layer 1 — Pairwise direction comparisons ($K(K-1)$ heads):** For each ordered pair $h \neq k$, a rank-1 head uses scalar query $a_1(u_h - u_k)^\top y$, key $b_i$, value $b_i$, and writes into coordinate $\text{wins}_h$. At token $N + 1$,
   $$
   c_h = \sum_{k \neq h} S_N\bigl(a_1(u_h - u_k)^\top y\bigr).
   $$
2. **Layer 2 — Winner gating ($K$ heads):** For each $h \in [K]$, a rank-1 head uses scalar query $a_2\bigl(c_h - (K - \tfrac{3}{2})\bigr)$, key $b_i$, value $b_i$, and writes $g_h = S_N\bigl(a_2(c_h - (K - \tfrac{3}{2}))\bigr)$ into $\text{gates}_h$. (The unique winner has $K - 1$ strict wins, while every loser has at most $K - 2$.)
3. **Layer 3 — Directional retrieval and baseline cancellation ($d(K+1)$ heads):** For each $h \in [K]$ and coordinate $j \in [d]$, a rank-1 head uses key $u_h^\top x_i$, scalar query $\beta g_h$, value $x_{ij}$, and writes into $\text{out}_j$; together these $d$ heads contribute $B_X(\beta g_h u_h)$. Another $d$ heads with constant key $1$, constant query $1$, and value $(1 - K)x_{ij}$ add $(1 - K)\mu_X$ into `out`. The final linear map reads `out`, yielding
   $$
   T(X, y) = \mu_X + \sum_{h=1}^K \bigl[B_X(\beta g_h u_h) - \mu_X\bigr].
   $$

Every head's query/key matrix is an outer product of two vectors (rank $1$), and every head's value/output matrix maps one scalar projection to one coordinate vector (rank $1$). At token $N + 1$, keys and values read from tokens $1, \dots, N$ depend only on their preserved data, constants, and parity bits, never on other tokens' scratch registers; moreover, $T(X, y)$ is invariant under permuting $x_1, \dots, x_N$. When $N = 1$, every retrieval head outputs $x_1$ and $\mu_X = x_1$, so $T(x_1, y) = x_1 = f(X; y)$ **exactly** for any gate values.

---

## 4. Uniform Error Analysis and Variants

**Theorem 1 (Uniform all-length approximation by a fixed rank-1 Transformer).** *For every $d \ge 2$ and $\epsilon > 0$, there exist finite $\beta, \rho, a_1, a_2$ (hence fixed width $2d + 2K + 2$ and fixed weights) such that the three-layer rank-1 softmax Transformer $T$ above satisfies*
$$
\sup_{N \ge 1} \mathbb{E}\bigl\|T(x_1, \dots, x_N, y) - f(X; y)\bigr\|_2^2 < \epsilon.
$$

*Proof.* Take $a_1 = a_2 = a$ (allowing distinct $a_1, a_2$ works identically). On the full-measure set of queries $y \in \mathbb{S}^{d-1}$ outside the finite union of hyperplanes $\bigcup_{h \neq k} \{y : (u_h - u_k)^\top y = 0\}$, every pairwise comparison is strict: $c_{h_*(y)} \to K - 1$ and $c_h \le K - 2 + o(1)$ for $h \neq h_*(y)$ as $a \to \infty$, uniformly over $N \ge 2$ since $\ell_N \in [0, \log 2]$. Once $c_{h_*(y)} \ge K - \tfrac{5}{4}$ and $c_h \le K - \tfrac{7}{4}$ for $h \neq h_*(y)$, the uniform offset bound $\ell_N \in [0, \log 2]$ forces
$$
e_a(y) := \sup_{N \ge 2} \max_{1 \le h \le K} \bigl|g_h - \mathbf{1}_{h = h_*(y)}\bigr| \longrightarrow 0
\qquad \text{as } a \to \infty
$$
for almost every $y \in \mathbb{S}^{d-1}$. Since $0 \le e_a(y) \le 1$, dominated convergence gives $\mathbb{E}[e_a(y)^2] \to 0$ as $a \to \infty$ for any fixed $\rho$-net.

For $N \ge 2$, using $\mu_X = B_X(0)$ and Lemma 1(2) on each term of $T(X, y) - B_X(\beta y)$,
$$
\bigl\|T(X, y) - B_X(\beta y)\bigr\|_2
\le \bigl\|B_X(\beta g_{h_*} u_{h_*}) - B_X(\beta y)\bigr\|_2 + \sum_{h \neq h_*} \bigl\|B_X(\beta g_h u_h) - B_X(0)\bigr\|_2
\le \beta\|g_{h_*} u_{h_*} - y\|_2 + (K - 1)\beta e_a(y)
\le K\beta e_a(y) + \beta\rho,
$$
since $\|g_{h_*} u_{h_*} - y\|_2 \le |g_{h_*} - 1| + \|u_{h_*} - y\|_2 \le e_a(y) + \rho$.
Now choose parameters in order:
1. By Lemma 1(1), choose $\beta > 0$ such that $\sup_{N \ge 1} \bigl(\mathbb{E}\|B_X(\beta y) - f(X; y)\|_2^2\bigr)^{1/2} < \sqrt{\epsilon}/3$.
2. Choose a $\rho$-net $\{u_1, \dots, u_K\} \subset \mathbb{S}^{d-1}$ with $\beta\rho < \sqrt{\epsilon}/3$.
3. Choose $a > 0$ large enough that $K\beta\bigl(\mathbb{E}[e_a(y)^2]\bigr)^{1/2} < \sqrt{\epsilon}/3$.

By Minkowski's inequality in $L^2$, for every $N \ge 2$,
$$
\bigl(\mathbb{E}\|T(X, y) - f(X; y)\|_2^2\bigr)^{1/2}
\le \bigl(\mathbb{E}\|B_X(\beta y) - f(X; y)\|_2^2\bigr)^{1/2} + \beta\rho + K\beta\bigl(\mathbb{E}[e_a(y)^2]\bigr)^{1/2}
< \sqrt{\epsilon}.
$$
Since $T(x_1, y) = f(x_1; y)$ exactly for $N = 1$, squaring gives $\sup_{N \ge 1} \mathbb{E}\|T(X, y) - f(X; y)\|_2^2 < \epsilon$. $\square$

**Corollary 2 (One rank-1 head per layer).** *Replacing each parallel layer by a sequence of single-head residual layers—one for each head in that stage—produces the exact same map $T(X, y)$ with fixed depth $K(K-1) + K + d(K+1)$, independent of $N$.*

*Proof.* Within each stage, no head reads the scratch coordinates written by that stage, so sequential residual updates leave every head's logits and values identical to the parallel evaluation at every token, and addition is associative. $\square$

**Corollary 3 (Position-free variant with a pointwise ReLU MLP).** *If a pointwise ReLU MLP is permitted alongside attention, the parity bits can be dropped entirely: compute directly at each token*
$$
g_h(y) = \min\!\Bigl(1,\; \max\!\Bigl(0,\; \min_{k \neq h} \frac{(u_h - u_k)^\top y}{\eta}\Bigr)\Bigr)
$$
*using affine maps and ReLUs (via $\min(s, t) = s - \operatorname{ReLU}(s - t)$), then apply Layer 3. For almost every $y$, $g_h(y) = \mathbf{1}_{h = h_*(y)}$ for all sufficiently small $\eta > 0$, and the same proof gives $\sup_{N \ge 1} \mathbb{E}\|T(X, y) - f(X; y)\|_2^2 < \epsilon$ using only content and shared constants.*

**Proposition 4 (Impossibility of worst-case-over-inputs approximation).** *At $N = 2$, any continuous map $T : (\mathbb{S}^{d-1})^3 \to \mathbb{R}^d$ satisfies $\sup_{x_1, x_2, y \in \mathbb{S}^{d-1}} \|T(x_1, x_2, y) - f(x_1, x_2; y)\|_2 \ge 1$.*

*Proof.* Fix $x_1 = e_1$ and $x_2 = -e_1$, and approach $y_0 = e_2$ (on the equator $y^\top e_1 = 0$) from the two hemispheres $y_\pm = \cos\theta\, e_2 \pm \sin\theta\, e_1$ ($\theta \downarrow 0$). Then $f(x_1, x_2; y_\pm) = \pm e_1$, while $T(x_1, x_2, y_\pm) \to T(x_1, x_2, y_0)$. By the triangle inequality, $\max\{\|T(x_1, x_2, y_0) - e_1\|_2, \|T(x_1, x_2, y_0) + e_1\|_2\} \ge 1$. $\square$

---

## 5. Computational Verification

Run from the repository root with Python and NumPy:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -B solutions/attention_rank_benefit/explore.py
```

The script (seed `240716153`) checks:
- **Exact layer equivalence:** $24$ literal diagonally-masked parallel and serial Transformer evaluations (at $N \in \{1, 2, 3, 4, 7, 16\}$, including a routing boundary query $y = (\cos\frac{\pi}{5}, \sin\frac{\pi}{5})$ and candidate reversal invariance) match the closed-form readout to within $2 \times 10^{-15}$.
- **Fixed-weight simulation across lengths $N = 1$ to $4097$:** Using $K = 64$ equally spaced directions for $d = 2$ and $K = 96$ Fibonacci-sphere directions for $d = 3$, with fixed parameters $a_1 = 30000$, $a_2 = 90$, $\beta = 300$ across every $N$:

| Candidates $N$ | Trials per $d$ | Rank-1 MSE ($d=2$) | Rank-1 MSE ($d=3$) |
|---:|---:|---:|---:|
| 1 | 1024 | 0.000000 | 0.000000 |
| 2 | 1024 | 0.009742 | 0.054536 |
| 3 | 1024 | 0.034967 | 0.147365 |
| 8 | 1024 | 0.009806 | 0.186605 |
| 31 | 1024 | 0.012650 | 0.122857 |
| 128 | 1024 | 0.004507 | 0.062665 |
| 511 | 256 | 0.000899 | 0.036309 |
| 2048 | 256 | 0.000846 | 0.024176 |
| 4097 | 256 | 0.000887 | 0.021389 |

The largest observed MSE is $0.1866$ (SE $0.0189$) for the coarse $96$-point grid in $d = 3$ at $N = 8$, and decreases steadily for larger $N$; the uniform all-length bound for arbitrary $\epsilon > 0$ follows from Theorem 1.
