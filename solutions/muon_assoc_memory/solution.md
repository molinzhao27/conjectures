# Partial Resolution of Exact Muon Rates and Refutation of Literal Newton Matching on Associative Memory

**Target paper:** Juno Kim, Eshaan Nichani, Denny Wu, Alberto Bietti, and Jason D. Lee, *Sharp Capacity Scaling of Spectral Optimizers in Learning Associative Memory*, [arXiv:2603.26554](https://arxiv.org/abs/2603.26554), Section 4 (Conjecture following Theorem `thm:multi`).  
**Companion files:** [`solutions/muon_assoc_memory/explore.py`](explore.py), [`solutions/muon_assoc_memory/proof.pdf`](proof.pdf).

---

## Verdict

This is a **partial resolution** of the exact-Muon capacity conjecture together with a **refutation of literal Newton matching**:

1. **Exact sparse-minibatch Muon rates: PROVED (Theorem 2).** For fixed $\alpha > 1$, $\epsilon > 0$, horizon $T = O(\log d)$, polynomial vocabulary $N \gtrsim d^{2\alpha+2}$, and sparse minibatches $B^{1/\alpha} \le d^{1/2-\epsilon}$ with $B/\log d \to \infty$, the **exact** softmax-gradient Muon iterates $W_{t+1} = W_t + \eta h_{\lambda_t}(G_t)$—including the unstabilized polar map $\lambda_t \equiv 0$ and every schedule $0 \le \lambda_t \le C\sqrt{d}/[B(\log d)^6]$—recover every item observed in the first $t$ batches with confidence $P_{ii}(W_t) \ge 1 - \exp(-c\log^2 d)$, recover the full prefix $i \le c'(tB/\log d)^{1/\alpha}$, and achieve population loss $L(W_t) \le O((tB/\log d)^{(1-\alpha)/\alpha}\log N) + \exp(-c\log^2 d)$ simultaneously for all $1 \le t \le T$. A matching high-probability lower bound $L(W_t) \gtrsim (tB)^{(1-\alpha)/\alpha}\log N$ shows the batch exponent is sharp up to logarithms.
2. **Dimension-limited multistep exponent ($t \ge 2$, including population Muon): OPEN.** Establishing the conjectured dimension-limited exponent $d_t = \widetilde\Theta(d^{2-(1-1/(2\alpha))^t})$ for exact dense or population Muon at $t \ge 2$ remains unresolved here; the uniform resolvent proof requires $m\delta \ll 1$, and controlling the data-dependent transition band where $1 - P_{ii}(W_t)$ varies across $(0,1)$ remains open.
3. **Literal matching between Muon and Newton: DISPROVED (Theorems 3–5).**
   - *One-step retrieval discrepancy (4-item shear):* At $d = 2, N = 4$ with $p_i \propto i^{-2}$ and sheared outputs $u_i = A v_i$, Newton retrieves **4/4** items after one step whereas Muon retrieves **2/4** for **every** $\lambda \ge 0$ and every positive step size. This persists on an open neighborhood of positive Gaussian probability. Moreover, for every $d \ge 2$ and $N \ge 2d$, the first Newton and Muon directions are non-collinear almost surely under isotropic Gaussian embeddings.
   - *Limiting dynamics discrepancy (3-item scalar memory and Gaussian local dynamics):* On a 3-item scalar memory ($d = 1$), unit-step polar Muon enters a persistent 2-cycle $\{-1, -2\}$, stabilized Muon converges at most linearly, and Newton converges quadratically to $w_* \approx -1.1994401007$. In the high-dimensional Gaussian population model ($N/d^2 \to \infty$), $L$ has a unique finite minimizer with high probability, where Newton enjoys local quadratic convergence while fixed-step polar Muon steps have Frobenius norm at least $\eta$ and cannot converge locally.
4. **Equality of typical Gaussian multistep capacity exponents: OPEN.** Neither counterexample rules out matching high-dimensional scaling exponents between suitably scheduled Muon and Newton on typical isotropic Gaussian memories; that asymptotic comparison remains open.

---

## 1. Model, Exact Gradient, and Conjecture Statement

Let input and output embeddings $v_1,\dots,v_N, u_1,\dots,u_N \in \mathbb{R}^d$ be i.i.d. $\mathcal{N}(0, I_d/d)$ with Zipf probabilities $p_i \propto i^{-\alpha}$ for fixed $\alpha > 1$. For a memory matrix $W \in \mathbb{R}^{d \times d}$, define scores $z_{ki} := u_k^\top W v_i$ and softmax probabilities $P_{ki}(W) := \exp(z_{ki}) / \sum_{j=1}^N \exp(z_{ji})$. Item $i$ is strictly recovered if $z_{ii} > \max_{k \neq i} z_{ki}$. The population cross-entropy loss is
$$
L(W) := -\sum_{i=1}^N p_i \log P_{ii}(W).
$$
Each minibatch draws $B$ independent indices from $p$ with empirical frequencies $q^{(t)} = (q_1^{(t)}, \dots, q_N^{(t)})$ (and $q^{(t)} = p$ for population updates). Writing $\mu_i(W) := \sum_{k=1}^N P_{ki}(W) u_k$, the exact negative gradient is
$$
G_t := -\nabla \widehat{L}_t(W_t) = \sum_{i=1}^N q_i^{(t)} \bigl(u_i - \mu_i(W_t)\bigr) v_i^\top.
$$
For a singular value decomposition $G = U \operatorname{diag}(s) V^\top$, the stabilized spectral map is $h_\lambda(G) := U \operatorname{diag}(h_\lambda(s)) V^\top$ where $h_\lambda(s) := s/\sqrt{s^2 + \lambda^2}$ for $\lambda > 0$, and at $\lambda = 0$ the partial polar factor $h_0(s) = \mathbf{1}_{s > 0}$. Starting from $W_0 = 0$, the exact Muon iterate is $W_{t+1} = W_t + \eta h_{\lambda_t}(G_t)$.

In [arXiv:2603.26554](https://arxiv.org/abs/2603.26554), Theorem `thm:multi` analyzes a **deflation surrogate** $\bar{G}_t = \sum_{i > d_t} q_i^{(t)} u_i v_i^\top$ (dropping both the softmax residuals $1 - P_{ii}$ and the wrong-class means $\mu_i$), obtaining for $\eta \asymp \sqrt{d}/(\log d)^4$, polynomial $N \gtrsim d^{2\alpha+2}$, and $t = O(\log d)$:
$$
d_0 = 0, \qquad
d_t = \widetilde\Theta\!\Bigl(\min\bigl\{d^{2 - (1 - 1/(2\alpha))^t},\, B^{1/\alpha}\bigr\}\Bigr), \qquad
L(W_t) \le \widetilde{O}\bigl(d_t^{1-\alpha}\bigr).
$$
*(Schedule note: Theorem `thm:multi` displays $\lambda_t = \widetilde\Theta(\sqrt{d}\,d_{t+1}^{-\alpha})$, whereas its appendix proof uses $\lambda_0$ from the one-step theorem and $\lambda_t \asymp d_t^{1/2-\alpha} d^{-1/2}\log d$ for $t \ge 1$. When the batch cap binds these can differ by powers of $d$; Theorem 2 below accommodates both.)*

**Conjecture ([arXiv:2603.26554](https://arxiv.org/abs/2603.26554), Section 4).** *The recovery and convergence rates of Theorem `thm:multi` also hold for the exact Muon iterates $W_{t+1} = W_t + \eta h_{\lambda_t}(G_t)$. Moreover, Muon matches Newton's method throughout training.*

---

## 2. Exact Multistep Muon in the Sparse-Minibatch Regime

### 2.1 Uniform Entrywise Spectral Perturbation

Write $\|M\|_{\max} := \max_{i,j} |M_{ij}|$.

**Lemma 1 (Uniform spectral perturbation).** *There exist universal constants $c_0, C > 0$ such that if $A, V \in \mathbb{R}^{d \times m}$ satisfy $\|A^\top A - I_m\|_{\max} \le \delta$ and $\|V^\top V - I_m\|_{\max} \le \delta$ with $m\delta \le c_0$, then for every positive diagonal matrix $D = \operatorname{diag}(D_1, \dots, D_m) \succ 0$ and every $\lambda \ge 0$,*
$$
h_\lambda(A D V^\top) = A \bigl[h_\lambda(D) + E\bigr] V^\top, \qquad \|E\|_{\max} \le C\delta.
$$
*Crucially, $C$ is independent of $\min_i D_i$ and $\lambda$, which is essential because well-learned items acquire exponentially small residual weights $q_i(1 - P_{ii})$.*

*Proof.* Put $C_0 := \operatorname{diag}(A, V) \in \mathbb{R}^{2d \times 2m}$, $K := C_0^\top C_0$, and $J := \begin{psmallmatrix} 0 & D \\ D & 0 \end{psmallmatrix}$. The symmetric dilation $H := C_0 J C_0^\top$ has $h_\lambda(A D V^\top)$ as the upper-right $d \times d$ block of $h_\lambda(H)$. Using the scalar identity $z/\sqrt{z^2+\lambda^2} = \frac{2}{\pi}\int_0^\infty \operatorname{Re}(z - is)^{-1}\,dx$ with $s := \sqrt{\lambda^2 + x^2}$ (valid for all $z \in \mathbb{R}$ and $\lambda \ge 0$, interpreting $0/0 = 0$ at $z = \lambda = 0$), we integrate $\operatorname{Re}(H - isI)^{-1}$ against $\frac{2}{\pi}\,dx$ on $(0, \infty)$.

On the column space of $C_0$, the resolvent equals $C_0 K^{-1} (J - is K^{-1})^{-1} K^{-1} C_0^\top$, while on the orthogonal complement it is $(0 - isI)^{-1} = \frac{i}{s}I$ (purely imaginary, contributing zero to the real part). Write $F := K^{-1} - I_{2m} = \operatorname{diag}((A^\top A)^{-1} - I_m,\, (V^\top V)^{-1} - I_m)$. Since products of $n$ matrices of entrywise norm at most $\delta$ have entrywise norm at most $m^{n-1}\delta^n$, the Neumann series for $K^{-1}$ yields $\|F\|_{\max} \le 2\delta$ and $\|F\|_{\mathrm{op}} \le 2m\delta$ for $m\delta$ small.

Set $R(s) := (J - isI)^{-1}$. Because $s\|R(s)\|_{\mathrm{op}} \le 1$, the expansion
$$
(J - is K^{-1})^{-1} = (J - isI - isF)^{-1} = \sum_{n=0}^\infty R(s)\bigl(isF\,R(s)\bigr)^n
$$
converges in operator norm for $2m\delta < 1$. The block matrix $R(s)$ is diagonal in the item index $i \in \{1,\dots,m\}$: within each $2 \times 2$ block, its diagonal entries are $is/(D_i^2 + s^2)$ and its off-diagonal (cross-block) entries are $D_i/(D_i^2 + s^2)$. Since $F$ is block-diagonal, any product contributing to the upper-right block of the $n$-th term must cross blocks via at least one factor of $R(s)$. Fix one cross-block factor at index $i$ with modulus $D_i/(D_i^2 + s^2)$, and bound each of the remaining $n$ resolvent factors by $1/s$, canceling all $n$ factors of $s$ from $(is)^n$. Integrating the retained factor gives
$$
\int_0^\infty \frac{D_i\,dx}{D_i^2 + \lambda^2 + x^2} = \frac{\pi}{2}\frac{D_i}{\sqrt{D_i^2 + \lambda^2}} \le \frac{\pi}{2}.
$$
For fixed endpoints and $n \ge 1$, there are at most $m^{n-1}$ intermediate item indices and $2^{n+1}$ block paths, so the integrated absolute entrywise contribution of order $n$ is at most $C_1^n m^{n-1} \delta^n$. For $m\delta \le c_0$ this series is summable (justifying termwise integration) and sums to $O(\delta)$ entrywise. Order $n = 0$ integrates to $h_\lambda(D)$ in the upper-right block. Finally, multiplying on left and right by the blocks of $I + F$ preserves the $O(\delta)$ entrywise error because $h_\lambda(D)$ is diagonal with entries in $[0, 1]$ and $\|F\|_{\max} \le 2\delta$. $\square$

### 2.2 Exact Sparse-Minibatch Guarantees and Sharp Lower Bound

**Theorem 2 (Exact sparse-minibatch Muon).** *Fix $\alpha > 1$, $\epsilon > 0$, $T = O(\log d)$, and polynomial $N \gtrsim d^{2\alpha+2}$. Suppose $B^{1/\alpha} \le d^{1/2-\epsilon}$ and $B/\log d \to \infty$. Run exact Muon $W_{t+1} = W_t + \eta h_{\lambda_t}(G_t)$ from $W_0 = 0$ with $\eta \asymp \sqrt{d}/(\log d)^4$ and any stabilization schedule satisfying*
$$
0 \le \lambda_t \le \frac{C\sqrt{d}}{B(\log d)^6}.
$$
*Then with high probability (at least $1 - O(d^{-M})$ for any fixed $M$, provided $B/\log d$ is sufficiently large), simultaneously for every $1 \le t \le T$:*
1. *Every item $i$ observed in the first $t$ batches satisfies $P_{ii}(W_t) \ge 1 - \exp(-c\log^2 d)$, and all off-diagonal logits satisfy $\max_{k \neq j} |z_{kj}(W_t)| = o(1)$.*
2. *The strictly recovered prefix contains $\{1, \dots, \lfloor c'(tB/\log d)^{1/\alpha} \rfloor\}$, and the population loss satisfies*
   $$
   L(W_t) \le O\!\left(\Bigl(\frac{tB}{\log d}\Bigr)^{(1-\alpha)/\alpha}\log N\right) + \exp(-c\log^2 d).
   $$
3. *(Matching batch-exponent lower bound) For each fixed $t \ge 1$, with high probability $L(W_t) \gtrsim (tB)^{(1-\alpha)/\alpha}\log N$, and the guaranteed recovered prefix is $O((tB)^{1/\alpha})$.*

*Note on schedules:* The upper bound $\lambda_t \le C\sqrt{d}/[B(\log d)^6]$ includes $\lambda_t \equiv 0$, a choice of Theorem `thm:multi`'s displayed soft order $\widetilde\Theta(\sqrt{d}\,B^{-1})$, and the appendix's actual schedule (where $B\lambda_0 = O((\log d)^2)$ and $B\lambda_t \lesssim d^{-1/2} B^{1/(2\alpha)} \operatorname{polylog}(d)$ for $t \ge 1$).

*Proof.* By Gaussian concentration and a union bound over polynomially many pairs, with probability $1 - O(d^{-M})$ both full dictionaries satisfy $|u_i^\top u_j - \delta_{ij}| \le \delta$ and $|v_i^\top v_j - \delta_{ij}| \le \delta$ with $\delta := C_M \sqrt{\log d / d}$, and the output matrix $U = [u_1,\dots,u_N]$ obeys $\|U\|_{\mathrm{op}} = O(\sqrt{N/d})$. In each minibatch of size $B$, the number $m$ of distinct observed items satisfies $m = O(B^{1/\alpha} + \log d)$ with probability $1 - O(d^{-M})$: ranks $i \le \lceil B^{1/\alpha}\rceil$ contribute at most $\lceil B^{1/\alpha}\rceil$, and the tail draw count is $\operatorname{Bin}(B, \sum_{i > \lceil B^{1/\alpha}\rceil} p_i)$ with mean $O(B^{1/\alpha})$, controlled by Chernoff and a union bound over $T = O(\log d)$ steps. Since $B^{1/\alpha} \le d^{1/2-\epsilon}$, we have $m\delta = o(1)$ simultaneously at all steps.

We induct on $t \in \{0, \dots, T\}$ to maintain $\max_{k \neq j} |z_{kj}(W_t)| \le C_2 t \eta \delta = o(1)$ (which holds trivially at $t = 0$). Fix step $t$ and let $S_t := \{i : q_i^{(t)} > 0\}$ ($|S_t| = m$). For each $i \in S_t$, decompose $\mu_i(W_t) = P_{ii} u_i + (1 - P_{ii}) b_i$, where $b_i := \sum_{k \neq i} \frac{P_{ki}}{1 - P_{ii}} u_k$ is the conditional output mean given a wrong class. Because all off-diagonal logits $z_{ki}$ ($k \neq i$) lie in an interval of length $o(1)$, every conditional weight $w_{ki} := P_{ki}/(1 - P_{ii})$ is $O(1/N)$, so $\|w_{\cdot, i}\|_2 = O(1/\sqrt{N})$ and therefore
$$
\|b_i\|_2 \le \|U\|_{\mathrm{op}}\, \|w_{\cdot, i}\|_2 = O\!\left(\frac{1}{\sqrt{d}}\right)
$$
uniformly in $i$, even when $P_{ii}$ is exponentially close to $1$. Factor the exact negative gradient over $S_t$ as $G_t = A D V^\top$ with active columns $a_i = u_i - b_i$, $v_i$, and positive weights $D_i = q_i^{(t)}(1 - P_{ii}(W_t)) > 0$. Because $\|b_i\|_2 = O(d^{-1/2}) = O(\delta)$ and $|u_k^\top b_i| \le w_{ki}\|u_k\|_2^2 + \max_{r \neq k} |u_k^\top u_r| = O(1/N + \delta) = O(\delta)$ for every $k \in [N]$, the active Gram matrices $A^\top A$ and $V^\top V$ satisfy Lemma 1 with error $O(\delta)$.

For every output $k \in [N]$ and input $j \in [N]$, the probe vectors $A^\top u_k$ and $V^\top v_j$ differ entrywise from the indicator vectors $\mathbf{1}_{k \in S_t} e_k$ and $\mathbf{1}_{j \in S_t} e_j$ by $O(\delta)$, and each has $\ell_1$-norm $O(1 + m\delta) = O(1)$. Applying Lemma 1 to $u_k^\top h_{\lambda_t}(G_t) v_j = (A^\top u_k)^\top [h_{\lambda_t}(D) + E] (V^\top v_j)$ yields
$$
u_k^\top h_{\lambda_t}(G_t) v_j = \mathbf{1}_{k = j,\, j \in S_t}\, h_{\lambda_t}(D_j) + O(\delta),
$$
where contracting the two probe errors against the bounded diagonal term $h_{\lambda_t}(D)$ costs only $O(m\delta^2) = o(\delta)$ and the error term $E$ contributes at most $\|E\|_{\max} \|A^\top u_k\|_1 \|V^\top v_j\|_1 = O(\delta)$. Because $T\eta\delta = O((\log d) \cdot (\sqrt{d}/(\log d)^4) \cdot \sqrt{\log d/d}) = O((\log d)^{-5/2}) = o(1)$, this closes the induction.

Before an item's first appearance, its diagonal logit is $O(T\eta\delta) = o(1)$, so $P_{ii} = O(1/N)$ and on its first active step $D_i \ge \frac{1}{2B}$. Consequently its diagonal logit receives a main-term increment at least
$$
\eta\, h_{\lambda_t}\!\left(\frac{1}{2B}\right) \ge c_1 \eta \min\!\left\{1,\, \frac{1}{B\lambda_t}\right\} \gtrsim (\log d)^2
$$
(with the minimum equal to $1$ when $\lambda_t = 0$). All subsequent diagonal main-term increments are nonnegative, and the cumulative uniform error across all $T$ steps stays $o(1)$. Hence every observed item satisfies $z_{ii}(W_t) - \max_{k \neq i} z_{ki}(W_t) \gtrsim \log^2 d$, giving $P_{ii}(W_t) \ge 1 - \exp(-c\log^2 d)$, while every unseen item has loss $\log N + o(1)$. Since $\mathbb{P}(i \text{ unseen in } t \text{ batches}) = (1 - p_i)^{tB} \le e^{-tB p_i}$, a union bound shows that all $i \le c'(tB/\log d)^{1/\alpha}$ appear in the first $t$ batches with high probability. Summing $\sum_{i > \lfloor c'(tB/\log d)^{1/\alpha}\rfloor} p_i (\log N + o(1))$ proves (1) and (2).

For the lower bound (3), take the block $[r, 2r)$ with $r := \lceil C_3 (tB)^{1/\alpha} \rceil$. Its expected number of draws in $tB$ samples is $O(tB r^{1-\alpha}) < r/4$ for $C_3$ sufficiently large, so with probability $1 - e^{-\Omega(r)}$ at least $k \ge r/2$ items in $[r, 2r)$ remain unseen. Conditional on the sampled batches, all output vectors $\{u_\ell\}$, and all observed input vectors, the matrix $W_t$ is fixed and the unseen inputs $v_i$ are i.i.d. $\mathcal{N}(0, I_d/d)$. The strict-recovery probabilities $a_i := \mathbb{P}(u_i^\top W_t v_i > \max_{\ell \neq i} u_\ell^\top W_t v_i \mid W_t, \{u_\ell\})$ are normal-cone probabilities for a common classifier and satisfy $\sum_{i=1}^N a_i \le 1$ (since any vector $v$ has at most one strict maximizer). By AM–GM, the probability that all $k$ unseen items in $[r, 2r)$ are strictly recovered is $\prod a_i \le (1/k)^k = k^{-k}$. Moreover, in the sparse regime every unseen item incurs loss $\log N + o(1)$ by the uniform logit bound and has weight $p_i \ge c_4 r^{-\alpha}$, yielding $L(W_t) \gtrsim r^{1-\alpha} \log N \asymp (tB)^{(1-\alpha)/\alpha}\log N$ with high probability. $\square$

### 2.3 Where the Dimension-Limited Argument Stops

At $B = \infty$ (population updates) or dense batches, all $N \gg d$ columns are active, so $m\delta \ll 1$ fails completely. The deflation surrogate bypasses this by imposing a deterministic rank cutoff $i > d_t$ independent of the embeddings in the tail. For the exact gradient $G_t$, the transition weights $1 - P_{ii}(W_t)$ and wrong-class means $\mu_i(W_t)$ depend on all embeddings simultaneously. Operator-norm closeness to a surrogate does not automatically transfer through $h_\lambda$: for instance, $\operatorname{diag}(1, e)$ and $\operatorname{diag}(1, -e)$ differ by $2e$ in operator norm, yet their polar factors differ by $2$ and their stabilized factors by $2e/\sqrt{e^2 + \lambda^2}$. For $\lambda > 0$, the general Lipschitz bound $\|h_\lambda(G) - h_\lambda(\bar G)\|_{\mathrm{op}} \le \|G - \bar G\|_{\mathrm{op}} / \lambda$ deteriorates as $\lambda_t$ shrinks. Controlling the directional geometry of the data-dependent transition band—and thereby settling the conjectured exponent $2 - (1 - 1/(2\alpha))^t$ at $t \ge 2$ for exact dense/population Muon—remains **open**.

---

## 3. Refutation of Literal Newton Matching

Differentiating the softmax loss yields the exact Hessian action on $X \in \mathbb{R}^{d \times d}$:
$$
H_W[X] = \sum_{i=1}^N q_i C_i(W) X v_i v_i^\top, \qquad
C_i(W) := \sum_{k=1}^N P_{ki}(W)\bigl(u_k - \mu_i(W)\bigr)\bigl(u_k - \mu_i(W)\bigr)^\top.
$$
At $W = 0$, $C_i(0) = \Sigma_u := \frac{1}{N}\sum_k (u_k - \bar{u})(u_k - \bar{u})^\top$ is independent of $i$, so $H_0[X] = \Sigma_u X M_v$ with $M_v := \sum_i q_i v_i v_i^\top$ and the Newton direction is $D_N = \Sigma_u^{-1} G M_v^{-1}$ whenever $\Sigma_u$ and $M_v$ are invertible. Note already that in the sparse regime of Theorem 2 ($m < d$), $M_v$ is singular and the classical Newton step does not exist.

### 3.1 One-Step Retrieval Discrepancy (4-Item Shear) and Generic Direction Mismatch

**Theorem 3 (4-item shear counterexample and generic directional mismatch).**
1. *Let $d = 2$, $N = 4$, $p_i \propto i^{-2}$, inputs $(v_1, v_2, v_3, v_4) = (e_1, -e_1, e_2, -e_2)$, and sheared outputs $u_i = A v_i$ with $A = \begin{psmallmatrix} 1 & 2 \\ 0 & 1 \end{psmallmatrix}$. From $W_0 = 0$, one Newton step with any step size $\eta_N > 0$ strictly recovers **4/4** items, whereas one Muon step with **any** $\lambda \ge 0$ and **any** step size $\eta > 0$ recovers only **2/4** items ($\pm e_2$, failing on $\pm e_1$).*
2. *For each fixed $\lambda \ge 0$, the strict margins and invertibility persist on an open neighborhood of embeddings, hence with strictly positive probability under finite-dimensional isotropic Gaussian sampling.*
3. *For every $d \ge 2$ and $N \ge 2d$, under i.i.d. isotropic Gaussian embeddings $u_i, v_i \sim \mathcal{N}(0, I_d/d)$ (population updates), the first-step Newton direction $D_N$ and any spectral update $h_\lambda(G_0)$ are non-collinear almost surely (even allowing data-dependent scalar step sizes).*

*Proof.* For (1), set $a := p_1 + p_2 > b := p_3 + p_4 > 0$. By symmetry $\bar{u} = 0$, so at $W_0 = 0$, $G = A\operatorname{diag}(a, b)$, $\Sigma_u = \frac{1}{2} A A^\top$, and $M_v = \operatorname{diag}(a, b)$. Thus the Newton direction is
$$
D_N = \Sigma_u^{-1} G M_v^{-1} = 2(A A^\top)^{-1} A = 2 A^{-\top} = \begin{pmatrix} 2 & 0 \\ -4 & 2 \end{pmatrix}.
$$
Its score matrix is $u_k^\top D_N v_i = 2 v_k^\top A^\top A^{-\top} v_i = 2 v_k^\top v_i$, giving correct logit $2$ against competitors $0, 0, -2$ on every input (4/4 recovered for all $\eta_N > 0$).

For Muon, put $S := G^\top G + \lambda^2 I_2$ and $r := \sqrt{\det S} = \sqrt{(a^2+\lambda^2)(5b^2+\lambda^2) - 4a^2 b^2} \ge ab > 0$. Using the $2 \times 2$ positive-definite square-root identity $S^{-1/2} = \frac{1}{r\sqrt{\operatorname{tr} S + 2r}}\bigl((\operatorname{tr} S + r) I_2 - S\bigr)$, the columns of $h_\lambda(G) = G S^{-1/2}$ are positively proportional (by a common factor) to
$$
\begin{pmatrix} a(b^2 + \lambda^2 + r) \\ -2ab^2 \end{pmatrix},
\qquad
\begin{pmatrix} 2b(\lambda^2 + r) \\ b(a^2 + \lambda^2 + r) \end{pmatrix}.
$$
On input $v_1 = e_1$, the margin of wrong output $u_3 = A e_2 = (2, 1)^\top$ over the true output $u_1 = A e_1 = (1, 0)^\top$ is a positive multiple of $(2 - 1) \cdot a(b^2 + \lambda^2 + r) + 1 \cdot (-2ab^2) = a(\lambda^2 + r - b^2)$. Since $a > b$, we have $r = \sqrt{a^2 b^2 + \lambda^2(a^2 + 5b^2) + \lambda^4} \ge ab > b^2$, so $a(\lambda^2 + r - b^2) > 0$. By sign symmetry, $-e_1$ also fails (wrong output $-Ae_2$ beats $-Ae_1$). On $v_3 = e_2$, the second column has strictly positive entries, so $u_3^\top h_\lambda(G) e_2 = 2 \cdot 2b(\lambda^2+r) + b(a^2+\lambda^2+r) > 2b(\lambda^2+r) = u_1^\top h_\lambda(G) e_2 > 0$, recovering $\pm e_2$ strictly. Thus Muon recovers 2/4 items for every $\lambda \ge 0$ and $\eta > 0$.

For (2), full rank and all strict margin inequalities are open conditions in $(u, v)$, giving a nonempty open set of positive Gaussian measure (without claiming a dimension-uniform lower bound).

For (3), every spectral update $S_G = h_\lambda(G)$ satisfies $G S_G^\top = S_G G^\top$ (symmetric). For Newton, $G D_N^\top = G M_v^{-1} G^\top \Sigma_u^{-1}$. Extending the 4-item witness to arbitrary $d \ge 2, N \ge 2d$ by placing $\pm e_j$ on axes $j = 3, \dots, d$, extending $A$ by the identity, and padding remaining items with zeros gives $G D_N^\top = \frac{N}{2} A M_v A^{-1}$, whose $(1,2)$-entry is $\frac{N}{2} \cdot 2(b - a) \neq 0$ while its $(2,1)$-entry is $0$, so $G D_N^\top$ is nonsymmetric. Clearing denominators by multiplying by $\det(\Sigma_u)\det(M_v)$ makes the antisymmetric part of $G D_N^\top$ a nonzero polynomial in $(u, v)$, whose zero set has Lebesgue/Gaussian measure zero. Since $\Sigma_u, M_v$ are invertible almost surely for $N \ge 2d$, $D_N$ and $h_\lambda(G_0)$ are non-collinear almost surely. $\square$

### 3.2 Limiting Dynamics Discrepancy: Scalar 2-Cycle and High-Dimensional Local Dynamics

**Theorem 4 (Scalar 3-item memory).** *Take $d = 1$, $v_1 = v_2 = v_3 = 1$, $u = (0, 1, 2)$, and $p_i \propto i^{-2}$ (so $p = (36, 9, 4)/49$). Then $L(w)$ has a unique global minimizer $w_* \in (-2, -1)$ ($w_* \approx -1.1994401007$). From $w_0 = 0$:*
1. *Unit-step polar Muon ($\lambda = 0, \eta = 1$) alternates forever in the 2-cycle $-1, -2, -1, -2, \dots$.*
2. *Unit-step Newton converges monotonically from above to $w_*$ with quadratic local rate.*
3. *Fixed-$\lambda > 0$ stabilized Muon has derivative $1 - \eta L''(w_*)/\lambda$ at $w_*$, giving at most linear local convergence whenever $|1 - \eta L''(w_*)/\lambda| \in (0, 1)$.*

*Proof.* Here $L'(w) = \mu(w) - \frac{17}{49}$ where $\mu(w) = (e^w + 2e^{2w})/(1 + e^w + e^{2w}) = 1 + \frac{\sinh w}{1/2 + \cosh w}$. Its derivative is the strictly positive softmax variance $\mu'(w) = (2\cosh w + 4)/(1 + 2\cosh w)^2 > 0$. Since $\mu(-2) \approx 0.135 < 17/49 \approx 0.347 < \mu(-1) \approx 0.354 < \mu(0) = 1$, the unique root $w_*$ lies in $(-2, -1)$, and $L'(0) > 0$, $L'(-1) > 0$, $L'(-2) < 0$. Hence unit-step polar Muon steps $0 \to -1 \to -2 \to -1 \to -2 \dots$. For $w \le 0$, $\mu''(w) = -2\sinh w\,(2\cosh w + 7)/(1 + 2\cosh w)^3 \ge 0$, so $\mu$ is increasing and convex on $(-\infty, 0]$, ensuring monotone Newton convergence from $w_0 = 0$ to $w_*$ with quadratic rate. For fixed $\lambda > 0$, the update map $w \mapsto w - \eta h_\lambda(L'(w))$ has derivative $1 - \eta L''(w_*)/\lambda$ at $w_*$. $\square$

**Theorem 5 (Local dynamics in the Gaussian population model).** *For i.i.d. $u_i, v_i \sim \mathcal{N}(0, I_d/d)$ with $p_i > 0$ and $N/d^2 \to \infty$, with probability $1 - o(1)$ the population loss $L(W)$ is strictly convex and coercive, attaining a unique finite minimizer $W_* \in \mathbb{R}^{d \times d}$ with positive definite Hessian $H_{W_*} \succ 0$. On this event, Newton's method enjoys local quadratic convergence to $W_*$, whereas fixed-step polar Muon ($\lambda = 0, \eta > 0$) has update norm $\|W_{t+1} - W_t\|_F = \eta\sqrt{\operatorname{rank}(G_t)} \ge \eta$ at every $W_t \neq W_*$ and therefore cannot converge locally to $W_*$.*

*Proof.* Let $D := d^2$ and $m := \lfloor N/2 \rfloor$, and form the $m$ independent centrally symmetric vectors $x_j := \operatorname{vec}((u_{2j-1} - u_{2j}) v_{2j-1}^\top) \in \mathbb{R}^D$ ($1 \le j \le m$). Any $D$ of them are linearly independent almost surely (their determinant is a nonzero polynomial, witnessed by selecting the $D$ matrix units). By Wendel's theorem (conditioning on the lines and applying independent sign flips, since $m$ central hyperplanes in general position partition $\mathbb{R}^D$ into at most $2\sum_{k=0}^{D-1}\binom{m-1}{k}$ open cells), the probability that all $x_j$ lie in a common open homogeneous half-space is at most $2^{1-m}\sum_{k=0}^{D-1}\binom{m-1}{k} = o(1)$ as $m/D \to \infty$. By general position, any nonzero weak separator places at most $D-1$ points on its boundary, which can be perturbed off to yield a strict separator; hence with probability $1 - o(1)$, for every unit $W \in \mathbb{R}^{d \times d}$ some pair has $(u_{2j-1} - u_{2j})^\top W v_{2j-1} < 0$. By compactness of the unit sphere, the worst paired margin is uniformly bounded below by $c_0 > 0$, so $L(tW) \ge (\min_i p_i)\log(1 + e^{c_0 t}) \to +\infty$ linearly along every ray. Since $u_1,\dots,u_N$ affinely span $\mathbb{R}^d$ and $v_1,\dots,v_N$ span $\mathbb{R}^d$ almost surely for $N > d$, every $C_i(W) \succ 0$ and hence $H_W \succ 0$ everywhere. Thus $L$ has a unique finite minimizer $W_*$ with $H_{W_*} \succ 0$, giving local quadratic convergence for Newton. At any $W \neq W_*$, $G(W) \neq 0$, so $\|h_0(G(W))\|_F = \sqrt{\operatorname{rank}(G(W))} \ge 1$ and $\|W_+ - W_*\|_F \ge \eta - \|W - W_*\|_F$, ruling out local convergence of fixed-step polar Muon. $\square$

*(Scope note: Theorems 3–5 refute literal trajectory, retrieval, and local-convergence matching between Muon and Newton. Whether suitably scheduled Muon and Newton share the same high-dimensional multistep capacity exponents on typical isotropic Gaussian memories—motivated by the paper's one-step Newton theorem at $B = \widetilde\Omega(d^\alpha)$—remains **open**.)*

---

## 4. Verification and Reproduction

Run from the repository root:

```sh
.venv/bin/python -B solutions/muon_assoc_memory/explore.py
```

The script requires NumPy and SciPy, checks the gradient and full $d^2 \times d^2$ Hessian by central differences, verifies invariance of Newton's trajectory under invertible output coordinate changes, and passes all assertions:

- **Finite counterexamples:** 4-item shear verified across $\lambda \in \{0, 10^{-4}, 0.2, 1, 100\}$ (Muon 2/4, Newton 4/4, $D_N = \begin{psmallmatrix} 2 & 0 \\ -4 & 2 \end{psmallmatrix}$ to $10^{-13}$); 3-item scalar memory verified over 10 steps (polar alternates between $-1.0$ and $-2.0$; Newton converges to $-1.1994401006992608$ with $|L'(w)| < 10^{-14}$).
- **Lemma 1 stress test:** Across 27 tests ($m = 8$, $\delta \in \{10^{-2}, 10^{-3}, 10^{-4}\}$, diagonal condition numbers up to $10^8$, $\lambda \in \{0, 10^{-5}, 1\}$), the maximum observed ratio $\|E\|_{\max}/\delta$ was **0.935444**.
- **Sparse minibatch run ($d = 128, N = 512, B = 12, \alpha = 1.5$):** Over 4 batches, cumulative observed item counts were `[6, 10, 13, 17]`; both polar ($\lambda = 0$) and stabilized ($\lambda = 1/B$) exact-gradient Muon strictly retained all observed items at every step.
- **Population conditioning sweep ($d = 12, N = 4d^2 = 576, \alpha = 1.5$, medians across 3 paired seeds, 3 steps):**

| Output $\kappa$ | Method | Prefix at steps 1, 2, 3 | Total correct at step 3 | Loss at step 3 |
|---:|---|---|---:|---:|
| 1 | Polar | 2, 3, 4 | 5 | 3.065747 |
| 1 | Stabilized exact gradient | 2, 3, 3 | 5 | 2.741540 |
| 1 | Deflation surrogate | 2, 1, 2 | 5 | 3.752249 |
| 1 | Newton | 2, 2, 2 | 5 | 3.819951 |
| 256 | Polar | 0, 0, 0 | 2 | 4.197713 |
| 256 | Stabilized exact gradient | 0, 0, 0 | 1 | 4.379866 |
| 256 | Deflation surrogate | 0, 0, 0 | 1 | 6.210624 |
| 256 | Newton | 2, 2, 2 | 5 | 3.819951 |
