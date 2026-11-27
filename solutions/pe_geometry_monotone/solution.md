# Disproof of the Positional Monotonicity Conjecture

**Target paper:** Giansalvo Cirrincione, *On the Geometry of Positional Encodings in Transformers*, [arXiv:2604.05217](https://arxiv.org/abs/2604.05217), Conjecture `conj:monotone` (Monotonicity) and Appendix A (NTK analysis).  
**Companion files:** [`solutions/pe_geometry_monotone/explore.py`](explore.py), [`solutions/pe_geometry_monotone/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:monotone` (Monotonicity): DISPROVED**, already at the minimal nontrivial sequence length $n = 3$ (length $L = 3$) and embedding dimension $d = 1$, and extended to $n \in \{3, 4, 5\}$ and every dimension $d \ge 1$.
- **Exact corpus and reversal:** Taking $n = 3$ positions with square-root token distributions given by shifts of $(1, 2, 3)/\sqrt{14}$ yields exact rational squared Hellinger distances $0 < 6/7 < 11/7 < 2$ at lags $0, 1, 2, \ge 3$, satisfying the full monotonicity hypothesis (including equal lags). Assigning binary token tags $g = (1, 0, 0, 1, 0, 1, 1)$ gives positive-label marginals $(r_1, \dots, r_n)$ equal to the first $n$ entries of $(1, 9, 4, 10, 13)/14$, so $r_1 < r_3 < r_2$. Under binary cross-entropy loss, the coercive, strictly convex objective has a unique global minimizer $P^*$ satisfying $p_1^* < p_3^* < p_2^*$, giving $|p_1^* - p_2^*| \approx 2.786624217 > 1.495878375 \approx |p_1^* - p_3^*|$ despite $|1 - 2| < |1 - 3|$.
- **Structural completeness:** The counterexample satisfies all hypotheses H1, H2, H3, and coercivity of the paper's Positional Separation Theorem, requires no parameter penalty at $d = 1$, and persists with an **active self-attention head** ($\varepsilon = 10^{-4}$), at lengths $n \in \{3, 4, 5\}$, and in every dimension $d \ge 1$.
- **Appendix A diagnosis:** In Appendix A, Hypothesis `A1` ($\alpha \succ 0$ with $\alpha_{ij} = f(d_H(\mu_i, \mu_j))$ and $f$ strictly increasing) is mathematically impossible for every $n \ge 2$ because $(e_i - e_j)^\top \alpha (e_i - e_j) = 2[f(0) - f(d_H(\mu_i, \mu_j))] \le 0$. Independently, the final step of `Lemma G3` invalidly deduces $\|p_i^* - p_j^*\| \le \|p_i^* - p_k^*\|$ from comparing upper bounds $C d_H(\mu_i, \mu_j) \le C d_H(\mu_i, \mu_k)$.
- **Raw-stress interpretation:** Under the paper's normalized raw-stress objective in $d = 1$, three distinct point masses (pairwise Hellinger distance $D = \sqrt{2}$) have global minimum stress $1/9$ at successive gaps $u = v = 2/3$ (in units of $D$) in every sorted order; ordering $(1, 3, 2)$ yields $|p_1^* - p_2^*| = 4D/3 > 2D/3 = |p_1^* - p_3^*|$, refuting the stress version as well.

---

## 1. Setup and Exact Problem Statement

In [arXiv:2604.05217](https://arxiv.org/abs/2604.05217), the corpus, token embeddings $E$, and all non-positional network parameters are fixed, and the expected task loss $\mathcal{L}(P)$ is minimized over the unconstrained positional encoding matrix $P = (p_1, \dots, p_n) \in \mathbb{R}^{n \times d}$. The Hellinger metric in `eq:hellinger` is the unnormalized chord distance on the positive orthant of the sphere:
$$
d_H(\mu, \nu) := \|\sqrt{\mu} - \sqrt{\nu}\|_2.
$$
Immediately following the Positional Separation Theorem (which assumes non-stationarity H1, order sensitivity H2, and attention non-degeneracy H3 on $M = W_Q W_K^\top$), the paper states:

> **Conjecture `conj:monotone` (Monotonicity).** *If $d_H(\mu_i, \mu_j) \le d_H(\mu_i, \mu_k)$ whenever $|i - j| \le |i - k|$, then every minimizer $P^*$ satisfies $\|p_i^* - p_j^*\| \le \|p_i^* - p_k^*\|$ under the same ordering.*

For $n \le 2$, the conclusion holds trivially for every matrix $P$; thus $n = 3$ is the minimal nontrivial sequence length.

---

## 2. Corpus, Hellinger Geometry, and Prediction Task

Fix $n \in \{3, 4, 5\}$ and vocabulary $\{1, \dots, n+2\}$. Draw tokens $T_1, \dots, T_n$ independently across positions, where at position $i \in \{1, \dots, n\}$ the tokens $(i, i+1, i+2)$ occur with probabilities $(1/14, 4/14, 9/14)$ and all other tokens have probability $0$.

**Lemma 1 (Exact rational Hellinger monotonicity).** *For every $n \in \{3, 4, 5\}$ and $i, j \in \{1, \dots, n\}$,*
$$
d_H(\mu_i, \mu_j)^2 = \begin{cases}
0 & \text{if } |i - j| = 0,\\
6/7 & \text{if } |i - j| = 1,\\
11/7 & \text{if } |i - j| = 2,\\
2 & \text{if } |i - j| \ge 3.
\end{cases}
$$
*In particular, $0 < 6/7 < 11/7 < 2$, so $d_H(\mu_i, \mu_j) \le d_H(\mu_i, \mu_k)$ whenever $|i - j| \le |i - k|$ (with strict inequality whenever $|i - j| < |i - k|$ and $|i - j| \le 2$).*

*Proof.* Each $\sqrt{\mu_i}$ is the shifted vector $(1, 2, 3)/\sqrt{14}$ supported on $\{i, i+1, i+2\}$. Therefore $\langle \sqrt{\mu_i}, \sqrt{\mu_j} \rangle$ equals $1$ at lag $0$, $(2\cdot 1 + 3\cdot 2)/14 = 8/14$ at lag $1$, $(3\cdot 1)/14 = 3/14$ at lag $2$, and $0$ at lag $\ge 3$. Since $d_H(\mu_i, \mu_j)^2 = 2 - 2\langle \sqrt{\mu_i}, \sqrt{\mu_j} \rangle$, the stated values follow immediately. $\square$

Now assign each vocabulary token $v \in \{1, \dots, n+2\}$ a binary tag $g(v) \in \{0, 1\}$ using the first $n + 2$ entries of
$$
g = (1, 0, 0, 1, 0, 1, 1).
$$
The prediction target is the ordered binary vector $Y_i := g(T_i)$ for $i = 1, \dots, n$ (equivalently, sequence classification over $\{0, 1\}^n$ with factorized output logits). By direct evaluation $r_i := \Pr(Y_i = 1) = \frac{1}{14}g(i) + \frac{4}{14}g(i+1) + \frac{9}{14}g(i+2)$, the positive-label marginals $(r_1, \dots, r_n)$ are the first $n$ entries of
$$
\frac{1}{14}(1, 9, 4, 10, 13).
$$
Crucially, $r_1 = 1/14 < r_3 = 4/14 < r_2 = 9/14$: Hellinger distance separates all distinct tokens, whereas the task depends only on which tokens share tag $1$.

---

## 3. Exact Global Minimizer and Verification of H1–H3 ($d = 1$)

Take $d = d_k = 1$, fixed token embedding $E(v) := \beta g(v)$ with $\beta := \log 2$, and $W_Q = W_K = 1$ (so $M = W_Q W_K^\top = 1$). First consider zero value projection, so the residual-stream logit is $h_i := E(T_i) + p_i$. Under summed binary cross-entropy, the exact expected loss is
$$
\mathcal{L}_0(P) = \sum_{i=1}^n F_{r_i}(p_i),
\qquad
F_r(x) := (1 - r)\log(1 + e^x) + r\log(1 + e^{-x - \beta}).
$$

**Theorem 2 (Unique non-monotone global minimizer).** *For every $n \in \{3, 4, 5\}$, $\mathcal{L}_0$ is coercive, strictly convex, and satisfies H1, H2, and H3. Its unique global minimizer $P^* = (p_1^*, \dots, p_n^*) \in \mathbb{R}^n$ is given in closed form by $p_i^* = \log z(r_i)$, where*
$$
z(r) := \frac{2r - 1 + \sqrt{(2r - 1)^2 + 8r(1 - r)}}{4(1 - r)}.
$$
*Moreover, $p_1^* < p_3^* < p_2^*$, so $|p_1^* - p_2^*| > |p_1^* - p_3^*|$ ($2.786624217 > 1.495878375$) even though $|1 - 2| < |1 - 3|$ and $d_H(\mu_1, \mu_2) < d_H(\mu_1, \mu_3)$.*

*Proof.* For $r \in (0, 1)$, $F_r(x) \to +\infty$ as $|x| \to \infty$ (coercivity), and
$$
F_r'(x) = (1 - r)\frac{e^x}{1 + e^x} - r\frac{1}{1 + e^{x+\beta}}
= (1 - r)\frac{z}{1 + z} - r\frac{1}{1 + 2z}
\qquad (z := e^x > 0),
$$
with $F_r''(x) = (1 - r)\sigma(x)(1 - \sigma(x)) + r\sigma(x+\beta)(1 - \sigma(x+\beta)) \in (0, 1/4]$ (strict convexity, where $\sigma(t) = (1 + e^{-t})^{-1}$). Clearing the positive denominator $(1 + z)(1 + 2z)$ in $F_r'(x) = 0$ yields the quadratic
$$
2(1 - r)z^2 + (1 - 2r)z - r = 0,
$$
whose unique positive root is $z(r)$ above, proving KKT uniqueness $p_i^* = \log z(r_i)$.

For fixed $x$, $F_r'(x)$ is strictly decreasing in $r$ (derivative $-\frac{z}{1+z} - \frac{1}{1+2z} < 0$), and for fixed $r$, $F_r'(x)$ is strictly increasing in $x$ (since $F_r'' > 0$). Hence the unique root $r \mapsto \log z(r)$ is strictly increasing on $(0, 1)$. Since $r_1 = 1/14 < r_3 = 4/14 < r_2 = 9/14$, we obtain $p_1^* < p_3^* < p_2^*$ and therefore
$$
|p_1^* - p_2^*| = p_2^* - p_1^* > p_3^* - p_1^* = |p_1^* - p_3^*|.
$$

It remains to verify H1–H3:
- **H1 (Non-stationarity):** The mean embeddings $\bar{e}_i = \mathbb{E}[E(T_i)] = \beta r_i$ are pairwise distinct because the first $n$ entries of $(1, 9, 4, 10, 13)/14$ are distinct.
- **H3 (Non-degeneracy):** $M = W_Q W_K^\top = 1 \neq 0$, so $M(\bar{e}_i - \bar{e}_j) \neq 0$ for all $i \neq j$.
- **H2 (Order sensitivity):** Fix any $P \in \mathbb{R}^n$ and transpose input tokens $T_i, T_j$ ($i \neq j$) while keeping the true ordered target $Y$ fixed. Set $D(x) := \log(1 + e^{x+\beta}) - \log(1 + e^x) \in (0, \beta)$. At position $i$, moving a tag-$1$ token onto a tag-$0$ target increases loss by $D(p_i)$ (probability $(1 - r_i)r_j$ by independence), while moving a tag-$0$ token onto a tag-$1$ target increases loss by $\beta - D(p_i)$ (probability $r_i(1 - r_j)$). Hence the expected loss increase at position $i$ is
  $$
  (1 - r_i)r_j D(p_i) + r_i(1 - r_j)(\beta - D(p_i)) \ge \beta\min\bigl((1 - r_i)r_j,\, r_i(1 - r_j)\bigr).
  $$
  Position $j$ contributes the same lower bound, so transposing $(i, j)$ strictly increases $\mathcal{L}_0(P)$ by at least
  $$
  2\beta\min\bigl((1 - r_i)r_j,\, r_i(1 - r_j)\bigr) \ge \frac{2\beta}{196} > 0
  $$
  at every fixed $P \in \mathbb{R}^n$. $\square$

---

## 4. Active Self-Attention Head and Arbitrary Dimension $d \ge 1$

To confirm that the reversal does not rely on a zero value projection or on $d = 1$, activate the attention head with scalar value $h_j = E(T_j) + p_j$ and perturbation $\varepsilon := 10^{-4}$:
$$
A_{ij} := \frac{e^{h_i h_j}}{\sum_{k=1}^n e^{h_i h_k}},
\qquad
s_i := h_i + \varepsilon\tanh\!\biggl(\sum_{j=1}^n A_{ij} h_j\biggr),
$$
and let $\mathcal{L}_\varepsilon(P)$ be the expected summed binary cross-entropy evaluated at the nonlinear logits $s_i$.

**Theorem 3 (Persistence with active attention and in every $d \ge 1$).** *For $n \in \{3, 4, 5\}$ and $\varepsilon = 10^{-4}$:*
1. *$\mathcal{L}_\varepsilon$ is coercive, continuous, satisfies H1–H3 (with strictly positive swap margin at every $P$), and attains a global minimum.*
2. ***Every** global minimizer $P^\star$ of $\mathcal{L}_\varepsilon$ satisfies $p_1^\star < p_3^\star < p_2^\star$, hence $|p_1^\star - p_2^\star| > |p_1^\star - p_3^\star|$.*
3. *For every $d > 1$, setting $E(v) = \beta g(v)e_1$, $W_Q = W_K = e_1$, reading values/logits from coordinate $1$, and adding $\frac{1}{2}\sum_{i=1}^n \sum_{m=2}^d p_{im}^2$ yields the identical reversal on $\mathbb{R}^{n \times d}$ with H1–H3 and coercivity preserved.*

*Proof.* Because binary cross-entropy is $1$-Lipschitz in its scalar logit and $|\varepsilon\tanh(\cdot)| \le \varepsilon$, we have $|\mathcal{L}_\varepsilon(P) - \mathcal{L}_0(P)| \le n\varepsilon$ pointwise in inputs, labels, and $P$. Thus $\mathcal{L}_\varepsilon$ remains coercive and attains a global minimum. Since $\beta = \log 2 > 1/2$, any swap increases $\mathcal{L}_\varepsilon(P)$ by at least $2\beta\min((1-r_i)r_j, r_i(1-r_j)) - 2n\varepsilon > \frac{1}{196} - \frac{1}{1000} > 0$, preserving H2.

To certify $p_1^\star < p_3^\star < p_2^\star$ at every global minimizer $P^\star$ of $\mathcal{L}_\varepsilon$, consider any pair $(a, b)$ with $r_a < r_b$ and any $x_a \ge x_b$. By monotonicity of $F'_{r_a}$, writing $z := e^{x_b} > 0$,
$$
F'_{r_a}(x_a) - F'_{r_b}(x_b)
\ge F'_{r_a}(x_b) - F'_{r_b}(x_b)
= (r_b - r_a)\biggl(\frac{z}{1 + z} + \frac{1}{1 + 2z}\biggr)
\ge \frac{r_b - r_a}{2},
$$
where the final inequality holds because $\frac{z}{1+z} + \frac{1}{1+2z} - \frac{1}{2} = \frac{1 + z + 2z^2}{2(1+z)(1+2z)} > 0$. Consequently, $\max(|F'_{r_a}(x_a)|, |F'_{r_b}(x_b)|) \ge (r_b - r_a)/4$. Because $F_r'' \le 1/4$, Taylor's theorem at the trial step $x - 4F'_r(x)$ gives
$$
F_r(x) - \min F_r \ge F_r(x) - F_r\bigl(x - 4F'_r(x)\bigr) \ge 4F'_r(x)^2 - \frac{1}{2}\cdot\frac{1}{4}\bigl(4F'_r(x)\bigr)^2 = 2F'_r(x)^2.
$$
Thus violating either $x_1 < x_3$ ($r_3 - r_1 = 3/14$) or $x_3 < x_2$ ($r_2 - r_3 = 5/14$) incurs an unperturbed excess loss of at least
$$
\mathcal{L}_0(P) - \mathcal{L}_0(P^*) \ge 2\Bigl(\frac{3/14}{4}\Bigr)^2 = \frac{9}{1568} \approx 5.74 \times 10^{-3}.
$$
On the other hand, comparing $P^\star$ with $P^*$ gives $\mathcal{L}_0(P^\star) - \mathcal{L}_0(P^*) \le 2n\varepsilon \le \frac{1}{1000} < \frac{9}{1568}$. Hence every global minimizer $P^\star$ satisfies $p_1^\star < p_3^\star < p_2^\star$.

Finally, for $d > 1$, the separable penalty $\frac{1}{2}\sum_{i=1}^n \sum_{m=2}^d p_{im}^2$ forces all transverse coordinates $p_{im} = 0$ ($m \ge 2$) at every minimizer while preserving domain $\mathbb{R}^{n \times d}$, coercivity, the swap margin, and H3 ($M = e_1 e_1^\top$, $M(\bar{e}_i - \bar{e}_j) = \beta(r_i - r_j)e_1 \neq 0$). $\square$

---

## 5. Diagnosis of Appendix A (`A1` and `Lemma G3`) and the Raw-Stress Interpretation

### 5.1 Appendix A: Impossibility of `A1` and gap in `Lemma G3`

Appendix A of [arXiv:2604.05217](https://arxiv.org/abs/2604.05217) claims to prove `conj:monotone` in the NTK regime under Hypotheses `A1`–`A4`. Two independent fatal obstructions invalidate that argument:

1. **Impossibility of Hypothesis `A1` for every $n \ge 2$:** Hypothesis `A1` requires $\alpha \succ 0$ and $\alpha_{ij} = f(d_H(\mu_i, \mu_j))$ for all $i, j \in \{1, \dots, n\}$ (including the diagonal $\alpha_{ii} = f(0)$) with $f : [0, \sqrt{2}] \to \mathbb{R}_+$ strictly increasing. For any distinct $i \neq j$, evaluating the quadratic form at $e_i - e_j$ yields
   $$
   (e_i - e_j)^\top \alpha (e_i - e_j)
   = \alpha_{ii} + \alpha_{jj} - 2\alpha_{ij}
   = 2\bigl(f(0) - f(d_H(\mu_i, \mu_j))\bigr) \le 0,
   $$
   contradicting $\alpha \succ 0$.
2. **Invalid inference in `Lemma G3`:** Even setting `A1` aside, the proof of `Lemma G3` derives an upper bound $\|p_i^* - p_j^*\| \le C\, d_H(\mu_i, \mu_j)$ for all $i, j$ (`eq:quantitative`) and then chains $\|p_i^* - p_j^*\| \le C\, d_H(\mu_i, \mu_j) \le C\, d_H(\mu_i, \mu_k)$ to conclude $\|p_i^* - p_j^*\| \le \|p_i^* - p_k^*\|$. Comparing upper bounds for two nonnegative quantities does not compare the quantities themselves: $\|p_i^* - p_k^*\|$ remains free to be strictly smaller than $\|p_i^* - p_j^*\|$.

### 5.2 Raw-stress interpretation ($u = v = 2/3$ counterexample)

One may alternatively ask whether monotonicity holds for minimizers of the paper's normalized raw-stress objective
$$
\mathrm{Stress}(P) := \frac{\sum_{1 \le i < j \le n} \bigl(\|p_i - p_j\| - d_H(\mu_i, \mu_j)\bigr)^2}{\sum_{1 \le i < j \le n} d_H(\mu_i, \mu_j)^2}.
$$

**Proposition 4 (Non-monotone global minimizer of normalized raw stress).** *For $n = 3$ distinct point-mass distributions $\mu_1 = \delta_1$, $\mu_2 = \delta_2$, $\mu_3 = \delta_3$ (so all pairwise Hellinger distances equal $D := \sqrt{2}$, satisfying the monotonicity premise) and $d = 1$, the global minimum of $\mathrm{Stress}(P)$ is $1/9$, attained in every permutation order with equal successive gaps $u = v = 2/3$ in units of $D$. In particular, placing positions in sorted order $(1, 3, 2)$ gives a global stress minimizer with $|p_1^* - p_2^*| = \frac{4}{3}D > \frac{2}{3}D = |p_1^* - p_3^*|$.*

*Proof.* On any sorted chamber with successive nonnegative gaps $uD, vD$ ($u, v \ge 0$), the three pairwise distances are $uD, vD, (u+v)D$, so
$$
\mathrm{Stress}(P) = S(u, v) := \frac{(u - 1)^2 + (v - 1)^2 + (u + v - 1)^2}{3}.
$$
Its Hessian is $\nabla^2 S = \frac{2}{3}\begin{psmallmatrix}2 & 1 \\ 1 & 2\end{psmallmatrix} \succ 0$ (strictly convex), and $\nabla S(u, v) = 0 \iff 2(u-1) + 2(u+v-1) = 2(v-1) + 2(u+v-1) = 0 \iff u = v = 2/3 > 0$, giving unique chamber minimum $S(2/3, 2/3) = 1/9$. Because all three target distances are equal, every permutation of $(1, 2, 3)$ achieves the same global minimum $1/9$. Taking the sorted order $p_1^* < p_3^* < p_2^*$ with $p_3^* - p_1^* = \frac{2}{3}D$ and $p_2^* - p_3^* = \frac{2}{3}D$ gives $|p_1^* - p_2^*| = \frac{4}{3}D > \frac{2}{3}D = |p_1^* - p_3^*|$. $\square$

*(Remark: when $d$ is at least the affine dimension of $\{\sqrt{\mu_1}, \dots, \sqrt{\mu_n}\}$ and the denominator is positive, zero stress is attainable and every stress minimizer is an exact isometry of the Hellinger distances, so monotonicity holds trivially in that unconstrained-dimension regime.)*

---

## 6. Exact Verification and Reproduction

Run `.venv/bin/python -B solutions/pe_geometry_monotone/explore.py` from the repository root (Python 3 standard library only, exact `fractions.Fraction` arithmetic). For $n = 3, 4, 5$, the script:
- enumerates all $3^n$ token sequences ($27 + 81 + 243 = 351$ total) to verify the exact rational tag marginals $(1, 9, 4, 10, 13)/14$;
- checks all $n^3$ Hellinger triples ($27 + 64 + 125 = 216$ total) and $d_H(\mu_1,\mu_2)^2 = 6/7 < 11/7 = d_H(\mu_1,\mu_3)^2$;
- verifies all $12$ KKT quadratic roots by exact symbolic substitution and certifies $p_1^* < p_3^* < p_2^*$ by rational bisection bracketing;
- checks all $3 + 6 + 10 = 19$ pairwise swap margins and active-head perturbation margins at $\varepsilon = 10^{-4}$;
- verifies the $u = v = 2/3$, $\mathrm{Stress} = 1/9$ raw-stress certificate.
