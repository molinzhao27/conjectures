# Sharp Universal Information Floor for Softmax Readouts

**Target paper:** Bruno Abrahao, *Neural Collapse Is Forbidden: Information Floors in Language Models*, [arXiv:2607.09487](https://arxiv.org/abs/2607.09487) (July 2026), Appendix, Conjecture `conj:generalfloor`.  
**Companion files:** [`solutions/lm_info_floor/explore.py`](explore.py), [`solutions/lm_info_floor/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:generalfloor`: PROVED** with the **sharp universal constant $c = 2$** (information in nats):
  $$
  R^2 D_k \ge 2 I_k,
  $$
  and the stronger diameter and directional-width chain
  $$
  I_k \le \tfrac{1}{8}\mathbb{E}\bigl[w(H - \mu)^2\bigr] \le \tfrac{L^2}{8} D_k \le \tfrac{R^2}{2} D_k.
  $$
- **No dependence on category size or small marginals:** Contrary to the paper's conjectured polynomial degradation $c(m, \bar{q}) = \Theta(\min_c \bar{q}_c / m)$, neither the category size $m \ge 2$ nor arbitrarily small marginal token probabilities weaken the constant.
- **Sharpness scope:** For every fixed category size $m \ge 2$, the constant $c = 2$ is unimprovable over admissible centered heads and context laws with **$\bar{q}$ free** (a two-context family attains $2 I_k / (R^2 D_k) \to 1$ with finite features and strictly positive marginals). Determining the optimal constant $c^*(m, \bar{q})$ when the marginal $\bar{q}$ is held fixed is a separate, finer question.

---

## 1. Setup and Main Inequality

Fix a category $S_k$ with $|S_k| = m \ge 2$ tokens and weight vectors $\{w_c\}_{c \in S_k} \subset \mathbb{R}^d$. Its centered residual rows are
$$
\delta_c := w_c - \frac{1}{m}\sum_{j \in S_k} w_j \qquad (c \in S_k),
$$
which satisfy $\sum_{c \in S_k} \delta_c = 0$. Let $R \ge \max_{c \in S_k} \|\delta_c\|_2$ bound their Euclidean norms, let $L := \max_{a, b \in S_k} \|\delta_a - \delta_b\|_2 \le 2R$ be the residual-row diameter, and define the directional width
$$
w(v) := \max_{c \in S_k} \delta_c^\top v - \min_{c \in S_k} \delta_c^\top v \qquad (v \in \mathbb{R}^d).
$$
Let $H$ be a random feature vector drawn from the category's conditional context law (including any weighted empirical distribution), with mean $\mu := \mathbb{E}[H]$. For $h \in \mathbb{R}^d$, define the within-category softmax distribution
$$
q_h(c) := \frac{\exp(\delta_c^\top h)}{\sum_{j \in S_k} \exp(\delta_j^\top h)} \qquad (c \in S_k).
$$
All three quantities in Conjecture `conj:generalfloor` are taken under the same context law for $H$:
$$
\bar{q} := \mathbb{E}[q_H], \qquad
I_k := \mathbb{E}\,\mathrm{KL}(q_H \| \bar{q}), \qquad
D_k := \mathbb{E}\bigl[\|H - \mu\|_2^2\bigr],
$$
with $\mathrm{KL}$ measured in nats. (Note that $D_k$ is total within-category feature dispersion, distinct from the spread of per-token feature means discussed elsewhere in the paper.)

**Theorem 1 (Sharp universal information floor).** *For every category size $m \ge 2$, every set of centered residual rows $\{\delta_c\}_{c \in S_k}$ with $\max_c \|\delta_c\|_2 \le R$, and every square-integrable context law $H$ with mean $\mu$,*
$$
I_k \le \frac{1}{8}\mathbb{E}\bigl[w(H - \mu)^2\bigr] \le \frac{L^2}{8} D_k \le \frac{R^2}{2} D_k.
\tag{1}
$$
*In particular, $R^2 D_k \ge 2 I_k$ and $L^2 D_k \ge 8 I_k$. The first inequality uses only fluctuations visible in within-category logit differences.*

---

## 2. Proof of Theorem 1

The proof combines three classical ingredients: the Bregman centroid (Pythagorean) identity and exponential-family Bregman duality (Banerjee, Merugu, Dhillon, and Ghosh, 2005) together with Popoviciu's (1935) bounded-variance inequality (the core variance step of Hoeffding's lemma, Hoeffding 1963).

1. **Centroid replacement ($\bar{q} \to q_\mu$).** For any strictly positive probability vector $r$ on $S_k$, splitting $\log(q_H(c)/r_c) = \log(q_H(c)/\bar{q}_c) + \log(\bar{q}_c/r_c)$ and taking expectations over $H$ yields the exact Bregman centroid identity (Banerjee et al., 2005):
   $$
   \mathbb{E}\,\mathrm{KL}(q_H \| r)
   = \mathbb{E}\,\mathrm{KL}(q_H \| \bar{q}) + \sum_{c \in S_k} \mathbb{E}[q_H(c)] \log\frac{\bar{q}_c}{r_c}
   = I_k + \mathrm{KL}(\bar{q} \| r).
   $$
   Since $\mathrm{KL}(\bar{q} \| r) \ge 0$, setting $r = q_\mu$ gives
   $$
   I_k \le \mathbb{E}\,\mathrm{KL}(q_H \| q_\mu).
   $$
   This accounts exactly for any discrepancy between $q_\mu$ and $\bar{q}$ without introducing a $1/\min_c \bar{q}_c$ factor.

2. **Exponential-family Bregman representation and Taylor remainder.** Let $A(h) := \log \sum_{c \in S_k} \exp(\delta_c^\top h)$ be the log-partition function of the exponential family $\{q_h\}$. Its gradient is $\nabla A(h) = \sum_c q_h(c)\,\delta_c = \mathbb{E}_{C \sim q_h}[\delta_C]$. For a fixed context $h \in \mathbb{R}^d$, put $v := \mu - h$ and $f(t) := A(h + tv)$ for $t \in [0, 1]$. Substituting $\log q_h(c) = \delta_c^\top h - A(h)$ and $\log q_\mu(c) = \delta_c^\top \mu - A(\mu)$ gives the reverse Bregman identity (Banerjee et al., 2005)
   $$
   \mathrm{KL}(q_h \| q_\mu)
   = \sum_{c \in S_k} q_h(c)\bigl(\delta_c^\top(h - \mu) + A(\mu) - A(h)\bigr)
   = A(\mu) - A(h) - \nabla A(h)^\top(\mu - h)
   = f(1) - f(0) - f'(0).
   $$
   Because $f$ is smooth on $[0, 1]$, Taylor's theorem with integral remainder gives
   $$
   \mathrm{KL}(q_h \| q_\mu) = \int_0^1 (1 - t)\,f''(t)\,dt.
   $$

3. **Uniform curvature bound via Popoviciu's inequality.** Differentiating the finite log-sum-exp $f(t)$ twice yields
   $$
   f'(t) = \mathbb{E}_{C \sim q_{h+tv}}[\delta_C^\top v], \qquad
   f''(t) = \operatorname{Var}_{C \sim q_{h+tv}}(\delta_C^\top v).
   $$
   For every $t \in [0, 1]$, the random variable $\delta_C^\top v$ takes values in the closed interval $[\min_c \delta_c^\top v,\, \max_c \delta_c^\top v]$ of length $w(v)$. By Popoviciu's inequality (1935) (the variance bound underlying Hoeffding's lemma, 1963), its variance is at most its mean squared distance from the interval midpoint, hence
   $$
   f''(t) \le \frac{w(v)^2}{4} \qquad (\forall t \in [0, 1]).
   $$
   Integrating against $(1 - t)\,dt$ on $[0, 1]$ (which integrates to $1/2$) gives
   $$
   \mathrm{KL}(q_h \| q_\mu) \le \frac{w(v)^2}{4}\int_0^1 (1 - t)\,dt = \frac{w(v)^2}{8}.
   $$

4. **Expectation, Cauchy–Schwarz, and diameter bound.** Taking expectations over $H$, noting $w(-v) = w(v)$ so $w(\mu - H) = w(H - \mu)$, yields the first inequality in (1). Next, for every $v \in \mathbb{R}^d$,
   $$
   w(v) = \max_{a, b \in S_k} (\delta_a - \delta_b)^\top v \le L \|v\|_2
   $$
   by Cauchy–Schwarz, and $L = \max_{a,b} \|\delta_a - \delta_b\|_2 \le 2 \max_c \|\delta_c\|_2 \le 2R$ by the triangle inequality. Squaring and taking expectations completes (1).

*Edge cases:* If $R = 0$ or $D_k = 0$, then $q_H$ is constant almost surely, so $I_k = 0$ and (1) holds with equality. If $\mu = \mathbb{E}[H]$ exists, $R > 0$, and $D_k = \infty$, then $R^2 D_k \ge 2 I_k$ holds trivially because $I_k \le H(\bar{q}) \le \log m < \infty$. $\square$

**Binary normalization comparison.** In the paper's binary proposition ($m = 2$), the symbol $R$ bounds the norm of the *difference* $u = \delta_a - \delta_b$ rather than the centered row norm. In those units, $L = \|u\|_2$ and $w(H - \mu) = |u^\top(H - \mu)|$, so (1) yields
$$
8 I_k \le \operatorname{Var}(u^\top H) \le \|u\|_2^2 D_k,
$$
strengthening the paper's binary coefficient $16 \bar{q}_a(1 - \bar{q}_a) \le 4$ by at least a factor of $2$ uniformly over $\bar{q}_a \in (0, 1)$. In the conjecture's centered row-radius units, $\delta_a = u/2$ and $\delta_b = -u/2$, so $R = \|u\|_2 / 2$ and $L^2/8 = R^2/2$, accounting for the exact factor-of-$4$ conversion.

---

## 3. Sharpness for Every Category Size $m \ge 2$

**Proposition 2 (Sharpness with $m$ fixed and $\bar{q}$ free).** *For every fixed $m \ge 2$,*
$$
\sup \frac{2 I_k}{R^2 D_k} = 1,
$$
*where the supremum runs over centered residual rows $\{\delta_c\}_{c=1}^m$ with $R = \max_c \|\delta_c\|_2 > 0$ and finite-support context laws $H$ with $D_k > 0$ and strictly positive marginals $\bar{q}_c > 0$.*

*Proof.* By Theorem 1, the ratio is at most $1$. Normalize to $R = 1$.

- **Case $m = 2$ (and even $m$ with uniform $\bar{q}$):** Take $\delta_1 = +1$, $\delta_2 = -1$ in $\mathbb{R}^1$ and $H \in \{+t, -t\}$ with probability $1/2$ each ($t > 0$). Then $\mu = 0$, $\bar{q} = (1/2, 1/2)$, $D_k = t^2$, and
  $$
  I_k = g(t) := t \tanh t - \log \cosh t = \frac{1}{2} t^2 + O(t^4) \qquad (t \to 0^+),
  $$
  so $2 I_k / D_k \to 1$. Repeating $+1$ and $-1$ equally often ($m/2$ times each) gives the same $I_k$ and $D_k$ with a uniform marginal for every even $m$.

- **Case $m > 2$ (all $m \ge 3$):** Fix $t \in (0, 1/2)$ and put $a := \sqrt{1 - t^2}$. Define two-dimensional rows
  $$
  \delta_1 = (a, t), \qquad \delta_2 = (-a, t), \qquad
  \delta_3 = \cdots = \delta_m = \Bigl(0,\, -\frac{2t}{m - 2}\Bigr).
  $$
  These rows sum to zero, satisfy $\|\delta_1\|_2 = \|\delta_2\|_2 = 1$, and $\|\delta_c\|_2 = 2t/(m - 2) < 1$ for $c \ge 3$ (since $0 < t < 1/2$ and $m - 2 \ge 1$), so $R = 1$. Let $H$ take the two values $(t, t^{-2})$ and $(-t, t^{-2})$ with equal probability $1/2$. Because the second coordinate is deterministic, $\mu = (0, t^{-2})$ and $D_k = t^2$. At both contexts, the combined probability assigned to $\{1, 2\}$ is the same:
  $$
  p_t := q_H(1) + q_H(2)
  = \frac{2 e^{1/t} \cosh(at)}{2 e^{1/t} \cosh(at) + (m - 2) e^{-2/((m-2)t)}}
  \xrightarrow[t \to 0^+]{} 1,
  $$
  and each remaining token $c \ge 3$ has a context-independent probability $q_H(c) = \bar{q}_c = (1 - p_t)/(m - 2) > 0$, contributing zero to $\mathrm{KL}(q_H \| \bar{q})$. On $\{1, 2\}$, flipping the sign of $H_1$ swaps $q_H(1)$ and $q_H(2)$, so $\bar{q}_1 = \bar{q}_2 = p_t / 2$ while $(q_H(1)/p_t, q_H(2)/p_t) = \operatorname{softmax}(\pm at, \mp at)$. Hence
  $$
  I_k = p_t\, g(at) = \frac{1}{2} p_t a^2 t^2 + O(t^4),
  $$
  and therefore $2 I_k / (R^2 D_k) = 2 p_t g(at)/t^2 \to 1$ as $t \to 0^+$.

Every member of the family has finite features and strictly positive marginal entries $\bar{q}_c > 0$. Note that for $m \ge 3$, $\min_c \bar{q}_c \to 0$ along this sequence; thus Proposition 2 establishes sharpness for fixed $m$ with $\bar{q}$ **free**. Determining the sharp constant $c^*(m, \bar{q})$ for a separately prescribed fixed marginal $\bar{q}$ remains an independent problem. $\square$

---

## 4. Reproducible Verification

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B solutions/lm_info_floor/explore.py
```

The script [`explore.py`](explore.py) tests 600 random finite laws across feature dimensions $1, 3, 7$, 15 differential-evolution finalists (three seeded runs per $m$ optimizing centered unit-radius heads and 3-atom context laws in $\mathbb{R}^2$), 25 sharp-family instances ($t \in \{0.2, 0.1, 0.05, 0.02, 0.01\}$), and 10 degenerate cases ($R = 0$ or $D_k = 0$). It verifies the entropy and centroid-KL identities, the reverse Bregman identity, the 64-node Gauss–Legendre curvature integral, Popoviciu's variance bound, every inequality in (1), and the exact closed form $p_t g(at)$ of the sharp family. Scores below report $2 I_k / (R^2 D_k) \le 1$ (sharp family at $t = 0.01$):

| Category size $m$ | Random laws tested | Best of 3 DE searches | Sharp family ($t = 0.01$) |
|---:|---:|---:|---:|
| 2 | 120 | 0.99960771 | 0.99995000 |
| 3 | 120 | 0.86421804 | 0.99985001 |
| 4 | 120 | 0.92866338 | 0.99985001 |
| 5 | 120 | 0.80378222 | 0.99985001 |
| 10 | 120 | 0.83939439 | 0.99985001 |

All assertions passed (NumPy 2.5.5, SciPy 1.18.1).

---

## References

- B. Abrahao, *Neural Collapse Is Forbidden: Information Floors in Language Models*, [arXiv:2607.09487](https://arxiv.org/abs/2607.09487), July 2026.
- A. Banerjee, S. Merugu, I. S. Dhillon, and J. Ghosh, *Clustering with Bregman Divergences*, Journal of Machine Learning Research 6:1705–1749, 2005.
- W. Hoeffding, *Probability Inequalities for Sums of Bounded Random Variables*, Journal of the American Statistical Association 58(301):13–30, 1963.
- T. Popoviciu, *Sur les équations algébriques ayant toutes leurs racines réelles*, Mathematica (Cluj) 9:129–145, 1935.
