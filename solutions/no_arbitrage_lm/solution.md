# Analysis and Refutation of Conjecture `conj:spread` (Equilibrium Spread)

**Target paper:** Daniel Dragonevskiy, *Understanding as No-Arbitrage: Bounded Dutch Books as a Definition and Training Objective for Language Models*, [arXiv:2609.39341](https://arxiv.org/abs/2609.39341) (September 2026), Section 4, Conjecture (`conj:spread`, *Equilibrium spread*).  
**Companion files:** [`solutions/no_arbitrage_lm/explore.py`](explore.py), [`solutions/no_arbitrage_lm/proof.pdf`](proof.pdf).

---

## Verdict

- **Specification status (honest framing):** Conjecture `conj:spread` is **underspecified** in arXiv:2609.39341: the paper leaves open the interval scoring rule / coverage payoff, the calibration bettor's information set and utility, the credibility level and interval convention, and the convergence mode (experiment E-A1b uses a *point* Brier anchor $\mu(p - t)^2$ with $\mu = 0.5$, and E-A2 leaves spread-valued credences open). Consequently, there is no single equilibrium width uniquely determined by the paper's prose alone.
- **Conjecture `conj:spread` under natural formalizations: FALSE.** Formalizing the stated opposing pressures (proper scoring + width penalty vs. logical traders and calibration bettors settling on verifiable binary anchors) refutes the proposed equality on four separate grounds:
  1. **Public-information Beta saddle (Theorem 1):** Against any bounded public-information calibration bettor (and any finite-budget logical trader), the unique minimax quote for every strictly proper binary score is the zero-width predictive mean $[m, m]$ (spread $0$), whereas every 95% credible interval of $\operatorname{Beta}(1, 1)$ has width $19/20$.
  2. **1-bit non-identifiability and repeated settlement (Proposition 2):** Any integrable payoff settled on a single binary outcome $Y$ depends on the posterior solely through its mean $m$, so posteriors with identical mean and wildly different credible widths induce identical games. In repeated settlement, sequential predictive means $m_t$ make every bounded predictable stake a zero-mean martingale with almost-sure $o(1)$ per-round gain, keeping equilibrium spread identically $0$ at every finite $t$.
  3. **Private-signal calibration bettor (Theorem 3):** Giving the bettor one extra private draw induces a positive equilibrium spread $[1/2 - d, 1/2 + d]$, but its width depends strictly on the scoring rule and bettor budget $B$, and is capped at $2D = 1/(2a+1)$ ($1/3 < 19/20$ for $\operatorname{Beta}(1, 1)$ even as $B \to \infty$).
  4. **Logical traders and anchor natural extension (Proposition 4):** By Walley's nonempty-core criterion (Walley 1991; Proposition `prop:walley`), any singleton inside the natural-extension interval $[L, U]$ avoids sure loss. For $\varphi \Rightarrow A$ with $P(A) = 1/2$, the anchor-optimal quote $1/2$ on $A$ paired with singleton $q(\varphi) = 1/4$ (width $0$) defeats all logical traders and public anchor bettors despite natural-extension width $1/2$.
- **Positive contrast:** Equal-tailed Beta credible intervals *are* uniquely elicited by the proper interval score $(u - l) + \frac{2}{\alpha}[(l - Z)_+ + (Z - u)_+]$ (Gneiting & Raftery 2007, Section 6.2), but only by settling directly on the latent parameter $Z = \theta$ and specifying a tail level $\alpha$—ingredients absent from single-bit anchor settlement.

---

## 1. Exact Statement and Missing Specifications

In Section 3–4 of arXiv:2609.39341, $W \subseteq \{0, 1\}^\Phi$ is the finite set of logically admissible worlds, $C := \operatorname{conv}(W)$ is the coherent polytope, and logical portfolios satisfy $\|x\|_1 \le 1$ with worst-world profit $\min_{w \in W} x \cdot (w - q)$. Proposition `prop:walley` (restating Walley 1991) states that interval credences $[l(\varphi), u(\varphi)]$ avoid sure loss iff there exists $p \in C$ with $l \le p \le u$, and defines the natural extension via the LP bounds $[\min, \max]\,p(\psi)$ over $\{p \in C : l \le p \le u \text{ on quoted propositions}\}$. Immediately afterward:

**Conjecture (`conj:spread`, arXiv:2609.39341, Section 4).** *Under a proper-scoring/coverage pressure (narrowing) opposed by logical traders and calibration bettors (Foster & Vohra 1998) settling on a verifiable anchor set (widening where unfounded), equilibrium spread width on $\varphi$ converges to the epistemic uncertainty consistent with the anchor and $\varphi$'s logical relations to it. In the exchangeable toy case (Beta posterior over a settleable event family), the equilibrium width equals the credible-interval width.*

The paper does not specify (i) how an interval $[l, u]$ is scored by a proper rule or coverage penalty, (ii) what information and utility the calibration bettor holds, (iii) which credibility level $1 - \alpha$ or interval convention (equal-tailed vs. HPD) is meant, or (iv) the convergence regime. Sections 2–4 below supply explicit, natural formalizations of each component.

---

## 2. Public-Information Beta Saddle and 1-Bit Non-Identifiability

Condition on the public history. Let $\theta \sim \operatorname{Beta}(a, b)$ ($a, b > 0$), draw the next verifiable outcome $Y \mid \theta \sim \operatorname{Bernoulli}(\theta)$, and set $m := \mathbb{E}[Y] = a/(a + b)$. The model quotes an interval $0 \le l \le u \le 1$. A public-information bettor chooses stake $x \in [-B, B]$ (buying at $u$ if $x > 0$, selling at $l$ if $x < 0$, or abstaining if $x = 0$), with settlement payoff
$$
G(l, u, x, Y) := x^+(Y - u) + x^-(l - Y),
$$
where $x^+ = \max(x, 0)$ and $x^- = \max(-x, 0)$. For this single proposition, both truth values $\{0, 1\}$ are admissible, so any finite-budget logical trader has optimal worst-world profit $0$ at every legal interval $[l, u] \subseteq [0, 1]$.

Let $S(q, Y)$ be any strictly proper binary loss with finite Bayes risk $R(q; m) := \mathbb{E}_m[S(q, Y)]$, and let $c > 0$ be a unit width charge. Define the model's expected loss (absorbing any positive adversary multiplier into $B$) by
$$
L(l, u, x) := \frac{1}{2}\bigl[R(l; m) + R(u; m)\bigr] + c(u - l) + \mathbb{E}[G(l, u, x, Y)].
$$

**Theorem 1 (Public-information Beta saddle).** *For every $a, b > 0$, every strictly proper binary loss $S$, every width cost $c > 0$, and every finite budget $B \ge 0$, the game $(l, u) \mapsto \max_{|x| \le B} L(l, u, x)$ has unique minimax quote $[m, m]$ (spread $0$), optimal bettor stake $x = 0$, and game value $R(m; m)$.*

*Proof.* Since $\mathbb{E}[G(l, u, x, Y)] = x^+(m - u) + x^-(l - m)$, maximizing over $|x| \le B$ gives $B\max(0, m - u, l - m) \ge 0$. At $[l, u] = [m, m]$, every stake $|x| \le B$ yields $\mathbb{E}[G] = 0$, so $L(m, m, x) = R(m; m)$. Conversely, against $x = 0$, strict propriety of $S$ and $c > 0$ give
$$
L(l, u, 0) = \frac{1}{2}\bigl[R(l; m) + R(u; m)\bigr] + c(u - l) \ge R(m; m),
$$
with equality if and only if $l = u = m$. Thus $L(m, m, x) = R(m; m) = L(m, m, 0) \le L(l, u, 0)$ for all admissible $(l, u, x)$. Because $x = 0$ is always feasible, $\max_{|x|\le B} L(l, u, x) \ge L(l, u, 0) \ge R(m; m)$ with unique equality at $[m, m]$. (The same proof applies to strictly proper midpoint scoring $R(\frac{l+u}{2}; m)$ plus any strictly increasing width penalty minimized at $0$.) $\square$

For $\operatorname{Beta}(1, 1)$, every $95\%$ credible interval has length $19/20 = 0.95$ (the equal-tailed interval is $[1/40, 39/40]$), and every positive-mass credible interval of any proper Beta law has strictly positive width—yet the minimax spread is identically $0$.

**Proposition 2 (1-bit non-identifiability and repeated settlement).**
1. *Single-bit non-identifiability:* For any fixed quote $(l, u)$, bettor action $x$, and integrable payoff $h(l, u, x, Y)$ settled on the next binary outcome $Y$, $\mathbb{E}[h(l, u, x, Y)] = (1 - m)h(l, u, x, 0) + m h(l, u, x, 1)$ depends on the posterior solely through its mean $m$. For example, $\operatorname{Beta}(1, 1)$ and $\operatorname{Beta}(20, 20)$ both have $m = 1/2$ while their $95\%$ equal-tailed widths are $0.950000$ and $0.304396$.
2. *Repeated settlement:* Let $\mathcal{F}_t$ be the public history after $t$ exchangeable Bernoulli draws and $m_t := (a + \sum_{i=1}^t Y_i)/(a + b + t)$. Quoting $[m_t, m_t]$ at every round makes $\mathbb{E}[x_t(Y_{t+1} - m_t) \mid \mathcal{F}_t] = 0$ for every bounded $\mathcal{F}_t$-predictable stake $x_t$ (including stakes selected by forecast bins, as in Foster & Vohra 1998). Cumulative bettor gain is a zero-mean martingale, and for uniformly bounded stakes the average gain per round converges to $0$ almost surely by Hoeffding–Azuma and Borel–Cantelli. Under additive per-round losses with exogenous draws, sequential predictive means $[m_t, m_t]$ and $x_t = 0$ form a saddle over every finite horizon; the ratio of equilibrium spread to Beta credible width is identically $0$ at every finite $t$.

---

## 3. Private-Signal Calibration Bettor: Score- and Budget-Dependent Cap

Suppose now we modify the game by giving the bettor private information. Start from a symmetric posterior $\operatorname{Beta}(a, a)$ ($a \ge 1$) with $c = 1/16$, and let the bettor privately observe one extra conditionally independent draw $X \mid \theta \sim \operatorname{Bernoulli}(\theta)$ after the public quote $[l, u]$ is posted and before betting on $Y$. The two conditional means $p_X = \mathbb{P}(Y = 1 \mid X) = \frac{1}{2} \pm D$ each occur with probability $1/2$, where $D := \frac{1}{2(2a + 1)}$. Against $[l, u]$, the informed bettor's best response earns conditional expected profit $B\max(0, p_X - u, l - p_X)$.

Consider binary log loss, binary Brier loss $(q - Y)^2$, and negative spherical score $-[Yq + (1-Y)(1-q)]/\sqrt{q^2 + (1-q)^2}$, with forecast-derivative slopes $g_S(d) := R'(\frac{1}{2} + d; \frac{1}{2})$:
$$
g_{\mathrm{log}}(d) = \frac{d}{1/4 - d^2}, \qquad
g_{\mathrm{Brier}}(d) = 2d, \qquad
g_{\mathrm{spherical}}(d) = \frac{d}{(1/2 + 2d^2)^{3/2}}.
$$

**Theorem 3 (Informed-bettor equilibrium).** *For each of the three scores above, any $a \ge 1$, $c = 1/16$, and $B \ge 0$, the unique subgame-perfect equilibrium quote is $[1/2 - d, 1/2 + d]$ with*
$$
d = \bigl(g_S|_{[0, D]}\bigr)^{-1}\!\Bigl(\min\bigl\{g_S(D),\, \max(0, B - 2c)\bigr\}\Bigr) \in [0, D].
$$
*In particular, the equilibrium spread $2d$ depends on both the scoring rule $S$ and the budget $B$, and is strictly capped at $2D = \frac{1}{2a + 1}$ for all $B$.*

*Proof.* Clipping $l, u$ into $[\frac{1}{2} - D, \frac{1}{2} + D]$ cannot increase width or any profitable trade against $p_X = \frac{1}{2} \pm D$, and strictly lowers $R(\cdot; \frac{1}{2})$ for any moved endpoint because $R'(q; \frac{1}{2})$ has the sign of $q - \frac{1}{2}$. Hence any optimum lies in $\frac{1}{2} - D \le l \le u \le \frac{1}{2} + D$. On $|d| \le D \le 1/6$, all three scores are strictly convex: for spherical loss, $g'_{\mathrm{spherical}}(d) = (\frac{1}{2} - 4d^2)/(\frac{1}{2} + 2d^2)^{5/2} > 0$ since $4D^2 \le 1/9 < 1/2$, and $g'_{\mathrm{log}}, g'_{\mathrm{Brier}} > 0$ immediately. The bettor's expected optimal profit is convex in $(l, u)$ as an average of maxima of affine functions. Thus the model objective is strictly convex with a unique minimizer, symmetric by invariance under $(l, u) \mapsto (1 - u, 1 - l)$. Substituting $l = \frac{1}{2} - d$, $u = \frac{1}{2} + d$ ($0 \le d \le D$) simplifies the objective to
$$
R\bigl(\tfrac{1}{2} + d; \tfrac{1}{2}\bigr) + 2cd + B(D - d),
$$
whose derivative $g_S(d) + 2c - B$ and boundary conditions at $d = 0, D$ yield the stated formula. $\square$

For $\operatorname{Beta}(1, 1)$, the $95\%$ credible width is $19/20 = 0.95$, whereas the informed-game spread never exceeds $2D = 1/3$ at any budget, and varies sharply across scores below saturation:

| Budget $B$ | Log width | Brier width | Spherical width | $\operatorname{Beta}(1,1)$ 95% width |
|---:|---:|---:|---:|---:|
| 0.125 | 0.000000 | 0.000000 | 0.000000 | 0.950000 |
| 0.250 | 0.062258 | 0.125000 | 0.089451 | 0.950000 |
| 0.500 | 0.181335 | 0.333333 | 0.302342 | 0.950000 |
| 1.000 | 0.333333 | 0.333333 | 0.333333 | 0.950000 |

---

## 4. Logical Traders, Anchor Natural Extension, and Interval Scoring

**Proposition 4 (Singleton quotes defeat all logical traders).** *Fix consistent anchor quotes with nonempty compact convex coherent set $K \subseteq C$, and let $[L, U] := \{p(\varphi) : p \in K\}$ be the natural extension of $\varphi$.*
1. *By Walley's nonempty-core criterion (Walley 1991; Proposition `prop:walley`), quoting $[l, u]$ on $\varphi$ together with the anchors avoids sure loss if and only if $[l, u] \cap [L, U] \neq \varnothing$. In particular, every singleton $[q, q]$ with $q \in [L, U]$ has zero logical exploitability (even against unrestricted logical portfolios).*
2. *Specifically, let $\varphi \Rightarrow A$ with verifiable anchor probability $P(A) = 1/2$. Then the natural extension of $\varphi$ is $[0, 1/2]$ (width $1/2$), yet quoting $1/2$ on $A$ and the singleton $q(\varphi) = 1/4$ (width $0$) defeats all logical traders and public anchor bettors while attaining every individual lower bound on model loss (proper score on $A$ plus width penalty on both quotes).*

*Proof.* Part (1) is immediate from convexity of $K$: $[L, U]$ is an interval, so $[l, u] \cap [L, U] \neq \varnothing$ iff there exists $p \in K$ with $l \le p(\varphi) \le u$. (Requiring the quote to contain *every* anchor-consistent extension is the strictly stronger condition $l \le L \le U \le u$, whose sharpest solution is $[L, U]$.) For part (2), in coordinate order $(A, \varphi)$ the admissible worlds are $\{00, 10, 11\}$. For any $r \in [0, 1/2]$, the distribution $(1/2, 1/2 - r, r)$ matches $P(A) = 1/2$ and gives $P(\varphi) = r$; in particular $r = 1/4$ is a coherent extension, so quoting $1/2$ on $A$ and singleton $1/4$ on $\varphi$ yields zero guaranteed logical profit and zero expected gain for public anchor bettors. All such laws generate identical anchor observations under independent repetition, preserving the width gap ($0$ vs. $1/2$) at every horizon. $\square$

**When Beta credible widths can be recovered (Gneiting & Raftery 2007).** To elicit an equal-tailed $(1 - \alpha)$ credible interval for a target $Z$, one must fix $\alpha \in (0, 1)$ and score $[l, u]$ directly against $Z$ using the proper interval loss (Gneiting & Raftery 2007, Section 6.2):
$$
S_\alpha(l, u; Z) := (u - l) + \frac{2}{\alpha}\bigl[(l - Z)_+ + (Z - u)_+\bigr].
$$
For a continuous target $Z \sim F$, the expected partial derivatives are $-1 + 2F(l)/\alpha$ and $1 - 2[1 - F(u)]/\alpha$, giving unique minimizer $[F^{-1}(\alpha/2), F^{-1}(1 - \alpha/2)]$. Taking $Z = \theta$ recovers the Beta equal-tailed credible interval—but requires both settling on the latent parameter $\theta$ (not a single binary draw $Y$) and committing explicitly to a tail level $\alpha$.

---

## 5. Reproducible Verification and References

Run from the repository root:

```sh
.venv/bin/python -B solutions/no_arbitrage_lm/explore.py
```

The script independently solves the full constrained endpoint-minimization epigraph over all pure contingent trading strategies (`SLSQP`) across 126 public-information cases (6 Beta laws $\times$ 3 scores $\times$ 7 budgets) and 63 private-signal cases (3 symmetric Beta laws $\times$ 3 scores $\times$ 7 budgets), matching Theorems 1 and 3 with maximum endpoint discrepancy $4.8 \times 10^{-7} < 5 \times 10^{-7}$. It also solves the two anchor extrema LPs (verifying natural extension $[0, 0.5]$) and three primal logical-trader LPs (verifying profit $0$ at $[0, 0.5]$ and singleton $[0.25, 0.25]$, vs. unit-budget profit $0.05$ at the disjoint quote $[0.6, 0.7]$).

### References
- D. P. Foster and R. V. Vohra, *Asymptotic calibration*, Biometrika **85**(2):379–390, 1998.
- T. Gneiting and A. E. Raftery, *Strictly proper scoring rules, prediction, and estimation*, Journal of the American Statistical Association **102**(477):359–378, 2007 (Section 6.2).
- P. Walley, *Statistical Reasoning with Imprecise Probabilities*, Chapman & Hall, 1991.
