# Exact Single-Round KL Contraction for the Gaussian Hypercube Channel

**Target paper:** Munsik Kim, *Information-Theoretic Lower Bounds for Bit-Constrained Stochastic Optimization via a Reduction to Compressed Gaussian Mean Estimation*, [arXiv:2606.00703](https://arxiv.org/abs/2606.00703), Appendix D.4, Conjecture `conj:singleround`.  
**Companion files:** [`solutions/bit_constrained_kl/explore.py`](explore.py), [`solutions/bit_constrained_kl/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:singleround`: PROVED with sharp absolute constant $C = 1$** (and for all $\mathrm{SNR} \ge 0$, not only $\mathrm{SNR} \le 1$).
- **Exact contraction identity (Theorem 1):** Write $s := \delta^2/\sigma^2 \ge 0$ for the per-coordinate SNR and $Z \sim \mathcal{N}(0, 1)$. For every dimension $d \ge 1$ and every $s \ge 0$, the distribution-dependent KL and $\chi^2$ contraction coefficients of the reverse channel $P_{V \mid Y}$ at $P_Y$ coincide exactly:
  $$
  \eta_{\mathrm{KL}}(P_Y, P_{V \mid Y})
  = \eta_{\chi^2}(P_Y, P_{V \mid Y})
  = r(s) := \mathbb{E}\bigl[\tanh^2(s + \sqrt{s}\,Z)\bigr].
  $$
  For every $s > 0$, $r(s) < \min\{s, 1\}$, and $\lim_{s \downarrow 0} r(s)/s = 1$; thus $C = 1$ is the sharp prefactor in $\eta_{\mathrm{KL}} \le C s$.
- **Single-round public-coin consequence (Corollary 2):** For any public-coin $B$-bit message $M$ generated from $Y$ and independent public randomness $U \perp (V, Y)$,
  $$
  I(V; M \mid U) \le r(s)\, I(Y; M \mid U) \le r(s)\, B \le s\, B \qquad (\text{in bits}).
  $$
- **Correction to Appendix D.4 and multi-round scope (Section D.5):** The identity $\eta_{\mathrm{KL}} = \eta_{\chi^2} = r(s)$ disproves the informal claim in Appendix D.4 of arXiv:2606.00703 that rare-event messages strictly exceed squared maximal correlation for this channel. As emphasized in Section D.5 of arXiv:2606.00703, however, **multi-round adaptive prefixes remain a separate problem**: conditioning on earlier messages $M_{1:t-1}$ can correlate the coordinates of $V$, breaking the product prior $P_V$, so the single-round tensorization proved here does not by itself telescope to $T > 1$ rounds.

---

## 1. Setup and Scalar Shift Identity

Let $V = (V_1, \dots, V_d)$ have independent Rademacher coordinates $\mathbb{P}(V_j = \pm 1) = 1/2$, and let $Y \mid V \sim \mathcal{N}(\delta V, \sigma^2 I_d)$, with per-coordinate SNR $s := \delta^2/\sigma^2 \ge 0$. For the reverse Markov kernel $K := P_{V \mid Y}$, the distribution-dependent KL and $\chi^2$ contraction coefficients at the marginal $P_Y$ (Polyanskiy and Wu, 2017, 2025) are
$$
\eta_{\mathrm{KL}}(P_Y, K) := \sup_{0 < D(Q \Vert P_Y) < \infty} \frac{D(QK \Vert P_V)}{D(Q \Vert P_Y)},
\qquad
\eta_{\chi^2}(P_Y, K) := \sup_{0 < \chi^2(Q \Vert P_Y) < \infty} \frac{\chi^2(QK \Vert P_V)}{\chi^2(Q \Vert P_Y)}.
$$
We work in nats throughout; changing logarithm base cancels in $\eta_{\mathrm{KL}}$. At $s = 0$, $V$ and $Y$ are independent, so both coefficients equal $0 = r(0)$. Assume $s > 0$ from now on.

Consider first $d = 1$. Rescale the observation to the sufficient statistic $X := \delta Y / \sigma^2 = s V + \sqrt{s}\, Z$ with $Z \sim \mathcal{N}(0, 1)$, and denote its symmetric marginal law by $P$ (an equal mixture of $\mathcal{N}(+s, s)$ and $\mathcal{N}(-s, s)$). The posterior log-odds are $2X$, so $W := \mathbb{E}[V \mid X] = \tanh(X)$, and by symmetry
$$
r := \mathbb{E}[W^2] = \mathbb{E}\bigl[\tanh^2(s + \sqrt{s}\,Z)\bigr] = r(s) \in (0, 1).
$$

**Lemma 1 (Integration identity and SNR bound).** *For every $f \in C^1([-1, 1])$,*
$$
\mathbb{E}[W f(W)] = \mathbb{E}\bigl[(1 - W^2) A(X) f'(W)\bigr],
\qquad
A(x) := \int_0^s e^{(s^2 - u^2)/(2s)} \frac{\cosh(xu/s)}{\cosh x}\,du. \tag{1}
$$
*Moreover, $A(x)$ is even and nonincreasing in $|x|$, $\mathbb{E}[(1 - W^2) A(X)] = r$, $r(s) < \min\{s, 1\}$ for all $s > 0$, and $\lim_{s \downarrow 0} r(s)/s = 1$.*

*Proof.* By the tower property and symmetry of $(V, X)$, $\mathbb{E}[W f(W)] = \mathbb{E}[V f(W)] = \frac{1}{2}(F(s) - F(-s)) = \frac{1}{2}\int_{-s}^s F'(u)\,du$, where $F(u) := \mathbb{E}[f(\tanh(u + \sqrt{s}\,Z))]$. Differentiating under the expectation (justified since $f'$ is bounded on $[-1, 1]$ and $\tanh'(x) = \operatorname{sech}^2(x) \le 1$) gives
$$
F'(u) = \mathbb{E}_{X_u \sim \mathcal{N}(u, s)}\bigl[(1 - \tanh^2(X_u)) f'(\tanh(X_u))\bigr].
$$
The likelihood ratio of $\mathcal{N}(u, s)$ relative to $P = \frac{1}{2}\mathcal{N}(s, s) + \frac{1}{2}\mathcal{N}(-s, s)$ is
$$
\frac{d\mathcal{N}(u, s)}{dP}(x)
= \frac{\exp\!\bigl(-(x - u)^2/(2s)\bigr)}{\frac{1}{2}\exp\!\bigl(-(x-s)^2/(2s)\bigr) + \frac{1}{2}\exp\!\bigl(-(x+s)^2/(2s)\bigr)}
= e^{(s^2 - u^2)/(2s)} \frac{e^{xu/s}}{\cosh x}.
$$
Averaging $u$ and $-u$ inside $\frac{1}{2}\int_{-s}^s F'(u)\,du$ gives (1).

For $a = u/s \in [0, 1]$ and $x \ge 0$, the logarithmic derivative of $\cosh(ax)/\cosh x$ is $a\tanh(ax) - \tanh x \le 0$, so $A(x)$ is even and nonincreasing in $|x|$. Taking $f(w) = w$ in (1) yields $\mathbb{E}[(1 - W^2)A(X)] = r$, while applying $f(w) = w$ directly to $\frac{1}{2}\int_{-s}^s F'(u)\,du$ and substituting $u = sv$ yields
$$
\frac{r(s)}{s} = \frac{1}{2}\int_{-1}^1 \mathbb{E}\bigl[\operatorname{sech}^2(sv + \sqrt{s}\,Z)\bigr]\,dv.
$$
For $s > 0$, $\operatorname{sech}^2(sv + \sqrt{s}\,Z) < 1$ almost surely, so $r(s)/s < 1$; and as $s \downarrow 0$, dominated convergence yields $r(s)/s \to 1$. Since $|W| < 1$ almost surely, $r(s) < 1$ as well. $\square$

---

## 2. Global Posterior CGF Bound and Exact Scalar Coefficients

Let $L(t) := \log \mathbb{E}[e^{tW}]$ be the cumulant generating function of $W = \tanh(X)$, let $\mathbb{E}_t$ denote expectation under the tilted measure $dP_t/dP = e^{tW - L(t)}$, and set $m(t) := L'(t) = \mathbb{E}_t[W]$.

**Lemma 2 (Global CGF bound).** *For all $t \in \mathbb{R}$,*
$$
L(t) \le \frac{1}{r} \log \cosh(rt). \tag{2}
$$

*Proof.* Set $k := r / (1 - r) > 0$. Define the probability measure $\tilde{P}$ with density $(1 - W^2)/(1 - r)$ relative to $P$. By Lemma 1, $\mathbb{E}_{\tilde{P}}[A(X)] = k$. Since $\tilde{P}$ is symmetric, $A(X)$ is nonincreasing in $|X|$, and $\cosh(tW)$ is nondecreasing in $|X|$ for $t \ge 0$, Chebyshev's association inequality (averaging $(A(X) - A(X'))(\cosh(tW) - \cosh(tW')) \le 0$ over i.i.d. copies $X, X' \sim \tilde{P}$) gives
$$
\mathbb{E}_{\tilde{P}}[A(X)\cosh(tW)] \le k\,\mathbb{E}_{\tilde{P}}[\cosh(tW)].
$$
Because $\tilde{P}$ is symmetric and $A(X)$ is even, we may replace $\cosh(tW)$ by $e^{tW}$ on both sides, obtaining $\mathbb{E}_t[(1 - W^2)A(X)] \le k\,\mathbb{E}_t[1 - W^2]$.

Applying (1) to $f(w) = e^{tw}$ yields $\mathbb{E}[W e^{tW}] = t\,\mathbb{E}[(1 - W^2)A(X) e^{tW}]$. Dividing by $e^{L(t)}$ and using $\mathbb{E}_t[W^2] = \operatorname{Var}_t(W) + m(t)^2 = m'(t) + m(t)^2$ gives, for every $t > 0$,
$$
m(t) = t\,\mathbb{E}_t[(1 - W^2)A(X)] \le tk\bigl(1 - m'(t) - m(t)^2\bigr),
$$
or equivalently
$$
m'(t) \le 1 - m(t)^2 - \frac{m(t)}{kt}. \tag{3}
$$
Compare $m(t)$ with $h(t) := \tanh(rt)$. Since $\sinh(2rt) \ge 2rt$ for $t > 0$, we have $\frac{h(t)}{rt} = \frac{\sinh(rt)\cosh(rt)}{rt\cosh^2(rt)} = \frac{\sinh(2rt)}{2rt}(1 - h(t)^2) \ge 1 - h(t)^2$. Multiplying by $1 - r = r/k$ yields $h(t)/(kt) \ge (1 - r)(1 - h(t)^2)$, i.e.,
$$
h'(t) = r(1 - h(t)^2) \ge 1 - h(t)^2 - \frac{h(t)}{kt}.
$$
Subtracting this from (3) for $g(t) := m(t) - h(t)$ yields
$$
g'(t) + \biggl(m(t) + h(t) + \frac{1}{kt}\biggr) g(t) \le 0 \qquad (t > 0).
$$
Set the integrating factor $\mu(t) := t^{1/k} \exp\!\bigl(\int_0^t (m(a) + h(a))\,da\bigr)$. Then $(\mu g)'(t) \le 0$ on $(0, \infty)$. Since $g$ is bounded near $0$ and $\mu(t) \to 0$ as $t \downarrow 0$, integrating from $0$ to $t$ gives $\mu(t) g(t) \le 0$, hence $m(t) \le h(t)$ for all $t > 0$. Integrating $m \le h$ from $0$ to $t$ gives (2) for $t \ge 0$, and evenness of $L$ gives (2) for all $t \in \mathbb{R}$. $\square$

**Proposition 3 (Exact scalar coefficients).** *For $d = 1$ and $s > 0$, $\eta_{\mathrm{KL}}(P, K_1) = \eta_{\chi^2}(P, K_1) = r(s)$.*

*Proof.* Let $Q$ be any distribution with $0 < D(Q \Vert P) < \infty$. Its output law $QK_1$ on $\{\pm 1\}$ has mean $b := \mathbb{E}_Q[W] \in (-1, 1)$ (or $b = \pm 1$, handled by continuity) and KL divergence from the fair sign law $\pi$ given by
$$
D(QK_1 \Vert \pi) = J(b) := \frac{1+b}{2}\log(1+b) + \frac{1-b}{2}\log(1-b) = \sup_{z \in \mathbb{R}} \{zb - \log \cosh z\}.
$$
By nonnegativity of $D(Q \Vert P_t)$ for every $t \in \mathbb{R}$, we have $D(Q \Vert P) \ge tb - L(t)$. Substituting (2) and $z = rt$ gives
$$
D(Q \Vert P) \ge \sup_{t \in \mathbb{R}} \Bigl\{tb - \frac{1}{r}\log\cosh(rt)\Bigr\} = \frac{J(b)}{r} = \frac{D(QK_1 \Vert \pi)}{r},
$$
so $\eta_{\mathrm{KL}}(P, K_1) \le r$.

For the lower bound, consider $dQ_\epsilon/dP = 1 + \epsilon W$ for $0 < |\epsilon| < 1$. Its output mean is $\epsilon r$. Since $|W| \le 1$, Taylor expansion gives $D(Q_\epsilon \Vert P) = \frac{1}{2}\epsilon^2 r + o(\epsilon^2)$ and $D(Q_\epsilon K_1 \Vert \pi) = J(\epsilon r) = \frac{1}{2}\epsilon^2 r^2 + o(\epsilon^2)$, whose ratio tends to $r$ as $\epsilon \to 0$. Thus $\eta_{\mathrm{KL}}(P, K_1) = r$.

For $\chi^2$, writing $dQ/dP = 1 + g$ (with $\mathbb{E}_P[g] = 0$), we have $\chi^2(Q \Vert P) = \mathbb{E}_P[g^2]$ and $\chi^2(QK_1 \Vert \pi) = (\mathbb{E}_P[gW])^2 \le \mathbb{E}_P[W^2]\,\mathbb{E}_P[g^2] = r\,\chi^2(Q \Vert P)$ by Cauchy–Schwarz, with equality at $g = \epsilon W$. $\square$

---

## 3. Tensorization to Arbitrary Dimension $d \ge 1$ and Public Coins

**Theorem 1 (Exact $d$-dimensional contraction coefficients).** *For every $d \ge 1$ and every $s \ge 0$,*
$$
\eta_{\mathrm{KL}}(P_Y, P_{V \mid Y}) = \eta_{\chi^2}(P_Y, P_{V \mid Y}) = r(s) = \mathbb{E}\bigl[\tanh^2(s + \sqrt{s}\,Z)\bigr].
$$

*Proof.* The case $s = 0$ is immediate (both sides vanish), so assume $s > 0$ (hence $0 < r < 1$). Because $(Y, V)$ has i.i.d. coordinates, the reverse kernel factors as $K = K_1^{\otimes d}$ with input marginal $P_Y = P_1^{\otimes d}$ and output marginal $P_V = \pi^{\otimes d}$.

*KL tensorization.* Let $Q$ be an arbitrary (possibly joint) law on $\mathbb{R}^d$ with $D(Q \Vert P_1^{\otimes d}) < \infty$, and define the joint law $Q_{Y^d, V^d} := Q_{Y^d} K_1^{\otimes d}$. By the chain rule for KL divergence and Proposition 3 applied coordinatewise,
$$
\begin{aligned}
D(Q_{V^d} \Vert \pi^{\otimes d})
&= \sum_{i=1}^d \mathbb{E}_{Q}\bigl[D(Q_{V_i \mid V_{<i}} \Vert \pi)\bigr] \\
&\le r \sum_{i=1}^d \mathbb{E}_{Q}\bigl[D(Q_{Y_i \mid V_{<i}} \Vert P_1)\bigr] \\
&\le r \sum_{i=1}^d \mathbb{E}_{Q}\bigl[D(Q_{Y_i \mid Y_{<i}} \Vert P_1)\bigr]
= r\, D(Q \Vert P_1^{\otimes d}).
\end{aligned}
$$
In the second inequality, $V_{<i}$ is generated solely from $Y_{<i}$ via $K_1^{\otimes (i-1)}$, so $Y_i - Y_{<i} - V_{<i}$ forms a Markov chain; hence $Q_{Y_i \mid V_{<i}}$ is a mixture of the conditionals $Q_{Y_i \mid Y_{<i}}$, and convexity of KL in its first argument gives $\mathbb{E}[D(Q_{Y_i \mid V_{<i}} \Vert P_1)] \le \mathbb{E}[D(Q_{Y_i \mid Y_{<i}} \Vert P_1)]$. Perturbing only coordinate $1$ (keeping coordinates $2, \dots, d$ at $P_1$) recovers the scalar supremum $r$, proving $\eta_{\mathrm{KL}}(P_Y, K) = r$.

*$\chi^2$ tensorization.* Let $dQ/dP_Y = 1 + g$ with $g \in L^2(P_Y)$ and $\mathbb{E}_{P_Y}[g] = 0$. The parity characters $v_S := \prod_{i \in S} v_i$ ($S \subseteq [d]$) form an orthonormal basis of $L^2(\pi^{\otimes d})$. The density perturbation of $QK$ relative to $\pi^{\otimes d}$ has Fourier coefficient $\mathbb{E}_{P_Y}[g(Y) W_S]$ at each nonempty $S$, where $W_S := \prod_{i \in S} W_i$ and $W_i := \mathbb{E}[V_i \mid Y_i]$. Because the $W_i$ are independent, mean-zero, and satisfy $\mathbb{E}[W_i^2] = r$, the family $\{W_S\}_{\varnothing \neq S \subseteq [d]}$ is orthogonal in $L^2(P_Y)$ with $\mathbb{E}[W_S^2] = r^{|S|}$. Normalizing to the orthonormal family
$$
e_S := \frac{W_S}{r^{|S|/2}} \qquad (\varnothing \neq S \subseteq [d]),
$$
Parseval's identity on $L^2(\pi^{\otimes d})$, the bound $r^{|S|} \le r$ (as $0 < r < 1$ and $|S| \ge 1$), and Bessel's inequality in $L^2(P_Y)$ yield
$$
\chi^2(QK \Vert P_V)
= \sum_{\varnothing \neq S \subseteq [d]} \bigl(\mathbb{E}[g W_S]\bigr)^2
= \sum_{\varnothing \neq S \subseteq [d]} r^{|S|} \bigl(\mathbb{E}[g e_S]\bigr)^2
\le r \sum_{\varnothing \neq S \subseteq [d]} \bigl(\mathbb{E}[g e_S]\bigr)^2
\le r\,\mathbb{E}[g^2]
= r\,\chi^2(Q \Vert P_Y).
$$
Taking $g = \epsilon W_1$ attains equality, proving $\eta_{\chi^2}(P_Y, K) = r$. $\square$

**Corollary 2 (Single-round public-coin bound).** *If $U \perp (V, Y)$ and $M \in \{0, 1\}^B$ is generated from $(Y, U)$, then $V - (Y, U) - M$ and applying Theorem 1 to $Q = P_{Y \mid M=m, U=u}$ and averaging over $(m, u)$ gives*
$$
I(V; M \mid U) \le r(s)\, I(Y; M \mid U) \le r(s)\, H(M \mid U) \le r(s)\, B \le s\, B \qquad (\text{in bits}).
$$

**Scope remark (multi-round adaptive prefixes, Section D.5).** Theorem 1 settles Conjecture `conj:singleround` completely ($T = 1$, sharp constant $C = 1$, all $s \ge 0$). As noted in Section D.5 of arXiv:2606.00703, extending this to $T > 1$ interactive rounds remains a separate step: conditioning on a transcript prefix $M_{1:t-1}$ can induce correlations among the coordinates of $V$, destroying the product form of $P_V$ required by the tensorization step above. (In the distributed-DPI literature, Braverman et al. (2016) handle interactive rounds via a truncated-Gaussian scalar SDPI coupled with an information-cost induction; the proof above handles the untruncated Gaussian mixture directly and determines the exact single-round constant.)

---

## 4. Verification and References

Running `.venv/bin/python -B solutions/bit_constrained_kl/explore.py` (Python 3, NumPy, SciPy) evaluates $r(s)$ by adaptive quadrature and cross-checks it against a 256-node Gauss–Hermite rule, 567 scalar exponential-tilt KL ratios across 7 positive SNRs, 28 exact 6-bin product singular-value checks ($d = 1, 2, 3, 4$), and 560 joint KL checks. All assertions pass.

| Per-coordinate SNR $s$ | Exact $\eta_{\mathrm{KL}} = \eta_{\chi^2} = r(s)$ (for every $d \ge 1$) | Ratio $r(s)/s$ |
|---:|---:|---:|
| $0$ | $0$ | — |
| $10^{-6}$ | $0.000000999999$ | $0.999999000002$ |
| $0.01$ | $0.009901624784$ | $0.990162478417$ |
| $0.1$ | $0.091340601205$ | $0.913406012049$ |
| $0.5$ | $0.350113404675$ | $0.700226809350$ |
| $1.0$ | $0.550400490793$ | $0.550400490793$ |

- **References:**
  - Y. Polyanskiy and Y. Wu (2017), *Strong data-processing inequalities for channels and Bayesian networks*, in *Convexity and Concentration*, IMA Vol. Math. Appl. 161, Springer, pp. 211–249.
  - Y. Polyanskiy and Y. Wu (2025), *Information Theory: From Coding to Learning*, Cambridge University Press.
  - M. Braverman, A. Garg, T. Ma, H. L. Nguyen, and D. P. Woodruff (2016), *Communication lower bounds for statistical estimation problems via a distributed data processing inequality*, STOC 2016, pp. 1011–1020.
