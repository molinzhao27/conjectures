# Characterization of Neyman–Pearson Optimality for Fisher–Rao Neutral Zones

**Target paper:** Soumojit Das, Nairanjana Dasgupta, and Prashanta Dutta, *Geometric Calibration and Neutral Zones for Uncertainty-Aware Multi-Class Classification*, [arXiv:2511.20960](https://arxiv.org/abs/2511.20960) (v2, November 2025), Section 3, Conjecture 1 (`conj:np`).  
**Companion files:** [`solutions/fisher_rao_neutral/explore.py`](explore.py), [`solutions/fisher_rao_neutral/proof.pdf`](proof.pdf).

---

## Verdict

- **Unqualified Conjecture `conj:np`: FALSE** as stated without a population calibration hypothesis. Even when the predicted class is Bayes-optimal at every input, the posterior correctness odds can strictly *decrease* with $R$, and fitting the paper's affine ALR recalibration map does not enforce the required population calibration identity.
- **Under perfect calibration: TRUE (via reduction to classical MSP thresholding).** Substituting the predicted simplex vertex $e_J$ into the Fisher–Rao metric yields $R(P) = \exp(-2\lambda \arccos\sqrt{S})$, a strictly increasing bijection of the maximum class probability (MSP) $S = \max_j P_j$. Thus Fisher–Rao geometry adds no decision-theoretic information beyond reparameterizing $S$, and under perfect calibration ($\Pr(Y = j \mid P) = P_j$), the strict odds ordering (on the extended half-line $[0, \infty]$) and Neyman–Pearson / reject-option optimality of thresholding $R$ reduce directly to classical MSP thresholding (Chow, 1970; Neyman–Pearson Lemma, Lehmann & Romano, 2005; El-Yaniv & Wiener, 2010; Franc, Pruša, & Voráček, 2023).
- **Exact necessary-and-sufficient criteria (Theorems 1–2):** Across all three observation levels—score-only ($Z = R$), full probability vector ($Z = P$), and full input/side information ($Z = X$)—randomized upper-threshold rules on $R$ are Neyman–Pearson optimal at *every* size (and equivalently optimal at *every* fixed deferral or coverage budget) **if and only if** $\Pr(C = 1 \mid Z) = g(S)$ almost surely for a nondecreasing function $g$. Four minimal exact-rational counterexamples strictly separate these regimes and also show that conditioning on the *true* class $Y = y$ can reverse the ordering even for true posteriors.

---

## 1. Reduction of the Fisher–Rao Score to Maximum Class Probability

Let $P = f(X) \in \Delta^{c-1}$ ($c \ge 2$) be the classifier's reported probability vector, let $J = J(P) \in \arg\max_j P_j$ be the predicted class under a fixed measurable tie-breaking rule, let $S := P_J = \max_{1 \le j \le c} P_j \in [1/c, 1]$, and let $C := \mathbf{1}_{\{Y = J\}}$ indicate prediction correctness. All conditional identities below hold almost surely; pointwise comparisons of conditional probabilities refer to a regular version on a set of full score measure.

**Proposition 1 (Exact bijection and calibration odds).** *For $c \ge 2$ and $\lambda > 0$, the Fisher–Rao reliability score $R := \exp\bigl(-\lambda\, d_{\mathrm{FR}}(P, e_J)\bigr)$ of [arXiv:2511.20960](https://arxiv.org/abs/2511.20960) satisfies*
$$
d_{\mathrm{FR}}(P, e_J) = 2\arccos\sqrt{S}, \qquad
R = h(S) := \exp\bigl(-2\lambda\arccos\sqrt{S}\bigr), \qquad
s(r) := h^{-1}(r) = \cos^2\!\left(\frac{-\log r}{2\lambda}\right).
\tag{1}
$$
*Here $h : [1/c, 1] \to [h(1/c), 1]$ is a strictly increasing bijection. Writing $\eta_j(p) := \Pr(Y = j \mid P = p)$, $u(p) := \eta_{J(p)}(p)$, and $m(s) := \mathbb{E}[u(P) \mid S = s]$,*
$$
\Pr(C = 1 \mid R = r) = m(s(r)).
\tag{2}
$$
*Under perfect calibration ($\eta_j(P) = P_j$ for all $j$), or more weakly under confidence calibration ($\mathbb{E}[C \mid S] = S$),*
$$
\Pr(C = 1 \mid R = r) = s(r), \qquad
\frac{\Pr(C = 1 \mid R = r)}{\Pr(C = 0 \mid R = r)}
= \frac{s(r)}{1 - s(r)}
= \cot^2\!\left(\frac{-\log r}{2\lambda}\right),
\tag{3}
$$
*which is strictly increasing on $[h(1/c), 1]$ in the extended half-line $[0, \infty]$ (attaining $+\infty$ at $r = 1$).*

*Proof.* Since $e_J$ has $1$ in entry $J$ and $0$ elsewhere, the Bhattacharyya coefficient is $\sum_j \sqrt{P_j (e_J)_j} = \sqrt{S}$, giving $d_{\mathrm{FR}}(P, e_J) = 2\arccos\sqrt{S}$ and (1). Strict monotonicity of $\sqrt{\cdot}$, $-\arccos(\cdot)$, and $\exp(\cdot)$ makes $h$ a strictly increasing bijection; thus $R$ and $S$ generate the exact same $\sigma$-algebra, rankings, and ties, and changing $\lambda > 0$ only relabels numeric thresholds. Conditioning $C$ on $P$ gives $u(P)$, and conditioning further on $S$ gives $m(S)$, yielding (2) by bijectivity of $h$. Under confidence calibration $m(s) = s$, substituting $s = s(r) = \cos^2(-\log r / (2\lambda))$ gives $s/(1-s) = \cos^2/\sin^2 = \cot^2$, which strictly increases on $[h(1/c), 1]$ in $[0, \infty]$. $\square$

*Two structural remarks:*
1. **No distribution-free relation exists:** Given *any* law of $P$ and *any* measurable function $v : \Delta^{c-1} \to [0, 1]$, assigning true label $J(p)$ with probability $v(p)$ and another label with probability $1 - v(p)$ realizes $u = v$ (and every measurable curve $m$ by taking $v = m \circ S$). Merely fitting the manuscript's affine ALR recalibration map does not impose $u(P) = S$ or $\mathbb{E}[C \mid S] = S$ in population.
2. **Sharper range bound for Theorem 2 of [arXiv:2511.20960](https://arxiv.org/abs/2511.20960):** Because $S \ge 1/c$, the true range of $R$ is $[h(1/c), 1] = [\exp(-2\lambda\arccos(1/\sqrt{c})), 1]$, strictly narrower than the diameter bound $[e^{-\lambda\pi}, 1]$ used in Theorem 2 of the target paper (e.g., for $c = 3$, $2\arccos(1/\sqrt{3}) \approx 1.911 < \pi$).

---

## 2. Exact Characterization of Neyman–Pearson Optimality

Let $Z$ denote the observation available to the decision rule, assuming $S$ (equivalently $R$) is measurable with respect to $Z$ (the three main cases are $Z = R$, $Z = P$, and $Z = X$). Let $U := \Pr(C = 1 \mid Z)$ and $\pi := \Pr(C = 1) \in (0, 1)$ so both hypotheses $H_0 : C = 0$ (error) and $H_1 : C = 1$ (correct) have positive probability. Denote by $Q$ the marginal law of $Z$ and by $Q_i$ its law given $C = i$ ($i \in \{0, 1\}$). By Bayes' rule,
$$
\frac{dQ_1}{dQ} = \frac{U}{\pi}, \qquad
\frac{dQ_0}{dQ} = \frac{1 - U}{1 - \pi}, \qquad
L := \frac{dQ_1 / dQ}{dQ_0 / dQ} = \frac{1 - \pi}{\pi}\,\frac{U}{1 - U} \in [0, \infty],
$$
where $L = +\infty$ on the singular event $\{U = 1\}$. A randomized test $\phi : Z \mapsto [0, 1]$ accepts the prediction (rejects $H_0$) with probability $\phi(Z)$, attaining size $\mathbb{E}_0[\phi] = \mathbb{E}[(1 - U)\phi]/(1 - \pi)$ (fraction of errors accepted) and power $\mathbb{E}_1[\phi] = \mathbb{E}[U\phi]/\pi$ (fraction of correct predictions accepted). A *score-threshold rule* has the form $\phi(Z) = \mathbf{1}_{\{S > \tau\}} + \gamma\,\mathbf{1}_{\{S = \tau\}}$ (including the constant rules $0$ and $1$) for $\tau \in [1/c, 1]$ and $\gamma \in [0, 1]$; by Proposition 1, these are identical to threshold rules on $R$.

**Theorem 1 (All-sizes Neyman–Pearson criterion).** *In the observation model $Z$ with $\pi \in (0, 1)$, score-threshold rules on $R$ (equivalently $S$) achieve maximum power among all $Z$-measurable tests at every size $\alpha \in [0, 1]$ **if and only if** there exists a nondecreasing function $g : [1/c, 1] \to [0, 1]$ such that*
$$
U = \Pr(C = 1 \mid Z) = g(S) \qquad \text{almost surely.}
\tag{4}
$$

*Proof.* Observe the exact linear identity for any $q \in (0, 1)$ and measurable $\phi : Z \mapsto [0, 1]$:
$$
\mathbb{E}\bigl[(U - q)\phi\bigr]
= (1 - q)\mathbb{E}[U\phi] - q\,\mathbb{E}[(1 - U)\phi]
= (1 - q)\pi\,\mathbb{E}_1[\phi] - q(1 - \pi)\,\mathbb{E}_0[\phi].
\tag{5}
$$
**Sufficiency ($\Leftarrow$).** Suppose $U = g(S)$ a.s. with $g$ nondecreasing. Let $\alpha \in [0, 1]$ and choose a score-threshold rule $\phi^*$ of exact size $\alpha$ via the $Q_0$-quantile of $S$ (taking $\phi^* = \mathbf{1}_{\{S > \tau\}}$ at $\alpha = 0$ if $\{U = 1\}$ has positive $Q_1$-mass and zero $Q_0$-mass, filling the entire $Q_0$-null upper tail $\{U = 1\}$ at zero cost; and taking $\phi^* \equiv 1$ at $\alpha = 1$). Let $\tau$ be the cutoff of $\phi^*$ and put $q := g(\tau)$. If $q = 1$, then $U = 1$ wherever $\phi^* > 0$, so $\alpha = 0$ and $\phi^*$ accepts the entire set $\{U = 1\}$ (every zero-size test vanishes $Q_0$-a.s., hence vanishes on $\{U < 1\}$, so $\phi^*$ is optimal). If $q = 0$, then $U = 0$ wherever $\phi^* < 1$, so $\phi^*$ attains power $1$, which is optimal. If $0 < q < 1$, monotonicity of $g$ implies $U \ge q$ on $\{\phi^* > 0\}$ and $U \le q$ on $\{\phi^* < 1\}$, so $\phi^*$ maximizes $\mathbb{E}[(U - q)\phi]$ pointwise over all $0 \le \phi \le 1$. By (5), any test $\phi$ with $\mathbb{E}_0[\phi] \le \mathbb{E}_0[\phi^*] = \alpha$ satisfies $\mathbb{E}_1[\phi] \le \mathbb{E}_1[\phi^*]$ (as $(1-q)\pi > 0$ and $q(1-\pi) > 0$). Thus $\phi^*$ is Neyman–Pearson optimal at size $\alpha$.

**Necessity ($\Rightarrow$).** Assume every size $\alpha \in [0, 1]$ admits an optimal score-threshold rule. Since $U$ has at most countably many atoms, the set $D := \{q \in (0, 1) : \Pr(U = q) = 0\}$ is dense in $(0, 1)$ (choose a countable dense subset $D_0 \subseteq D$). For each $q \in D_0$, the rule $\phi_q := \mathbf{1}_{\{U > q\}}$ has strict pointwise sign $(U - q)\phi_q > (U - q)\phi$ wherever $\phi \neq \phi_q$ and $U \neq q$; since $\Pr(U = q) = 0$, $\phi_q$ is the $Q$-a.s. *unique* maximizer of $\mathbb{E}[(U - q)\phi]$. Let $\psi_q(S)$ be an optimal score-threshold rule at size $\alpha_q := \mathbb{E}_0[\phi_q]$. Because $\mathbb{E}_1[\psi_q] \ge \mathbb{E}_1[\phi_q]$ and $\mathbb{E}_0[\psi_q] \le \mathbb{E}_0[\phi_q]$, identity (5) implies $\mathbb{E}[(U - q)\psi_q] \ge \mathbb{E}[(U - q)\phi_q]$, forcing $\psi_q(S) = \phi_q$ $Q$-a.s. Define
$$
g(s) := \sup\bigl(\{0\} \cup \{q \in D_0 : \psi_q(s) > 1/2\}\bigr) \qquad (s \in [1/c, 1]).
$$
Because each score-threshold function $\psi_q$ is nondecreasing in $s$, $g$ is nondecreasing. On the full-measure event $\bigcap_{q \in D_0} \{\psi_q(S) = \mathbf{1}_{\{U > q\}}\}$, we have $\psi_q(S) > 1/2 \iff U > q$ for every $q \in D_0$, and density of $D_0$ in $(0, 1)$ gives $g(S) = U$ almost surely. $\square$

**Corollary 1 (The three information levels).**
1. **Score-only experiment ($Z = R$):** The strict posterior-odds ordering of `conj:np` holds iff $m$ is strictly increasing on a full-measure set of confidence values; weak MLR holds iff $m$ is nondecreasing there, which by Theorem 1 is necessary and sufficient for score-threshold optimality at every size. (Flat odds permit optimal thresholds without the strict inequality.)
2. **Full probability vector ($Z = P$):** Score thresholds are NP-optimal among all vector-based tests at every size iff $u(P) = g(S)$ a.s. for a nondecreasing $g$—equivalently, $m$ is nondecreasing and $C \perp\!\!\!\perp P \mid S$ (a binary variable's conditional law is fixed by its conditional mean). Under perfect calibration ($u(P) = S$), this holds identically, recovering classical MSP optimality (Chow, 1970; Lehmann & Romano, 2005).
3. **Full input / side information ($Z = X$):** Under perfect vector calibration, score thresholds are optimal among all $X$-measurable tests at every size iff $\Pr(C = 1 \mid X) = S$ a.s. (conditioning $g(S)$ on $P$ forces $g(S) = S$), which holds in particular when $P$ is the true posterior $\Pr(Y = \cdot \mid X)$.

---

## 3. Deferral Budgets and Selective-Risk Neutral Zones

Besides NP test size $\mathbb{E}_0[\phi] = \alpha$, [arXiv:2511.20960](https://arxiv.org/abs/2511.20960) considers both (i) a cap $b \in [0, 1]$ on total deferral $\mathbb{E}[D] \le b$ maximizing captured error mass $\mathbb{E}[(1 - U)D]$, and (ii) a selective-risk cap $\Pr(C = 0 \mid \text{accept}) \le \alpha$ maximizing coverage $a = \mathbb{E}[A]$ (where $A = 1 - D$ is the acceptance probability). As established in the selective-classification literature (Chow, 1970; El-Yaniv & Wiener, 2010; Franc, Pruša, & Voráček, 2023), all three formulations share the same optimal ranking by $U$:

**Theorem 2 (All-budget deferral and coverage criterion).** *In the observation model $Z$, randomized score-threshold rules on $R$ maximize captured error mass at every deferral budget $b \in [0, 1]$ (equivalently, minimize selective error at every coverage $a \in (0, 1]$) **if and only if** $U = g(S)$ almost surely for a nondecreasing function $g$.*

*Proof.* For sufficiency, at any cutoff $k \in [0, 1]$, deferring where $1 - U > k$ maximizes $\mathbb{E}[(1 - U - k)D]$ over $0 \le D \le 1$, and accepting where $U > q$ maximizes $\mathbb{E}[(U - q)A]$ over $0 \le A \le 1$; when $U = g(S)$ is nondecreasing, lower-score deferral (upper-score acceptance) with boundary randomization attains every budget $b$ (coverage $a$) optimally. For maximal coverage under selective risk $\le \alpha$, the risk at coverage $a > 0$ is $1 - \mathbb{E}[U A]/a$, and the maximal captured correctness mass is $1$-Lipschitz (hence continuous) in $a$, so the supremum of feasible coverages is attained by an upper-threshold rule whenever positive coverage is feasible. (Under perfect calibration, a deterministic cutoff $S \ge 1 - \alpha$ guarantees risk $\mathbb{E}[1 - S \mid S \ge 1 - \alpha] \le \alpha$ when nonempty, though an optimal region can include lower scores while keeping average risk $\le \alpha$.)

For necessity, take any atom-free $q \in D_0$ and fix coverage $a_q := \Pr(U > q)$. On the fixed-coverage set $\{\mathbb{E}[A] = a_q\}$, maximizing $\mathbb{E}[U A]$ is identical to maximizing $\mathbb{E}[(U - q)A]$, whose unique maximizer is $\phi_q = \mathbf{1}_{\{U > q\}}$. Hence an optimal score-threshold rule $\psi_q(S)$ at coverage $a_q$ equals $\phi_q$ a.s., and the same supremum reconstruction as in Theorem 1 yields $U = g(S)$ a.s. with $g$ nondecreasing. Complementing $A = 1 - D$ gives the deferral statement. $\square$

---

## 4. Minimal Exact-Rational Counterexamples

All four examples specify the exact population law on finite state spaces (no sampling error):

1. **Reversed odds despite Bayes-optimal predicted labels.** Report binary vectors $(3/5, 2/5)$ and $(9/10, 1/10)$ with probability $1/2$ each, while the true class-$1$ probabilities are $9/10$ and $3/5$, respectively. Class $1$ is the Bayes-optimal label at every state, yet the correctness odds *fall* from $9$ to $3/2$ as $S$ increases from $3/5$ to $9/10$. At deferral budget $b = 1/2$, low-score deferral captures $1/5$ of errors whereas high-score deferral captures $4/5$. In the NP test, accepting only the low-score state has size $1/5$ and power $3/5$, whereas the upper score threshold of size $1/5$ accepts $1/4$ of the high-score state and attains power only $1/10$. (Conversely, true class-$1$ probabilities $7/10$ and $4/5$ yield strict MLR and full-vector optimality despite miscalibration; $7/10$ and $7/10$ yield optimal thresholds with flat odds.)
2. **Exact confidence calibration + strict score MLR still loses to the full vector.** Report $(3/5, 2/5)$, $(2/5, 3/5)$, and $(4/5, 1/5)$ with masses $1/4, 1/4, 1/2$ and conditional correctness probabilities $3/10, 9/10, 4/5$. Then $m(3/5) = \frac{1}{2}(3/10 + 9/10) = 3/5$ and $m(4/5) = 4/5$, so the classifier is *exactly* confidence-calibrated and satisfies strict score MLR. Yet accepting only the second vector $(2/5, 3/5)$ achieves correct mass $9/40$ at error mass $1/40$, while the upper score threshold at error mass $1/40$ accepts $1/4$ of the third state for correct mass only $1/10 < 9/40$.
3. **Side information $X$ beats a perfectly calibrated score.** Always report $(3/4, 1/4)$, while the input $X$ reveals two equally likely subpopulations with class-$1$ probabilities $1$ and $1/2$. The reported vector is perfectly calibrated, yet accepting only the first subpopulation achieves NP size $0$ and power $2/3$, whereas every score-only test of size $0$ has power $0$.
4. **Predicted-class vs. true-class conditioning.** Under perfect calibration, conditioning on the *predicted* class preserves $\Pr(C = 1 \mid S, J) = S$ by the tower property (even when predicted classes have different average errors). Conditioning on the *true* class $Y = y$, however, satisfies
   $$
   \Pr(C = 1 \mid S, Y = y) = \frac{\mathbb{E}[\mathbf{1}_{\{J = y\}} P_y \mid S]}{\mathbb{E}[P_y \mid S]}
   $$
   whenever the denominator is positive, which can reverse even for true posteriors. Report $(3/5, 2/5)$ and $(1/10, 9/10)$ with equal probability $1/2$ and draw $Y \sim P$. Within $Y = 1$, both scores occur with positive probability, yet $\Pr(C = 1 \mid S = 3/5, Y = 1) = 1 > 0 = \Pr(C = 1 \mid S = 9/10, Y = 1)$. The class-conditional Bayes errors are $1/7$ for class $1$ and $4/13$ for class $2$, while pooled correctness still increases from $3/5$ to $9/10$.

---

## 5. Reproducible Exact Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/fisher_rao_neutral/explore.py
```

The script [`explore.py`](explore.py) uses Python's standard library (`fractions.Fraction` for exact rational frontier comparisons, floating-point only for the trigonometric bijection check):
- **441 geometric checks:** Verifies strict monotonicity and exact inversion $s(h(s)) = s$ across $c \in \{2, \dots, 8\}$, $\lambda \in \{0.1, 0.75, 2.0\}$, and 21 confidence grid points each.
- **62 exact rational finite experiments under both budgets:** Over a 3-state model with tied score levels, exhaustively verifies that score-threshold and full-vector frontiers match iff $a = b \le c$ and score-threshold and pooled score-only frontiers match iff $(a+b)/2 \le c$, for both NP error-cost and coverage-cost budgets.
- **All four explicit counterexamples:** Verifies every exact fraction reported in Section 4.

All assertions pass with exit status `0`.

---

## References

- C. K. Chow, *On Optimum Recognition Error and Reject Tradeoff*, IEEE Transactions on Information Theory 16(1):41–46, 1970.
- S. Das, N. Dasgupta, and P. Dutta, *Geometric Calibration and Neutral Zones for Uncertainty-Aware Multi-Class Classification*, [arXiv:2511.20960](https://arxiv.org/abs/2511.20960) (v2), November 2025.
- R. El-Yaniv and Y. Wiener, *On the Foundations of Noise-Free Selective Classification*, Journal of Machine Learning Research 11:1605–1641, 2010.
- V. Franc, D. Pruša, and V. Voráček, *Optimal Strategies for Reject Option Classifiers*, Journal of Machine Learning Research 24(11):1–49, 2023.
- E. L. Lehmann and J. P. Romano, *Testing Statistical Hypotheses*, 3rd ed., Springer, 2005.
