# Refutation and Exact Characterization of the Scale Half of the Non-Additive Kernel Identifiability Dichotomy

**Target paper:** Jiarui Qi, *Shape without scale: an identifiability dichotomy for a bounded tail observed through a non-additive measurement kernel*, [arXiv:2609.27384](https://arxiv.org/abs/2609.27384) (September 2026), Section 4.3, Conjecture 4.8 (`conj:dichotomy`).  
**Companion files:** [`solutions/nonadd_kernel_ident/explore.py`](explore.py), [`solutions/nonadd_kernel_ident/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 4.8 (`conj:dichotomy`, the scale half of the sharpened dichotomy): FALSE** in its necessity direction.
  - Sufficiency of Anchor C (`(C-a) + (C-b)`) holds as stated in the paper, but the fixed-scale injectivity clause `(C-b)` is **strictly stronger than necessary**—even if injectivity is tested only on regular latent laws inside a single nonempty fiber $\mathcal{M}_{\mathcal{K}}(\ell_0)$.
- **Automatic identification of $(\alpha, L)$ under `(C-a)` (Proposition 1):** On every nonempty candidate fiber $\mathcal{M}_{\mathcal{K}}(\ell_0)$, clause `(C-a)` alone already identifies both the shape index $\alpha$ and the tail scale $L$ without requiring `(C-b)` or any leading edge-profile constant.
- **Explicit scaled-uniform counterexample (Theorem 2):** We construct a one-parameter candidate family $\{K_\theta : \theta \in [1, 2]\}$ with a single uniform jitter, strictly positive bounded widths, and no escape to infinity that satisfies `(C-a)` and identifies $(\alpha, L, p_\tau)$ on **every** fiber $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$, yet fails `(C-b)` on every fiber—including on a pair of regular members $(F_1, K_\theta), (F_2, K_\theta) \in \mathcal{M}_{\mathcal{K}}(\ell_\theta)$ sharing $(\alpha, L, p_\tau)$ and inducing the exact same observed mixture $K_\theta F_1 = K_\theta F_2$. (A companion family with width $w_0(s) = 4 - \min(s, 3)$ satisfies `(C-b)`.)
- **Exact necessary and sufficient replacement for `(C-b)` (Theorem 3):** Under `(C-a)`, $(L, p_\tau)$ is identifiable on every fiber $\mathcal{M}_{\mathcal{K}}(\ell_0)$ if and only if for every parameter $\theta$ with nonempty fiber, every finite signed measure $\nu \in \mathcal{V} \cap \ker K_\theta$ supported on $[0, \bar s_{\mathcal{K}}]$ with $\nu(\{0\}) = 0$ and vanishing density on $(0, \bar s_0)$ satisfies $\nu([0, \tau]) = 0$.

---

## 1. Setup, Candidate Families, and Forward Direction

We follow Definitions `def:M`, `def:psi`, Assumption `ass:C` (Anchor C), and Equation `def:fiber` of arXiv:2609.27384, keeping the paper's notation $\bar s_0$ for the shared near-zero density floor and $\bar s_{\mathcal{K}}$ for the candidate family's finite severity cap (avoiding collision with the Euler Beta function $B(\cdot, \cdot)$).

A member $(F_S, K)$ consists of a Borel probability law $F_S$ on $[0, \infty)$ and a Borel Markov kernel $K(\cdot \mid s)$ observed only through the marginal mixture $Q = K F_S := \int K(\cdot \mid s)\, F_S(ds)$. Write $\ell(s)$ and $u(s)$ for the finite lower and upper support endpoints of $K(\cdot \mid s)$ and $w(s) := u(s) - \ell(s)$ for its width. In the regularity class $\mathcal{M}_{\mathrm{reg}}$ with shared class constants $(\bar s_0, s_A, s_R, \tau, \beta, c_-, c_+, \bar b)$ satisfying $0 < \tau < s_R \le \bar s_0$:
- **A1:** $F_S$ has lower endpoint $0$, $F_S(\{0\}) = 0$, and a density version on $(0, \bar s_0]$ with $f_S(s) = L s^\alpha(1 + o(1))$ as $s \downarrow 0$, for member-specific $L > 0$ and $\alpha \ge 0$. The target estimand is $\psi = (\alpha, L, p_\tau)$ with $p_\tau := F_S([0, \tau])$.
- **A2–A4, NA:** $0 < w(s) < \infty$ everywhere with no escape to infinity on $[0, s_A]$ (A2); midpoint bias $m(s) := (\ell(s) + u(s))/2 - s$ satisfies $|m(s)| \le \bar b$ and $m \not\equiv 0$ (A3); $w(s)/s \to \infty$ as $s \downarrow 0$ (A4); and $K(\cdot \mid s)$ does not agree $F_S$-a.e. with a single additive error law (NA).
- **A5–A6:** $\ell$ is nondecreasing on $[0, \infty)$ with secant slopes in $[c_-, c_+]$ on $[0, s_R]$ (A5); and $K([\ell(s), \ell(s) + h] \mid s) = c_K(s) h^{\beta+1}(1 + o(1))$ uniformly on $s \in (0, s_R]$ as $h \downarrow 0$, with $c_K(s)$ bounded above and away from zero near $0$ (A6).

A *candidate family* $\mathcal{K} = \{K_\theta : \theta \in \Theta\}$ ($\Theta \subseteq \mathbb{R}^k$) carries strictly increasing lower edge maps $\ell_{K_\theta}$ on $[0, \infty)$, a finite severity cap $\bar s_{\mathcal{K}} < \infty$, and clause **(C-a)**: all non-$\theta$ structure and the observed edge $\ell_0 = \ell_{K_\theta}$ are known, and $\ell_{K_\theta}$ uniquely determines $\theta$. Each admissible edge $\ell_0$ defines the fiber
$$
\mathcal{M}_{\mathcal{K}}(\ell_0) := \bigl\{(F_S, K) \in \mathcal{M}_{\mathrm{reg}} : K \in \mathcal{K},\; \ell_K = \ell_0,\; \operatorname{supp} F_S \subseteq [0, \bar s_{\mathcal{K}}]\bigr\}.
$$
Clause **(C-b)** asserts that for every $\theta \in \Theta$, the mixture map $F \mapsto K_\theta F$ is injective on *all* Borel probability measures supported on $[0, \bar s_{\mathcal{K}}]$. Conjecture 4.8 (`conj:dichotomy`) proposes that for every candidate family $\mathcal{K}$, $(L, p_\tau)$ is identifiable on every fiber $\mathcal{M}_{\mathcal{K}}(\ell_0)$ if and only if $\mathcal{K}$ satisfies `(C-b)`.

The *sufficiency* direction (`(C-b)` $\Rightarrow$ fiberwise identifiability of $(L, p_\tau)$) is immediate: under `(C-a)`, two members of the same fiber share the same kernel $K_\theta$; under `(C-b)`, $K_\theta F = K_\theta F'$ forces $F = F'$, hence equal $p_\tau$ and equal $(\alpha, L)$ (via $F([0, t]) \sim L t^{\alpha+1}/(\alpha+1)$ as $t \downarrow 0$).

---

## 2. Automatic Identification of $(\alpha, L)$ Under `(C-a)`

**Proposition 1.** *Let $\mathcal{K}$ be any candidate family (satisfying `(C-a)` only) and let $\mathcal{M}_{\mathcal{K}}(\ell_0)$ be a nonempty fiber with common kernel $K$. If $(F, K), (F', K) \in \mathcal{M}_{\mathcal{K}}(\ell_0)$ satisfy $KF = KF'$, then their A1 parameters coincide: $(\alpha, L) = (\alpha', L')$.*

*Proof.* Suppose for contradiction that $(\alpha, L) \neq (\alpha', L')$. If $\alpha \neq \alpha'$, then $f(s)/f'(s) \to 0$ or $\infty$ as $s \downarrow 0$; if $\alpha = \alpha'$ and $L \neq L'$, then $f(s)/f'(s) \to L/L' \neq 1$. Swapping $F$ and $F'$ if needed, there exists $d \in (0, s_R)$ such that $f(s) > f'(s)$ for all $s \in (0, d)$.

Pick $t \in (0, d)$ and set $E := [0, \ell(t)]$. Since $\ell$ is strictly increasing on $[0, \infty)$ (standing structure of candidate families in Assumption `ass:C`):
- For every $s > t$, $\ell(s) > \ell(t)$, so $K(E \mid s) = 0$.
- For every $s \in (0, t)$, $\ell(s) < \ell(t)$, and every neighborhood of the finite lower support endpoint $\ell(s)$ has strictly positive mass, so $K(E \mid s) > 0$.

By A1, $F(\{0\}) = F'(\{0\}) = 0$, and both laws have densities on $(0, s_R]$ so $\{t\}$ has measure zero under both. Therefore
$$
(KF)(E) - (KF')(E) = \int_0^t K(E \mid s)\bigl(f(s) - f'(s)\bigr)\,ds > 0,
$$
contradicting $KF = KF'$. $\square$

---

## 3. Scaled-Uniform Counterexample to the Necessity of `(C-b)`

**Theorem 2 (Counterexample to Conjecture 4.8).** *Set shared class constants $\bar s_{\mathcal{K}} = 3$, $s_A = s_R = \bar s_0 = 1$, $\tau = 1/2$, $\beta = 0$, $c_- = c_+ = 1$, and $\bar b = 4$. Define the Borel width functions*
$$
w_0(s) := 4 - \min(s, 3), \qquad
w(s) := \begin{cases}
s - 2, & 2 < s \le 3,\\
w_0(s), & \text{otherwise},
\end{cases}
$$
*and for $\theta \in \Theta := [1, 2]$ let $K_\theta(\cdot \mid s) := \operatorname{Law}(\theta + s + w(s)U)$ with $U \sim \mathrm{Unif}[0, 1]$. Then:*
1. *$\mathcal{K} = \{K_\theta : \theta \in [1, 2]\}$ is a valid candidate family satisfying `(C-a)`, with every fiber $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$ nonempty and realizing non-constant $(L, p_\tau)$.*
2. *$(\alpha, L, p_\tau)$ is identifiable on every fiber $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$.*
3. *Clause `(C-b)` fails on every fiber $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$, even when restricted to regular members $(F_1, K_\theta), (F_2, K_\theta) \in \mathcal{M}_{\mathcal{K}}(\ell_\theta)$.*

*Proof.* **1. Admissibility and `(C-a)`.** The map $(s, u) \mapsto \theta + s + w(s)u$ is jointly Borel measurable, so $K_\theta$ is a Borel Markov kernel. (Note that $w$ has a single jump discontinuity at $s = 2$, strictly outside $[0, s_R] = [0, 1]$; neither $\mathcal{M}_{\mathrm{reg}}$ nor Assumption `ass:C` requires continuity of $w$, only Borel measurability and bounded positive widths.) Its lower edge $\ell_\theta(s) = \theta + s$ is strictly increasing on $[0, \infty)$ and uniquely determines $\theta = \ell_\theta(0)$, so `(C-a)` holds. For every $\theta \in [1, 2]$:
- **A2:** $0 < w(s) \le 4$ for all $s \ge 0$ (indeed $w(s) \in [1, 4]$ on $[0, 2] \cup (3, \infty)$ and $w(s) = s - 2 \in (0, 1]$ on $(2, 3]$), with no escape to infinity on $[0, 1]$.
- **A3:** The midpoint bias is $m_\theta(s) = \theta + w(s)/2 \in (1, 4]$, so $0 < |m_\theta(s)| \le 4 = \bar b$ everywhere.
- **A4:** For $s \in (0, 1]$, $w(s) = 4 - s \to 4$, so $w(s)/s \to \infty$ as $s \downarrow 0$.
- **A5:** $\ell_\theta(s) = \theta + s$ has secant slope $1 \in [c_-, c_+]$ everywhere.
- **A6:** For $s \in (0, 1]$ and $h \in (0, 3]$, $K_\theta([\ell_\theta(s), \ell_\theta(s) + h] \mid s) = h/(4 - s)$ identically, giving exact uniform A6 with $\beta = 0$ and $c_K(s) = 1/(4 - s) \in [1/4, 1/3]$ (strictly bounded above and away from zero).
- **NA:** For every A1 law $F$, $F$ has a strictly positive density on an interval $(0, \delta) \subseteq (0, 1)$ where $w(s) = 4 - s$ is strictly decreasing, whereas any fixed additive error law has constant conditional width. Hence NA holds for every A1 law paired with $K_\theta$.

**2. Fiberwise identifiability of $(\alpha, L, p_\tau)$.** Fix $\theta \in [1, 2]$ and $(F, K_\theta) \in \mathcal{M}_{\mathcal{K}}(\ell_\theta)$. For any $x = \theta + t$ with $t \in (0, 1)$, only severities $s \le t$ satisfy $\ell_\theta(s) \le x$, and for every $s \in [0, t] \subset [0, 1)$ the upper endpoint is $\theta + s + (4 - s) = \theta + 4 > \theta + t$. By Tonelli's theorem, $Q = K_\theta F$ is absolutely continuous on $(\theta, \theta + 1)$ with right-continuous density version
$$
H(t) := \int_{[0, t]} \frac{F(ds)}{4 - s} \qquad (0 < t < 1).
$$
Right continuity determines $H$ uniquely on $(0, 1)$ from $Q$. Its Lebesgue–Stieltjes measure on $(0, 1)$ is $dH(s) = F(ds)/(4 - s)$, so $F(ds) = (4 - s)\,dH(s)$ on $(0, 1)$. Since A1 gives $F(\{0\}) = 0$, $Q$ uniquely determines $F|_{(0, 1)}$, hence $p_\tau = F([0, 1/2])$ and $(\alpha, L)$.

**3. Failure of `(C-b)` inside every fiber.** Set $s_n := 2 + 2^{1 - n} \in (2, 3]$ for $n \ge 1$ and define the probability measure $G := \sum_{n=1}^\infty 2^{-n} \delta_{s_n}$ supported on $[0, 3]$. At $s = 2$, $K_\theta(\cdot \mid 2) = \mathrm{Unif}[\theta + 2,\, \theta + 4]$, which has density $1/2$ on $(\theta + 2, \theta + 4)$. At $s_n$, $w(s_n) = 2^{1 - n}$, so $K_\theta(\cdot \mid s_n) = \mathrm{Unif}[\theta + 2 + 2^{1 - n},\, \theta + 2 + 2^{2 - n}]$. These dyadic subintervals partition $\theta + (2, 4]$ up to endpoints, and weighting the $n$-th tile of length $2^{1 - n}$ by $2^{-n}$ gives uniform density $2^{-n}/2^{1 - n} = 1/2$ on every tile. Hence
$$
K_\theta G = K_\theta \delta_2 = \mathrm{Unif}[\theta + 2,\, \theta + 4],
$$
while $G(\{2\}) = 0 \neq 1 = \delta_2(\{2\})$.

To witness non-injectivity inside the regular fiber $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$ itself, let $D := \mathrm{Unif}[0, 1]$ and for any $r \in (0, 1)$ define
$$
F_1 := r D + (1 - r)\delta_2, \qquad F_2 := r D + (1 - r)G.
$$
Both are supported on $[0, 3]$ and have constant density $r$ on $(0, 1)$, so both satisfy A1 (with $\alpha = 0$, $L = r$, $p_\tau = r/2$) and therefore lie in $\mathcal{M}_{\mathcal{K}}(\ell_\theta)$. Yet $K_\theta F_1 = K_\theta F_2$ while $F_1 \neq F_2$. Varying $r \in (0, 1)$ also confirms that both $L$ and $p_\tau$ vary nontrivially across the fiber. $\square$

**Remark (Injective comparison family).** Replacing $w$ by $w_0(s) = 4 - \min(s, 3)$ everywhere yields a companion candidate family $\{K_\theta^0 : \theta \in [1, 2]\}$ satisfying the exact same regularity axioms and constants. On the entire severity cap $[0, 3]$, every conditional support ends at $\theta + 4$, so $K_\theta^0 F$ has right-continuous density $H(t) = \int_{[0, \min(t, 3)]} (4 - s)^{-1} F(ds)$ on $[0, 4)$ (and $0$ for $t < 0$). Then $(4 - s)\,dH(s) = F(ds)$ on $[0, 3]$—including possible atoms at $0$ and $3$ via the jumps of $H$—so $\{K_\theta^0\}$ satisfies `(C-b)`.

---

## 4. Exact Necessary and Sufficient Replacement for `(C-b)`

For any candidate family $\mathcal{K} = \{K_\theta : \theta \in \Theta\}$ with severity cap $\bar s_{\mathcal{K}}$ and shared density floor $\bar s_0$, let $\mathcal{V}$ be the linear space of finite signed Borel measures $\nu$ supported on $[0, \bar s_{\mathcal{K}}]$ such that
$$
\nu(\{0\}) = 0, \qquad
\nu|_{(0, \bar s_0)} = g(s)\,ds, \qquad
\lim_{s \downarrow 0} g(s) = 0
$$
for some density version $g$. Extend $K_\theta$ linearly to signed measures via $(K_\theta \nu)(A) := \int K_\theta(A \mid s)\,\nu(ds)$.

**Theorem 3 (Exact fiberwise identifiability characterization).** *Let $\mathcal{K} = \{K_\theta : \theta \in \Theta\}$ be a candidate family (satisfying `(C-a)`). Then $(L, p_\tau)$ is identifiable on every fiber $\mathcal{M}_{\mathcal{K}}(\ell_0)$ if and only if for every $\theta \in \Theta$ whose fiber $\mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$ is nonempty,*
$$
\nu \in \mathcal{V} \quad \text{and} \quad K_\theta \nu = 0 \quad \Longrightarrow \quad \nu([0, \tau]) = 0.
\tag{1}
$$

*Proof.* **Sufficiency.** Assume (1), and let $(F, K_\theta), (F', K_\theta) \in \mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$ satisfy $K_\theta F = K_\theta F'$. By Proposition 1, $F$ and $F'$ share the same A1 parameters $(\alpha, L)$ with $\alpha \ge 0$. Set $\nu := F - F'$. Then $\nu$ is supported on $[0, \bar s_{\mathcal{K}}]$, $\nu(\{0\}) = 0$, $K_\theta \nu = 0$, and on $(0, \bar s_0)$ its density is $f(s) - f'(s) = o(s^\alpha) = o(1)$ as $s \downarrow 0$ since $\alpha \ge 0$. Thus $\nu \in \mathcal{V} \cap \ker K_\theta$, so (1) yields $F([0, \tau]) - F'([0, \tau]) = \nu([0, \tau]) = 0$.

**Necessity.** Suppose there exist $\theta \in \Theta$ with nonempty fiber $\mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$ and a signed measure $\nu \in \mathcal{V}$ such that $K_\theta \nu = 0$ and $\nu([0, \tau]) \neq 0$. Because $K_\theta(\mathbb{R} \mid s) = 1$ for all $s$, evaluating $(K_\theta \nu)(\mathbb{R}) = 0$ yields $\nu([0, \bar s_{\mathcal{K}}]) = 0$. Writing the Jordan decomposition $\nu = \nu^+ - \nu^-$, both parts are supported on $[0, \bar s_{\mathcal{K}}]$ with equal positive total mass $m := \nu^+([0, \bar s_{\mathcal{K}}]) = \nu^-([0, \bar s_{\mathcal{K}}]) > 0$ (since $\nu \neq 0$). Moreover $\nu^\pm(\{0\}) = 0$, and on $(0, \bar s_0)$ their densities $g^+(s) = \max(g(s), 0)$ and $g^-(s) = \max(-g(s), 0)$ satisfy $\lim_{s \downarrow 0} g^\pm(s) = 0$.

Fix a member $(F_*, K_\theta) \in \mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$. Nonemptiness implies $\bar s_{\mathcal{K}} > 0$ (since $F_*$ has lower endpoint $0$ and no atom at $0$). Choose $d \in (0, \min(\bar s_{\mathcal{K}}, \bar s_0))$, let $D := \mathrm{Unif}[0, d]$, and define the two probability measures
$$
F_+ := \frac{1}{3}\bigl(F_* + D + \nu^+ / m\bigr), \qquad
F_- := \frac{1}{3}\bigl(F_* + D + \nu^- / m\bigr).
$$
Both $F_+$ and $F_-$ are supported on $[0, \bar s_{\mathcal{K}}]$, have no atom at $0$, and possess densities on $(0, \bar s_0]$. As $s \downarrow 0$, the density of $F_*$ approaches a finite nonnegative limit ($L_*$ if $\alpha_* = 0$, and $0$ if $\alpha_* > 0$), the uniform component $D$ contributes $1/d > 0$ on $(0, d)$, and $g^\pm(s)/m \to 0$. Hence both $F_+$ and $F_-$ satisfy A1 with exponent $\alpha = 0$ and positive scale $(L_* \mathbf{1}_{\{\alpha_* = 0\}} + 1/d)/3$. The kernel-only axioms A2–A6 hold because $(F_*, K_\theta) \in \mathcal{M}_{\mathrm{reg}}$. Finally, $F_* \ll F_\pm$, so any additive representation of $K_\theta$ holding $F_\pm$-almost everywhere would also hold $F_*$-almost everywhere, contradicting NA for $(F_*, K_\theta)$; hence both $(F_+, K_\theta)$ and $(F_-, K_\theta)$ lie in $\mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$. Since $F_+ - F_- = \nu/(3m)$, we have $K_\theta F_+ = K_\theta F_-$ while $F_+([0, \tau]) - F_-([0, \tau]) = \nu([0, \tau])/(3m) \neq 0$, so $p_\tau$ is not identifiable on $\mathcal{M}_{\mathcal{K}}(\ell_{K_\theta})$. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/nonadd_kernel_ident/explore.py
```

The script [`explore.py`](explore.py) performs exact rational arithmetic (`fractions.Fraction`) checks across $\theta \in \{1, 3/2, 2\}$:
- **1,578 exact mixture CDF comparisons** verifying $K_\theta G = K_\theta \delta_2$ (summing the full infinite dyadic tail analytically) at uniform grid points and at $80$ dyadic tile endpoints and midpoints.
- **20 exact local recovery checks** verifying $(4 - t)\,H(t) + \int_0^t H(s)\,ds = F([0, t])$ and density inversion $(4 - t)\,H'(t) = f(t)$ for polynomial-weighted latent densities.
- **Global atomic inversions** for the injective comparison family $w_0(s) = 4 - \min(s, 3)$, recovering atoms at $s \in \{0, 1/2, 2, 3\}$ exactly from jumps of $H$.

All assertions passed.

---

## References

- J. Qi, *Shape without scale: an identifiability dichotomy for a bounded tail observed through a non-additive measurement kernel*, [arXiv:2609.27384](https://arxiv.org/abs/2609.27384), September 2026.
- A. Meister, *Deconvolution Problems in Nonparametric Statistics*, Lecture Notes in Statistics 193, Springer, 2009.
- W. K. Newey and J. L. Powell, *Instrumental Variable Estimation of Nonparametric Models*, Econometrica 71(5):1565–1578, 2003.
