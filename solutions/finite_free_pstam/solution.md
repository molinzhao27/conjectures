# Cubic Phase Transition, Refutation of Subcritical Bifurcation, and All-Degree Symmetric-Score Strictness for the Finite Free $p$-Stam Inequality

**Target paper:** Baran Hashemi, *Spectral Structure in Finite Free Information Inequalities and $p$-Stam Phase Transitions*, [arXiv:2604.11922](https://arxiv.org/abs/2604.11922) (April 2026), Conjectures `conj:extremizer_transition`, `conj:phase`, `conj:pstam_holds`, `conj:strict` (with prior-art attribution for `conj:spectrum`, `rem:En_sym`, and `rem:uniqueness`).  
**Companion files:** [`solutions/finite_free_pstam/explore.py`](explore.py), [`solutions/finite_free_pstam/proof.pdf`](proof.pdf).

---

## Verdict and Novelty Demarcation

| Target in `arXiv:2604.11922` | Status | Contribution / Attribution |
|---|---|---|
| `conj:extremizer_transition` | **FALSE** (refuted in degree $n = 3$) | **New.** Unrestricted and fixed-total-variance infima are $0$ (unattained on simple roots); even on the fixed equal-variance slice, the matching Hermite pair uniquely minimizes both $g_p$ and $\rho_p$ for all $7/4 \le p \le 2$ (Theorem 2). Existence of non-Hermite variance-constrained minimizers for $n \ge 4$ at small $p$ remains open. |
| `conj:strict` (closed domain) | **FALSE** (literal boundary) | **New.** Under the continuous extension to multiple roots, $x^n \boxplus_n g = g$ yields $g_p(x^n, g) = 0$ for all $n \ge 3$, $p \in (1, 2)$. All positive strictness statements therefore apply to simple real-rooted polynomials. |
| `conj:phase`, `conj:pstam_holds`, `conj:strict` ($n = 3$, simple roots) | **PROVED** | **New.** Strict concavity of $U(a) = \|s(P_a)\|_p^{-2}$ on $(-1, 1)$ for $1 < p \le 2$ proves superadditivity of $\|s\|_p^{-2}$, hence strict $p$-Stam for all simple cubics when $1 < p < 2$ and uniqueness at $p = 2$ (Theorem 1). |
| `conj:pstam_holds`, `conj:strict` ($n \ge 3$, one symmetric score) | **PROVED** (partial, all degrees) | **New.** For every $n \ge 3$ and $1 < p < 2$, if $s_f$ admits a sign-reversing permutation involution $S s_f = -s_f$ (e.g., any reflection-symmetric input paired with an arbitrary simple partner), $F_p(f \boxplus_n g) \ge \eta^{-(2-p)/(p-1)} F_p(f) + F_p(g) > F_p(f) + F_p(g)$ (Theorem 3). Also proved uniformly on compact sets of simple pairs near $p = 1^+$ and (using `arXiv:2609.10367v1`) near $p = 2^-$ (Proposition 4). |
| `conj:pstam_holds`, `conj:phase`, `conj:strict` ($n \ge 4$, arbitrary asymmetric simple pairs) | **OPEN** | We prove an explicit structural obstruction: the stronger centered Jacobian $\ell^p$-contraction $\|J(u, v)\|_p^p \le \|u\|_p^p + \|v\|_p^p$ fails at simple pairs for $n = 64, p = 3/2$ (Proposition 5). |
| `conj:spectrum`, `rem:En_sym` (dyadic spectrum $\{2^{-k/2}\}_{k=1}^{n-1}$ and symmetry of $E_n$) | **PRIOR ART** | Previously proved in **Hashemi & Hyun**, *When Finite Free Curves Split* ([arXiv:2609.10367v1](https://arxiv.org/abs/2609.10367v1), Appendix A) and Machina Mathematica (`freeconv-spectrum`). A brief self-contained reproof is included in Section 6 for completeness. |
| `rem:uniqueness` (Hermite uniqueness at $p = 2$) | **PRIOR ART** | Main theorem of **Hashemi & Hyun** ([arXiv:2609.10367v1](https://arxiv.org/abs/2609.10367v1)), building on the $p = 2$ finite free Stam framework of **Garza-Vargas, Srivastava, and Stier** ([arXiv:2602.15822](https://arxiv.org/abs/2602.15822)). |
| Supercritical failure ($p > 2$) at the Hermite pair | **PRIOR ART** | Already proved in **Hashemi** ([arXiv:2604.11922](https://arxiv.org/abs/2604.11922), Proposition 5.5 and Lemma 5.6). |

---

## 1. Definitions and Boundary Domain

For a monic degree-$n$ polynomial $f(x) = \prod_{i=1}^n (x - a_i)$ with distinct ordered real roots $a_1 < \cdots < a_n$, define its score vector, $\ell^p$-Fisher information ($p > 1$, dual exponent $q := p / (p - 1)$), and inverse $p$-Fisher functional by
$$
s_i(a) := \sum_{j \neq i} \frac{1}{a_i - a_j} = \frac{f''(a_i)}{2 f'(a_i)}, \qquad
\Phi_p(f) := \sum_{i=1}^n |s_i(a)|^p, \qquad
F_p(f) := \Phi_p(f)^{-1/(p-1)} = \|s(a)\|_p^{-q}.
$$
Scores sum to zero, satisfy $s_n(a) > 0$, are translation-invariant, and satisfy $F_p(c_* f) = c^q F_p(f)$ under root dilation by $c > 0$. The finite free additive convolution $C = f \boxplus_n g$ averages $\prod_{i=1}^n (x - a_i - b_{\pi(i)})$ over $\pi \in S_n$; writing descending monic coefficients (signs included) as $f_i, g_j$, the coefficient of $x^{n-k}$ in $f \boxplus_n g$ is
$$
(f \boxplus_n g)_k = \sum_{i + j = k} \frac{(n - i)!\,(n - j)!}{n!\,(n - k)!}\, f_i g_j.
$$
The additive and scale-free $p$-Stam deficits are
$$
g_p(f, g) := F_p(f \boxplus_n g) - F_p(f) - F_p(g), \qquad
\rho_p(f, g) := \frac{g_p(f, g)}{F_p(f) + F_p(g)}.
$$
(These match `arXiv:2604.11922`, Definitions 2.2–2.3; multiplying $\Phi_p$ by a positive constant rescales $g_p$ by a positive constant and leaves $\rho_p$ unchanged.)

**Boundary failure of literal `conj:strict`.** Under the continuous extension to monic real-rooted polynomials with repeated roots, $F_p(f) = 0$ whenever $f$ has a multiple root: in any isolated merging cluster of $m \ge 2$ roots, the leftmost root's score tends to $-\infty$ because all internal terms are negative and external terms stay bounded. Because $x^n \boxplus_n g = g$, we obtain $g_p(x^n, g) = F_p(g) - 0 - F_p(g) = 0$ for every simple $g$, every $n \ge 3$, and every $p \in (1, 2)$. Thus `conj:strict` as literally quantified over the closed real-rooted domain is false; all positive strictness theorems below apply on the interior of simple real-rooted polynomials.

---

## 2. Complete Proof of the $p$-Stam Phase Transition and Strictness for Cubics ($n = 3$)

Every centered simple cubic is a positive dilation $d > 0$ of $P_a(x) := x^3 - 3x - 2a$ with shape parameter $a \in (-1, 1)$. Set
$$
U(a) := \|s(P_a)\|_p^{-2} \qquad (a \in (-1, 1)),
$$
extended continuously by $U(\pm 1) = 0$. Note that $F_p(d_* P_a) = d^q U(a)^{q/2}$.

**Theorem 1 (Strict cubic $p$-Stam and phase transition).** *For $n = 3$ and every $p \in (1, 2]$:*
1. *$U$ is even and strictly concave on $(-1, 1)$.*
2. *For all simple real-rooted cubics $f, g$, the stronger inverse-squared-norm superadditivity holds:*
   $$
   \|s(f \boxplus_3 g)\|_p^{-2} \ge \|s(f)\|_p^{-2} + \|s(g)\|_p^{-2}.
   \tag{1}
   $$
3. *Raising (1) to the power $q / 2 = p / (2p - 2) \ge 1$ proves $g_p(f, g) > 0$ for all simple cubics when $1 < p < 2$, and $g_2(f, g) \ge 0$ with equality if and only if both $f$ and $g$ are affine Hermite cubics ($a = b = 0$). Combined with the Hermite counterexample $g_p(H_3, H_3) = (2^{q/2} - 2) F_p(H_3) < 0$ for $p > 2$ (`arXiv:2604.11922`), this establishes `conj:phase`, `conj:pstam_holds`, and simple-root `conj:strict` in degree $n = 3$.*

*Proof.* Write $a = \cos t$ with $t \in (0, \pi)$. The roots of $P_a$ are $r_j = 2 \cos((t + 2\pi j)/3)$ ($j = 0, 1, 2$), and since $P_a'(r_j) = 3(r_j^2 - 1)$ and $P_a''(r_j) = 6 r_j$,
$$
s_j = \frac{r_j}{r_j^2 - 1} = \frac{\sin(2(t + 2\pi j)/3)}{\sin t}.
$$
Hence $w(t) := s \sin t$ traces a circle of constant radius $\sqrt{3/2}$ in the zero-sum plane $\mathbf{1}^\perp \subset \mathbb{R}^3$ with polar angle $2t/3$ in an oriented orthonormal basis. Negating $a$ reflects $t \mapsto \pi - t$ and negates the scores up to permutation, so $U$ is even.

Fix $a \in (-1, 1) \setminus \{0\}$, so all three scores $s_j$ are nonzero. Let $z \in \mathbb{R}^3$ be the unique $\ell^p$-norming functional of $w(t)$, so $z_j \propto \operatorname{sgn}(w_j) |w_j|^{p-1}$ and $\|z\|_q = 1$. Freezing $z$ while varying $a$ near the chosen point yields a smooth upper bound
$$
U(a) \le \frac{1 - a^2}{(z \cdot w(t))^2},
$$
with equality at the chosen $a$. Writing $\delta := 2t/3 - \phi$, where $\phi$ is the polar angle of the orthogonal projection of $z$ onto $\mathbf{1}^\perp$, this upper bound is a positive constant times $(1 - a^2) \sec^2 \delta$. Differentiating twice in $a$ (using $dt/da = -1/\sin t$, $d\delta/da = -\frac{2}{3\sin t}$) and dividing by $\sec^2 \delta$ gives the quadratic polynomial in $T := \tan \delta$:
$$
L(T) = -\frac{10}{9} + 4 \cot(t)\, T + \frac{8}{3}\, T^2.
$$

We show that the projected normal direction $\phi$ always lies between the radial direction $w(t)$ ($\delta = 0$) and the projected $\ell^1$ normal $\operatorname{sgn}(w(t))$, within an angular span $< \pi/2$. Permuting coordinates and flipping a common sign, write $w(t)$ proportional to $(-1, u, v)$ with $u + v = 1$ and $u \ge v > 0$. Setting $b := p - 1 \in (0, 1]$, $A := u^b$, $B := v^b$, the slope of the projected $\ell^p$ normal relative to the axis $(-2, 1, 1)$ in $\mathbf{1}^\perp$ is proportional (with the same positive constant for all $p$) to $(A - B)/(2 + A + B)$. At the $\ell^1$ endpoint ($b = 0$) this slope is $0$; at the radial direction ($b = 1$) it is $(u - v)/3$. Since $b \in (0, 1]$, $A + B = u^b + (1 - u)^b \ge 1$, and $u^b - (1 - u)^b$ is convex on $[1/2, 1]$ (its second derivative is $b(b - 1)(u^{b-2} - (1-u)^{b-2}) \ge 0$ for $u \ge 1/2$), so it lies below its chord $2u - 1 = u - v$ connecting $u = 1/2$ to $u = 1$. Thus
$$
0 \le \frac{A - B}{2 + A + B} \le \frac{u - v}{3}.
$$
Hence the projected $\ell^p$ normal lies in the closed angular wedge between the $\ell^1$ and radial normals (which spans $< \pi/6 < \pi/2$).

At the radial endpoint, $L(0) = -10/9 < 0$. At the $\ell^1$ endpoint, the signs of $s$ are locally constant, so the frozen bound equals $U_1(a) := \|s(P_a)\|_1^{-2}$ locally. For any three zero-sum numbers, $\|s\|_1 = 2 \max_j |s_j|$, attained at the unique coordinate with sign opposite to the other two; this belongs to an outer root $r$ with $|r| > 1$ (if the middle score is positive, the leftmost score dominates; if negative, the rightmost dominates). At that root, $|s| = |r| / (r^2 - 1)$ gives $4 U_1 = (r - r^{-1})^2$. Since $a = (r^3 - 3r)/2$ satisfies $da/dr = \frac{3}{2}(r^2 - 1)$,
$$
\frac{d(4 U_1)}{da} = \frac{4}{3}(r^{-1} + r^{-3}), \qquad
\frac{d^2(4 U_1)}{da^2} = -\frac{8(r^{-2} + 3 r^{-4})}{9(r^2 - 1)} < 0.
$$
Thus $L(T) < 0$ at both endpoints of the $T$-interval; since $L$ is convex in $T$, $L(T) < 0$ at the $\ell^p$ normal as well, proving $U''(a) < 0$ for all $a \in (-1, 1) \setminus \{0\}$. For $p > 1$, $a \mapsto \|s(P_a)\|_p^p$ is $C^1$ at $a = 0$ (where the middle score vanishes smoothly and $|x|^p$ is $C^1$), and evenness gives $U'(0) = 0$, so $U$ is strictly concave across all of $(-1, 1)$.

Now take two centered simple cubics with scales $d, e > 0$ and shapes $a, b \in (-1, 1)$. By the cubic coefficient rule for $\boxplus_3$, their convolution has scale $R := \sqrt{d^2 + e^2}$ and shape $c = A^3 a + B^3 b$, where $A := d/R$, $B := e/R$ satisfy $A^2 + B^2 = 1$. Using concavity of $U$ and the fact that $U$ achieves its strict maximum at $0$ (hence $U(Aa) \ge U(a)$ for $0 < A < 1$),
$$
U(c) = U\bigl(A^2(Aa) + B^2(Bb)\bigr) \ge A^2 U(Aa) + B^2 U(Bb) \ge A^2 U(a) + B^2 U(b).
$$
Multiplying by $R^2$ proves (1). Since $q/2 > 1$ for $1 < p < 2$ and both summands in (1) are strictly positive, raising to $q/2$ gives strict inequality $F_p(f \boxplus_3 g) > F_p(f) + F_p(g)$. At $p = 2$, equality in the chain forces $U(Aa) = U(a)$ and $U(Bb) = U(b)$, hence $a = b = 0$ by strict concavity. $\square$

---

## 3. Refutation of Subcritical Extremizer Bifurcation (`conj:extremizer_transition`) in Degree $n = 3$

**Theorem 2 (Cubic refutation of `conj:extremizer_transition`).** *In degree $n = 3$:*
1. *On the unrestricted simple-root domain, $\inf_{f,g} g_p(f, g) = 0$ for all $p \in (1, 2)$ and is not attained. Even under fixed total variance $d^2 + e^2 = \mathrm{const}$, taking two Hermite cubics ($a = b = 0$) with $e \to 0^+$ drives both $g_p \to 0$ and $\rho_p \to 0$ with pair mismatch $d_{PQ} \equiv 0$.*
2. *On the fixed equal-variance slice ($d = e = 1$ after common dilation), for every $p \in [7/4, 2]$ the matching Hermite pair $(a, b) = (0, 0)$ (equally spaced roots) is the **unique** global minimizer of both $g_p$ and $\rho_p$.*

*Proof.* Part (1) follows immediately from Theorem 1 and the scaling formula $g_p(d_* H_3, e_* H_3) = \bigl((d^2 + e^2)^{q/2} - d^q - e^q\bigr) F_p(H_3) \to 0$ as $e \to 0^+$ at fixed $d > 0$ (while the baseline $(d^q + e^q) F_p(H_3) \to d^q F_p(H_3) > 0$).

For part (2), set $d = e = 1$, so $R = \sqrt{2}$ and $c = r(a + b)/2$ with $r := 1/\sqrt{2}$. Put $M := U(0) > 0$, $K(x) := M - U(x) \ge 0$, and $k := q/2 = p / (2p - 2) \ge 1$. Because $K$ is convex, even, and satisfies $K(0) = 0$,
$$
K(c) = K\!\left(r \cdot \frac{a + b}{2} + (1 - r) \cdot 0\right) \le r\, K\!\left(\frac{a + b}{2}\right) \le \frac{r}{2}\bigl(K(a) + K(b)\bigr).
$$
Because $x \mapsto x^k$ is convex on $[0, M]$ ($k \ge 1$), the tangent inequality at $M$ gives $M^k - U(c)^k \le k M^{k-1} K(c)$, while the chord inequality from $0$ to $M$ gives $M^k - U(x)^k \ge M^{k-1} K(x)$ for $x \in \{a, b\}$. Writing $g_p(a, b) = 2^k U(c)^k - U(a)^k - U(b)^k$ and subtracting $g_p(0, 0) = (2^k - 2) M^k$, we obtain
$$
g_p(a, b) - g_p(0, 0)
= \bigl(M^k - U(a)^k\bigr) + \bigl(M^k - U(b)^k\bigr) - 2^k\bigl(M^k - U(c)^k\bigr)
\ge M^{k-1}\bigl(1 - 2^{k-1} k r\bigr)\bigl(K(a) + K(b)\bigr).
$$
For $7/4 \le p \le 2$, we have $1 \le k \le 7/6$, so
$$
2^{k-1} k r \le 2^{1/6} \cdot \frac{7}{6} \cdot 2^{-1/2} = \frac{7}{6} \cdot 2^{-1/3} < 1 \qquad (\text{since } 7^3 = 343 < 432 = 2 \cdot 6^3).
$$
Since $K(x) > 0$ for all $x \neq 0$, $(a, b) = (0, 0)$ uniquely minimizes $g_p(a, b)$ on the equal-variance slice. Moreover, the positive baseline $A_p(a, b) = U(a)^k + U(b)^k$ is uniquely *maximized* at $(0, 0)$, so $\rho_p(a, b) = g_p(a, b) / A_p(a, b)$ is likewise uniquely minimized at $(0, 0)$. $\square$

*Scope note.* Conjecture `conj:extremizer_transition` quantifies over all $n \ge 3$ and all $p \in (1, 2)$, and is therefore refuted as stated by Theorem 2. Whether non-Hermite variance-constrained minimizers exist in degrees $n \ge 4$ at small $p \in (1, 2)$ remains open.

---

## 4. All-Degree Subcritical Strictness with One Symmetric Score and Near Endpoints

Let $J = (A, B) \in \mathbb{R}^{n \times 2n}$ denote the full Jacobian of the ordered roots of $C = f \boxplus_n g$ with respect to the ordered simple roots $(a, b)$ of $(f, g)$. We first record the three root-map properties (versions of the estimates in Garza-Vargas, Srivastava, and Stier, `arXiv:2602.15822`, with the centering term kept explicit):
$$
A, B \text{ are entrywise positive doubly stochastic}, \qquad
A s_f = B s_g = s_C, \qquad
\|A u + B v\|_2^2 \le \|u\|_2^2 + \|v\|_2^2 + 2 n \bar{u} \bar{v},
\tag{2}
$$
where $\bar{u} = \frac{1}{n}\sum_i u_i$ and $\bar{v} = \frac{1}{n}\sum_i v_i$.

*Self-contained proof of (2).*
1. **Real-rootedness and simplicity:** The symmetric multiaffine polarization of a real-rooted degree-$n$ polynomial $g$ is nonzero on the upper half-plane $\mathbb{H}^n$: inductively split a variable $w$ of multiplicity $m$ into $z$ via $P \mapsto P + (z - w) P_w' / m$. For prior arguments in $\mathbb{H}$, all roots $c_j$ in $w$ lie in $\operatorname{Im} \le 0$; for $y = \operatorname{Im} w > 0$, each $(w - c_j)^{-1}$ lies in the closed disk $-\operatorname{Im} \zeta \ge y |\zeta|^2$, and so does their average $L = P_w'/(mP)$ (padded with zeros if $\deg_w P < m$), so any zero $z = w - 1/L$ satisfies $\operatorname{Im} z \le 0$. Evaluating at $(x - a_1, \dots, x - a_n)$ gives $f \boxplus_n g$ and proves real-rootedness. For fixed monic $g$, $f \mapsto f \boxplus_n g$ is an invertible affine map on monic coefficient space (its matrix is triangular with unit diagonal), hence maps the open interior (simple real-rooted polynomials) into itself; thus one simple input suffices for simple output.
2. **Majorization and $\ell^2$ estimate:** For $u, v \in \mathbb{R}^n$, let $r_1(t) < \cdots < r_n(t)$ be the ordered roots of the convolution along $(a + tu, b + tv)$, and let $\lambda_1 \le \cdots \le \lambda_n$ be the ordered roots of the convolution of the polynomials with roots $u, v$. As $t \to +\infty$, $r_i(t)/t \to \lambda_i$ (and reversed as $t \to -\infty$). For each $1 \le k \le n$, the monic polynomial $Q_k(x, t)$ whose roots are the $N = \binom{n}{k}$ $k$-subset sums of $r_i(t)$ has total degree $\le N$ in $(x, t)$. Pick $M$ strictly above the sum of the top $k$ slopes $\lambda_i$. Every branch of $Q_k$ minus $Mt$ tends to $+\infty$ as $t \to -\infty$ and $-\infty$ as $t \to +\infty$. Its derivative at $t = 0$ cannot be positive: otherwise a generic horizontal level $x = Mt + c$ (avoiding the finitely many exceptional values of the squarefree part of $Q_k$) would be crossed at least $3$ times by this branch and at least once by every other branch (with full multiplicity on persistent coincidences), giving $> N$ roots of the degree-$\le N$ nonzero polynomial $Q_k(Mt + c, t)$. Letting $M$ decrease gives all partial-sum inequalities $\sum_{i \in I} r_i'(0) \le \sum_{j=n-k+1}^n \lambda_j$, with equal totals, so $r'(0)$ is majorized by $\lambda$. Consequently $\sum_i (r_i'(0) - z)_+ \le \sum_i (\lambda_i - z)_+$ for all $z$, and integrating in $z$ gives $\|r'(0)\|_2^2 \le \|\lambda\|_2^2 = \|u\|_2^2 + \|v\|_2^2 + 2n \bar{u}\bar{v}$ (by the degree-1 and degree-2 coefficient rules).
3. **Positive doubly stochastic $A, B$ and score transport:** Varying one root of $f$ has slopes $(1, 0, \dots, 0)$, so majorization gives $A_{ij} \ge 0$, with column sums $1$ (sum of roots) and row sums $1$ (translation equivariance). Perturbing $a_j \mapsto a_j + t$ produces $C - t Q$, where $Q$ is the degree-$(n-1)$ convolution of $f(x)/(x - a_j)$ with $g'/n$; since $g'$ is simple by Rolle, $Q$ is simple. If $A_{ij} = 0$, then $Q(c_i) = 0$ and setting $t = C'(c_i)/Q'(c_i)$ would force a double root at $c_i$, impossible since $g$ remains simple. Hence $A_{ij} > 0$, and symmetrically $B_{ij} > 0$. Finally, perturbing $f$ by $-t f''/2$ yields root velocity $s_f$; since $\boxplus_n$ commutes with $d^2/dx^2$, the output perturbs by $-t C''/2$ with velocity $s_C$, giving $A s_f = s_C$ and symmetrically $B s_g = s_C$. $\square$

**Theorem 3 (All-degree strict subcritical $p$-Stam with one symmetric score).** *Let $n \ge 3$ and $1 < p < 2$. Let $f, g$ be simple monic real-rooted polynomials of degree $n$, and suppose there exists a permutation involution $S$ ($S^2 = I$) such that $S s_f = -s_f$ (in particular, this holds whenever $f$ is reflection-symmetric about its mean, with $g$ arbitrary). Setting $P := (I - S)/2$ and $\eta := \|A P\|_{1 \to 1}$, we have $\eta \in (0, 1)$ and*
$$
F_p(f \boxplus_n g) \ge \eta^{-(2 - p)/(p - 1)} F_p(f) + F_p(g) > F_p(f) + F_p(g).
\tag{3}
$$

*Proof.* The operator $P = (I - S)/2$ is an orthogonal projection in $\mathbb{R}^n$ with $P s_f = s_f$ and $\mathbf{1}^\top P = 0$ (mean-zero range). Each column of $AP$ is half the difference of two columns of the entrywise positive stochastic matrix $A$, so its $\ell^1$-norm is strictly less than $1$; hence $\eta := \|AP\|_{1 \to 1} < 1$, while $AP s_f = A s_f = s_C \neq 0$ gives $\eta > 0$.

Because $Pu$ has mean zero for every $u \in \mathbb{R}^n$, the cross term $2n \overline{(Pu)} \bar{v}$ in (2) vanishes identically on $\mathbb{R}^{2n}$, giving
$$
\|A P u + B v\|_2^2 \le \|P u\|_2^2 + \|v\|_2^2 \le \|u\|_2^2 + \|v\|_2^2 \qquad (\forall u, v \in \mathbb{R}^n),
$$
and the same norm bound $\le 1$ extends to $\mathbb{C}^{2n}$ by splitting real and imaginary parts. At the $\ell^1$ endpoint, $\|\eta^{-1} A P u + B v\|_1 \le \|u\|_1 + \|v\|_1$.

Consider the analytic operator family $T_z := (\eta^{z - 1} AP,\; B)$ on the closed strip $0 \le \operatorname{Re} z \le 1$. On $\operatorname{Re} z = 0$, $\|T_z\|_{1 \to 1} \le 1$; on $\operatorname{Re} z = 1$, $\|T_z\|_{2 \to 2} \le 1$. By Riesz–Thorin complex interpolation (applied to the bilinear pairing of $T_z$ with unit vectors $d \in \ell^p(\mathbb{C}^{2n})$, $e \in \ell^q(\mathbb{C}^n)$ via the analytic exponents $|d_j|^{p(1-z/2)}$ and $|e_i|^{q z/2}$ and the three-lines theorem), at $\theta := 2 - 2/p \in (0, 1)$ we obtain $\|T_\theta\|_{p \to p} \le 1$.

Applying $T_\theta$ to $d = (\eta^{1 - \theta} \lambda s_f,\; (1 - \lambda) s_g)$ for $\lambda \in (0, 1)$ yields $T_\theta d = \lambda A P s_f + (1 - \lambda) B s_g = s_C$, hence
$$
\Phi_p(C) = \|s_C\|_p^p \le \eta^{p(1 - \theta)} \lambda^p \Phi_p(f) + (1 - \lambda)^p \Phi_p(g)
= \eta^{2 - p} \lambda^p \Phi_p(f) + (1 - \lambda)^p \Phi_p(g).
$$
Minimizing $a \lambda^p + b (1 - \lambda)^p$ over $\lambda \in (0, 1)$ for $a, b > 0$ gives $(a^{-1/(p-1)} + b^{-1/(p-1)})^{1 - p}$. Raising both sides to $-1/(p - 1)$ yields (3). $\square$

**Proposition 4 (Uniform endpoint strictness on compact sets).** *Let $n \ge 3$ and let $\mathcal{K}$ be any compact set of pairs of simple monic real-rooted degree-$n$ polynomials.*
1. *There exists $\delta_1(\mathcal{K}) > 0$ such that $g_p(f, g) > 0$ for all $(f, g) \in \mathcal{K}$ and all $p \in (1, 1 + \delta_1(\mathcal{K}))$.*
2. *Assuming the $p = 2$ Hermite uniqueness theorem of Hashemi and Hyun ([arXiv:2609.10367v1](https://arxiv.org/abs/2609.10367v1)), there exists $\delta_2(\mathcal{K}) > 0$ such that $g_p(f, g) > 0$ for all $(f, g) \in \mathcal{K}$ and all $p \in (2 - \delta_2(\mathcal{K}), 2)$.*

*Proof.*
1. Because $A$ and $B$ are entrywise positive doubly stochastic and $s_f, s_g$ have both positive and negative coordinates, strict cancellation in every row gives $\|s_C\|_1 < \min(\|s_f\|_1, \|s_g\|_1)$. By continuity and compactness, there exists $r < 1$ and $\delta_1 > 0$ such that $\|s_C\|_p / \|s_f\|_p \le r$ and $\|s_C\|_p / \|s_g\|_p \le r$ on $\mathcal{K} \times [1, 1 + \delta_1]$. Shrinking $\delta_1$ so that $2 r^q < 1$ (since $q = p/(p-1) \to \infty$ as $p \to 1^+$) gives $F_p(f)/F_p(C) + F_p(g)/F_p(C) \le 2 r^q < 1$, i.e., $g_p(f, g) > 0$.
2. By Hashemi and Hyun (`arXiv:2609.10367v1`), $g_2(f, g) = 0$ on simple pairs iff $(f, g)$ is an affine Hermite pair with scales $d, e > 0$. At any such pair, with $R := \sqrt{d^2 + e^2}$, $u := d/R$, $v := e/R$,
   $$
   g_p(f, g) = F_p(H_n)\, R^q (1 - u^q - v^q),
   \qquad
   \left.\frac{\partial g_p}{\partial p}\right|_{p=2}
   = F_2(H_n)\, R^2 \bigl(u^2 \log u + v^2 \log v\bigr) < 0,
   $$
   since $q'(2) = -1$ and $u, v \in (0, 1)$. Because $x \mapsto |x|^p \log |x|$ extends continuously by $0$, $\partial_p g_p$ is continuous in $(f, g, p)$ near $p = 2$. Thus every $p = 2$ equality pair has a neighborhood where $\partial_p g_p < 0$, forcing $g_p > g_2 \ge 0$ immediately below $2$; every other pair has $g_2 > 0$ and remains positive on a neighborhood. Compactness of $\mathcal{K}$ yields a uniform $\delta_2(\mathcal{K}) > 0$. $\square$

---

## 5. Obstruction to Centered Jacobian $\ell^p$-Contraction in Higher Degrees

One might hope to prove `conj:pstam_holds` for arbitrary asymmetric simple pairs in degrees $n \ge 4$ by establishing the operator contraction $\|J(u, v)\|_p^p \le \|u\|_p^p + \|v\|_p^p$ on centered velocities $u, v \in \mathbf{1}^\perp$. The next proposition rules out this entire strategy.

**Proposition 5 (Counterexample to centered Jacobian $\ell^p$-contraction).** *At $n = 64$ and $p = 3/2$, there exists a simple diagonal pair $(f, f)$ with centered root vector $\tilde{a} \in \mathbf{1}^\perp$ whose convolution Jacobian $J = (A, A)$ satisfies*
$$
\|J(\tilde{a}, \tilde{a})\|_p^p > \|\tilde{a}\|_p^p + \|\tilde{a}\|_p^p = 2 \|\tilde{a}\|_p^p.
$$

*Proof.* Start with the degenerate centered vector $a = (-1, \dots, -1, n - 1) \in \mathbf{1}^\perp$. Before centering, the polynomial is $x^{n-1}(x - n)$, whose self-convolution by the coefficient formula is
$$
\bigl(x^{n-1}(x - n)\bigr) \boxplus_n \bigl(x^{n-1}(x - n)\bigr) = x^{n-2}\bigl(x^2 - 2nx + n(n - 1)\bigr),
$$
with roots $0$ (multiplicity $n - 2$) and $n \pm \sqrt{n}$. Centering by $-2$ gives the output root vector $c = (-2, \dots, -2,\, n - 2 - \sqrt{n},\, n - 2 + \sqrt{n})$. For $n = 64$ and $p = 3/2$, convexity of $x \mapsto x^{3/2}$ on the two outer roots $62 \pm 8$ gives $(62 - 8)^{3/2} + (62 + 8)^{3/2} \ge 2 \cdot 62^{3/2}$, so
$$
\|c\|_{3/2}^{3/2} - 2 \|a\|_{3/2}^{3/2}
\ge 62 \cdot 2^{3/2} + 2 \cdot 62^{3/2} - 2\bigl(63 + 63^{3/2}\bigr)
= 62 \cdot 2^{3/2} - 126 - 2\bigl(63^{3/2} - 62^{3/2}\bigr).
$$
Since $63^{3/2} - 62^{3/2} < \frac{3}{2}\sqrt{63} < 12$ and $2^{3/2} = 2\sqrt{2} > 14/5$, we have $62 \cdot 2^{3/2} - 126 - 24 > 173.6 - 150 = 23.6 > 0$. Now split the repeated roots of $a$ by an arbitrarily small centered perturbation $\tilde{a} \in \mathbf{1}^\perp$ with distinct roots, producing a simple polynomial $f$ and simple output $\tilde{c}$. By continuity, $\|\tilde{c}\|_{3/2}^{3/2} > 2 \|\tilde{a}\|_{3/2}^{3/2}$ persists. By Euler's homogeneity identity for common dilation of $(f, f)$, the Jacobian satisfies $J(\tilde{a}, \tilde{a}) = \tilde{c}$ identically, giving the strict violation. $\square$

Thus any proof of `conj:pstam_holds` for general asymmetric pairs in degrees $n \ge 4$ must exploit specific properties of score vectors beyond centering.

---

## 6. Self-Contained Reproof of the Hermite Spectrum (Prior Art: `arXiv:2609.10367v1`)

For completeness, we include a concise reproof of `conj:spectrum` and `rem:En_sym` (previously proved in Hashemi and Hyun, [arXiv:2609.10367v1](https://arxiv.org/abs/2609.10367v1), Appendix A, and Machina Mathematica `freeconv-spectrum`).

Write $D := d/dx$, $T_t := \exp(-t D^2/2)$ on polynomials (truncating on finite degree), and let $H_m := T_1 x^m$ be the probabilists' Hermite polynomials. From the coefficient formula, $f \boxplus_n g = \sum_j g_j \frac{(n - j)!}{n!} D^j f$, so convolution with $H_n$ acts precisely as $T_1$; in particular,
$$
T_1 H_m(x) = T_2 x^m = 2^{m/2} H_m(x / \sqrt{2}).
$$
Since common positive rescaling leaves the partial root Jacobian $E_n$ unchanged, we may evaluate $E_n$ at the unnormalized Hermite roots $h_1 < \cdots < h_n$ of $H_n$. Perturbing the first input to $H_n + \varepsilon H_m$ ($0 \le m < n$) produces root derivative $-v_m$, where $v_m(i) := H_m(h_i) / H_n'(h_i)$, while the output $T_1 H_n + \varepsilon T_1 H_m$ has root derivative $-2^{(m - n + 1)/2} v_m$ at $\sqrt{2}\, h$. Thus $E_n v_m = 2^{-(n - 1 - m)/2} v_m$.

To verify symmetry and singular values, the generating function $e^{xt - t^2/2}$ gives $H_m' = m H_{m-1}$ and $H_{m+1} = x H_m - m H_{m-1}$. Telescoping the recurrence gives the Christoffel–Darboux identity
$$
\sum_{m=0}^{n-1} \frac{H_m(x) H_m(y)}{m!} = \frac{H_n(x) H_{n-1}(y) - H_{n-1}(x) H_n(y)}{(n - 1)!\,(x - y)}.
$$
At distinct roots $x = h_i \neq h_j = y$ this vanishes, while at $x = y = h_i$ L'Hôpital's rule gives $H_n'(h_i)^2 / n!$. Hence the $n \times n$ matrix with columns $\sqrt{n!/m!}\, v_m$ has orthonormal rows and therefore orthonormal columns. Because $v_{n-1} = \mathbf{1}/n$, deleting $m = n - 1$ yields an orthonormal eigenbasis of $E_n$ on $W = \mathbf{1}^\perp$. Therefore $E_n$ is symmetric positive definite with singular values (= eigenvalues) $\{2^{-k/2} : 1 \le k \le n - 1\}$ on $W$. $\square$

---

## 7. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/finite_free_pstam/explore.py
```

The script [`explore.py`](explore.py) performs exact rational SymPy verification of all Hermite tangent actions and Euclidean Gram matrices for $3 \le n \le 12$, direct Jacobian SVDs and 30 strengthened symmetric-input interpolation bounds (Theorem 3), 4,608 random pair/exponent checks (3,072 with $p \le 2$, all nonnegative; 1,536 with $p > 2$, 62 negative), 72 Hermite sign checks, 675 asymmetric Laguerre checks up to degree $n = 1{,}024$ (using $(1 - D)^m x^n$ and the tridiagonal Laguerre recurrence), 6 cubic curvature grids, and 3 equal-variance cubic deficit grids confirming unique minimization at $(0, 0)$. All assertions passed.

---

## References

- J. Garza-Vargas, N. Srivastava, and Z. Stier, *The Finite Free Stam Inequality*, [arXiv:2602.15822](https://arxiv.org/abs/2602.15822), February 2026.
- B. Hashemi, *Spectral Structure in Finite Free Information Inequalities and $p$-Stam Phase Transitions*, [arXiv:2604.11922](https://arxiv.org/abs/2604.11922), April 2026.
- B. Hashemi and S. Hyun, *When Finite Free Curves Split*, [arXiv:2609.10367v1](https://arxiv.org/abs/2609.10367v1), September 2026 (Appendix A: Hermite spectrum and symmetry; main theorem: $p = 2$ Hermite uniqueness).
- Machina Mathematica, *Spectrum of the Linearized Finite Free Convolution at the Hermite Diagonal*, `freeconv-spectrum`.
