# Sharp Cubic Rademacher Stability Constant

**Target paper:** Peigan Gao and Jian Qian, *Fourth-Moment Geometry of Rademacher Sums*, [arXiv:2608.17802](https://arxiv.org/abs/2608.17802) (August 2026), Conjecture `eq:conjecture`.  
**Companion files:** [`solutions/rademacher_fourth_moment/explore.py`](explore.py), [`solutions/rademacher_fourth_moment/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `eq:conjecture`: PROVED (full and sharp for all $n \ge 1$)** with optimal dimension-free constant
  $$
  C = 5\sqrt{3} - 6\sqrt{2} \approx 0.174972663606.
  $$
  Specifically, for every integer $n \ge 1$ and every unit vector $a \in S^{n-1}$,
  $$
  \mathbb{E}\bigl|\overline{S}_n\bigr|^3 - \mathbb{E}\!\left|\sum_{i=1}^n a_i \varepsilon_i\right|^3 \ge (5\sqrt{3} - 6\sqrt{2})\,\Delta_n(a),
  $$
  where $\varepsilon_1, \dots, \varepsilon_n$ are i.i.d.\ uniform $\pm 1$ signs, $\overline{S}_n = n^{-1/2}\sum_{i=1}^n \varepsilon_i$, and $\Delta_n(a) = \sum_{i=1}^n (a_i^2 - 1/n)^2$.
- **Sharpness:** Equality holds at the nonflat vector $n = 3$, $a = (1, 1, 0)/\sqrt{2}$, where $\mathbb{E}|\overline{S}_3|^3 = 5/(2\sqrt{3})$, $\mathbb{E}|a \cdot \varepsilon|^3 = \sqrt{2}$, and $\Delta_3(a) = 1/6$, giving ratio $6(5/(2\sqrt{3}) - \sqrt{2}) = 5\sqrt{3} - 6\sqrt{2}$.
- **Dimension-specific optimal constants for $2 \le n \le 5$:** Certified by exact integer-arithmetic polynomial cone decompositions: $C_{2,1} = 2(\sqrt{2}-1)$, $C_{3,2} = 5\sqrt{3} - 6\sqrt{2}$, $C_{4,2} = 6 - 4\sqrt{2}$, and $C_{5,4} = \frac{27\sqrt{5}}{2} - 30$.
- **Proof structure:** Computer-assisted with 100% deterministic integer and exact-rational certificates (zero dependence on floating-point `mpmath` or numerical optimization): 47 polynomial cones ($n = 2, 3, 4, 5$), 6 exact rational far-region endpoints (even $n \ge 6$ and far odd $n \ge 7$), and 6,609 integer-arithmetic interval boxes covering all odd $n \ge 7$ simultaneously via the continuous reciprocal dimension parameter $z = 1/(n - 3) \in [0, 1/8]$.

---

## 1. Setup and Exact Rational Preliminaries

By sign symmetry, we may assume $a_i \ge 0$ and write $x_i = a_i^2 \ge 0$ with $\sum_{i=1}^n x_i = 1$, $q = \sum_{i=1}^n x_i^2$, $\Delta_n(a) = q - 1/n$, $F(x) = \mathbb{E}\bigl|\sum_{i=1}^n \sqrt{x_i}\,\varepsilon_i\bigr|^3$, and $M_n = \mathbb{E}|\overline{S}_n|^3$. We prove
$$
M_n - F(x) \ge C\,\Delta_n(a) \qquad (\forall n \ge 1,\; x \in \Delta^{n-1}),
\tag{1}
$$
where $C = 5\sqrt{3} - 6\sqrt{2}$. Set the rational comparison constant $c := 7/40$. All terminating decimals below denote exact rationals.

**Elementary exact inequalities (zero floating-point reliance).**
- Using $3.14159 < \pi < 3.14160$:
  - $\frac{4}{\pi}\bigl(\frac{0.64}{4} - \frac{1}{32}\bigr) > \frac{4}{3.14160}\cdot\frac{103}{800} = \frac{10300}{62832} > 0.16$;
  - $1.595^2 = 2.544025 < \frac{8}{3.14160} < \frac{8}{\pi} < \frac{8}{3.14159} < 2.547216 = 1.596^2$, so $1.595 < \mu := \sqrt{8/\pi} < 1.596$;
  - $0.797^2 = 0.635209 < \frac{2}{3.14160} < \frac{2}{\pi}$, so $\sqrt{2/\pi} > 0.797$.
- By exact integer squaring:
  - $C = 5\sqrt{3} - 6\sqrt{2} < \frac{7}{40} \iff 5\sqrt{3} < 6\sqrt{2} + \frac{7}{40} \iff \frac{4751}{1600} < \frac{3360}{1600}\sqrt{2} \iff 4751^2 = 22{,}572{,}001 < 22{,}579{,}200 = 2 \cdot 3360^2$;
  - $C_{2,1} = 2(\sqrt{2} - 1) > \frac{7}{40} \iff 8 \cdot 40^2 = 12{,}800 > 7{,}569 = 87^2$;
  - $C_{4,2} = 6 - 4\sqrt{2} > \frac{7}{40} \iff 233^2 = 54{,}289 > 51{,}200 = 32 \cdot 40^2$;
  - $C_{5,4} = \frac{27\sqrt{5}}{2} - 30 > \frac{7}{40} \iff 5 \cdot 540^2 = 1{,}458{,}000 > 1{,}456{,}849 = 1207^2$.

---

## 2. Scalar Fourier Majorant

Define for $s > 0$ and $x > 0$:
$$
p(s) := \frac{2}{\pi}\int_0^{\pi/2} \cos^s t\,dt, \qquad h(s) := 2\sqrt{s}\,p(s), \qquad g(x) := x\,h(1/x),\quad g(0) := 0.
$$

**Proposition 1 (Fourier majorant).** *For every $x \in \Delta^{n-1}$,*
$$
F(x) \le \sum_{i=1}^n g(x_i), \qquad \text{with } M_n = h(n) \text{ for all even } n.
\tag{2}
$$

*Proof.* Put $J := \int_0^\infty (\cos t - 1 + t^2/2)t^{-4}\,dt \in (0, \infty)$. Since the integrand is nonnegative, scaling gives $J^{-1}\int_0^\infty (\cos(ut) - 1 + u^2 t^2/2)t^{-4}\,dt = |u|^3$ for all $u \in \mathbb{R}$, and Tonelli's theorem gives $F(x) = J^{-1}\int_0^\infty \bigl(\prod_i \cos(\sqrt{x_i}\,t) - 1 + t^2/2\bigr)t^{-4}\,dt$. By weighted AM–GM over nonzero coordinates,
$$
\prod_{i=1}^n \cos(\sqrt{x_i}\,t) \le \prod_{i : x_i > 0} |\cos(\sqrt{x_i}\,t)| \le \sum_{i : x_i > 0} x_i |\cos(\sqrt{x_i}\,t)|^{1/x_i},
$$
with equality at the flat vector $x_i = 1/n$ when $n$ is even (since $\cos^n(t/\sqrt{n}) = |\cos(t/\sqrt{n})|^n$). Subtracting $1 - t^2/2$, integrating against $J^{-1}t^{-4}\,dt$, and rescaling reduce (2) to the identity
$$
J^{-1}\int_0^\infty \bigl(|\cos t|^s - 1 + \tfrac{1}{2}st^2\bigr)t^{-4}\,dt = 2s^2 p(s) \qquad (s > 0).
\tag{3}
$$
For $s > 6$, $f(t) = |\cos t|^s \in C^6(\mathbb{R})$ is $\pi$-periodic and even, so its Fourier series $f(t) = \sum_{j \in \mathbb{Z}} b_j e^{2ijt}$ has $b_{-j} = b_j$, $b_0 = p(s)$, and $b_j = O_s(|j|^{-6})$. Differentiating $f'\cos t = -s f\sin t$ yields the recurrence $(u + j + 1)b_{j+1} = (u - j)b_j$ with $u = s/2$. Setting $P(j) := \frac{1}{2}[j^3 + (u - 1)j^2 + u^2]$, algebraic expansion confirms $(u + j + 1)P(j) - (u - j)P(j + 1) = (u + j + 1)j^3$, hence $j^3 b_j = P(j)b_j - P(j + 1)b_{j+1}$. Telescoping gives $\sum_{j \ge 0} j^3 b_j = P(0)b_0 = \frac{1}{2}u^2 b_0$. Since $f(0) = 1$ and $f''(0) = -s$, substituting the Fourier series into (3) (justified by $\sum |j|^3|b_j| < \infty$ and the nonnegative kernel) yields $\sum_{j \in \mathbb{Z}} |2j|^3 b_j = 16 \cdot \frac{1}{2}u^2 b_0 = 2s^2 p(s)$. Both sides of (3) are holomorphic on $\{\Re s > 0\}$: on compact subsets of $\{\Re s > 0\}$, $|\cos t|^s - 1 + \frac{1}{2}st^2 = O_K(t^4)$ near $0$ and is bounded by $(2 + \frac{1}{2}|s|t^2)$ on $[1, \infty)$, so dominated convergence and the identity theorem extend (3) to all $s > 0$. $\square$

---

## 3. Curvature, Even Dimensions $n \ge 6$, and the Far Odd Region

Integration by parts gives $p(s + 2) = \frac{s + 1}{s + 2}p(s)$, and rescaling by $\sqrt{s}$ with $\cos t \le e^{-t^2/2}$ gives $h(s) \to \mu = \sqrt{8/\pi}$ as $s \to \infty$.

**Lemma 2 (Log-derivative and curvature lower bounds).** *For every $s > 0$,*
$$
(\log h)'(s) = \frac{1}{2}\int_0^\infty e^{-st}\tanh(t/2)\,dt.
\tag{4}
$$
*Moreover, for $x = 1/s \in (0, 1]$, setting $A_s := s\int_0^\infty u e^{-u}\tanh(u/(2s))\,du$ and $B_s := \int_0^\infty e^{-u}\tanh(u/(2s))\,du$,*
$$
-\frac{g''(x)}{2} = h(s)\left(\frac{A_s}{4} - \frac{s B_s^2}{8}\right) \ge \max\bigl\{0.16,\; L_0(x)\bigr\},
\tag{5}
$$
*where $L_0(x) := 1.595(1 - x/4)\bigl((1 - x^2)/4 - x/32\bigr)$.*

*Proof.* Set $L(s) := \log\mu - \frac{1}{2}\int_0^\infty e^{-st}\tanh(t/2)\,dt/t$. Differentiating gives $(L(s) - L(s+2))' = \frac{1}{2}\int_0^\infty e^{-st}(1 - e^{-t})^2\,dt = \frac{1}{2s} + \frac{1}{2(s+2)} - \frac{1}{s+1} = \bigl(\log\frac{\sqrt{s(s+2)}}{s+1}\bigr)'$. Since both $L(s) - L(s+2)$ and $\log(h(s)/h(s+2))$ vanish as $s \to \infty$ and satisfy the same recurrence, $L - \log h$ is $2$-periodic with limit $0$, proving (4).

Using $g''(x) = s^3 h''(s)$ and differentiating (4) yields the identity in (5). Since $\tanh$ is concave on $[0, \infty)$ with $\tanh 0 = 0$, $A_s$ increases with $s \ge 1$, so $A_s \ge A_1 = 1 - 2\sum_{j \ge 1} (-1)^{j-1}/(j + 1)^2$. The alternating 25-term partial sum gives $A_1 > 0.64$ (verified in exact rational arithmetic). Combined with $B_s \le 1/(2s)$ (from $\tanh v \le v$) and $h(s) \ge h(1) = 4/\pi$, this gives $-g''(x)/2 \ge \frac{4}{\pi}(0.64/4 - 1/32) > 0.16$. For the second bound, $\tanh v \ge v - v^3/3$ gives $A_s \ge 1 - s^{-2}$, while (4) and $\tanh v \le v$ give $h(s) \ge \mu e^{-1/(4s)} \ge 1.595(1 - 1/(4s))$; multiplying proves $-g''(x)/2 \ge L_0(x)$ whenever the bracket is positive (and $0.16$ covers the remaining region). $\square$

**Theorem 3 (Even $n \ge 6$ and far odd $n \ge 7$).** *Inequality (1) holds with the stronger constant $c = 7/40 > C$ for:*
1. *Every even $n \ge 6$ and every $x \in \Delta^{n-1}$.*
2. *Every odd $n \ge 7$ whenever $D := n^2 \Delta_n(a) = \sum_{i=1}^n (n x_i - 1)^2 \ge H(n)$, where $H(7) = 1.75$, $H(9) = 1.45$, and $H(n) = 1.24$ for odd $n \ge 11$.*

*Proof.* At $x_0 = 1/n$, Taylor's remainder gives $g(x_0) + g'(x_0)(x - x_0) - g(x) = (x - x_0)^2 \int_0^1 2(1 - t)\,[-g''(x_0 + t(x - x_0))/2]\,dt$ (extending to $x = 0$ by continuity). The subinterval $t \in [0, 1/3]$ carries weight $5/9$ and stays in $[0, \frac{2}{3n} + \frac{1}{3}]$. Because $L_0$ decreases on $[0, 1/2]$, the tangent deficit coefficient is at least $T_6 := (5 L_0(4/9) + 4 \cdot 0.16)/9 > 7/40 = c$ for $n \ge 6$, and at least $T_7 := (5 L_0(3/7) + 4 \cdot 0.16)/9 > 0.2219$ for $n \ge 7$. Summing over $i$ cancels the linear terms $\sum_i (x_i - 1/n) = 0$; by Proposition 1, this settles all even $n \ge 6$.

For odd $n = k + 1$ ($k$ even), let $P_k := 2^{-k}\binom{k}{k/2} = p(k)$ and $S_k := \sum_{i=1}^k \varepsilon_i$. By (2), $\mathbb{E}|S_k|^3 = 2k^2 P_k$, and telescoping $(2j - k)\binom{k}{j} = k\bigl[\binom{k-1}{j-1} - \binom{k-1}{j}\bigr]$ gives $\mathbb{E}|S_k| = k P_k$. Conditioning on $S_k$ adds $3\mathbb{E}|S_k| + P_k$ to the unnormalized cubic moment, yielding
$$
M_n = \frac{(2n - 1)P_k}{\sqrt{n}} = h(n - 1)\frac{2n - 1}{2\sqrt{n(n - 1)}}.
$$
Set $\delta_n := n^2(h(n) - M_n) \ge 0$. Integrating (4) on $[n - 1, n]$ bounds $\log(h(n)/h(n - 1)) \le \frac{1}{4n(n-1)}$, while $\bigl(\frac{2n-1}{2\sqrt{n(n-1)}}\bigr)^2 = 1 + \frac{1}{4n(n-1)}$. Using $\log(1 + v) \ge v - v^2/2$, $1 - e^{-v} \le v$, and $h(n) \le \mu < 1.596$ gives
$$
\delta_n \le 1.596\left(\frac{n}{8(n - 1)} + \frac{1}{64(n - 1)^2}\right),
\tag{6}
$$
which is strictly decreasing in $n$ and bounded by $e_7 = 0.234$, $e_9 = 0.225$, and $e_{11} = 0.220$ for $n = 7$, $n = 9$, and $n \ge 11$. For $D \ge 6$ and $n \ge 7$, $(T_7 - c) \cdot 6 > 0.234 > \delta_n$, settling $D \ge 6$. For $D \in [H(n), 6]$, every coordinate satisfies $x_i \le x(D) := (1 + \sqrt{D})/N < 1/2$ (where $N \in \{7, 9, 11\}$ is the row's minimal dimension), so (5) along each full tangent segment suffices provided $\psi_N(D) := D[L_0(x(D)) - c] \ge e_N$. On $[0, 1/2]$, $L_0' < 0$ and $L_0'' < 0$, so $\psi_N''(D) = \frac{3}{2}L_0'(x)x' + D L_0''(x)(x')^2 < 0$ on $(0, 6]$; thus $\psi_N$ is minimized on $[H(N), 6]$ at an endpoint. Rounding $\sqrt{D}$ upward (valid since $L_0$ decreases) and evaluating in exact rational arithmetic verifies $\psi_N(D) > e_N$ at all six endpoints ($N \in \{7, 9, 11\}$, $D \in \{H(N), 6\}$). $\square$

---

## 4. Near-Flat Odd Dimensions $n \ge 7$

Assume $n \ge 7$ is odd and $D \le H(n)$. Rescale to $u_i := \sqrt{n x_i} \ge 0$ (so $\sum u_i^2 = n$). Fix a least coordinate $d$, and among the remaining $n - 1$ coordinates let $A$ be a largest and $B$ a smallest. Note $B > 1/5$, since two coordinates $\le 1/5$ would force $D \ge 2(1 - 1/25)^2 = 1.8432 > 1.75$.

**Lemma 3 (Two-coordinate derivative and central-slice Jensen lower bound).** *When $A > B$, set $h_+ := A + B$, $l := A - B$, and for $r \ge 0$,*
$$
K(r) := \frac{(h_+ - r)_+^2}{h_+} - \frac{(l - r)_+^2}{l} \qquad (\text{second term } 0 \text{ at } l = 0).
$$
*Let $m := n - 3$ (even), $z := 1/m$, let $b_1, \dots, b_m \in [B, A]$ be the coordinates other than $d, A, B$, and set $w := m^{-1}\sum_{i=1}^m b_i^2 = 1 + z(3 - d^2 - B^2 - A^2)$. Then:*
1. *With $R$ the residual signed sum of the $n - 2$ coordinates outside $\{A, B\}$,*
   $$
   \frac{\partial_{x_B} F - \partial_{x_A} F}{2(x_A - x_B)} = \frac{3\sqrt{n}}{8AB}\,\mathbb{E}\bigl[K(|R|)\bigr].
   \tag{7}
   $$
2. *The function $K$ is nonnegative and nonincreasing, and $v \mapsto K(\sqrt{v})$ is $C^1$ and convex on $[0, \infty)$.*
3. *With $E := H(n) - (d^2 - 1)^2 - (B^2 - 1)^2 - (A^2 - 1)^2$,*
   $$
   v := \min\!\left\{\frac{E}{(1 - z)(1 + B)^2},\; \frac{(A - B)^2}{4z(1 - z)}\right\}, \qquad
   Y_0 := \sqrt{d^2 + v}, \qquad
   Y_2 := \sqrt{(2\sqrt{w} - d)^2 + (1 - 4z^2)v},
   $$
   *the left-hand side of (7) is bounded below by*
   $$
   \frac{3}{8AB}\cdot 0.797\,\sqrt{\frac{1 + 3z}{1 + z}}\left[K(Y_0) + \frac{K(Y_2)}{1 + 2z}\right].
   \tag{8}
   $$

*Proof.* (1) Put $H(r, t) := \frac{1}{2}(|r + t|^3 + |r - t|^3)$. Conditional on $R$, the unnormalized two-coordinate moment is $\frac{1}{2}[H(|R|, h_+) + H(|R|, l)]$, whose gradient difference in $(A^2, B^2)$ divided by $2(A^2 - B^2)$ is $[H_t(|R|, h_+)/h_+ - H_t(|R|, l)/l]/(8AB)$. Using $H_t(r, t)/t = 6r + 3(t - r)_+^2/t$ and rescaling by $n^{-3/2} \cdot n^2 = \sqrt{n}$ gives (7).

(2) On $[0, l^2]$, $K(\sqrt{v}) = h_+ - l + (\frac{1}{h_+} - \frac{1}{l})v$ is affine with negative slope; on $[l^2, h_+^2]$, $K(\sqrt{v}) = h_+ - 2\sqrt{v} + v/h_+$ has continuous matching derivative $1/h_+ - 1/l$ at $l^2$ and $0$ at $h_+^2$, with second derivative $\frac{1}{2}v^{-3/2} > 0$; on $[h_+^2, \infty)$ it vanishes identically.

(3) Write $\bar{b} = m^{-1}\sum b_i$ and $V = \frac{m}{m - 1}\sum (b_i - \bar{b})^2$. Conditioning on the unweighted sign sum $j = \sum_{i=1}^m \varepsilon_i$, exchangeability gives $\operatorname{Var}(\varepsilon_i \mid j) = 1 - j^2/m^2$ and zero variance for $\sum \varepsilon_i$, so $\mathbb{E}[\sum b_i \varepsilon_i \mid j] = j\bar{b}$ and $\operatorname{Var}(\sum b_i \varepsilon_i \mid j) = (1 - j^2/m^2)V$. On the zero slice $j = 0$ (probability $P_m$), averaging over the sign of $d$ gives $\mathbb{E}[R^2 \mid j = 0] = d^2 + V$. On the slices $j = \pm 2$, keeping only the sign of $d$ opposite to $j$ retains total probability $P_m/(1 + 2z)$ and conditional second moment $(2\bar{b} - d)^2 + (1 - 4z^2)V$, with $0 \le 2\bar{b} - d \le 2\sqrt{w} - d$. Because $\sum (b_i - \bar{b})^2 \le \sum (b_i - 1)^2 \le E/(1 + B)^2$ and also $\sum (b_i - \bar{b})^2 \le m(A - B)^2/4$, we have $V \le v$. Applying Jensen's inequality to the convex decreasing function $v \mapsto K(\sqrt{v})$ on these slices, discarding other slices by $K \ge 0$, and using $P_m \sqrt{m + 1} \downarrow \sqrt{2/\pi} > 0.797$ (ratio $\sqrt{(m+1)(m+3)}/(m+2) < 1$) and $\sqrt{n}/P_m^{-1} = \sqrt{\frac{1+3z}{1+z}}\,P_m\sqrt{m+1}$ yields (8). $\square$

**Theorem 4 (Near-flat odd $n \ge 7$).** *Inequality (1) holds for every odd $n \ge 7$ whenever $D \le H(n)$.*

*Proof.* **Step 1 (Universal-$n$ interval certificate for (8) $\ge 7/40$).** Every feasible configuration satisfies $d \in [0, 1]$, $B \in [1/5, 3/2]$ (since $(n - 1)B^2 \le n \implies B \le \sqrt{7/6} < 3/2$), $A \in [1, 8/5]$ (since $A^2 \le 1 + \sqrt{1.75} < (8/5)^2$), $d \le B \le A$, $E \ge 0$, and $B^2 \le w \le A^2$. Partition all odd $n \ge 7$ into three $z$-intervals: $[0.25, 0.25]$ ($n = 7$, $H = 1.75$), $[0.166666, 0.166667]$ ($n = 9$, $H = 1.45$), and $[0, 0.125]$ (all odd $n \ge 11$, $H = 1.24$, covering every $1/(n - 3) \in (0, 1/8]$). Using integer multiples of $10^{-6}$ with outward integer rounding (`near_certificate` in [`explore.py`](explore.py), omitting the singular second branch of $v$ when $z_- = 0$), exhaustive branch-and-bound certifies (8) $\ge 7/40$ across all three boxes in $435$, $427$, and $5{,}747$ visited boxes respectively. Hence $(\partial_{x_B} F - \partial_{x_A} F)/(2(x_A - x_B)) \ge c > C$ whenever $A > B$.

**Step 2 (Equalization path and endpoint comparison).** Transfer squared mass continuously from the largest unpinned level to the smallest unpinned level, splitting equally within tied groups, merging levels as they meet. Throughout this path, $\sum x_i = 1$, $q$ strictly decreases (keeping $D \le H(n)$), $d$ remains a minimum, every unpinned coordinate stays $\ge B > 1/5 > 0$ (so $F$ is $C^2$ between mergers), and Step 1 ensures $F + Cq$ is nondecreasing. After finitely many mergers we reach a vector proportional to $(1, \dots, 1, t)$ with $k = n - 1$ (even) ones and $t \in [0, 1]$. By the binomial identities of Section 3 (only $S_k = 0$ contributes $t^3$), its moment and fourth-power mass are
$$
f(t) = P_k \frac{2k^2 + 3kt^2 + t^3}{(k + t^2)^{3/2}}, \qquad Q(t) = \frac{k + t^4}{(k + t^2)^2}.
$$
On $(0, 1)$, $Q'(t) < 0$ and
$$
\frac{f'(t)}{-Q'(t)} = \frac{3P_k}{4}\,\frac{t\sqrt{k + t^2}}{1 + t}
$$
is strictly increasing in $t$, so $f(t) + CQ(t)$ is quasiconvex along the monotone parameter $Q(t)$ and attains its maximum at an endpoint $t \in \{0, 1\}$. Comparing $t = 1$ with $t = 0$ gives the secant slope
$$
\frac{M_{k+1} - M_k}{1/k - 1/(k + 1)} = k(k + 1)(M_{k+1} - M_k) = P_k\sqrt{k}\,\frac{r}{(1 + r)^2}, \qquad r = \sqrt{1 + 1/k}.
$$
As even $k \ge 2$ increases, $P_k \sqrt{k}$ strictly increases (ratio $(k + 1)/\sqrt{k(k + 2)} > 1$) and $r$ decreases toward $1$, strictly increasing $r/(1 + r)^2$. At $k = 2$, the value equals $6(M_3 - M_2) = 5\sqrt{3} - 6\sqrt{2} = C$. Thus $f(0) + CQ(0) \le f(1) + CQ(1) = M_n + C/n$, proving (1). $\square$

---

## 5. Exact Cone Certificates for Dimensions $1 \le n \le 5$

For $n = 1$, both sides of (1) vanish. For $2 \le n \le 5$, let $a_1 \ge \cdots \ge a_n \ge 0$ be possibly unnormalized, put $N = \sum a_i^2$, $Q = \sum a_i^4$, $b = 2^{n-1}$, $\mathcal{S} = \{1\} \times \{\pm 1\}^{n-1}$, and $h_0 = \sum_{\sigma \in \mathcal{S}} |\sum_i \sigma_i|^3$, so $M_n = h_0/(b n \sqrt{n})$. For each $(n, k)$ in the table below, write $\sqrt{n}\,C_{n,k} = (A + B\sqrt{nk})/d$ with integers $A, B, d$ ($d > 0$). Inequality $M_n - F(x) \ge C_{n,k}\Delta_n(a)$ is equivalent to
$$
\sqrt{n}\,n d \sqrt{N}\sum_{\sigma \in \mathcal{S}} |\sigma \cdot a|^3 \le d h_0 N^2 + (A + B\sqrt{nk})\,b(N^2 - n Q),
\tag{9}
$$
whose right-hand side is strictly positive for $N > 0$ (since $M_n \ge 1$ by Jensen and $0 < C_{n,k} < 1$).

On any simplicial cone $a = \sum_{j=1}^n t_j r_j$ ($t_j \ge 0$), split each linear form $\sigma \cdot a = P_\sigma(t) - N_\sigma(t)$ into its nonnegative- and negative-coefficient parts in $t$, and majorize $|\sigma \cdot a|^3 \le P_\sigma^3 + N_\sigma^3$ to form $U := \sum_{\sigma \in \mathcal{S}} (P_\sigma^3 + N_\sigma^3)$. Nonnegativity of every monomial coefficient of
$$
\bigl[d h_0 N^2 + (A + B\sqrt{nk})\,b(N^2 - n Q)\bigr]^2 - d^2 n^3 N U^2 \in \mathbb{Z}[\sqrt{nk}][t_1, \dots, t_n]
$$
(checked via exact sign-preserving integer squaring for $u + v\sqrt{nk} \ge 0$) certifies (9) on the entire cone. Starting from the ordered cone $\{r_j = (1^j, 0^{n-j}) : 1 \le j \le n\}$ and recursively subdividing along the first mixed-sign hyperplane $\sigma \cdot a = 0$ (or bisecting the longest Euclidean edge on $\sum a_i = 1$ if no mixed sign remains) terminates with the exact leaf counts below (`chamber_certificate` in [`explore.py`](explore.py)):

| $n$ | Optimal support $k$ | Optimal constant $C_{n,k} = \frac{M_n - M_k}{1/k - 1/n}$ | Certified cones | Relation to $C$ |
|---:|---:|---:|---:|---|
| $2$ | $1$ | $2(\sqrt{2} - 1) \approx 0.828427$ | $1$ | $> 7/40 > C$ |
| $3$ | $2$ | $5\sqrt{3} - 6\sqrt{2} \approx 0.174973$ | $2$ | $= C$ |
| $4$ | $2$ | $6 - 4\sqrt{2} \approx 0.343146$ | $3$ | $> 7/40 > C$ |
| $5$ | $4$ | $\frac{27\sqrt{5}}{2} - 30 \approx 0.186918$ | $41$ | $> 7/40 > C$ |

Combining Theorems 3 and 4 with the exact cone certificates for $n \le 5$ completes the proof of Conjecture `eq:conjecture` for all $n \ge 1$.

---

## 6. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/rademacher_fourth_moment/explore.py
```

The script verifies all $47$ exact polynomial cones ($n = 2, 3, 4, 5$), the $6$ exact rational far-region endpoints, and all $6{,}609$ integer-arithmetic interval boxes ($435$, $427$, and $5{,}747$) covering all odd $n \ge 7$. Independent of the proof, it also performs 32 seeded BFGS sphere searches per dimension across $n = 2, \dots, 13$ with exact integer re-evaluation at 75-digit precision; the global minimum found is $0.174972663606$ at $n = 3$, with minimizing supports $4, 6, 8, 10, 12$ for odd $n = 5, 7, 9, 11, 13$. All assertions passed.

---

## References

- P. Gao and J. Qian, *Fourth-Moment Geometry of Rademacher Sums*, [arXiv:2608.17802](https://arxiv.org/abs/2608.17802), August 2026.
