# Resolution of Curto–Morrison CTLN Open Questions 11, 13, and 14

**Target paper:** Carina Curto and Katherine Morrison, *Graph rules for recurrent neural network dynamics* (extended version), [arXiv:2301.12638](https://arxiv.org/abs/2301.12638) (Jan 2023; *Notices of the AMS* 70(4), April 2023), Open Questions 11, 13, and 14.  
**Companion files:** [`lean_proofs/Conjectures/CTLNDynamics.lean`](../../lean_proofs/Conjectures/CTLNDynamics.lean), [`solutions/ctln_dynamics/explore.py`](explore.py), [`solutions/ctln_dynamics/proof.pdf`](proof.pdf).

---

## Summary of Verdicts

A threshold-linear network (TLN) evolves on $\mathbb{R}_{\ge 0}^n$ via $\dot{x}_i = -x_i + [\theta + (Wx)_i]_+$ where $[u]_+ := \max(0, u)$. A combinatorial TLN (CTLN) associated with a simple directed graph $G$ on $\{1, \dots, n\}$ has $W_{ii} = 0$, $W_{ij} = -1 + \varepsilon$ if $j \to i$, and $W_{ij} = -1 - \delta$ if $j \not\to i$, with legal parameters $\theta > 0$, $\delta > 0$, and $0 < \varepsilon < \delta/(\delta+1)$.

1. **Open Question 13 ("Proper sources die"): PROVED** (for every directed graph $G$, every $\delta > 0$, $0 < \varepsilon \le 1$, $\theta > 0$, and every nonnegative initial condition $x(0) \ge 0$).
   - Every proper source $j$ (in-degree $0$, out-degree $\ge 1$) enters a permanent decay trap in finite time and thereafter decays **identically** as $x_j(t) = x_j(t_*) e^{-(t - t_*)}$.
   - Note that the legal bound $\varepsilon < \delta/(\delta+1)$ is never even needed. This settles the full conjecture left open after Caitlin Lienkaemper's thesis ([arXiv:2209.07583](https://arxiv.org/abs/2209.07583), Prop. 10.2 / Thm. 10.1 / Lemma 10.3).

2. **Open Question 11 (Gaudi attractor constant total activity + general classification): DISPROVED + GENERAL RIGIDITY / NO-GO THEOREM.**
   - *Gaudi counterexample:* For the 5-node circulant Gaudi network from Figure 8 ($j \to j+1, j+2 \pmod 5$, $(\varepsilon, \delta, \theta) = (0.1, 0.12, 1)$), a 448-bit Arb interval-arithmetic proof certifies the attracting rotating-wave orbit with period $T = 238.5904619456428291\dots$, minimum total activity $S_{\min} = 0.99206349206349206349\dots = 125/126$, maximum total activity $S_{\max} = 0.99634376409659372132\dots$, and certified peak-to-peak excursion $0.004280272033101658$ (error $< 2.1 \times 10^{-50}$). (On all-active flights, $\dot{S} = 5 - \frac{126}{25} S$ relaxes exponentially to $125/126$ within $\sim 10^{-41}$ for $\sim 55\%$ of each period, explaining why $S(t)$ looks visually flat in numerical plots.)
   - *General answer to Part 2 (Rigidity Theorem):* In **any** ReLU TLN, any solution $x(t)$ with constant total activity $\sum_i x_i(t) \equiv S_0$ remains in a **single closed linear chamber** $\overline{\mathcal{C}_\sigma}$ for all $t$ and evolves under a single affine ODE $\dot{x} = A_\sigma x + b_\sigma$. Consequently, a non-static constant-sum trajectory requires $(W - I)|_\sigma$ to have a purely imaginary (or zero) eigenvalue. In particular, **no isolated periodic orbit, no hyperbolic attractor, and no chaotic attractor** can conserve total population activity (any constant-sum periodic orbit sits in a continuum $\bar{x} + \lambda(x(t) - \bar{x})$, $\lambda \in [0, 1]$). For the Gaudi network, $\gcd(p_\sigma(z), p_\sigma(-z)) = 1$ across all 31 nonempty active sets rules out *every* non-static constant-sum trajectory purely algebraically.

3. **Open Question 14 (Linear interpolation persistence of attractors): DISPROVED in the strict reading (both for CTLNs and for the general TLN convex-hull conjecture).**
   - Juliana Londono Alvarez's thesis ([arXiv:2410.11012](https://arxiv.org/abs/2410.11012), Oct 2024, §7.1, Lemma 48 & Corollaries 49–50) proved that a shared fixed point or trajectory $x(t)$ of two TLNs persists as an exact invariant set along $W_s = (1-s)W_0 + sW_1$. Our counterexamples prove that **persistence holds as an invariant set, but fails as an attractor**:
     - *5-node static TLN convex-hull counterexample:* $W_0 = -\frac{3}{2} P$, $W_1 = -\frac{3}{2} Q$, $b = \mathbf{1}$ share the exact strictly positive fixed point $x^* = \frac{2}{5}\mathbf{1}$ with spectral abscissa $-0.25 < 0$ at $s = 0, 1$, which destabilizes to $+0.0738 > 0$ at $s = 1/2$.
     - *7-node CTLN periodic-orbit counterexample (Arb-certified):* Two non-isomorphic 7-node CTLNs at legal parameters $(\varepsilon, \delta, \theta) = (0.3, 1.5, 1)$ share the **exact same** synchronous attracting periodic trajectory in the exact same labeled coordinates ($T = 5.635227\dots$, block-on fraction $p = 0.507488\dots$). Its largest transverse Floquet exponent is $-0.1880$ (multiplier $0.3466$) at both endpoints $s = 0, 1$, but becomes strictly positive ($+0.00915$, multiplier $1.0529$) at the midpoint $s = 1/2$.
   - *Honest scope caveats:*
     1. At $s = 1/2$, trajectories leave the synchronous orbit and settle onto a *nearby desynchronized periodic attractor* (clone peaks $0.120$–$0.142$ vs. synchronous $0.132$, with nodes $6, 7$ unchanged; a stronger non-isomorphic pair $f_0 = (2,4,1,1,2)$, $f_1 = (3,1,5,3,4)$ gives exponent $+0.0749$ at $s = 0.58$ and two coexisting desynchronized attractors). Thus the 7-node pair refutes the strict statement (the shared attracting orbit ceases to be an attractor), while under a loose "nearly identical attractor up to small symmetry-breaking deformation" reading, a nearby deformed cycle remains.
     2. Both 7-node graphs contain bidirectional edges inside block $A$; the restricted $n = 5$ oriented no-sink dictionary (Figure 14 of the paper) was not tested exhaustively. (Note: when $W_0$ and $W_1$ differ only in columns of neurons that remain strictly silent on the attractor — as in Figure 14 `att 4` — the trajectory and all its Floquet multipliers are identically independent of $s$.)

---

## 1. Proof of Open Question 13 (Every Proper Source Dies)

**Theorem 1 (Eventual exact exponential decay of proper sources).** *Let $G$ be any directed graph on $n \ge 2$ vertices, let $\delta > 0$, $0 < \varepsilon \le 1$, $\theta > 0$, and let $x(t)$ be the solution of the CTLN $\dot{x}_i = -x_i + [\theta + (Wx)_i]_+$ from any initial state $x(0) \in \mathbb{R}_{\ge 0}^n$. If $j$ is a proper source of $G$ (in-degree $0$ and some outgoing edge $j \to k$, $k \neq j$), then there exists $t_* \ge 0$ such that $x_j(t) = x_j(t_*) e^{-(t - t_*)}$ for all $t \ge t_*$; in particular $\lim_{t \to \infty} x_j(t) = 0$.*

*Proof.* Set $c := 1 + \delta > 1$, $h := \delta + \varepsilon > 0$, $S(t) := \sum_{i=1}^n x_i(t)$, and $D(t) := \theta - c S(t)$. Since $W \le 0$ entrywise and $x(0) \ge 0$, the orthant $\mathbb{R}_{\ge 0}^n$ is forward invariant and every coordinate satisfies $\dot{x}_i \le \theta - x_i$, so $0 \le x_i(t) \le \max(x_i(0), \theta)$ for all $t \ge 0$.

**Step 1 (Uniform lower bound on total activity).** Since $[u]_+ \ge u$, $W_{ii} = 0$, $W_{i\ell} \ge -c$ for $\ell \neq i$, and $x \ge 0$:
$$
\dot{S} = \sum_{i=1}^n \bigl([\theta + (Wx)_i]_+ - x_i\bigr)
\ge \sum_{i=1}^n \bigl(\theta - c(S - x_i) - x_i\bigr)
= n\theta - \bigl(1 + (n - 1)c\bigr)S.
$$
Set $r := 1 + (n-1)c > 0$ and $L := n\theta / r$. Since $c > 1$, we have $n c > 1 + (n-1)c = r$, so $L > \theta/c$. Fix $\eta := \frac{1}{2}(c L - \theta) > 0$. By Grönwall's inequality, $S(t) \ge L(1 - e^{-rt})$, so there exists $T \ge 0$ (uniform over all $x(0) \ge 0$) such that $D(t) = \theta - c S(t) < -\eta$ for all $t \ge T$.

**Step 2 (Permanent decay trap and infinite firing measure).** Because $j$ has no incoming edges, every $W_{j\ell} = -c$ ($\ell \neq j$), so its preactivation is
$$
u_j(t) = \theta - c\sum_{\ell \neq j} x_\ell(t) = D(t) + c x_j(t).
$$
Set $m := \eta / c > 0$. Whenever $t \ge T$ and $x_j(t) \le m$, we have $u_j(t) < -\eta + c m = 0$, so $\dot{x}_j(t) = -x_j(t) \le 0$. Thus $\{x_j \le m\}$ is forward invariant for $t \ge T$, and once $x_j(t_*) \le m$ at some $t_* \ge T$, we have $u_j(t) < 0$ and hence $x_j(t) = x_j(t_*) e^{-(t - t_*)}$ for all $t \ge t_*$.

Suppose for contradiction that $x_j(t) > m$ for all $t \ge T$. Let $E := \{t \ge T : u_j(t) > 0\}$. Because $W_{j\ell} \le 0$ and $x \ge 0$, we have $u_j(t) \le \theta$, so $\dot{x}_j(t) \le -m + \theta \mathbf{1}_E(t)$ for all $t \ge T$. Integrating on $[T, t]$ gives
$$
0 < x_j(t) \le x_j(T) - m(t - T) + \theta\,|E \cap [T, t]|,
$$
hence $|E \cap [T, t]| \ge \frac{m}{\theta}(t - T) - \frac{x_j(T)}{\theta} \to \infty$ as $t \to \infty$.

**Step 3 (Out-neighbor $k$ catches and dominates $j$).** Pick an outgoing edge $j \to k$ ($k \neq j$). Then $W_{kj} = -1 + \varepsilon = -c + h$, while $W_{k\ell} \ge -c$ for $\ell \notin \{k, j\}$, so
$$
u_k(t) = \theta + (Wx)_k \ge D(t) + c x_k(t) + h x_j(t).
$$
Consider $d(t) := x_k(t) - x_j(t)$ for $t \ge T$:
- *On $E$ ($u_j > 0$):* $[u_j]_+ = D + c x_j$ and $[u_k]_+ \ge u_k$, so
  $$
  \dot{d} = ([u_k]_+ - x_k) - (D + c x_j - x_j) \ge (c - 1)x_k + (h - c + 1)x_j = \delta x_k + \varepsilon x_j \ge \varepsilon m > 0.
  $$
- *Off $E$ ($u_j \le 0$):* $[u_j]_+ = 0$ and $[u_k]_+ \ge 0$, so $\dot{d} \ge -x_k + x_j = -d$.

Notice that wherever $d(t) \le 0$ (on $[T, \infty)$), we have $\dot{d}(t) \ge 0$, with $\dot{d}(t) \ge \varepsilon m > 0$ on $E$ and $\dot{d}(t) \ge -d(t) > 0$ when $d(t) < 0$ off $E$.
- *Finite-time catch-up:* If $d(t) < 0$ for all $t \ge T$, then $\dot{d}(t) \ge 0$ everywhere on $[T, \infty)$ and $\dot{d}(t) \ge \varepsilon m$ on $E$, so $d(t) \ge d(T) + \varepsilon m |E \cap [T, t]| \to +\infty$, contradicting $d < 0$. Thus there exists $T_1 \ge T$ with $d(T_1) \ge 0$.
- *Forward invariance of $\{d \ge 0\}$:* Suppose $d(t_1) < 0$ for some $t_1 > T_1$, and let $t_0 := \sup\{t \in [T_1, t_1] : d(t) \ge 0\}$. By continuity $d(t_0) = 0$ and $d(t) < 0$ on $(t_0, t_1]$, where $\dot{d}(t) > 0$, giving $d(t_1) = \int_{t_0}^{t_1} \dot{d}(s)\,ds > 0$, a contradiction. Hence $x_k(t) \ge x_j(t) > m$ for all $t \ge T_1$.

**Step 4 (Ratio blow-up contradiction).** For $t \ge T_1$, let $r(t) := x_k(t) / x_j(t) \ge 1$. Then
$$
\dot{r} = \frac{\dot{x}_k x_j - x_k \dot{x}_j}{x_j^2} = \frac{[u_k]_+ - r [u_j]_+}{x_j}.
$$
- *Off $E$ ($u_j \le 0$):* $[u_j]_+ = 0$ and $[u_k]_+ \ge 0$, so $\dot{r} \ge 0$.
- *On $E$ ($u_j > 0$):* Since $D(t) < -\eta < 0$ and $r(t) \ge 1$,
  $$
  \dot{r} \ge \frac{(D + c x_k + h x_j) - r(D + c x_j)}{x_j} = h + \frac{(r - 1)(-D)}{x_j} \ge h.
  $$
Integrating over $[T_1, t]$ gives $r(t) \ge r(T_1) + h\,|E \cap [T_1, t]| \to +\infty$ as $t \to \infty$, contradicting the uniform bound $r(t) \le \max(x_k(0), \theta)/m$. $\square$

---

## 2. Proof of Open Question 11 (Gaudi Counterexample and Constant-Sum Rigidity)

**Theorem 2 (Rigidity of constant-sum trajectories in arbitrary ReLU TLNs).** *Let $\dot{x} = -x + [Wx + b]_+$ be any ReLU TLN on $\mathbb{R}^n$, and let $x : \mathbb{R} \to \mathbb{R}^n$ be a complete solution whose total activity $S(t) := \sum_{i=1}^n x_i(t)$ is constant. Then:*
1. *There exists an activation pattern $\sigma \in \{0, 1\}^n$ such that $x(t)$ remains for all $t \in \mathbb{R}$ in the single closed linear chamber $\overline{\mathcal{C}_\sigma} := \{z : (Wz+b)_i \ge 0 \text{ if } \sigma_i = 1,\ (Wz+b)_i \le 0 \text{ if } \sigma_i = 0\}$, and satisfies the single affine ODE $\dot{x} = A_\sigma x + b_\sigma$ there (where $A_\sigma = P_\sigma W - I$, $b_\sigma = P_\sigma b$, $P_\sigma = \operatorname{diag}(\sigma)$).*
2. *In particular, if $x(t)$ is periodic of period $T > 0$ and non-constant, then its time average $\bar{x} := \frac{1}{T}\int_0^T x(t)\,dt$ lies in $\overline{\mathcal{C}_\sigma}$ and satisfies $A_\sigma \bar{x} + b_\sigma = 0$, so for every $\lambda \in [0, 1]$, $x_\lambda(t) := \bar{x} + \lambda(x(t) - \bar{x})$ is an exact periodic solution of the TLN. Hence **no isolated periodic orbit** (and no hyperbolic or chaotic attractor) of any ReLU TLN can have constant total activity.*
3. *Any non-static bounded constant-sum trajectory requires $A_\sigma$ (hence $(W - I)|_\sigma$) to have a purely imaginary eigenvalue.*

*Proof.* The closed chambers $\{\overline{\mathcal{C}_\sigma}\}_{\sigma \in \{0,1\}^n}$ form a finite closed cover of $\mathbb{R}^n$, so their preimages $\{t : x(t) \in \overline{\mathcal{C}_\sigma}\}$ cover $\mathbb{R}$. By the Baire category theorem, some $\sigma$ has a preimage containing a nonempty open interval $(t_1, t_2)$. Let $z(t)$ be the unique global solution of the affine ODE $\dot{z} = A_\sigma z + b_\sigma$ with $z(t_1) = x(t_1)$. Then $z(t) = x(t)$ on $[t_1, t_2]$, so the real-analytic function $t \mapsto \mathbf{1}^\top z(t)$ is constant on $[t_1, t_2]$ and hence constant on all of $\mathbb{R}$: $\mathbf{1}^\top z(t) \equiv S_0 = \mathbf{1}^\top x(t)$.

Write the TLN vector field relative to chamber $\sigma$ as
$$
-x + [Wx + b]_+ = A_\sigma x + b_\sigma + q(x),
\qquad
q(x) := [Wx + b]_+ - P_\sigma(Wx + b).
$$
For active coordinates ($\sigma_i = 1$), $q_i(x) = [y_i]_+ - y_i \ge 0$; for inactive coordinates ($\sigma_i = 0$), $q_i(x) = [y_i]_+ \ge 0$; and $q(x) = 0 \iff x \in \overline{\mathcal{C}_\sigma}$.
Choose $\tau_0 > 0$ (depending only on $A_\sigma$) such that every entry of the row vector $\mathbf{1}^\top e^{A_\sigma s}$ is strictly positive for all $s \in [0, \tau_0]$ (possible since $\mathbf{1}^\top e^{0} = \mathbf{1}^\top > 0$). Whenever $x(t_*) = z(t_*)$, variation of constants gives, for $s \in [0, \tau_0]$,
$$
x(t_* + s) - z(t_* + s) = \int_0^s e^{A_\sigma (s - v)} q(x(t_* + v))\,dv.
$$
Left-multiplying by $\mathbf{1}^\top$ yields
$$
0 = \mathbf{1}^\top\bigl(x(t_* + s) - z(t_* + s)\bigr) = \int_0^s \bigl(\mathbf{1}^\top e^{A_\sigma(s - v)}\bigr) q(x(t_* + v))\,dv.
$$
Because $\mathbf{1}^\top e^{A_\sigma(s - v)} > 0$ componentwise and $q \ge 0$ componentwise, this forces $q(x(t_* + v)) = 0$ for all $v \in [0, \tau_0]$, hence $x(t_* + s) = z(t_* + s)$ on $[t_*, t_* + \tau_0]$. Iterating forward from $t_2$ in fixed steps of $\tau_0$ gives $x(t) = z(t) \in \overline{\mathcal{C}_\sigma}$ for all $t \ge t_1$. Repeating symmetrically backward in time (replacing $A_\sigma$ by $-A_\sigma$, where $\mathbf{1}^\top e^{-A_\sigma s} > 0$ for small $s \ge 0$) gives $x(t) = z(t) \in \overline{\mathcal{C}_\sigma}$ for all $t \in \mathbb{R}$, proving~(1).

For~(2), if $x$ is $T$-periodic, integrating $\dot{x} = A_\sigma x + b_\sigma$ over $[0, T]$ gives $A_\sigma \bar{x} + b_\sigma = 0$. Since $\overline{\mathcal{C}_\sigma}$ is convex, $\bar{x} \in \overline{\mathcal{C}_\sigma}$ and $x_\lambda(t) = (1-\lambda)\bar{x} + \lambda x(t) \in \overline{\mathcal{C}_\sigma}$ for all $\lambda \in [0, 1]$ and all $t$, and $\dot{x}_\lambda = \lambda(A_\sigma x + b_\sigma) = A_\sigma x_\lambda + b_\sigma$, so $x_\lambda$ is an exact TLN periodic orbit. Part~(3) follows immediately from the spectral representation of $z(t) = \bar{x} + e^{A_\sigma t}(x(0) - \bar{x})$. $\square$

**Application to the Gaudi attractor (`arXiv:2301.12638`, Figure 8):**
1. *Exact algebraic exclusion:* For the Gaudi network ($j \to j+1, j+2 \pmod 5$, $\varepsilon = 0.1, \delta = 0.12, \theta = 1$), exact rational computation verifies $\gcd(p_\sigma(z), p_\sigma(-z)) = 1$ for every nonempty active block $\sigma \subseteq \{1, \dots, 5\}$, excluding both zero and purely imaginary eigenvalues of $A_\sigma$. Thus **no** non-static constant-sum trajectory exists anywhere in the Gaudi network.
2. *Arb-certified Gaudi orbit:* Using 448-bit ball arithmetic (`python-flint 0.9`, `uv run solutions/ctln_dynamics/explore.py certificate`), we certify the attracting $\mathbb{Z}_5$-rotating-wave periodic orbit of Figure 8 via a contraction mapping on its 11-dimensional shooting map (Lipschitz bound $6.39 \times 10^{-55}$), a Schur–Cohn stability check on the reduced characteristic polynomial, and 474 Taylor-enclosure itinerary intervals:
   - Period $T = 238.5904619456428291\dots$,
   - $S_{\min} = 0.99206349206349206349\dots = 125/126$, $S_{\max} = 0.99634376409659372132\dots$,
   - Certified peak-to-peak total-activity oscillation $S_{\max} - S_{\min} = 0.004280272033101658$ (absolute error $< 2.1 \times 10^{-50}$).

---

## 3. Disproof of Open Question 14 (Strict Interpolation Persistence Fails)

**Theorem 3 (Failure of attractor persistence under linear weight interpolation).**
1. *(General TLN convex-hull conjecture — 5-node static counterexample):* Let $P, Q \in \{0, 1\}^{5 \times 5}$ be the row-selection matrices of $f_0 = (2,3,1,1,2)$ and $f_1 = (3,5,4,1,4)$ (1-based). Set $W_0 := -\frac{3}{2} P$, $W_1 := -\frac{3}{2} Q$, $b := \mathbf{1} \in \mathbb{R}^5$, and $W_s := (1-s)W_0 + sW_1$. Then $x^* := \frac{2}{5}\mathbf{1} > 0$ is an exact strictly positive fixed point of $(W_s, b)$ for every $s \in [0, 1]$. Its Jacobian $A_s = -I - \frac{3}{2} R_s$ ($R_s = (1-s)P + sQ$) has spectral abscissa $-0.25 < 0$ at both endpoints $s = 0, 1$ (exponentially attracting), but $+0.0738\dots > 0$ at the midpoint $s = 1/2$ (unstable).
2. *(CTLN conjecture — 7-node periodic-orbit counterexample):* Fix legal CTLN parameters $(\varepsilon, \delta, \theta) = (0.3, 1.5, 1)$ ($\varepsilon = 0.3 < \delta/(\delta+1) = 0.6$). Partition $\{1, \dots, 7\}$ into $A = \{1, \dots, 5\}$, $B = \{6\}$, $C = \{7\}$, include all edges $A \to B \to C \to A$ and no reverse interblock edges, and inside $A$ include every off-diagonal edge except the single nonedge $f(i) \to i$ per receiver, taking $f_0 = (2,3,1,1,2)$ for $G_0$ and $f_1 = (3,5,4,1,4)$ for $G_1$. Then:
   - $G_0$ and $G_1$ are loopless and non-isomorphic (column counts of missing edges in $A$ have 2 zeros for $G_0$ vs. 1 zero for $G_1$).
   - Every interpolated TLN $W_s = (1-s)W_0 + sW_1$ ($s \in [0, 1]$) leaves the synchronous subspace $\mathcal{M} = \{x_1 = \dots = x_5\}$ invariant with the **identical** 3-node quotient matrix
     $$
     W_{\mathrm{red}} = \begin{pmatrix} -23/5 & -5/2 & -7/10 \\ -7/2 & 0 & -5/2 \\ -25/2 & -7/10 & 0 \end{pmatrix}.
     $$
   - The quotient TLN has an Arb-certified attracting periodic orbit $\gamma$ of period $T = 5.635227195185912\dots$ on which block $A$ is active for fraction $p = 0.507488130425927\dots \in (0.5074, 0.5076)$ of the period (contraction bound $1.93 \times 10^{-57}$, 192 intervals). Thus $\gamma_{\mathrm{sync}}$ is the **exact same** periodic trajectory in the exact same labeled coordinates for all $s \in [0, 1]$.
   - Along $\gamma_{\mathrm{sync}}$, the normal variational equation on the transverse quotient $\mathbb{R}^5 / \operatorname{span}(\mathbf{1})$ switches simultaneously between $-\varepsilon I - (\delta+\varepsilon)R_s = -0.3 I - 1.8 R_s$ (when $A$ is active) and $-I$ (when $A$ is inactive). Because the ReLU vector field is continuous across the switching hyperplane and piecewise linear, perturbations of size $|\xi|$ experience $O(|\xi|)$ vector-field difference for $O(|\xi|)$ time, contributing $O(|\xi|^2)$ to the return map; hence the saltation matrix at the simultaneous clone crossing is the identity $I$.
   - Since the active and inactive transverse matrices commute, an eigenvalue $\mu$ of $R_s$ on $\mathbb{R}^5 / \operatorname{span}(\mathbf{1})$ gives exact per-unit-time normal Floquet exponent
     $$
     \Lambda(\mu) = -(1 - p) + p(-0.3 - 1.8 \operatorname{Re}\mu) = -1 + p(0.7 - 1.8 \operatorname{Re}\mu).
     $$
   - Both $P$ and $Q$ have characteristic polynomial $z^2(z-1)(z^2+z+1)$, whose transverse eigenvalues are $0, 0, e^{\pm 2\pi i/3}$ ($\min \operatorname{Re}\mu = -1/2$), yielding max normal Floquet exponent $-1 + 1.6p \approx -0.1880 < 0$ (multiplier $\approx 0.3466 < 1$) at both endpoints $s = 0, 1$ — so $\gamma_{\mathrm{sync}}$ is exponentially attracting at both ends.
   - At $s = 1/2$, $\frac{1}{2}(P+Q)$ has characteristic polynomial $\frac{1}{16}(z-1)(16z^4 + 16z^3 + 8z^2 + 2z - 1)$. The quartic $q(z) = 16z^4 + 16z^3 + 8z^2 + 2z - 1$ satisfies $q(-0.716) > 0 > q(-0.715)$, so it has a real transverse eigenvalue $\mu_* \in (-0.716, -0.715)$ ($\mu_* \approx -0.71586$). Thus at $s = 1/2$, $\Lambda(\mu_*) = -1 + p(0.7 - 1.8\mu_*) > 0$ ($\approx +0.00915$, multiplier $\approx 1.0529 > 1$), so $\gamma_{\mathrm{sync}}$ is **unstable** and ceases to be an attractor. $\square$

---

## 4. Verification, Formalization, and Attribution

- **Lean 4 formalization** ([`lean_proofs/Conjectures/CTLNDynamics.lean`](../../lean_proofs/Conjectures/CTLNDynamics.lean), Lean `v4.24.0` + Mathlib, zero `sorry`, standard axioms only): machine-checks the OQ 13 inequalities (`ctln_sum_lower_bound`, `ctln_asymptotic_D_neg`, `oq13_source_decay_trap`, `oq13_diff_on_E`, `oq13_diff_off_E`, `oq13_ratio_growth_on_E`, `oq13_ratio_nonneg_off_E`), OQ 11 chamber defect (`oq11_q_nonneg_and_zero_iff`), and the full OQ 14 algebraic/spectral core (`oq14_params_legal`, `oq14_no_self_loops`, `oq14_non_isomorphic`, `oq14_synchronous_reduction`, `oq14_static_tln_fixed_point`, `oq14_charpoly_P`, `oq14_charpoly_Q`, `oq14_charpoly_midpoint`, `oq14_midpoint_real_root`, `oq14_floquet_exponent_flip`, `oq14_static_abscissa_flip`).
- **Python & Arb verification** ([`solutions/ctln_dynamics/explore.py`](explore.py)):
  - `.venv/bin/python solutions/ctln_dynamics/explore.py all` runs 120 proper-source trials, Gaudi shooting/algebraic checks, and 7-node/static interpolation checks.
  - `uv run solutions/ctln_dynamics/explore.py certificate` executes both 448-bit Arb interval-arithmetic proofs (Gaudi orbit & 7-node quotient orbit).
- **Literature note:** Prior partial results on OQ 13 appear in Lienkaemper's thesis ([arXiv:2209.07583](https://arxiv.org/abs/2209.07583), §10); prior invariant-set persistence results on OQ 14 appear in Londono Alvarez's thesis ([arXiv:2410.11012](https://arxiv.org/abs/2410.11012), §7.1, Lemma 48 & Corollaries 49–50). No prior full resolution of OQ 13, OQ 11, or stability counterexample for OQ 14 was found as of November 2026.
