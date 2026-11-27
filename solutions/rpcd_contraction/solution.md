# Contraction Bounds for Random-Permutation Coordinate Descent on General Quadratics

**Target paper:** Donghwa Kim, Jaewook Lee, and Chulhee Yun, *Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent*, ICML 2025 (PMLR 267), [arXiv:2505.23152](https://arxiv.org/abs/2505.23152), Section 4, Conjecture 4.1 (`conj:general`).  
**Companion files:** [`solutions/rpcd_contraction/explore.py`](explore.py), [`solutions/rpcd_contraction/_certificate.py`](_certificate.py), [`solutions/rpcd_contraction/_groups.py`](_groups.py), [`solutions/rpcd_contraction/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 4.1 (`conj:general`): PROVED FOR FOUR BROAD REGIMES:**
  1. **Every normalized Hessian in dimension $n \le 3$** (Theorem 2), with the strictly stronger single-epoch energy contraction $L(A) \preceq (1 - s/2)^4 A$ certified by an exact integer-arithmetic polynomial positivity certificate (19,986 positive, 967 zero, 0 negative monomial coefficients).
  2. **Every dimension $n \ge 1$ whenever $\lambda_{\max}(A) \le 2$** (Theorem 1), via the exact expected Gram matrix $\mathbb{E}[\Gamma_p \Gamma_p^\top] = A + \frac{1}{3}F^2 + \frac{1}{6}\operatorname{diag}(F^2)$ and operator convexity.
  3. **Every Hessian sign-conjugate to one with nonpositive off-diagonal entries** (Proposition 3), via a deterministic weighted-$\ell_\infty$ per-epoch contraction bound $\rho(|I - A|)^2 = (1 - s)^2 \le B_n(s)$.
  4. **All block-diagonal direct sums** of Hessians satisfying Conjecture 4.1 (Proposition 4).
- **Scope caveat & structural barrier:** Unrestricted Conjecture 4.1 remains **open** for $n \ge 4$ with $\lambda_{\max}(A) > 2$. As noted in Appendix E.2 of Kim, Lee, and Yun (2025), the single-epoch energy contraction factor $\|A^{-1/2} L(A) A^{-1/2}\|$ exceeds $B_n(\sigma)$ for large $n$ and $\sigma > 1/2$, so settling the general case requires a multi-epoch analysis rather than a single-epoch energy bound.
- **Exhaustive-permutation evidence:** Across $8{,}079$ numerical instances (2,415 dense matrices in $n = 3, \dots, 8$; 3,744 differential-evolution evaluations in $n = 3, \dots, 6$; and 1,920 two-group models up to $n = 258$) plus $4$ exact rational algebraic spectral-radius certificates ($n = 3, 4, 5, 6$), every instance stays strictly below $B_n(s)$.

---

## 1. Setup and Spectral-Radius Equivalence

For a symmetric positive definite Hessian $A \succ 0$ with $D = \operatorname{diag}(A)$, changing coordinates to $D^{1/2}x$ conjugates each coordinate-descent update and preserves asymptotic squared-norm rates by norm equivalence (Kim, Lee, and Yun, 2025, Section 2). Henceforth assume $A \succ 0$ has unit diagonal and write $s := \sigma := \lambda_{\min}(A) \in (0, 1]$. Conjecture 4.1 (`conj:general`) asserts that the asymptotic per-epoch expected squared-norm contraction rate is at most
$$
B_n(s) := \max\!\left\{\left(1 - \frac{1}{n}\right)^{\!n},\; \left(1 - \frac{s}{n}\right)^{\!2n}\right\}.
$$

Write $U_i := I - e_i e_i^\top A$. An RPCD epoch draws an independent uniform permutation $p \in S_n$ and applies $M_p := U_{p(n)} \cdots U_{p(1)}$. Let $\Gamma_p$ be the matrix with unit diagonal and off-diagonal entry $a_{ij}$ in position $(i, j)$ if $j$ precedes $i$ in $p$ (and $0$ otherwise). The vector of coordinate decrements satisfies the triangular system $\Gamma_p(x - M_p x) = Ax$, so $M_p = I - \Gamma_p^{-1} A$. Define the epoch second-moment operator $L(X) := \mathbb{E}_p[M_p^\top X M_p]$, so $\mathbb{E}\|x_K\|_2^2 = x_0^\top L^K(I) x_0$.

**Lemma 1 (Spectral radius and existence of root limits).**
1. *For every $K \ge 1$ and $X \in \mathbb{R}^{n \times n}$, $\|L^K(X)\| \le \|X\|\,\|L^K(I)\|$ in spectral norm, with equality at $X = I$. Consequently $\lim_{K \to \infty} (\operatorname{tr} L^K(I))^{1/K} = \rho(L)$, and restricting $L$ to any $L$-invariant real subspace containing $I$ preserves $\rho(L)$.*
2. *For every deterministic $x_0 \neq 0$, the limit $\lim_{K \to \infty} (\mathbb{E}\|x_K\|_2^2)^{1/K}$ exists, is at most $\rho(L)$, and is attained with equality by at least one standard basis vector $e_i$.*

*Proof.* (1) For a product $T$ of $K$ independent epochs, Cauchy–Schwarz on $\mathbb{E}[(Tu)^\top X (Tv)]$ gives $\|L^K(X)\| \le \|X\|\,\|L^K(I)\|$, attained at $X = I$. Since $L^K(I) \succeq 0$, $\|L^K(I)\| \le \operatorname{tr} L^K(I) \le n\|L^K(I)\|$; taking $K$th roots gives $\rho(L)$ by Gelfand's formula, and the same argument holds on any invariant subspace containing $I$.

(2) Conjugating by $A^{1/2}$ maps $U_i$ to the orthogonal projection $I - v_i v_i^\top$ ($v_i = A^{1/2}e_i$), so reversing a permutation $p$ transposes the conjugated epoch matrix. Uniform reversal invariance makes the $A^{1/2}$-conjugated operator self-adjoint in the Frobenius inner product. Therefore $\mathbb{E}[x_{2k}^\top A x_{2k}]$ is a finite sum of real coefficients times nonnegative eigenvalue squares raised to the power $k$; collecting equal bases shows its $(2k)$th root converges (unless the sequence eventually vanishes). Since energy decreases at every coordinate step, successive even terms sandwich the intervening odd term, and norm equivalence transfers the root limit to $\mathbb{E}\|x_K\|_2^2$. Finally, $\operatorname{tr} L^K(I) = \sum_{i=1}^n e_i^\top L^K(I) e_i$ implies some $e_i$ attains $\rho(L)$. $\square$

---

## 2. All-Dimensional Theorem for $\lambda_{\max}(A) \le 2$

Because $a_{ii} = 1$ and each coordinate update zeroes the updated gradient entry, a coordinate decrement $d_i$ lowers $x^\top A x$ by exactly $d_i^2$. Summing across an epoch yields the exact energy identity
$$
A - L(A) = A H A, \qquad H := \mathbb{E}_p\bigl[\Gamma_p^{-\top}\Gamma_p^{-1}\bigr] = \mathbb{E}_p\bigl[(\Gamma_p \Gamma_p^\top)^{-1}\bigr].
\tag{1}
$$
Thus $\lambda_{\min}(A^{1/2} H A^{1/2}) \ge 1 - q$ implies $L(A) \preceq q A$ and hence $\rho(L) \le q$.

**Theorem 1 (All dimensions with $\lambda_{\max}(A) \le 2$).** *For every $n \ge 1$ and every unit-diagonal $A \succ 0$ with $\lambda_{\max}(A) \le 2$, Conjecture 4.1 holds.*

*Proof.* The case $n = 1$ is exact in one update; assume $n \ge 2$. Write $F := A - I$ (zero diagonal). We compute $\mathbb{E}_p[\Gamma_p \Gamma_p^\top]$ entrywise: for $i = j$, the diagonal term is $1 + \sum_{k \neq i} \Pr[k \prec i]\,F_{ik}^2 = 1 + \frac{1}{2}(F^2)_{ii}$; for $i \neq j$, the off-diagonal term is $F_{ij}(\Pr[j \prec i] + \Pr[i \prec j]) + \sum_{k \neq i, j} \Pr[k \prec i \text{ and } k \prec j]\,F_{ik} F_{jk} = F_{ij} + \frac{1}{3}(F^2)_{ij}$. Hence
$$
\mathbb{E}_p[\Gamma_p \Gamma_p^\top] = A + J, \qquad J := \frac{1}{3}F^2 + \frac{1}{6}\operatorname{diag}(F^2).
$$
By the variational identity $v^\top G^{-1} v = \max_z (2 v^\top z - z^\top G z)$ for $G = \Gamma_p \Gamma_p^\top \succ 0$, taking expectations and optimizing over deterministic $z$ gives operator convexity $H = \mathbb{E}[G^{-1}] \succeq (\mathbb{E}[G])^{-1} = (A + J)^{-1}$. Putting $h := \lambda_{\max}(A^{-1/2} J A^{-1/2})$, we have $A^{1/2} H A^{1/2} \succeq (I + A^{-1/2} J A^{-1/2})^{-1} \succeq (1 + h)^{-1} I$, so the single-epoch energy contraction factor is at most $h/(1 + h)$.

Put $t := 1 - s \in [0, 1)$. Since $\operatorname{spec}(F) \subseteq [-t, 1]$, the scalar quadratic inequality $y^2 \le (1 - t)y + t$ on $[-t, 1]$ gives $F^2 \preceq (1 - t)F + t I$. Reading diagonal entries (using $\operatorname{diag}(F) = 0$) gives $\operatorname{diag}(F^2) \preceq t I$. Therefore
$$
h \le \max_{z \in [s, 2]} \frac{(z - 1)^2/3 + t/6}{z}.
$$
The objective is convex on $z > 0$, so its maximum on $[s, 2]$ occurs at an endpoint:
$$
h_s := \frac{t(3 - 2s)}{6s}, \qquad h_2 := \frac{2 + t}{12} \le \frac{1}{4}.
$$
At $z = 2$, $h_2/(1 + h_2) \le 1/5 < 1/4$. At $z = s$: if $s \ge 1/2$, then $h_s \le 1/3$ and $h_s/(1 + h_s) \le 1/4$; if $s \le 1/2$, then $h_s/(1 + h_s) \le t^2$ because
$$
t^2(3 + s + 2s^2) - t(3 - 2s) = s^2(1 - s)(1 - 2s) \ge 0.
$$
Finally, $B_n(s) \ge (1 - 1/n)^n \ge 1/4$ (increasing for $n \ge 2$) and $B_n(s) \ge (1 - s/n)^{2n} \ge (1 - s)^2 = t^2$ (Bernoulli's inequality). Thus $h/(1 + h) \le \max\{1/4, (1 - s)^2\} \le B_n(s)$. (In particular, all $2 \times 2$ unit-diagonal Hessians satisfy $\operatorname{tr}(A) = 2 \implies \lambda_{\max}(A) < 2$ and are covered.) $\square$

---

## 3. All Hessians in Dimension Three

**Theorem 2 (All $3 \times 3$ normalized Hessians).** *For every $3 \times 3$ unit-diagonal $A \succ 0$ with $s = \lambda_{\min}(A) \in (0, 1]$,*
$$
L(A) \preceq \left(1 - \frac{s}{2}\right)^{\!4} A.
\tag{2}
$$
*Since $(1 - s/2)^2 \le (1 - s/3)^3$ on $[0, 1]$ (as $(1 - s/3)^3 - (1 - s/2)^2 = s^2(\frac{1}{12} - \frac{s}{27}) \ge 0$), this yields $\rho(L) \le (1 - s/2)^4 \le (1 - s/3)^6 \le B_3(s)$ and proves Conjecture 4.1 for all $n \le 3$.*

*Proof.* If $s = 1$, then $A = I$ and $L(A) = 0$. Assume $s \in (0, 1)$, so $G := (A - s I)/(1 - s) \succeq 0$ has unit diagonal and rank $\le 2$. Represent $G$ as the Gram matrix of three unit vectors in $\mathbb{R}^2$. Modulo sign, their directions lie on the projective circle $\mathbb{R}/\pi\mathbb{Z}$ of circumference $\pi$. Cutting the circle inside a largest gap $\gamma \ge \pi/3$ and ordering the remaining directions from left to right gives two consecutive gaps $\alpha, \beta \in [0, \pi/2]$ (each at most the omitted gap $\gamma$, including coincident directions). Choosing signs along this arc conjugates $A$ by a signed permutation (an orthogonal symmetry of RPCD), reducing to $G_{12} = \cos\alpha$, $G_{23} = \cos\beta$, $G_{13} = \cos(\alpha + \beta)$ with $\alpha, \beta \in [0, \pi/2]$.

By continuity of both sides of (2) in $(\alpha, \beta)$, it suffices to prove (2) for $\alpha, \beta \in [0, \pi/2)$ (handling the boundary values $\alpha = \pi/2$ or $\beta = \pi/2$ by taking limits, and $s = 1$ separately above). Parameterize $(s, \alpha, \beta) \in (0, 1) \times [0, \pi/2)^2$ rationally by $u, v, w \ge 0$:
$$
s = \frac{u}{1 + u}, \qquad \tan\frac{\alpha}{2} = \frac{v}{1 + v}, \qquad \tan\frac{\beta}{2} = \frac{w}{1 + w}.
$$
Put $(b, c, d) := ((1 + v)^2 + v^2,\; 1 + 2v,\; 2v(1 + v))$ and $(f, g, h) := ((1 + w)^2 + w^2,\; 1 + 2w,\; 2w(1 + w))$, and set $D := (1 + u)bf$. Then $N := DA$ is an integer-coefficient matrix polynomial with diagonal $D$ and off-diagonal entries $N_{12} = cf$, $N_{23} = bg$, $N_{13} = cg - dh$. For each ordering $p = (i, j, k) \in S_3$, $V := D^2 \Gamma_p^{-1}$ has diagonal $D^2$ and at most three nonzero off-diagonal entries $V_{ji} = -D N_{ji}$, $V_{kj} = -D N_{kj}$, and $V_{ki} = N_{kj} N_{ji} - D N_{ki}$. Setting $q := (1 - s/2)^4$, form the integer-coefficient matrix polynomial
$$
P(u, v, w) := 16\!\left(\sum_{p \in S_3} V_p^\top V_p\right)\!N - 6\bigl([2(1 + u)]^4 - (2 + u)^4\bigr)(1 + u)(bf)^5 I = 96 D^5\bigl[HA - (1 - q)I\bigr].
$$
Because $HA$ is similar to the symmetric matrix $A^{1/2} H A^{1/2}$, all three eigenvalues of $P$ are real. Write $\det(z I + P) = z^3 + e_1 z^2 + e_2 z + e_3$. If $e_1, e_2, e_3 \ge 0$ everywhere on $[0, \infty)^3$, then $\det(z I + P) > 0$ for every $z > 0$, excluding negative eigenvalues of $P$ and proving $A^{1/2} H A^{1/2} \succeq (1 - q)I$, i.e., (2) via (1).

Directly from the formulas above, every entry of $P$ has degree at most $(5, 10, 10)$ in $(u, v, w)$, so $e_k$ has degree at most $(5k, 10k, 10k)$. Evaluating $(e_1, e_2, e_3) = (\operatorname{tr} P,\, \frac{1}{2}[(\operatorname{tr} P)^2 - \operatorname{tr}(P^2)],\, \det P)$ on the integer grid $\{0, \dots, 5k\} \times \{0, \dots, 10k\} \times \{0, \dots, 10k\}$ (exploiting $(v, w)$-swap symmetry) and recovering the exact monomial coefficients via scaled integer Lagrange interpolation along each axis (`_certificate.py`, asserting every division by $d!$ is exact) yields **zero negative coefficients**:

| Invariant | Degree box $(d_u, d_v, d_w)$ | Positive coefficients | Zero coefficients | Negative coefficients |
|---|---:|---:|---:|---:|
| $e_1 = \operatorname{tr} P$ | $(5, 10, 10)$ | $726$ | $0$ | $0$ |
| $e_2 = \frac{1}{2}[(\operatorname{tr} P)^2 - \operatorname{tr}(P^2)]$ | $(10, 20, 20)$ | $4{,}848$ | $3$ | $0$ |
| $e_3 = \det P$ | $(15, 30, 30)$ | $14{,}412$ | $964$ | $0$ |

Hence $e_1, e_2, e_3 \ge 0$ throughout $[0, \infty)^3$, establishing (2) on the entire continuous parameter domain. $\square$

---

## 4. Coordinate Comparison and Block Direct-Sum Closure

**Proposition 3 (Entrywise comparison and sign-conjugate M-matrices).** *Let $C := |I - A|$ entrywise. If $r := \rho(C) < 1$, then for every deterministic sequence of permutations the per-epoch squared-norm contraction rate is at most $r^2$. In particular, if $A$ is sign-conjugate to a matrix with all off-diagonal entries nonpositive (a symmetric M-matrix; cf.\ Varga 2000), then $r = 1 - s$ and $\rho(L) \le (1 - s)^2 \le B_n(s)$.*

*Proof.* For any $r' \in (r, 1)$, the Neumann series gives $w := (I - C/r')^{-1}\mathbf{1} > 0$ with $Cw \le r' w$. If $|x_j| \le \alpha w_j$ for all $j$ before an update, then each updated coordinate satisfies $|x_i'| = |\sum_{j \neq i} a_{ij} x_j| \le \sum_j C_{ij} \alpha w_j \le r' \alpha w_i$. Thus after one full epoch every coordinate satisfies $|x_i^{(1)}| \le r' \alpha w_i$. Iterating in the weighted $\ell_\infty$ norm, passing to squared Euclidean norm, and letting $r' \downarrow r$ proves contraction $\le r^2$ for every permutation order. If $DAD$ has nonpositive off-diagonal entries for a diagonal sign matrix $D$, then $C = I - DAD$ is orthogonally similar to $I - A$ and entrywise nonnegative, so by Perron–Frobenius $\rho(C) = \lambda_{\max}(I - A) = 1 - s$, giving $(1 - s)^2 \le B_n(s)$. $\square$

**Proposition 4 (Direct-sum closure).** *If $A = A_1 \oplus \cdots \oplus A_m$ is a block-diagonal direct sum of unit-diagonal blocks each satisfying Conjecture 4.1, then $A$ satisfies Conjecture 4.1.*

*Proof.* Restricting an i.i.d.\ uniform permutation of $\{1, \dots, n\}$ to block $j$ (of size $n_j$) produces i.i.d.\ uniform permutations of that block. Expected squared norms add across blocks, so $\rho(L_A) = \max_j \rho(L_{A_j}) \le \max_j B_{n_j}(\lambda_{\min}(A_j))$. Because $\lambda_{\min}(A_j) \ge s$ and $B_n(s)$ is nondecreasing in $n$ (since $(1 - a/n)^n$ increases in $n \ge a$ for $a \in [0, 1]$) and nonincreasing in $s$, we obtain $\rho(L_A) \le B_n(s)$. $\square$

---

## 5. Exact Algebraic Certificates and Exhaustive Permutation Searches

Run from the repository root:

```sh
.venv/bin/python -B solutions/rpcd_contraction/explore.py
```

The script [`explore.py`](explore.py) (importing [`_certificate.py`](_certificate.py) and [`_groups.py`](_groups.py)) executes:
1. **Dimension-three exact polynomial certificate:** Verifies all $19{,}986$ positive, $967$ zero, and $0$ negative coefficients of $e_1, e_2, e_3$.
2. **Exact rational algebraic spectral-radius certificates ($n = 3, 4, 5, 6$):** For $A = (25 I + W W^\top)/50$ ($s = 1/2$) using the first $n$ rows of $(5,0),(3,4),(4,-3),(0,5),(4,3),(3,-4)$, computes the exact integer characteristic polynomial of the symmetric-subspace operator $C_n$ (divisor $d_n = n!\,50^{2n}$) and verifies via exact shifted polynomial coefficients (negative constant term, all higher coefficients strictly positive, positive value at right endpoint) that $\rho(L)$ lies in $(0.146414221, 0.146414222)$ ($n = 3$), $(0.157833435, 0.157833436)$ ($n = 4$), $(0.187002136, 0.187002137)$ ($n = 5$), and $(0.198144651, 0.198144652)$ ($n = 6$), all strictly below $(1 - 1/(2n))^{2n}$.
3. **Dense and differential-evolution full-permutation searches:** $2{,}415$ dense matrices ($n = 3, \dots, 8$; worst $(\rho - B_n(s))/s = -0.169414$ at $n = 7$) and $3{,}744$ differential-evolution evaluations over planar Gram angles ($n = 3, \dots, 6$; worst $-0.185616$ at $n = 6$).
4. **Two-group exact reduced-operator searches (`_groups.py`):** Exact dynamic programming over the $5$-dimensional invariant quadratic subspace (matching full tensor enumeration within $10^{-11}$ on $9$ controls) across $1{,}920$ trials in $13$ size pairs up to $n = 258$; worst $(\rho - B_n(s))/s = -0.01292736$ at $(4, 150)$.

All assertions passed.

---

## References

- D. Kim, J. Lee, and C. Yun, *Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent*, ICML 2025, PMLR 267 ([arXiv:2505.23152](https://arxiv.org/abs/2505.23152)).
- C.-P. Lee and S. J. Wright, *Random Permutations Fix a Worst Case for Cyclic Coordinate Descent*, IMA Journal of Numerical Analysis 39(3):1246–1275, 2019.
- R. Sun, Z.-Q. Luo, and Y. Ye, *On the Efficiency of Random Permutation for ADMM and Coordinate Descent*, Mathematics of Operations Research 45(1):233–271, 2020.
- R. S. Varga, *Matrix Iterative Analysis*, 2nd ed., Springer, 2000.
- S. J. Wright and C.-P. Lee, *Analyzing Random Permutations for Cyclic Coordinate Descent*, Mathematics of Computation 89:2217–2248, 2020.
