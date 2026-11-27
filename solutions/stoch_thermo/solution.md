# Analytical Resolution & Refutation of Upper-Bound Conjectures in Wampler & Barato (arXiv:2110.14540)

- **Conjecture Source**: Taylor Wampler and André C. Barato, *"Skewness and Kurtosis in Stochastic Thermodynamics"*, *J. Phys. A: Math. Theor.* 55 (2022) 014002, [arXiv:2110.14540](https://arxiv.org/abs/2110.14540).
- **Domain**: Biophysics / Stochastic Thermodynamics (Unicyclic Enzymatic & Molecular-Motor Markov Networks).
- **Novelty & Attribution Note**: On Nov 22, 2026, B. Li & E. Yeroushalmi ([github.com/brandonlign/first-passage](https://github.com/brandonlign/first-passage)) independently posted a preprint proving the **lower-envelope** conjectures for every $\Omega$ and containing an algebraically equivalent closed-form SCGF, while explicitly leaving the **upper-bound** conjectures open. Accordingly, our primary novel contributions are the **exact $\Omega = 2$ analytical upper bounds**, the **refutation of Wampler–Barato's Table 1 and "kink" claim** (for $\Omega = 2, 3$ and numerically across the entire Table 1 through $\Omega = 8$), and the **exact $\Omega \to \infty$ continuum bifurcation thresholds**.
- **Summary Verdict**:
  1. **Complete Exact Upper-Bound Solution for $\Omega = 2$ & Refutation of Table 1 and the "Kink" Claim ($\Omega = 2, 3$ and $\Omega \to \infty$)**:
     - For $\Omega = 2$, the exact onset of violation of $S \le S_{\mathrm{ARW}}(A,2)$ and $K \le K_{\mathrm{ARW}}(A,2)$ is $A_*^S(2) = 2\ln(2+\sqrt{3}) \approx 2.6339158$ (vs. Table 1's $2.84$) and $A_*^K(2) = 4\operatorname{artanh}(\sqrt{\sqrt{2}-1}) \approx 3.0571418$ (vs. Table 1's $3.27$), and both transitions are **$C^1$-smooth (no kink)** with quadratic onset $O((A - A_*)^2)$. Explicit counterexamples violate Table 1 at $A = 2.75 < 2.84$ ($S \approx 4.01883 > 4.01522 = S_{\mathrm{ARW}}(2.75,2)$) and $A = 3.15 < 3.27$ ($K \approx 26.6752 > 26.6466 = K_{\mathrm{ARW}}(3.15,2)$).
     - For $\Omega = 3$, the exact Hessian bifurcation at the uniform ARW occurs at $A_{\mathrm{bif}}^S(3) \approx 3.03745 < 3.15$ and $A_{\mathrm{bif}}^K(3) \approx 3.52807 < 3.64$, again refuting Table 1 with explicit perturbations at $A = 3.10$ and $A = 3.58$. (Independent numerical Hessian checks confirm every row $\Omega = 4, \dots, 8$ of Table 1 also overestimates the true bifurcation threshold.)
     - As $\Omega \to \infty$, the continuum bifurcation thresholds are $A_{\mathrm{bif}}^S(\infty) = 2\pi/\sqrt{3} \approx 3.62760$ and $A_{\mathrm{bif}}^K(\infty) = 2\pi/5^{1/4} \approx 4.20182$.
  2. **Lower Bounds & Closed-Form FPT SCGF (Independent Partial Results; Superseded by Li–Yeroushalmi, Nov 22 2026)**:
     - Independent derivation of the closed-form FPT SCGF $\rho(s)$ via Gallavotti–Cohen symmetrization, complete $\Omega = 2$ lower bounds, and universal small-$A$ lower bounds ($A \le 2.39936$ for skewness, $A \le 2.77038$ for kurtosis, and all $R \ge R_{\mathrm{ARW}}(A, 2)$). Full lower bounds for all $\Omega \ge 1, A > 0$ are now proved by Li & Yeroushalmi (Nov 22, 2026).

---

## 1. Exact Closed-Form FPT SCGF via Gallavotti–Cohen Symmetrization

For an $\Omega$-state unicyclic Markov jump process with forward rates $k_i > 0$, backward rates $\bar{k}_i > 0$, $\Pi_+ = \prod_{i=1}^\Omega k_i$, $\Pi_- = \prod_{i=1}^\Omega \bar{k}_i$, and affinity $A = \ln(\Pi_+/\Pi_-) > 0$, the tilted generator $\mathbf{L}(z)$ has characteristic polynomial
$$\Xi(s, z) = \det(s\mathbb{I} - \mathbf{L}(z)) = P(s) - \Pi_+(e^{-Az} - 1) - \Pi_-(e^{Az} - 1), \qquad P(s) := \det(s\mathbb{I} - \mathbf{L}(0)) = \sum_{k=1}^\Omega b_k s^k.$$
At $z = 1/2$, the off-diagonal entries $L_{i+1,i}(1/2) = L_{i,i+1}(1/2) = \sqrt{k_i \bar{k}_i}$ are already symmetric! Thus $H := -\mathbf{L}(1/2)$ is real symmetric and strictly positive definite for $A > 0$, because its quadratic form is
$$x^\top H x = \sum_{i=1}^\Omega \bigl(\sqrt{k_i}\,x_i - \sqrt{\bar{k}_i}\,x_{i+1}\bigr)^2 > 0 \qquad (x \neq 0).$$
Let $\nu_1, \dots, \nu_\Omega > 0$ be the eigenvalues of $H$ and $\tau_i := 1/\nu_i > 0$. Evaluating $\Xi(s, 1/2)$ gives
$$P(s) + (\sqrt{\Pi_+} - \sqrt{\Pi_-})^2 = \det(s\mathbb{I} + H) = (\sqrt{\Pi_+} - \sqrt{\Pi_-})^2 \prod_{i=1}^\Omega (1 + \tau_i s).$$
Solving $\Xi(s, \rho(s)) = 0$ yields the **exact closed-form FPT SCGF for every unicyclic network**:
$$\rho(s) = \frac{1}{2} - \frac{2}{A}\operatorname{arsinh}\!\left(\sinh\!\left(\frac{A}{4}\right)\prod_{i=1}^\Omega \sqrt{1 + \tau_i s}\right).$$

Setting $J_{\mathrm{diff}} = \Pi_+ - \Pi_-$, $c = \coth(A/2)$, $\eta = \coth(A/4)$, normalized weights $w_i = \tau_i / \sum_j \tau_j > 0$, and $\gamma_k := b_k J_{\mathrm{diff}}^{k-1}/b_1^k = \eta^{k-1} e_k(w_1, \dots, w_\Omega) > 0$, with $y_2 := c - 2\gamma_2 > 0$:
$$
\begin{aligned}
R &= A y_2, \\
S &= \sqrt{A}\,\frac{3c y_2 - 1 + 6\gamma_3}{y_2^{3/2}}, \\
K &= A\,\frac{3c y_2^2 + (12c^2 - 6)y_2 - 3c + 24(c\gamma_3 - \gamma_4)}{y_2^2}.
\end{aligned}
$$

---

## 2. Complete Exact Solution for $\Omega = 2$ & Refutation of Table 1

For $\Omega = 2$, $\gamma_3 = \gamma_4 = 0$, and by AM–GM on $b_1 = (k_1 + k_2) + (\bar{k}_1 + \bar{k}_2) \ge 2(e^{A/2} + 1)\sqrt{\Pi_-}$, the exact achievable range of $y_2$ is $[y_{\min}, c) = [\frac{1}{2}\coth(A/4),\, \coth(A/2))$.

1. **Lower Bounds (Proved):**
   - $S'(y_2) = \frac{3\sqrt{A}}{2 y_2^{5/2}}(1 - c y_2)$ changes sign once from $+$ to $-$ at $y_*^S = 1/c = \tanh(A/2)$, so $S$ is strictly quasiconcave in $y_2$ and minimized at an endpoint: $S \ge \min(S_{\mathrm{ARW}}(A,1), S_{\mathrm{ARW}}(A,2)) \ge S_{\min}$.
   - In $u = 1/y_2$, $K(u) = A(3c + (12c^2-6)u - 3cu^2)$ is strictly concave quadratic, hence minimized at an endpoint: $K \ge \min(K_{\mathrm{ARW}}(A,1), K_{\mathrm{ARW}}(A,2)) \ge K_{\min}$.

2. **Exact Closed-Form Upper Bounds for ALL $A > 0$ (Refuting Table 1 and the "Kink" Claim):**
   - **Skewness:** The unique maximizer $y_*^S = \tanh(A/2)$ enters $[y_{\min}, c)$ iff $\tanh(A/4) \ge 1/\sqrt{3}$, i.e.
     $$A_*^S(2) = 4\operatorname{artanh}(1/\sqrt{3}) = 2\ln(2+\sqrt{3}) \approx \mathbf{2.63391579} < 2.84 \text{ (Table 1)},$$
     with exact global maximum
     $$S_{\max}(A,2) = \begin{cases} S_{\mathrm{ARW}}(A,2), & 0 < A \le 2\ln(2+\sqrt{3}), \\ 2\sqrt{A}\,(\coth\frac{A}{2})^{3/2}, & A \ge 2\ln(2+\sqrt{3}). \end{cases}$$
   - **Kurtosis:** The unique maximizer $u_*^K = (2c^2-1)/c$ enters $(1/c, 2\tanh(A/4)]$ iff $\tanh(A/4) \ge \sqrt{\sqrt{2}-1}$, i.e.
     $$A_*^K(2) = 4\operatorname{artanh}\bigl(\sqrt{\sqrt{2}-1}\bigr) \approx \mathbf{3.05714184} < 3.27 \text{ (Table 1)},$$
     with exact global maximum
     $$K_{\max}(A,2) = \begin{cases} K_{\mathrm{ARW}}(A,2), & 0 < A \le 4\operatorname{artanh}(\sqrt{\sqrt{2}-1}), \\ \dfrac{3A(4\coth^4(A/2) - 3\coth^2(A/2) + 1)}{\coth(A/2)}, & A \ge 4\operatorname{artanh}(\sqrt{\sqrt{2}-1}). \end{cases}$$
   - Both transitions are **$C^1$-smooth (no kink)** by the envelope theorem, with quadratic excess $O((A - A_*)^2)$. Explicit verified counterexamples to Table 1 occur at $A = 2.75 < 2.84$ ($S \approx 4.01883 > 4.01522 = S_{\mathrm{ARW}}(2.75,2)$) and $A = 3.15 < 3.27$ ($K \approx 26.6752 > 26.6466 = K_{\mathrm{ARW}}(3.15,2)$).

---

## 3. Exact Bifurcation Theory for $\Omega = 3$ and Continuum Limit $\Omega \to \infty$

- **$\Omega = 3$:** Evaluating the exact second-order Hessian at the uniform ARW in $v = 2\cosh(A/6)$ and $u = 2\cosh(A/3)$ shows the ARW loses local maximality at the unique root $v_* \approx 2.26180$ of $v^6 + 3v^5 - 3v^4 - 15v^3 - 12v^2 + 2 = 0$ for skewness, giving
  $$A_{\mathrm{bif}}^S(3) = 6\operatorname{arcosh}(v_*/2) \approx \mathbf{3.03745} < 3.15 \text{ (Table 1)},$$
  and at $u_* \approx 3.5496$ of $(u^4 + 4u^3 - 19u^2 - 40u + 18) + \frac{\sqrt{u+2}}{2}(7u^3 - 8u^2 - 61u + 26) = 0$ for kurtosis, giving
  $$A_{\mathrm{bif}}^K(3) = 3\operatorname{arcosh}(u_*/2) \approx \mathbf{3.52807} < 3.64 \text{ (Table 1)}.$$
  Explicit perturbations strictly exceed $S_{\mathrm{ARW}}(A,3)$ at $A = 3.10 < 3.15$ and $K_{\mathrm{ARW}}(A,3)$ at $A = 3.58 < 3.64$.
- **Continuum Limit ($\Omega \to \infty$):** Second-order resolvent perturbation of the uniform cycle in Fourier modes gives the asymptotic bifurcation thresholds
  $$A_{\mathrm{bif}}^S(\infty) = \frac{2\pi}{\sqrt{3}} \approx \mathbf{3.62759873}, \qquad A_{\mathrm{bif}}^K(\infty) = \frac{2\pi}{5^{1/4}} \approx \mathbf{4.20181930},$$
  consistent with numerical Hessian thresholds for $\Omega = 4, \dots, 16$ (with $O(1/\Omega^2)$ convergence).

---

## 4. Universal Lower Bounds for All $\Omega \ge 1$

By Maclaurin's inequality $\frac{e_1 e_3}{e_4} \ge \frac{4\Omega}{\Omega - 3} > 4$ ($\Omega \ge 4$) on the positive normalized weights $w_i > 0$, we have universally:
$$\gamma_3 \ge 0, \qquad c\gamma_3 - \gamma_4 = \eta^2\bigl(c e_3(w) - \eta e_4(w)\bigr) \ge \eta(\eta^2 + 2) e_4(w) \ge 0.$$
Consequently, for **every** $\Omega \ge 1$ and every $A > 0$, $S \ge S_{\Omega=2}(y_2)$ and $K \ge K_{\Omega=2}(y_2)$, proving:
1. Whenever $R \ge R_{\mathrm{ARW}}(A, 2)$, $S \ge \min(S_{\mathrm{ARW}}(A,1), S_{\mathrm{ARW}}(A,2)) \ge S_{\min}$ and $K \ge \min(K_{\mathrm{ARW}}(A,1), K_{\mathrm{ARW}}(A,2)) \ge K_{\min}$.
2. Combining with the universal TUR $y_2 = R/A \ge 2/A$, for **all $\Omega \ge 1$**:
   - $S \ge S_{\mathrm{ARW}}(A,1) = \min_{N\le\Omega} S_{\mathrm{ARW}}(A,N)$ for all $0 < A \le A_0^S \approx 2.399357$ (root of $\frac{A}{2}\tanh\frac{A}{2} = 1$).
   - $K \ge K_{\mathrm{ARW}}(A,1) = \min_{N\le\Omega} K_{\mathrm{ARW}}(A,N)$ for all $0 < A \le A_0^K \approx 2.7702$ (root of $A = 4\coth\frac{A}{2} - 2\tanh\frac{A}{2}$).
3. In the irreversible limit $A \to \infty$, $S^* \in [2/\sqrt{\Omega}, 2]$ and $K^* \in [6/\Omega, 6]$ by Hölder/Cauchy–Schwarz.

Computational verification: [`verify_stoch_thermo.py`](./verify_stoch_thermo.py).
