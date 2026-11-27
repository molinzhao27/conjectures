# Resolution of the Three Similarity Matching PCA Conjectures

**Target paper:** Veronica Centorrino, Francesco Bullo, and Giovanni Russo, *Similarity Matching Networks: Hebbian Learning and Convergence Over Multiple Time Scales*, [arXiv:2506.06134](https://arxiv.org/abs/2506.06134) (v1, June 2025; published in *Neural Computation* 38(5):725–764, April 2026, DOI `10.1162/neco.a.1509`), Conjectures 1, 2, and 3 (`conj_full_row_rank`, `conj_full_strict_saddle`, `conj_lin_prog`).  
**Companion files:** [`lean_proofs/Conjectures/SimMatchingPCA.lean`](../../lean_proofs/Conjectures/SimMatchingPCA.lean), [`solutions/sim_matching_pca/explore.py`](explore.py), [`solutions/sim_matching_pca/proof.pdf`](proof.pdf).

---

## Summary of Verdicts

Let $C := C_X \in \mathbb{R}^{n \times n}$ be symmetric positive definite ($C \succ 0$), $1 \le m \le n$, $\mathcal{R}_{m,n} := \{W \in \mathbb{R}^{m \times n} : \operatorname{rank}(W) = m\}$, and consider the similarity-matching cost and continuous-time gradient flow on $\mathcal{R}_{m,n}$:
$$
S_3(W) = 2\|W\|_F^2 - 3\operatorname{tr}\!\bigl((WCW^\top)^{2/3}\bigr),
\qquad
\dot{W} = -\nabla S_3(W) = -4W + 4(WCW^\top)^{-1/3}WC.
$$

1. **Conjecture 1 (`conj_full_row_rank` — Full-rank invariance): PROVED.**
   If $W_0 \in \mathcal{R}_{m,n}$, the unique solution $W(t)$ exists for all $t \ge 0$, remains in $\mathcal{R}_{m,n}$, and satisfies the explicit uniform singular-value envelope
   $$
   \min\{\sigma_{\min}(W_0),\, \lambda_{\min}(C)\}
   \le \sigma_{\min}(W(t))
   \le \sigma_{\max}(W(t))
   \le \max\{\sigma_{\max}(W_0),\, \lambda_{\max}(C)\}
   \qquad (\forall t \ge 0).
   $$
   (For $m = n$, $\mathcal{R}_{n,n} = \mathrm{GL}_n(\mathbb{R})$ and the same proof applies; for $m > n$, $\mathcal{R}_{m,n} = \varnothing$.)

2. **Conjecture 2 (`conj_full_strict_saddle` — Global minima and strict saddles): INTENDED STATEMENT PROVED (printed set has a matrix-permutation typo).**
   - *Typo in arXiv v1:* The printed stationary set $\mathcal{S}$ uses $(P V_C)_m^\top$ (permuting rows of $V_C$), whereas the paper's own derivation gives $V_W = V_C P^\top$ (permuting columns). A $1 \times 2$ witness in the printed set has nonzero gradient norm $4(2^{2/3} - 1) \approx 2.3496$. Additionally, if $V_C$ denotes a single fixed eigenbasis, cutoff eigenvalue ties ($\lambda_m = \lambda_{m+1}$) produce continuous manifolds of global minima outside a single basis choice.
   - *Corrected theorem:* The full-rank stationary points of $S_3$ are precisely $W = U \Sigma V^\top$ where $U \in O_m$, $V \in \mathbb{R}^{n \times m}$ has orthonormal columns spanning a $C$-invariant $m$-subspace with $CV = V\Sigma$, and $\Sigma \succ 0$ is diagonal. Every global minimum of $S_3$ on $\mathbb{R}^{m \times n}$ has full row rank (in fact, no rank-deficient matrix is even a local minimum), achieved precisely when the columns of $V$ span a top-$m$ eigenspace of $C$ ($\beta \le \alpha$, where $\alpha = \min_i \Sigma_{ii}$ and $\beta = \lambda_{\max}(C|_{V^\perp})$). Every other full-rank stationary point ($\beta > \alpha$) is a strict saddle with an explicit negative Hessian direction of curvature $4(1 - \beta/\alpha) < 0$ and a positive radial curvature $8/3 > 0$.
   - *Prior-art note:* In substance, the landscape structure of $S_3$ connects closely to Pehlevan–Sengupta–Chklovskii ([arXiv:1703.07914v2](https://arxiv.org/abs/1703.07914), Eq. 38 and Theorem 1 / Eq. 65) and Lipshutz–Lipshutz ([arXiv:2601.13170](https://arxiv.org/abs/2601.13170), Lemma 1). What is added here is the rank-deficient exclusion proof, arbitrary-multiplicity handling, direct strict-saddle Hessian calculation on $S_3$, and identification of the printed-set typo.

3. **Conjecture 3 (`conj_lin_prog` — Orthogonal diagonalization): FALSE; exact iff proved.**
   Assume $\Lambda_1 \in \mathbb{R}^{m \times n}$ is diagonal with nonzero diagonal entries $a_1, \dots, a_m$ (required so $(\Lambda_1 U \Lambda_2 U^\top \Lambda_1^\top)^{-1/3}$ is defined), $\Lambda_2 \succ 0$ is diagonal with distinct entries, and $U \in O_n$.
   - *Exact criterion:* $H := U^\top \Lambda_1^\top (\Lambda_1 U \Lambda_2 U^\top \Lambda_1^\top)^{-1/3} \Lambda_1 U$ is diagonal **if and only if** $U^\top (\Lambda_1^\top \Lambda_1) U$ is diagonal — $\Lambda_2$ drops out completely!
   - *Decisive refutation (kernel rotation):* Whenever $n - m \ge 2$, taking $U = \operatorname{diag}(I_m, R)$ for any non-monomial orthogonal $R \in O_{n-m}$ yields a strictly diagonal $H$ for **every** $\Lambda_1$ and $\Lambda_2$. This refutes Conjecture 3 across all five $(m, n)$ test dimensions in the paper (which random sampling missed because block-diagonal rotations have measure zero).
   - *Exact signed-permutation criterion:* $U$ is forced to be a signed permutation matrix if and only if $|a_1|, \dots, |a_m|$ are pairwise distinct and $n - m \le 1$ (and even then, sign flips prevent forcing a pure permutation matrix).
   - *Survival of the paper's alternative theorem:* Although Conjecture 3 fails, the paper's trace-maximization lemma (`lem:pita`) still holds unconditionally by Horn's log-majorization, so the paper's alternative minimizer characterization survives by a different proof.

---

## 1. Proof of Conjecture 1 (Full-Rank Invariance and Forward Completeness)

Let $c := \lambda_{\min}(C) > 0$ and $d := \lambda_{\max}(C) \ge c$, so $c I_n \preceq C \preceq d I_n$. On the open set $\mathcal{R}_{m,n}$, the matrix $A(W) := WCW^\top$ is symmetric positive definite, and $A \mapsto A^{-1/3}$ is real-analytic on $\mathbb{S}^m_{++}$; hence the vector field $F(W) = -4W + 4A(W)^{-1/3}WC$ is real-analytic on $\mathcal{R}_{m,n}$. Given $W_0 \in \mathcal{R}_{m,n}$, there is a unique maximal solution $W : [0, T_{\max}) \to \mathcal{R}_{m,n}$.

Set $G(t) := W(t)W(t)^\top \succ 0$ and $A(t) := W(t)CW(t)^\top \succ 0$. Differentiating $G(t)$ along $\dot{W} = -4W + 4A^{-1/3}WC$ gives
$$
\dot{G} = \dot{W}W^\top + W\dot{W}^\top = -8G + 4A^{-1/3}A + 4AA^{-1/3} = -8G + 8A^{2/3}.
$$
Let $\ell(t) := \lambda_{\min}(G(t)) > 0$ and $L(t) := \lambda_{\max}(G(t)) > 0$. Since $cI_n \preceq C \preceq dI_n$, for every $x \in \mathbb{R}^m$ we have
$$
c\ell(t)\|x\|^2 \le c x^\top G(t) x \le x^\top A(t) x \le d x^\top G(t) x \le d L(t)\|x\|^2,
$$
so $c\ell(t) I_m \preceq A(t) \preceq dL(t) I_m$. By spectral mapping of the increasing function $s \mapsto s^{2/3}$ on $(0,\infty)$ (no Löwner–Heinz needed, since the bounds are scalar multiples of $I_m$),
$$
(c\ell(t))^{2/3} I_m \preceq A(t)^{2/3} \preceq (dL(t))^{2/3} I_m.
$$

**Lower bound via strict sublevels and Weyl's inequality.** Fix any $b$ with $0 < b < \min\{\ell(0), c^2\}$, and suppose for contradiction that $\ell(t)$ reaches $b$ in $[0, T_{\max})$. Let $t_* := \inf\{t \in [0, T_{\max}) : \ell(t) \le b\}$. By continuity, $t_* > 0$, $\ell(t_*) = b < c^2$, and $\ell(t) > b$ for all $t \in [0, t_*)$.

For small $h > 0$ (with $t_* - h \ge 0$ and $h < 1/8$), Taylor expansion at $t_*$ yields
$$
G(t_* - h) = G(t_*) - h \dot{G}(t_*) + o(h) = (1 + 8h)G(t_*) - 8h A(t_*)^{2/3} + o(h).
$$
Wait: let's evaluate backward cleanly! Better: let $v$ be a unit eigenvector of $G(t_*)$ with eigenvalue $\ell(t_*) = b$. Then the smooth scalar function $\phi(t) := v^\top G(t) v$ satisfies $\phi(t) \ge \ell(t)$ for all $t$, $\phi(t_*) = b$, and
$$
\phi'(t_*) = v^\top \dot{G}(t_*) v = -8b + 8 v^\top A(t_*)^{2/3} v \ge 8\bigl((cb)^{2/3} - b\bigr) > 0,
$$
because $b < c^2 \iff b^3 < c^2 b^2 = (cb)^2 \iff b < (cb)^{2/3}$.
(Equivalently, without using a single eigenvector, for $0 < h < 1/8$, $(1-8h)G(t) + 8hA(t)^{2/3} \succeq [(1-8h)\ell(t) + 8h(c\ell(t))^{2/3}]I_m$, so Weyl's inequality gives the lower right Dini derivative $D_+\ell(t) \ge 8((c\ell(t))^{2/3} - \ell(t))$ even at repeated eigenvalues.)
Because $\phi(t_*) = b$ and $\phi'(t_*) > 0$, for sufficiently small $h > 0$ we have $\ell(t_* - h) \le \phi(t_* - h) < b$, contradicting the definition of $t_*$. Hence $\ell(t) > b$ for all $t \in [0, T_{\max})$. Letting $b \uparrow \min\{\ell(0), c^2\}$ gives $\ell(t) \ge \min\{\ell(0), c^2\}$.

**Upper bound.** Symmetrically, fix any $B > \max\{L(0), d^2\}$, and let $t^* := \inf\{t \in [0, T_{\max}) : L(t) \ge B\}$. For a unit eigenvector $v$ of $G(t^*)$ with $G(t^*)v = Bv$, $\psi(t) := v^\top G(t) v \le L(t)$ satisfies $\psi(t^*) = B$ and
$$
\psi'(t^*) = -8B + 8 v^\top A(t^*)^{2/3} v \le 8\bigl((dB)^{2/3} - B\bigr) < 0,
$$
since $B > d^2 \iff B > (dB)^{2/3}$. Thus for small $h > 0$, $L(t^* - h) \ge \psi(t^* - h) > B$, contradicting minimality of $t^*$. Letting $B \downarrow \max\{L(0), d^2\}$ yields $L(t) \le \max\{L(0), d^2\}$.

**Global existence ($T_{\max} = \infty$).** Taking square roots gives the singular-value bounds $\alpha_0 \le \sigma_{\min}(W(t)) \le \sigma_{\max}(W(t)) \le \beta_0$ with $\alpha_0 = \min\{\sigma_{\min}(W_0), c\} > 0$ and $\beta_0 = \max\{\sigma_{\max}(W_0), d\}$. The set $\{W \in \mathbb{R}^{m \times n} : \alpha_0 \le \sigma_{\min}(W) \le \sigma_{\max}(W) \le \beta_0\}$ is compact and contained in the domain of real-analyticity $\mathcal{R}_{m,n}$. By standard ODE escape-from-compacts theory, $T_{\max} = \infty$. $\square$

---

## 2. Proof of Conjecture 2 (Corrected Landscape Theorem and Typo Refutation)

### 2.1 Printed-set typo counterexample (`arXiv:2506.06134` v1)
In `main.tex` (line 645), the printed stationary set writes $(P V_C)_m^\top$ instead of $(V_C P^\top)_m^\top$ (compare `main.tex`, line 972, which derives $V_W = V_C P^\top$). For $m = 1, n = 2$, $V_C = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1\\ 1 & -1\end{pmatrix}$, $D = \operatorname{diag}(2, 1)$, $C = V_C D V_C^\top = \frac{1}{2}\begin{pmatrix}3 & 1\\ 1 & 3\end{pmatrix}$, and $P = \begin{pmatrix}0 & 1\\ 1 & 0\end{pmatrix}$, the printed formula gives
$$
W = (P D P^\top)_{1,1}\bigl((P V_C)_{:,1}\bigr)^\top = \frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1\end{pmatrix}.
$$
Then $WW^\top = 1$, $WC = 2W$, and $WCW^\top = 2$, so
$$
-\nabla S_3(W) = -4W + 4(2)^{-1/3}(2W) = 4(2^{2/3} - 1)W \neq 0,
$$
with Frobenius norm $4(2^{2/3} - 1) \approx 2.3496$. Thus $W$ is not even stationary.

### 2.2 Corrected full-rank stationary set
Let $W \in \mathcal{R}_{m,n}$ and $K := (WCW^\top)^{1/3} \succ 0$. Stationarity $\nabla S_3(W) = 0$ is equivalent to $KW = WC$. Right-multiplying by $W^\top$ gives $K(WW^\top) = WCW^\top = K^3$, hence $WW^\top = K^2$ and $K = (WW^\top)^{1/2}$.

Write a thin SVD $W = U \Sigma V^\top$ with $U \in O_m$, $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_m) \succ 0$, and $V \in \mathbb{R}^{n \times m}$ ($V^\top V = I_m$). Then $K = (WW^\top)^{1/2} = U \Sigma U^\top$, so $KW = WC \iff U \Sigma^2 V^\top = U \Sigma V^\top C \iff \Sigma V^\top = V^\top C \iff CV = V\Sigma$.
Thus the columns of $V$ are orthonormal eigenvectors of $C$ with eigenvalues $\sigma_1, \dots, \sigma_m$, and $S_3(W) = 2\sum_{i=1}^m \sigma_i^2 - 3\sum_{i=1}^m (\sigma_i^3)^{2/3} = -\sum_{i=1}^m \sigma_i^2$.

### 2.3 No rank-deficient matrix is a local minimum
Suppose $W \in \mathbb{R}^{m \times n}$ has $\operatorname{rank}(W) = r < m$. Choose a unit vector $u \in \ker(W^\top) \subset \mathbb{R}^m$ (so $u^\top W = 0$, existing because $r < m$) and a unit vector $v \in \ker(WC) \subset \mathbb{R}^n$ (existing because $\operatorname{rank}(WC) = r < m \le n$). Set $W_t := W + t u v^\top$ and $\gamma := v^\top C v > 0$ (since $C \succ 0$). Because $u^\top W = 0$ and $WCv = 0$:
$$
\|W_t\|_F^2 = \|W\|_F^2 + t^2, \qquad
W_t C W_t^\top = W C W^\top + t^2 \gamma\, u u^\top.
$$
Since $u \in \ker(W^\top) = \ker(WCW^\top)$, the rank-1 term $t^2 \gamma\, u u^\top$ acts orthogonally to the range of $WCW^\top$, so the nonzero spectrum of $W_t C W_t^\top$ is that of $WCW^\top$ together with the single new eigenvalue $t^2 \gamma$. Therefore
$$
S_3(W_t) - S_3(W) = 2t^2 - 3|t|^{4/3}\gamma^{2/3} < 0
\qquad \text{for all } 0 < |t| < \Bigl(\frac{3}{2}\Bigr)^{3/2}\gamma.
$$
Hence no rank-deficient point is a local minimum. Since $S_3(W) \ge 2\|W\|_F^2 - 3 d^{2/3} m^{1/3} \|W\|_F^{4/3} \to +\infty$ as $\|W\|_F \to \infty$, $S_3$ attains its global minimum on $\mathbb{R}^{m \times n}$, and every global minimum lies in $\mathcal{R}_{m,n}$ and therefore in the stationary set above.

### 2.4 Global minima vs. strict saddles
At a full-rank stationary point $W = U \Sigma V^\top$ with $CV = V\Sigma$, the orthogonal complement $V^\perp$ is $C$-invariant (as $C$ is symmetric). Let $\alpha := \min_{1 \le i \le m} \sigma_i$ and $\beta := \lambda_{\max}(C|_{V^\perp})$ (with $\beta = -\infty$ if $m = n$).
- **Global optimality $\iff \beta \le \alpha$:** Since $S_3(W) = -\sum_{i=1}^m \sigma_i^2$ at stationary points and the global minimum is attained at a stationary point, $W$ is a global minimum if and only if $\{\sigma_1, \dots, \sigma_m\}$ equals the top-$m$ eigenvalues of $C$ (with multiplicity), which holds if and only if $\beta \le \alpha$.
- **Strict saddle when $\beta > \alpha$:** Pick $i \in \{1, \dots, m\}$ with $\sigma_i = \alpha$, let $u_i$ be column $i$ of $U$, and pick a unit eigenvector $v_\perp \in V^\perp$ of $C$ with $C v_\perp = \beta v_\perp$. Along the perturbation $W_\varepsilon := W + \varepsilon u_i v_\perp^\top$, because $V^\top C v_\perp = \beta V^\top v_\perp = 0$, the matrix $W_\varepsilon C W_\varepsilon^\top = U \operatorname{diag}(\sigma_1^3, \dots, \sigma_i^3 + \varepsilon^2 \beta, \dots, \sigma_m^3) U^\top$ remains diagonal in the $U$ basis! Expanding $S_3(W_\varepsilon)$ in $\varepsilon$:
  $$
  S_3(W_\varepsilon) - S_3(W) = 2\varepsilon^2 - 3\Bigl((\alpha^3 + \varepsilon^2 \beta)^{2/3} - \alpha^2\Bigr)
  = \frac{1}{2} \cdot 4\Bigl(1 - \frac{\beta}{\alpha}\Bigr)\varepsilon^2 + O(\varepsilon^4).
  $$
  Thus $\frac{d^2}{d\varepsilon^2}\big|_{\varepsilon=0} S_3(W_\varepsilon) = 4(1 - \beta/\alpha) < 0$.
- **Positive curvature direction:** Along the radial perturbation $W_\varepsilon = (1 + \varepsilon)W$, $S_3(W_\varepsilon) = (2(1+\varepsilon)^2 - 3(1+\varepsilon)^{4/3})\sum \sigma_i^2$, so $\frac{d^2}{d\varepsilon^2}\big|_{\varepsilon=0} S_3((1+\varepsilon)W) = \frac{8}{3}\sum_{i=1}^m \sigma_i^2 > 0$ (and $\frac{8}{3}$ per unit-norm radial perturbation along a single mode $u_i v_i^\top$). Thus every non-global full-rank stationary point is a strict saddle (and never a local maximum). $\square$

---

## 3. Refutation and Exact Characterization of Conjecture 3

Let $\Lambda_1 = [\operatorname{diag}(a_1, \dots, a_m),\ 0] \in \mathbb{R}^{m \times n}$ with $a_i \neq 0$ for all $i = 1, \dots, m$ (if some $a_i = 0$, then $\Lambda_1 U \Lambda_2 U^\top \Lambda_1^\top$ is singular and its $-1/3$ power is undefined), let $\Lambda_2 = \operatorname{diag}(d_1, \dots, d_n) \succ 0$ have distinct diagonal entries, and let $U \in O_n$. Define
$$
H := U^\top \Lambda_1^\top \bigl(\Lambda_1 U \Lambda_2 U^\top \Lambda_1^\top\bigr)^{-1/3} \Lambda_1 U.
$$

**Theorem 3 (Exact diagonalization criterion).** *$H$ is diagonal if and only if $K_U := U^\top (\Lambda_1^\top \Lambda_1) U$ is diagonal. In particular, this condition is completely independent of $\Lambda_2$.*

*Proof.* Set $B := \Lambda_1 U \Lambda_2^{1/2} \in \mathbb{R}^{m \times n}$. Since $a_i \neq 0$ and $\Lambda_2 \succ 0$, $B$ has full row rank $m$. Let $B = P S Q^\top$ be a thin SVD ($P \in O_m$, $S \in \mathbb{R}^{m \times m}$ positive diagonal, $Q \in \mathbb{R}^{n \times m}$ with $Q^\top Q = I_m$). Then $BB^\top = P S^2 P^\top$ and $B^\top B = Q S^2 Q^\top$, so
$$
\Lambda_2^{1/2} H \Lambda_2^{1/2}
= B^\top (BB^\top)^{-1/3} B
= Q S P^\top (P S^{-2/3} P^\top) P S Q^\top
= Q S^{4/3} Q^\top
= (B^\top B)^{2/3}.
$$
(Equivalently, writing $K = (BB^\top)^{1/3}$ and $N = B^\top K^{-1} B$, we have the purely algebraic identity $N^3 = (B^\top B)^2$.)
Since $N = \Lambda_2^{1/2} H \Lambda_2^{1/2}$ and $B^\top B = \Lambda_2^{1/2} K_U \Lambda_2^{1/2}$ are symmetric positive semidefinite and the map $X \mapsto X^{2/3}$ is a bijection on $\mathbb{S}^n_+$ preserving diagonal matrices (with inverse $Y \mapsto Y^{3/2}$), we obtain:
$$
H \text{ diagonal}
\iff \Lambda_2^{1/2} H \Lambda_2^{1/2} \text{ diagonal}
\iff (B^\top B)^{2/3} \text{ diagonal}
\iff B^\top B \text{ diagonal}
\iff K_U = U^\top(\Lambda_1^\top \Lambda_1)U \text{ diagonal}. \quad \square
$$

**Corollary (Counterexamples and exact signed-permutation iff).**
1. *Kernel rotation ($n - m \ge 2$):* $\Lambda_1^\top \Lambda_1 = \operatorname{diag}(a_1^2, \dots, a_m^2, 0, \dots, 0)$ has a kernel of dimension $n - m \ge 2$. Taking $U = \operatorname{diag}(I_m, R)$ for any non-monomial $R \in O_{n-m}$ leaves $K_U = \Lambda_1^\top \Lambda_1$ diagonal, so $H$ is strictly diagonal for **every** $\Lambda_1$ and $\Lambda_2$.
2. *Active ties and sign flips:* Any rotation within a block of equal $|a_i|$ leaves $K_U$ diagonal, and any diagonal sign matrix $U = \operatorname{diag}(\pm 1, \dots, \pm 1)$ leaves $K_U$ diagonal.
3. *Exact iff:* $H$ diagonal forces $U$ to be a signed permutation matrix **if and only if** $|a_1|, \dots, |a_m|$ are pairwise distinct and $n - m \le 1$ (since those are precisely the cases in which $\Lambda_1^\top \Lambda_1$ has simple spectrum, forcing the columns of $U^\top$ to be signed standard basis vectors).

**Survival of the paper's trace-maximization lemma (`lem:pita`).** Even though Conjecture 3 fails, the paper's trace bound $\operatorname{tr}((\Lambda_1 U \Lambda_2 U^\top \Lambda_1^\top)^{2/3}) \le \sum_{i=1}^m (a_i^2 d_{(i)})^{2/3}$ still holds unconditionally via Horn's log-majorization (the singular values of $\Lambda_1 U \Lambda_2^{1/2}$ are weakly log-majorized by the ordered products $|a|^\downarrow \odot (d^{1/2})^\downarrow$, and $x \mapsto \sum x_i^{4/3}$ is Schur-convex on $\mathbb{R}_{\ge 0}^m$). Thus the paper's alternative minimizer characterization survives independently of Conjecture 3.

---

## 4. Verification, Formalization, and Attribution

- **Lean 4 formalization** ([`lean_proofs/Conjectures/SimMatchingPCA.lean`](../../lean_proofs/Conjectures/SimMatchingPCA.lean), Lean `v4.24.0` + Mathlib, zero `sorry`, standard axioms only): machine-checks the Gram ODE identity (`gram_ode_identity`), strict barrier drift signs (`lower_barrier_pos`, `upper_barrier_neg`), stationarity (`stationary_implies_gram_eq_sq`), the printed-set counterexample (`printed_set_not_stationary`), rank-1 escape and strict-saddle curvatures (`rank_one_escape_negative`, `exchange_curvature_neg`, `radial_curvature_pos`), and the Conjecture 3 cube identity and counterexamples (`conj3_cube_identity`, `conj3_kernel_rotation_counterexample`, `conj3_active_tie_counterexample`).
- **Python verification** ([`solutions/sim_matching_pca/explore.py`](explore.py)): verifies the Gram ODE residual, singular-value envelope across hard trajectories (down to $\sigma_{\min}(W_0) = 10^{-6}$), printed-set gradient norm, cutoff ties, 200 random critical-point Hessians, Conjecture 3 kernel rotations up to $(10, 50)$, 500 random trials of the exact iff, and 2,000 random tests of the Horn log-majorization bound.
- **Literature note:** arXiv:2506.06134 v1 was published in *Neural Computation* 38(5):725–764 (April 2026); the publisher PDF returned HTTP 403 and could not be checked directly, so "as printed" refers to arXiv v1. For Conjecture 2, Pehlevan–Sengupta–Chklovskii ([arXiv:1703.07914v2](https://arxiv.org/abs/1703.07914)) and Lipshutz–Lipshutz ([arXiv:2601.13170](https://arxiv.org/abs/2601.13170)) contain closely related landscape results and critical-point characterizations; Conjecture 1 and the Conjecture 3 refutation appear new as of November 2026.
