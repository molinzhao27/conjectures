# Refutation of Three Conjectures on Group-Averaged Spectral Estimation

**Target paper:** Mitchell A. Thornton, *Algebraic Diversity: Group-Theoretic Spectral Estimation from Single Observations*, [arXiv:2604.03634](https://arxiv.org/abs/2604.03634) (local v17 draft, June 2026), Conjectures `conj:blind`, `conj:nadh`, and `conj:gaat`.  
**Companion files:** [`solutions/group_spectral_est/explore.py`](explore.py), [`solutions/group_spectral_est/proof.pdf`](proof.pdf).

---

## Verdict

All three conjectures in the local v17 draft of arXiv:2604.03634 are **FALSE as stated**:

- **Conjecture `conj:blind` (Blind Group Selection): FALSE**, even when all candidate groups are transitive and regular on $\{1, \dots, M\}$ (or even conjugate cyclic groups) and the signal comes from a polynomial graph filter.
  - With $G = \mathbb{Z}_4$ and $H = V_4$ (both order-$4$ regular transitive groups), the blind selector chooses the mismatched group $H$ with probability tending to **$1$** as $\mathrm{SNR} \to \infty$ (agreement probability $\to 0$).
  - With two conjugate cyclic groups $G = \langle(1\,2\,3\,4)\rangle$ and $K = \langle(1\,3\,2\,4)\rangle$ and the polynomial graph-filtered Gaussian signal $y = (2I - A)w$ on $C_4$, the blind selector chooses the wrong group with limiting probability strictly bounded away from $0$, refuting $\Pr \to 1$. The same failure occurs under the population-twirl oracle $[T_G(R), R]$.
- **Conjecture `conj:nadh` (Non-Abelian Dominance Hypothesis, refined): FALSE as printed.**
  - Because the printed comparator places $\max_{U \in U(n)}$ *inside* the expectation, $\mathbb{E}[\max_{U \in U(n)} \psi(U^H \mathbb{Z}_n U, x)]$, choosing $U \in U(n)$ per observation so that $U(x/\|x\|) = \mathbf{1}/\sqrt{n}$ makes $U^H \mathbb{Z}_n U$ fix $x$ and achieves $\psi = 1$ pointwise (hence expectation $1 \ge \mathbb{E}[\psi(G, x)]$).
  - On the 6-vertex 3-sunlet graph (triangle with one pendant leaf per vertex, $\mathrm{Aut}(\mathcal{G}) \cong S_3$ of order $6$) with polynomial filter $h(A) = 3I + A$, both spectral premises of `conj:nadh` hold while $\psi(S_3, x) < 1$ almost surely, reversing the claimed strict inequality.
  - **Scope caveat:** This refutation relies on the printed quantifier order $\mathbb{E}[\max_U \psi(\dots)]$ (which matches the author's per-trial random-unitary sweep in §13.4 Test A of v17). The alternative **fixed-$U$ variant** placing the maximum *outside* the expectation, $\max_{U \in U(n)} \mathbb{E}[\psi(U^H \mathbb{Z}_n U, x)]$, is **not** refuted by this argument and remains open.
- **Conjecture `conj:gaat` (General Algebraic Averaging): FALSE as stated**, due to both omitted regularity/linearity hypotheses and structural ill-posedness of $d_{\mathrm{eff}}(G, f)$:
  - *Omitted regularity & noise-dependence:* $f(z) = \exp(|z|^2/2)$ has finite mean $2$ and infinite variance at $\sigma = 1$ (refuting (ii)–(iv), and likewise on the 6-vertex graph for any deterministic $s$ via $f(x) = \exp(\|x\|^2/2)$); $f(z) = \exp(|z|^2)$ lacks an expectation (refuting (i)); $f(z) = |z|^2$ has $\mathrm{Var}(f) = \sigma^4$, refuting a $\sigma$-independent constant $C(f, \mathbf{s})$; and a nonlinear output involution refutes (i) unless $\tilde{\pi}$ is linear.
  - *Structural defect:* Definition `def:deff` provides no general formula for $d_{\mathrm{eff}}(G, f)$ outside the outer product, and any variance-ratio definition $\mathrm{Var}(Y)/\mathrm{Var}(PY)$ violates the stated orbit-count upper bound (e.g., $f(z) = z$ under $\mathbb{Z}_2$).
  - *Positive fixed-$\sigma$ covariance bound (Proposition 5):* Under linear unitary $\tilde{\pi}$ and square-integrable $f$, (i), (iii) given (ii), and (iv) hold; for the outer product $f(x) = xx^*$ at fixed $\sigma$, we prove the exact bound $\mathrm{Var}(F_G(x)) \le M^2(\sigma^4 + 2\sigma^2\|s\|^2)/d_{\mathrm{eff}}$ (with equality at $s = 0$).

---

## 1. Definitions and Oracle Conventions

For a finite unitary representation $\pi : G \to U(M)$, define the group twirl and single-observation estimator
$$
T_G(B) := \frac{1}{|G|} \sum_{g \in G} \pi_g B \pi_g^*, \qquad F_G(x) := T_G(xx^*).
$$
Following `def:commut_residual` and Eq. (953) of the v17 draft, the sample commutativity residual and spectral concentration score are
$$
\delta(G, x, R) := \frac{\|[F_G(x), R]\|_F}{\|F_G(x)\|_F \|R\|_F},
\qquad \psi(G, x) := \frac{\lambda_{\max}(F_G(x))}{\|x\|^2},
$$
where $\operatorname{tr}(F_G(x)) = \|x\|^2$, so $0 < \psi(G, x) \le 1$ for every $x \ne 0$. Because `conj:blind` abbreviates the residual as $\delta(G, R)$, we establish counterexamples under all natural oracle readings: the sample residual $\delta(G, x, R)$, its expectation $\mathbb{E}[\delta(G, x, R)]$, exact group commutation $[\pi_g, R] = 0$, and the population-twirl commutator $[T_G(R), R]$.

---

## 2. Disproof of Blind Group Selection (`conj:blind`)

**Theorem 1 (Failure with two regular transitive groups of order 4).** *Let $G = \langle P \rangle \cong \mathbb{Z}_4$ where $P = (1\,2\,3\,4)$ acts by $P e_j = e_{P(j)}$, and let $H \cong V_4$ consist of the identity and the three double transpositions on $\{1, 2, 3, 4\}$. Both groups act regularly and transitively on $\{1, 2, 3, 4\}$. Define*
$$
u := \frac{1}{2}(1, 1, -1, -1)^\top, \qquad v := Pu = \frac{1}{2}(-1, 1, 1, -1)^\top, \qquad w_\pm := \frac{u \pm i v}{\sqrt{2}}.
$$
*Let $\epsilon \in \{\pm 1\}$ be uniform, $\eta \in S^1 \subset \mathbb{C}$ be an independent uniform phase, and $z \sim \mathcal{CN}(0, I_4)$ be independent noise. For $s := \eta(2w_+ + \epsilon w_-)$ and $x_\sigma := s + \sigma z$ with covariance $R_\sigma = R_0 + \sigma^2 I_4$ ($R_0 = 4 w_+ w_+^* + w_- w_-^*$):*
1. *$G$ is the unique oracle optimum under $\delta(G, x_\sigma, R_\sigma)$, under $\mathbb{E}[\delta(G, x_\sigma, R_\sigma)]$, and under elementwise commutation with $R_\sigma$.*
2. *As $\sigma \downarrow 0$ ($\mathrm{SNR} \to \infty$), $\Pr[\arg\max_{J \in \{G, H\}} \psi(J, x_\sigma) = G] \to 0$.*

*Proof.* On $\operatorname{span}\{u, v\}$, $Pu = v$ and $Pv = -u$, so $w_\pm$ are orthonormal eigenvectors of $P$ with distinct eigenvalues $\mp i$ (all four characters of $G$ are distinct). The vectors $u, v$ are eigenvectors of $H$ with distinct characters. In a character basis of any finite Abelian group with distinct 1D characters, twirling deletes all off-diagonal entries: $F_J(x) = \sum_k |\langle e_k, x\rangle|^2 e_k e_k^*$.

For $G$, both $F_G(x)$ and $R_\sigma$ are diagonal in $G$'s character basis for every $x$, so $\delta(G, x_\sigma, R_\sigma) \equiv 0$, and $F_G(s)$ has nonzero eigenvalues $4, 1$. For $H$, the squared components of $s$ along $u, v$ are $(2+\epsilon)^2/2$ and $(2-\epsilon)^2/2$, so for either sign $\epsilon \in \{\pm 1\}$, $F_H(s)$ has nonzero eigenvalues $9/2$ and $1/2$. Therefore,
$$
\psi(G, s) = \frac{4}{5} < \frac{9}{10} = \psi(H, s).
$$
In the $(u, v)$ basis, $R_0$ has off-diagonal entries of modulus $3/2$, giving $\|[F_H(s), R_0]\|_F = 6\sqrt{2}$, $\|F_H(s)\|_F^2 = 41/2$, $\|R_0\|_F^2 = 17$, and hence $\delta(H, s, R_0) = 12/\sqrt{697} > 0$. Moreover, a double transposition with opposite signs on $u$ and $v$ fails to commute with $R_0$.

Coupling all $\sigma > 0$ on the same $(s, z)$ gives $x_\sigma \to s$ almost surely. By continuity of $\psi$ and $\delta$ at $(s, R_0)$, the strict opposite rankings $\delta(G, x_\sigma, R_\sigma) < \delta(H, x_\sigma, R_\sigma)$ and $\psi(G, x_\sigma) < \psi(H, x_\sigma)$ hold eventually on almost every draw. Bounded convergence yields $\Pr[\text{agreement}] \to 0$ (regardless of tie breaking), and since $\delta \le 2$, $\mathbb{E}[\delta(H, x_\sigma, R_\sigma)] \to 12/\sqrt{697} > 0 = \mathbb{E}[\delta(G, x_\sigma, R_\sigma)]$. $\square$

**Proposition 2 (Failure with conjugate cyclic groups and a graph-filtered Gaussian signal).** *Keep $G = \langle(1\,2\,3\,4)\rangle$, let $K = \langle(1\,3\,2\,4)\rangle$ be its conjugate cyclic group, and let $A = P + P^\top$ be the $C_4$ adjacency matrix. For the polynomial graph-filtered signal $y = (2I - A)w$ ($w \sim \mathcal{CN}(0, I_4)$, covariance $R_0 = (2I - A)^2$) and noisy observations $x_\sigma = y + \sigma z$ ($R_\sigma = R_0 + \sigma^2 I_4$), $\liminf_{\sigma \downarrow 0} \Pr[\arg\max_{J \in \{G, K\}} \psi(J, x_\sigma) \ne G] \ge p > 0$.*

*Proof.* Let $b := \frac{1}{2}(1, -1, 1, -1)^\top$. The filter $2I - A$ vanishes on $\mathbf{1}$ and has eigenvalues $2, 2, 4$ on $u, v, b$, so $y$ has full positive Gaussian density on $\operatorname{span}_{\mathbb{C}}\{u, v, b\}$ with $R_0$ eigenvalues $4, 4, 16$. At the support point $y_0 := 3u + v + i b$, the $G$-Fourier squared magnitudes are $5, 5, 1$, giving $\psi(G, y_0) = 5/11$ and $[F_G(y_0), R_0] = 0$. Under $K$'s generator $P_K$, $P_K u = -u$, $P_K v = -b$, $P_K b = v$, so $v + i b$ is a character vector of $K$ with eigenvalue $+i$. Hence $F_K(y_0)$ has nonzero eigenvalues $9, 2$, yielding $\psi(K, y_0) = 9/11 > 5/11 = \psi(G, y_0)$, while its $(v, b)$ block $(v + ib)(v + ib)^*$ has a nonzero off-diagonal entry coupling the distinct $R_0$ eigenvalues $4$ and $16$, so $\delta(K, y_0, R_0) > 0 = \delta(G, y_0, R_0)$. Both strict inequalities persist on an open relative neighborhood of $y_0$ of probability $p > 0$; Fatou's lemma as $\sigma \downarrow 0$ gives limiting error probability at least $p > 0$. $\square$

**Proposition 3 (Population-twirl oracle $[T_J(R), R]$).** *For the same conjugate cyclic pair $\{G, K\}$ and $s = \eta a$ with $a = (2, 1, -2, -1)^\top$, $R = aa^* + \sigma^2 I_4$, we have $T_J(R) = F_J(a) + \sigma^2 I_4$. In $G$'s cycle order, $a$ has equal nonzero Fourier powers $5, 5$, so $F_G(a) a = 5a$ and $[T_G(R), R] = 0$. In $K$'s cycle order, $(2, -2, 1, -1)^\top$ has unequal nonzero Fourier powers $9, 1/2, 1/2$, so $a$ is not an eigenvector of $F_K(a)$ and $[T_K(R), R] \ne 0$. Yet $\psi(G, a) = 1/2 < 9/10 = \psi(K, a)$, so agreement probability tends to $0$ as $\sigma \downarrow 0$.*

---

## 3. Disproof of the Non-Abelian Dominance Hypothesis (`conj:nadh`)

**Theorem 2 (Disproof of `conj:nadh` as printed).** *For every $n \ge 1$ and every $x \in \mathbb{C}^n \setminus \{0\}$,*
$$
\max_{U \in U(n)} \psi(U^H \mathbb{Z}_n U, x) = 1.
$$
*Consequently, for any graph $\mathcal{G}$ and any observation law with $\Pr[x = 0] = 0$,*
$$
\mathbb{E}\bigl[\psi(G, x)\bigr] \le 1 = \mathbb{E}\!\left[\max_{U \in U(n)} \psi(U^H \mathbb{Z}_n U, x)\right].
$$
*Moreover, the 6-vertex 3-sunlet graph (a triangle with one pendant leaf attached to each vertex) has non-Abelian automorphism group $G = \mathrm{Aut}(\mathcal{G}) \cong S_3$ of order $6 = n$, and under the polynomial filter $h(A) = 3I + A$ ($x = h(A)w + n$, $w \sim \mathcal{CN}(0, I_6)$, independent white noise $n$), the covariance satisfies both spectral premises of `conj:nadh` while $\mathbb{E}[\psi(S_3, x)] < 1$, strictly reversing the conjectured inequality.*

*Proof.* Given $x \ne 0$, pick $U \in U(n)$ mapping $x / \|x\|$ to $\mathbf{1}/\sqrt{n}$. Because every cyclic shift fixes $\mathbf{1}/\sqrt{n}$, every element of $U^H \mathbb{Z}_n U$ fixes $x$, so $F_{U^H \mathbb{Z}_n U}(x) = xx^*$ and $\psi(U^H \mathbb{Z}_n U, x) = 1$. Since $\psi \le 1$ universally, the pointwise maximum equals $1$.

For the 3-sunlet graph, vertex degrees ($3$ on the triangle, $1$ on the leaves) are preserved by automorphisms and each triangle permutation uniquely determines the leaf permutation, so $\mathrm{Aut}(\mathcal{G}) \cong S_3$ (non-Abelian, order $6$). Its adjacency matrix is
$$
A = \begin{pmatrix} J_3 - I_3 & I_3 \\ I_3 & 0 \end{pmatrix}.
$$
Since $J_3 - I_3$ has eigenvalue $2$ on $\mathbf{1}_3$ and $-1$ on $\mathbf{1}_3^\perp$, eigenvectors $(\lambda q, q)$ of $A$ satisfy $\lambda^2 - \mu\lambda - 1 = 0$ ($\mu \in \{2, -1\}$), giving simple eigenvalues $1 \pm \sqrt{2}$ and two multiplicity-$2$ eigenvalues $(-1 \pm \sqrt{5})/2$. On each 2D eigenspace, a 3-cycle in $S_3$ has distinct eigenvalues $e^{\pm 2\pi i / 3}$ whose eigenlines are swapped by any transposition; hence both 2D eigenspaces carry the irreducible standard $S_3$ representation. Because every adjacency eigenvalue exceeds $-3$, the filter $h(A) = 3I + A$ yields a positive definite covariance $h(A)^2 + \sigma^2 I_6$ preserving all eigenspaces and strict eigenvalue order, with simple top eigenvalue $(4 + \sqrt{2})^2 + \sigma^2$. Thus both spectral premises of `conj:nadh` hold. Finally, $\psi(S_3, x) = 1$ requires all orbit vectors $\{g x : g \in S_3\}$ to be collinear, forcing $x$ to lie in an eigenspace of a nonidentity transposition; this proper union of subspaces has probability zero under the full-rank Gaussian law, so $\psi(S_3, x) < 1$ almost surely and $\mathbb{E}[\psi(S_3, x)] < 1$. $\square$

**Important scope remark (quantifier order):** Theorem 2 refutes `conj:nadh` verbatim, with $\max_{U}$ inside the expectation (matching §13.4 Test A of the v17 draft). If the maximum is instead placed *outside* the expectation—comparing $\mathbb{E}[\psi(G, x)]$ against $\max_{U \in U(n)} \mathbb{E}[\psi(U^H \mathbb{Z}_n U, x)]$—then the pointwise construction above no longer applies, and the fixed-$U$ variant remains open.

---

## 4. Disproof and Repair of General Algebraic Averaging (`conj:gaat`)

In `conj:gaat`, $x = s + n \in \mathbb{C}^M$ with deterministic $s$ and $n \sim \mathcal{CN}(0, \sigma^2 I_M)$, $f(\pi_g x) = \tilde{\pi}_g f(x)$, and $\hat{\theta}_G = |G|^{-1} \sum_{g \in G} f(\pi_g x)$.

### 4.1 Counterexamples to the printed statement

1. **Finite mean, infinite variance (fixed noise $\sigma = 1$):** Take $M = 1$, $s = 0$, $\sigma = 1$, $G = \{e\}$, and $f(z) = \exp(|z|^2/2)$ (also invariant under any finite phase group). Since $T = |z|^2 \sim \mathrm{Exp}(1)$,
   $$
   \mathbb{E}[f] = \int_0^\infty e^{-t/2}\,dt = 2, \qquad \mathbb{E}[f^2] = \int_0^\infty 1\,dt = \infty.
   $$
   Here $\hat{\theta}_G = f$ and $d_{\mathrm{eff}} = 1$, so no finite $C(f, 0)$ satisfies (ii). For any finite $L \ge 1$, the i.i.d. average $\hat{\theta}_{G, L} \ge f(x_1)/L$ still has infinite variance, refuting (iii) and the variance claim in (iv). The exact same divergence holds on the 6-vertex graph for any deterministic $s \in \mathbb{C}^6$ using the $\mathrm{Aut}(\mathcal{G})$-invariant statistic $f(x) = \exp(\|x\|^2/2)$ (completing the square gives finite mean, whereas in $\mathbb{E}[f^2]$ the quadratic decay cancels, leaving $\int_{\mathbb{C}^6} c\exp(2\operatorname{Re} x^* s)\,dx = \infty$). Without integrability, $f(z) = \exp(|z|^2)$ even lacks an expectation, refuting (i).
2. **Noise-level dependence of $C(f, s)$:** For the polynomial outer product $f(z) = |z|^2$ ($M = 1, s = 0, G = \{e\}, d_{\mathrm{eff}} = 1$), $\mathrm{Var}(f) = \sigma^4$, which is unbounded as $\sigma$ varies, refuting any constant $C(f, s)$ independent of $\sigma$.
3. **Nonlinearity of the output action $\tilde{\pi}$:** Take $G = \{\pm 1\}$ acting by sign flip on $\mathbb{C}$, $s = 0$, $\sigma = 1$, and $f(z) = h(\operatorname{Re} z)$ where $h(t) = t + \max(t, 0)$ is a bijection $\mathbb{R} \to \mathbb{R}$. Under the nonlinear involution $\tilde{\pi}_{-1}(y) := h(-h^{-1}(y))$, $f$ is equivariant with invariant set $\{0\}$, yet $\hat{\theta}_G(z) = |\operatorname{Re} z|/2$ has strictly positive mean, refuting (i).
4. **Structural ill-posedness of $d_{\mathrm{eff}}(G, f)$:** Definition `def:deff` provides an explicit formula only for $f(x) = xx^*$ ($d_{\mathrm{eff}} = M^2 / \dim_{\mathbb{C}} \mathcal{C}(\pi)$) and states that the distinct orbit-value count upper-bounds $d_{\mathrm{eff}}$. However, for a variance-ratio definition $\mathrm{Var}(Y)/\mathrm{Var}(PY)$, the linear statistic $f(z) = z$ under $\mathbb{Z}_2$ has orbit size $\le 2$ while $PY \equiv 0$ (infinite variance reduction).

### 4.2 Positive results under regularity hypotheses

**Proposition 4 (Linear unitary actions and finite moments).** *If $\tilde{\pi}$ is a unitary representation on a finite-dimensional inner-product space $\mathcal{V}$, then $P := |G|^{-1}\sum_{g \in G} \tilde{\pi}_g$ is the orthogonal projector onto $\mathcal{V}^G$ and $\hat{\theta}_G = P f(x)$. Hence (i) holds for integrable $f$. If $Y = f(x)$ is square-integrable with covariance operator $\Sigma_Y$, then $\mathrm{Var}(\hat{\theta}_G) = \operatorname{tr}(P \Sigma_Y P^*) \le \operatorname{tr}(\Sigma_Y)$, and for $L$ independent snapshots $\mathrm{Var}(\hat{\theta}_{G, L}) = \mathrm{Var}(\hat{\theta}_G)/L$, recovering (iii) from any valid one-snapshot bound and (iv).*

**Proposition 5 (Fixed-$\sigma$ outer-product variance bound).** *For $f(x) = xx^*$ equipped with the Frobenius norm, $D := \dim_{\mathbb{C}} \mathcal{C}(\pi)$, and $d_{\mathrm{eff}} := M^2 / D$,*
$$
\mathrm{Var}(F_G(x)) \le D(\sigma^4 + 2\sigma^2\|s\|^2) = \frac{M^2(\sigma^4 + 2\sigma^2\|s\|^2)}{d_{\mathrm{eff}}},
$$
*with equality in the first relation when $s = 0$.*

*Proof.* Twirling $T_G$ is the Frobenius orthogonal projector onto the adjoint-closed commutant $\mathcal{C}(\pi)$, which admits a Frobenius-orthonormal basis $\{B_1, \dots, B_D\}$ of Hermitian matrices (complexifying its real Hermitian subspace). Each basis coefficient is $x^* B_j x$. Diagonalizing Hermitian $B$ and using rotational invariance of $n \sim \mathcal{CN}(0, \sigma^2 I_M)$, each independent scalar coordinate $w_k \sim \mathcal{CN}(m_k, \sigma^2)$ satisfies $\mathrm{Var}(|w_k|^2) = \sigma^4 + 2\sigma^2 |m_k|^2$, whence
$$
\mathrm{Var}(x^* B x) = \sigma^4 \operatorname{tr}(B^2) + 2\sigma^2 \|Bs\|^2.
$$
Since $\|B_j\|_F = 1$ implies $\|B_j\|_{\mathrm{op}} \le 1$, summing over $j = 1, \dots, D$ gives $\mathrm{Var}(F_G(x)) = \sum_{j=1}^D (\sigma^4 + 2\sigma^2 \|B_j s\|^2) \le D(\sigma^4 + 2\sigma^2 \|s\|^2)$, with equality when $s = 0$. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/group_spectral_est/explore.py
```

*(Notation alignment: `explore.py` defines the cyclic generator via `shift = permutation([1, 2, 3, 0])`, which acts as $e_j \mapsto e_{j-1}$, i.e., $P^{-1}$ relative to the prose convention $P e_j = e_{P(j)}$. Both generate the exact same group $G = \langle(1\,2\,3\,4)\rangle$, swapping $w_+$ and $w_-$ while leaving every score, residual, and covariance invariant.)*

| Conjecture | Verification test in `explore.py` | Observed result |
|---|---|---|
| `conj:blind` ($G=\mathbb{Z}_4, H=V_4$) | Exact scores/residuals + 256 noisy draws ($\sigma = 10^{-3}$) | $\psi = 4/5 < 9/10$, $\delta = 0 < 12/\sqrt{697}$; **256/256** chose rival |
| `conj:blind` ($C_4$ filter, $G$ vs. $K$) | Exact $y_0$ check + population-twirl check + 2,048 Gaussian draws | $\psi(y_0) = 5/11 < 9/11$; **765/2,048** chose rival cycle |
| `conj:nadh` (3-sunlet, $S_3$) | Exhaustive $6! = 720$ automorphism check + spectrum + 128 draws | $|\mathrm{Aut}| = 6$, mean $S_3$ score **0.524225**, cyclic max **1.000000** |
| `conj:gaat` | Truncated $\mathbb{E}[f], \mathbb{E}[f^2]$ at $T \le 1, 4, 16, 64$; $\sigma^4$ scaling; $L=2$ halving | Truncated $\mathbb{E}[f^2] = 1, 4, 16, 64$ (divergent); all identities verified |

All assertions passed.

---

## References

- M. A. Thornton, *Algebraic Diversity: Group-Theoretic Spectral Estimation from Single Observations*, [arXiv:2604.03634](https://arxiv.org/abs/2604.03634) (local v17 draft), June 2026.
