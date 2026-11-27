# Resolution of the Recursive and Rectangular Symmetric Tensor Unfolding Conjectures

**Target paper:** Tom Szwagier, Xiaoyi Mai, and José Henrique de Morais Goulart, *A Random Matrix Analysis of Symmetric Tensor Unfolding*, [arXiv:2611.05137](https://arxiv.org/abs/2611.05137), Appendix D (Conjecture `conj:recursive`, *Recursive Unfolding*) and Appendix E (Conjecture `conj:transition`, *Phase Transition* for rectangular unfoldings).  
**Companion files:** [`solutions/sym_tensor_unfolding/explore.py`](explore.py), [`solutions/sym_tensor_unfolding/proof.pdf`](proof.pdf).

---

## Summary of Verdicts

Let $\mathcal{X} = \lambda v^{\otimes p} + \mathcal{Z}$ be a rank-one spiked symmetric tensor of fixed order $p \ge 2$, planted unit signal $v \in \mathbb{S}^{n-1}$, deterministic SNR $\lambda > 0$, and $\mathcal{Z} \sim \mathrm{GOTE}_{p,n}$. All overlap limits below are almost sure as $n \to \infty$.

1. **Conjecture `conj:recursive` (Recursive Unfolding, Appendix D): FALSE as printed (at $q = 1$, i.e., $p = 2$); supercritical exact recovery PROVED for every fixed $q \ge 2$ (even $p = 2q \ge 4$); fixed $\tau \le 1$ ($q \ge 2$) remains OPEN.**
   - *Quantifier and counterexample at $q = 1$ ($p = 2$):* Section 2.1 of arXiv:2611.05137 fixes the standing assumption $p = 2q$ for an arbitrary positive integer $q \ge 1$ (explicitly including $p = 2$, $q = 1$, $\alpha_{2,1} = 1$ in Section 2.2 and Section 3.3), and `conj:recursive` states no $q \ge 2$ restriction. At $p = 2$, $\operatorname{Mat}_1(\hat u) = \hat u \in \mathbb{R}^{n \times 1}$, so $\hat v = \pm \hat u$ and $|\hat v^\top v|^2 \to (1 - \tau^{-2})^+$ almost surely for every fixed $\tau > 0$ (e.g., $3/4 < 1$ at $\tau = 2$), refuting `conj:recursive` as stated.
   - *Supercritical theorem for $q \ge 2$ (Theorem 1):* For every fixed $q \ge 2$ (even $p \ge 4$) and fixed $\tau = \lambda / (\alpha_{p,q} n^{(p-2)/4}) > 1$, the leading left singular vector $\hat v$ of $\operatorname{Mat}_1(\hat u)$ satisfies $|\hat v^\top v|^2 \to 1$ almost surely.
   - *Open cases and critical-sequence note:* The fixed subcritical and critical regimes $\tau \le 1$ for $q \ge 2$ remain **open** here (the outlier spectral gap closes at $\tau \le 1$). If the critical statement were interpreted as covering arbitrary sequences $\tau_n \to 1$, it fails for every $q \ge 2$ as well via diagonalization along a deterministic sequence $\tau_n \downarrow 1$ (Proposition 2).

2. **Conjecture `conj:transition` (Rectangular Phase Transition, Appendix E): PROVED in full for every fixed integer $1 \le k < p/2$ (including odd $p \ge 3$ and even $p \ge 4$).**
   - For the compact $n_k \times n_{p-k}$ unfolding $\Phi_k(\mathcal{X})$ with leading singular vectors $(\hat v_\ell, \hat v_r)$ and normalized signal strengths
     $$
     \tau_\ell = \frac{\lambda}{\gamma_{\ell,p,k}\, n^{(p-2)/4}}, \quad
     \gamma_{\ell,p,k} = \left(\frac{k!(p-k)!}{(p-1)!^2}\right)^{1/4}, \qquad
     \tau_r = \frac{\lambda}{\gamma_{r,p,k}\, n^{(p-k-1)/2}}, \quad
     \gamma_{r,p,k} = \left(\frac{k!}{(p-1)!}\right)^{1/2},
     $$
     Theorem 5 establishes that almost surely
     $$
     |\hat v_\ell^\top v^{\odot k}|^2 \to (1 - \tau_\ell^{-4})^+
     \qquad \text{and} \qquad
     |\hat v_r^\top v^{\odot (p-k)}|^2 \to \frac{\tau_r^2}{1 + \tau_r^2}
     $$
     for all fixed $\tau_\ell \ge 0$ (including the critical point $\tau_\ell = 1$) and all fixed $\tau_r \ge 0$, with uniform tracking across scales.

---

## 1. Normalizations and Inputs from `arXiv:2611.05137`

Let $H_j := \operatorname{Sym}^j(\mathbb{R}^n)$ with the inherited Euclidean tensor inner product, $n_j := \dim(H_j) = \binom{n+j-1}{j}$, and $P_j : (\mathbb{R}^n)^{\otimes j} \to H_j$ the orthogonal projector. For a nondecreasing multi-index $A$ of length $j$ with index multiplicities $(A_1, \dots, A_n)$, let $d_A := j! / \prod_{i=1}^n A_i!$ be the size of its permutation orbit. The canonical orthonormal basis of $H_j$ consists of the normalized orbit sums $d_A^{-1/2} \sum_{\mathbf{j} \in \mathfrak{S}_A} E_{\mathbf{j}}$; let $U_j : \mathbb{R}^{n_j} \to (\mathbb{R}^n)^{\otimes j}$ be the associated isometry ($U_j^\top U_j = I_{n_j}$, $U_j U_j^\top = P_j$), and write $v^{\odot j} := U_j^\top v^{\otimes j} \in \mathbb{S}^{n_j - 1}$.

A noise tensor $\mathcal{Z} \sim \mathrm{GOTE}_{p,n}$ is given by $\mathcal{Z} = \sqrt{p/n}\, P_p \mathcal{G}$ for an ordered i.i.d. $\mathcal{N}(0,1)$ array $\mathcal{G} \in (\mathbb{R}^n)^{\otimes p}$; equivalently, its coordinates in the $U_p$-basis of $H_p$ are i.i.d. $\mathcal{N}(0, p/n)$. For $1 \le k \le p/2$ and $b := p - k$, the ordinary unfolding $\operatorname{Mat}_k : (\mathbb{R}^n)^{\otimes p} \to \mathbb{R}^{n^k \times n^b}$ groups the first $k$ modes into rows and the remaining $b$ modes into columns, while the compact symmetric unfolding $\Phi_k(\mathcal{T}) := U_k^\top \operatorname{Mat}_k(\mathcal{T}) U_b \in \mathbb{R}^{n_k \times n_b}$ has entries $(\Phi_k(\mathcal{T}))_{A,B} = \sqrt{d_A d_B}\, \mathcal{T}_{A+B}$.

For even $p = 2q$, set $\alpha_{p,q} := \sqrt{q!/(p-1)!}$, $c_n := \alpha_{p,q} n^{(p-2)/4}$, $M := \Phi_q(\mathcal{Z}) / c_n$, and $u := v^{\odot q}$. Our proof of Theorem 1 uses three results proved in arXiv:2611.05137:
- the isotropic deterministic equivalent $(M - zI)^{-1} \leftrightarrow s_{\mathrm{sc}}(z) I_{n_q}$ at fixed $z \in \mathbb{C} \setminus \mathbb{R}$, where $s_{\mathrm{sc}}(z) = -(z - \sqrt{z^2 - 4})/2$ (`thm:DE_symmetric`),
- the spectral norm convergence $\|M\|_{\mathrm{op}} \to 2$ almost surely (`prop:spectral_norm`), and
- the square-unfolding phase transition (`thm:transition`): for fixed $\tau = \lambda / c_n > 1$, the top eigenvalue $\hat z$ of $M + \tau u u^\top$ satisfies $\hat z \to z_* := \tau + \tau^{-1}$ and $|\hat u^\top u|^2 \to 1 - \tau^{-2}$ almost surely (and $\hat z \to 2$, $|\hat u^\top u|^2 \to 0$ for $\tau \le 1$).

Our proof of `conj:transition` (Section 3) is self-contained and relies only on the classical rank-one GOE transition of Capitaine, Donati-Martin, and Féral (2009), [arXiv:0706.0136](https://arxiv.org/abs/0706.0136).

---

## 2. Recursive Unfolding (`conj:recursive`)

### 2.1 Counterexample at $q = 1$ ($p = 2$)

In arXiv:2611.05137, Section 2.1 stipulates $p = 2q$ for an arbitrary fixed positive integer $q \ge 1$, and `conj:recursive` imposes no further restriction on $q$. When $q = 1$ ($p = 2$), we have $\alpha_{2,1} = 1$, $c_n = 1$, and $\tau = \lambda$. The first-mode matricization $\operatorname{Mat}_1(\hat u) \in \mathbb{R}^{n \times 1}$ of the leading eigenvector $\hat u \in \mathbb{S}^{n-1}$ is simply the column vector $\hat u$ itself, so its leading left singular vector is $\hat v = \pm \hat u$. By the classical rank-one GOE transition (`thm:transition` at $p = 2$),
$$
|\hat v^\top v|^2 = |\hat u^\top v|^2 \longrightarrow (1 - \tau^{-2})^+ \qquad \text{almost surely.}
$$
In particular, at $\tau = 2$, $|\hat v^\top v|^2 \to 3/4 < 1$, refuting `conj:recursive` as printed.

### 2.2 Supercritical Exact Recovery for Even $p \ge 4$ ($q \ge 2$)

**Theorem 1 (Supercritical recursive unfolding for $q \ge 2$).** *Fix an integer $q \ge 2$ ($p = 2q \ge 4$) and a deterministic signal strength $\tau = \lambda / c_n > 1$. Let $\hat u \in \mathbb{S}^{n_q - 1}$ be a leading eigenvector of $M + \tau u u^\top$ (equivalently, $U_q \hat u$ is a leading symmetric eigenvector of $\operatorname{Mat}_q(\mathcal{X})$), and let $\hat v \in \mathbb{S}^{n-1}$ be a leading left singular vector of $L(\hat u) := \operatorname{Mat}_1(U_q \hat u) \in \mathbb{R}^{n \times n^{q-1}}$. Then $|\hat v^\top v|^2 \to 1$ almost surely.*

*Proof.* Note that $L : \mathbb{R}^{n_q} \to \mathbb{R}^{n \times n^{q-1}}$ is linear and preserves Frobenius norm ($\|L(h)\|_F = \|U_q h\|_F = \|h\|_2$, hence $\|L(h)\|_{\mathrm{op}} \le \|h\|_2$). Let $R(z) := (M - zI)^{-1}$. We first show that for every fixed $z \in \mathbb{C}$ with $\eta := \Im z > 0$,
$$
\bigl\|L\bigl((R(z) - s_{\mathrm{sc}}(z)I)u\bigr)\bigr\|_{\mathrm{op}} \longrightarrow 0 \qquad \text{almost surely.} \tag{1}
$$

*Step 1 (Deterministic bias).* We claim $\|(\mathbb{E} R(z) - s_{\mathrm{sc}}(z)I)u\|_2 \to 0$. If not, along a subsequence the real or imaginary part of this deterministic vector has norm bounded away from zero; normalizing it yields deterministic unit test vectors $a_n \in \mathbb{S}^{n_q - 1}$ for which $|a_n^\top (\mathbb{E} R(z) - s_{\mathrm{sc}}(z)I)u|$ stays bounded away from zero. However, $|a_n^\top R(z) u| \le \eta^{-1}$ is uniformly bounded and $a_n^\top (R(z) - s_{\mathrm{sc}}(z)I)u \to 0$ almost surely by `thm:DE_symmetric`, so bounded convergence forces $a_n^\top (\mathbb{E} R(z) - s_{\mathrm{sc}}(z)I)u \to 0$, a contradiction. Since $\|L(h)\|_{\mathrm{op}} \le \|h\|_2$, the deterministic bias in (1) vanishes.

*Step 2 (Concentration over a product net).* Write $\mathcal{Z} = \sqrt{p/n}\, U_p g$ with $g \sim \mathcal{N}(0, I_{n_p})$. Because $U_p g \in H_p \subset H_q \otimes H_q$, Proposition `prop:eigenstructure` implies $\|\Phi_q(U_p g)\|_F = \|\operatorname{Mat}_q(U_p g)\|_F = \|g\|_2$. Hence $g \mapsto M(g)$ has Frobenius Lipschitz constant
$$
\frac{\sqrt{p/n}}{c_n} = \sqrt{\frac{(2q)!}{q!}}\, n^{-q/2} = O(n^{-q/2}).
$$
For any fixed $a \in \mathbb{R}^{n_q}$ with $\|a\|_2 \le 1$, the resolvent identity $R(M_1) - R(M_2) = R(M_1)(M_2 - M_1)R(M_2)$ and $\|R(z)\|_{\mathrm{op}} \le \eta^{-1}$ give
$$
|a^\top (R(M_1) - R(M_2))u| \le \eta^{-2} \|M_1 - M_2\|_F = \eta^{-2}\frac{\sqrt{p/n}}{c_n}\|g_1 - g_2\|_2.
$$
Thus the real and imaginary parts of $a^\top R(z) u$ are $L_n$-Lipschitz functions of $g$ with $L_n = O(\eta^{-2} n^{-q/2})$.

Let $\mathcal{N}_1 \subset \mathbb{S}^{n-1}$ and $\mathcal{N}_2 \subset \mathbb{S}^{n^{q-1}-1}$ be $1/4$-nets of cardinalities $|\mathcal{N}_1| \le 9^n$ and $|\mathcal{N}_2| \le 9^{n^{q-1}}$. For $(x, y) \in \mathcal{N}_1 \times \mathcal{N}_2$, set $a_{x,y} := U_q^\top (x \otimes y)$; then $\|a_{x,y}\|_2 \le \|x \otimes y\|_2 = 1$ and $x^\top L(h) y = a_{x,y}^\top h$. For any fixed $\varepsilon > 0$, Gaussian Lipschitz concentration yields
$$
\Pr\bigl(|a_{x,y}^\top (\Re R(z) - \mathbb{E}\Re R(z)) u| > \varepsilon\bigr) \le 2\exp(-c_{q,z,\varepsilon} n^q),
$$
and identically for $\Im R(z)$, for a constant $c_{q,z,\varepsilon} > 0$. Since $q \ge 2$, the product net has size $|\mathcal{N}_1||\mathcal{N}_2| \le 9^{n + n^{q-1}} = \exp(O(n^{q-1}))$, so the union bound over $\mathcal{N}_1 \times \mathcal{N}_2$ is summable in $n$. Standard $1/4$-net approximation bounds the operator norm of any real matrix $A \in \mathbb{R}^{n \times n^{q-1}}$ by $2\max_{(x,y) \in \mathcal{N}_1 \times \mathcal{N}_2} |x^\top A y|$. Applying Borel–Cantelli over rational $\varepsilon > 0$ and combining with Step 1 proves (1).

*Step 3 (Passing to the real outlier).* By `prop:spectral_norm` and `thm:transition`, almost surely $\|M\|_{\mathrm{op}} \to 2$ and $\hat z \to z_* = \tau + \tau^{-1} > 2$, so $z_*$ and $\hat z$ are eventually separated from $\operatorname{Spec}(M)$ by at least $\delta := (z_* - 2)/2 > 0$. Apply (1) simultaneously at $z_j := z_* + i/j$ for all integers $j \ge 1$. On the event $\|M\|_{\mathrm{op}} \le 2 + \delta$, the resolvent identity gives
$$
\|R(z_*) - R(z_j)\|_{\mathrm{op}} \le \frac{1}{j\,\delta^2},
$$
while $|s_{\mathrm{sc}}(z_*) - s_{\mathrm{sc}}(z_j)| \to 0$ as $j \to \infty$. Because $\|L(h)\|_{\mathrm{op}} \le \|h\|_2$, letting $n \to \infty$ and then $j \to \infty$ transfers (1) to $z_*$. Replacing $z_*$ by $\hat z \to z_*$ via $\|R(\hat z) - R(z_*)\|_{\mathrm{op}} \le |\hat z - z_*|/\delta^2 \to 0$ yields
$$
\bigl\|L\bigl((R(\hat z) - s_{\mathrm{sc}}(z_*)I)u\bigr)\bigr\|_{\mathrm{op}} \longrightarrow 0 \qquad \text{almost surely,}
$$
where $s_{\mathrm{sc}}(z_*) = -1/\tau$.

*Step 4 (Conclusion).* Orient $\hat u$ so that $\rho_n := \langle u, \hat u \rangle \ge 0$; by `thm:transition`, $\rho_n \to \sqrt{1 - \tau^{-2}} > 0$ almost surely. When $\hat z \notin \operatorname{Spec}(M)$, the eigenvalue equation $(M + \tau u u^\top)\hat u = \hat z \hat u$ rearranges to $\hat u = -\tau \rho_n R(\hat z) u$. Using $L(u) = v (v^{\otimes(q-1)})^\top$ and $-\tau s_{\mathrm{sc}}(z_*) = 1$, we obtain
$$
\bigl\|L(\hat u) - \rho_n v (v^{\otimes(q-1)})^\top\bigr\|_{\mathrm{op}}
= \tau \rho_n \bigl\|L\bigl((R(\hat z) - s_{\mathrm{sc}}(z_*)I)u\bigr)\bigr\|_{\mathrm{op}} \longrightarrow 0 \qquad \text{almost surely.}
$$
Hence the first-mode Gram matrix $L(\hat u) L(\hat u)^\top$ differs in operator norm from $\rho_n^2 v v^\top$ by $o(1)$ with $\rho_n^2$ bounded away from zero, so its leading eigenvector $\hat v$ satisfies $|\hat v^\top v|^2 \to 1$ almost surely. $\square$

### 2.3 Remarks on Remainder Rate, Open Cases, and Critical Sequences

- **Orthogonal remainder lower bound:** Writing $\hat u = \rho_n u + \sqrt{1 - \rho_n^2}\, w$ with $w \perp u$ and $\|w\|_2 = 1$, Theorem 1 yields $\|L(w)\|_{\mathrm{op}} = o(1)$. Because $\|L(w)\|_F = 1$ and $\operatorname{rank}(L(w)) \le n$, one always has $\|L(w)\|_{\mathrm{op}} \ge n^{-1/2}$ (consistent with the $O(n^{-1/2})$ scale established in the asymmetric model by Montanari and Richard (2014), Theorem 5).
- **Fixed $\tau \le 1$ ($q \ge 2$) remains open:** At $\tau \le 1$, the spectral gap separating $\hat z$ from the noise bulk closes, so Step 3 does not apply. Vanishing lifted overlap $|\hat u^\top u|^2 \to 0$ alone does not rule out alignment after matricization, since both signal and remainder components can have vanishing operator norm.
- **Proposition 2 (Sequences approaching $\tau = 1$ from above):** *For every fixed $q \ge 2$, there exists a deterministic sequence $\tau_n \downarrow 1$ such that the recursive unfolding estimator satisfies $|\hat v^\top v|^2 \to 1$ almost surely.*  
  *Proof.* Couple the countable family of fixed strengths $1 + 1/j$ ($j \ge 1$) on the same sequence of noise tensors $\mathcal{Z}_n$. By Theorem 1, for each $j$ there exists a deterministic $N_j$ (chosen strictly increasing) such that $\Pr(\inf_{n \ge N_j} |\hat v_{n, 1+1/j}^\top v|^2 < 1 - 1/j) \le 2^{-j}$. Setting $\tau_n := 1 + 1/j$ for $N_j \le n < N_{j+1}$ yields $\tau_n \downarrow 1$ and $|\hat v_n^\top v|^2 \to 1$ almost surely by Borel–Cantelli. (This approaches $\tau = 1$ from above and leaves the fixed choice $\tau = 1$ open.) $\square$

---

## 3. Rectangular Phase Transition (`conj:transition`)

Fix $p \ge 3$ and an integer $1 \le k < p/2$, and set $b := p - k > k$. Put $m := n_k = \binom{n+k-1}{k}$, $N := n_b = \binom{n+b-1}{b}$, $D := \binom{p}{k}$, and $s := \sqrt{mN}$. Note that $m/N = \Theta(n^{-(b-k)}) \to 0$ polynomially. Rescale to unit noise coordinates:
$$
\sigma^2 := \frac{k!\,b!}{n(p-1)!} = \frac{p}{nD}, \qquad
W := \frac{\Phi_k(\mathcal{Z})}{\sigma} \in \mathbb{R}^{m \times N}, \qquad
\ell := \frac{\lambda}{\sigma}.
$$
Then $W = \sqrt{D}\,\Phi_k(U_p g)$ for $g \sim \mathcal{N}(0, I_{n_p})$. For multi-indices $A$ ($|A| = k$) and $B$ ($|B| = b$),
$$
\operatorname{Var}(W_{A,B}) = D\,\frac{d_A d_B}{d_{A+B}} = \prod_{i=1}^n \binom{A_i + B_i}{A_i},
$$
and distinct rows in the same column $B$ have disjoint multiset sums $A + B$ and therefore zero covariance. Summing $\operatorname{Var}(W_{A,B})$ over all $|B| = b$ extracts the coefficient of $z^b$ in $\prod_{i=1}^n (1 - z)^{-1 - A_i} = (1 - z)^{-(n+k)}$, giving the exact finite-dimensional centering identity
$$
\mathbb{E}[W W^\top] = h I_m, \qquad h := \binom{n+p-1}{b}, \qquad \frac{h}{N} \longrightarrow 1. \tag{2}
$$
We center by the exact $h$ (since $h - N$ can exceed $s = \sqrt{mN}$).

### 3.1 Uniform Covariance Estimate

For a unit vector $x \in \mathbb{R}^m \cong H_k$, define $T_x : H_p \to H_b$ by $T_x g := W^\top x$. Its adjoint is $T_x^* f = \sqrt{D}\, P_p(x \otimes f)$ (identifying vectors with symmetric tensors via $U_k, U_b, U_p$), so $\|T_x\|_{\mathrm{op}} \le \sqrt{D}$.

**Lemma 3 (Block-exchange covariance bound).** *Uniformly over unit vectors $x, y \in H_k$,*
$$
\|T_x T_y^* - \langle x, y \rangle I_N\|_F \le C_p\, n^{(b-1)/2}, \tag{3}
$$
*and setting $B_x := T_x^* T_x$ (which is positive semidefinite of rank at most $N$, operator norm at most $D$, and trace $h$),*
$$
\operatorname{tr}(B_x B_y) = \|T_x T_y^*\|_F^2 = N\bigl(\langle x, y \rangle^2 + O_p(n^{-1/2})\bigr). \tag{4}
$$

*Proof.* On $H_b$, $T_x T_y^* = D\, (\langle x| \otimes I_{n^b})\, P_p\, (|y\rangle \otimes I_{n^b})$. Partition the $p!$ permutations in $P_p$ by the number $j \in \{0, \dots, k\}$ of positions exchanged between the first $k$ modes and the last $b$ modes. Because $x, y$ and inputs/outputs in $H_b$ are symmetric, permutations internal to the blocks leave the operator unchanged; the number of mode-partition placements with $j$ exchanges is $\binom{k}{j}\binom{b}{j} / D$, so multiplying by $D$ yields weight $\binom{k}{j}\binom{b}{j}$.
- For $j = 0$, the term is $\langle x, y \rangle I_N$.
- For $1 \le j \le k$, the $k - j$ unexchanged modes of $x$ and $y$ contract into the $n^j \times n^j$ matrix $(C_j)_{I,J} = \sum_H y_{IH} x_{JH}$ satisfying $\|C_j\|_F \le \|y\|_F \|x\|_F = 1$, while the remaining $b - j$ modes act as $I_{n^{b-j}}$. The bilinear form on $H_b$ is the compression of $C_j \otimes I_{n^{b-j}}$ (up to transpose) to $H_b$, whose Frobenius norm is at most $\|C_j \otimes I_{n^{b-j}}\|_F \le n^{(b-j)/2} \le n^{(b-1)/2}$.

Summing over $j = 1, \dots, k$ gives (3). Cyclicity gives $\operatorname{tr}(B_x B_y) = \|T_x T_y^*\|_F^2$. Expanding $\|(T_x T_y^* - \langle x, y \rangle I_N) + \langle x, y \rangle I_N\|_F^2$ with (3) and $N = \Theta_p(n^b)$ yields (4). $\square$

### 3.2 Deformed Upper Edge via Gaussian Interpolation

Define the centered Gram matrix $H := (W W^\top - h I_m)/s$. Fix a deterministic unit vector $a \in \mathbb{S}^{m-1}$ and $d \ge 0$, and let
$$
f(d) := \begin{cases}
2, & 0 \le d \le 1, \\
d + d^{-1}, & d > 1.
\end{cases}
$$

**Proposition 4 (Deformed upper edge).** *Almost surely, simultaneously for all fixed $d \ge 0$,*
$$
\lambda_{\max}(H + d\, a a^\top) \longrightarrow f(d). \tag{5}
$$

*Proof.* Fix $0 < \varepsilon < 1/4$ and let $\mathcal{N} \subset \mathbb{S}^{m-1}$ be an $\varepsilon$-net of cardinality $K \le (1 + 2/\varepsilon)^m$. On $\mathcal{N}$, compare the two centered random fields
$$
F_x := x^\top H x = \frac{g^\top B_x g - \operatorname{tr}(B_x)}{s}, \qquad
Y_x := x^\top G_m x,
$$
where $G_m$ is an independent standard $m \times m$ GOE matrix (off-diagonal variance $1/m$, diagonal variance $2/m$), so $\operatorname{Cov}(Y_x, Y_y) = C_{xy} := 2\langle x, y \rangle^2 / m$.

*Interpolation error on the net.* For any smooth $u$ with bounded first derivatives, integrating one Gaussian factor of $g$ by parts gives
$$
\mathbb{E}[F_x u(F)] = \sum_{y \in \mathcal{N}} \mathbb{E}[\Gamma_{xy}\,\partial_y u(F)],
\qquad \Gamma_{xy} := \frac{2\, g^\top B_x B_y g}{s^2}.
$$
By (4), $|\mathbb{E}\Gamma_{xy} - C_{xy}| \le C_p / (m\sqrt{n})$. The symmetric part $A_{xy} := \frac{1}{2}(B_x B_y + B_y B_x)$ has $\|A_{xy}\|_{\mathrm{op}} \le D^2$ and $\|A_{xy}\|_F \le D^2 \sqrt{N}$. By the Hanson–Wright tail bound for Gaussian quadratic forms (obtained by diagonalizing $A_{xy}$ and bounding the centered $\chi^2$ MGF),
$$
\Pr\bigl(|g^\top A g - \operatorname{tr}(A)| > C(\|A\|_F \sqrt{t} + \|A\|_{\mathrm{op}} t)\bigr) \le 2e^{-t} \qquad (\forall t \ge 0).
$$
Taking a union bound over the $K^2 \le (1 + 2/\varepsilon)^{2m}$ pairs $(x, y) \in \mathcal{N}^2$ and integrating the tail yields
$$
\mathbb{E}\max_{x, y \in \mathcal{N}} |\Gamma_{xy} - C_{xy}| \le \frac{C_{p,\varepsilon}}{m}\left(n^{-1/2} + \sqrt{\frac{m}{N}} + \frac{m}{N}\right). \tag{6}
$$
Consider the smooth maximum $S_\beta(z) := \beta^{-1} \log \sum_{x \in \mathcal{N}} e^{\beta(z_x + d\langle a, x \rangle^2)}$ and the interpolation $Z_t := \sqrt{t}\,F + \sqrt{1-t}\,Y$ ($t \in (0, 1)$), following the smooth-maximum Gaussian interpolation method (see Koike, 2017/2021). The softmax Hessian satisfies $\sum_{x,y} |\partial_x \partial_y S_\beta(z)| \le 2\beta$. Differentiating $\mathbb{E} S_\beta(Z_t)$ and applying integration by parts conditionally on $Y$ for $F$ and conditionally on $F$ for $Y$ yields
$$
\frac{d}{dt}\,\mathbb{E} S_\beta(Z_t) = \frac{1}{2}\sum_{x, y \in \mathcal{N}} \mathbb{E}\bigl[(\Gamma_{xy} - C_{xy})\, (\partial_x \partial_y S_\beta)(Z_t)\bigr].
$$
Integrating over $t \in [0, 1]$ and using $0 \le S_\beta(z) - \max_{x \in \mathcal{N}}(z_x + d\langle a, x \rangle^2) \le \beta^{-1} \log K$, we obtain
$$
\Bigl|\mathbb{E}\max_{x \in \mathcal{N}}(F_x + d\langle a, x \rangle^2) - \mathbb{E}\max_{x \in \mathcal{N}}(Y_x + d\langle a, x \rangle^2)\Bigr|
\le \frac{2\log K}{\beta} + \beta\,\mathbb{E}\max_{x,y}|\Gamma_{xy} - C_{xy}|.
$$
Choosing $\beta := m \log n$ makes the first term $O_\varepsilon(1/\log n)$ and the second term $O_{p,\varepsilon}((\log n)(n^{-1/2} + \sqrt{m/N} + m/N)) \to 0$ because $m/N = \Theta(n^{-(b-k)})$ decays polynomially.

*Passage off the net and expectation limit.* Using a $1/4$-net of size $\le 9^m$, the quadratic tail bound applied to $B_x$ ($\|B_x\|_F \le D\sqrt{N}$, $\|B_x\|_{\mathrm{op}} \le D$) at $t = m\log 9 + m + u$ gives, outside probability $2e^{-m-u}$,
$$
\|H\|_{\mathrm{op}} \le \frac{C_p}{s}\bigl(\sqrt{N(m+u)} + m + u\bigr) \qquad (\forall u \ge 0).
$$
Integrating in $u$ gives $\sup_n \mathbb{E}\|H\|_{\mathrm{op}}^2 < \infty$, and setting $u = 0$ gives $\|H\|_{\mathrm{op}} = O(1)$ almost surely; the same moment bound holds for $\|G_m\|_{\mathrm{op}}$. For either deformed matrix, replacing $\lambda_{\max}$ by the $\varepsilon$-net maximum incurs error at most $2\varepsilon$ times the deformed operator norm. Letting $n \to \infty$ and then $\varepsilon \downarrow 0$ shows $\mathbb{E}\lambda_{\max}(H + d aa^\top) - \mathbb{E}\lambda_{\max}(G_m + d aa^\top) \to 0$. By the rank-one GOE phase transition of Capitaine, Donati-Martin, and Féral (2009) (which includes the critical point $d = 1$) and uniform integrability from the second-moment bound, $\mathbb{E}\lambda_{\max}(H + d aa^\top) \to f(d)$.

*Almost-sure convergence.* Write $S := \|[W,\ \sqrt{sd}\, a]\|_{\mathrm{op}}$, so that $\lambda_{\max}(H + d aa^\top) = (S^2 - h)/s$. Since unfolding preserves Frobenius norm up to $\sqrt{D}$, $g \mapsto S(g)$ is $\sqrt{D}$-Lipschitz. Gaussian concentration gives $|S - \mathbb{E} S| = O(\sqrt{\log n})$ almost surely and $\operatorname{Var}(S) = O(1)$, while (2) and $\mathbb{E}\|H\|_{\mathrm{op}} = O(1)$ give $\mathbb{E} S = O(\sqrt{N})$. Thus $(S^2 - \mathbb{E}[S^2])/s = O(\sqrt{N\log n}/s) = O(\sqrt{(\log n)/m}) \to 0$ almost surely, proving (5) for each fixed $d \ge 0$. Intersecting over rational $d \ge 0$ and using the $1$-Lipschitz continuity of $d \mapsto \lambda_{\max}(H + d aa^\top)$ establishes (5) simultaneously for all $d \ge 0$. $\square$

### 3.3 Proof of Conjecture `conj:transition`

**Theorem 5 (Rectangular phase transition).** *Fix $p \ge 3$ and $1 \le k < p/2$, and let $(\hat v_\ell, \hat v_r)$ be leading left and right singular vectors of $\Phi_k(\mathcal{X})$ with $\mathcal{X} = \lambda v^{\otimes p} + \mathcal{Z}$. With $\tau_\ell = \lambda / (\gamma_{\ell,p,k} n^{(p-2)/4})$ and $\tau_r = \lambda / (\gamma_{r,p,k} n^{(p-k-1)/2})$ as in `conj:transition`, almost surely*
$$
|\hat v_\ell^\top v^{\odot k}|^2 \longrightarrow (1 - \tau_\ell^{-4})^+
\qquad \text{and} \qquad
|\hat v_r^\top v^{\odot (p-k)}|^2 \longrightarrow \frac{\tau_r^2}{1 + \tau_r^2}
$$
*for every fixed $\tau_\ell \ge 0$ (including $\tau_\ell = 1$) and every fixed $\tau_r \ge 0$. More generally, for any deterministic sequence $\lambda_n \ge 0$, $|\hat v_\ell^\top v^{\odot k}|^2 - (1 - \tau_{\ell,n}^{-4})^+ \to 0$ and $|\hat v_r^\top v^{\odot (p-k)}|^2 - \tau_{r,n}^2/(1+\tau_{r,n}^2) \to 0$ almost surely.*

*Proof.* Set $a := v^{\odot k} \in \mathbb{S}^{m-1}$, $c := v^{\odot b} \in \mathbb{S}^{N-1}$, $V := W + \ell a c^\top = \Phi_k(\mathcal{X})/\sigma$, $t_n := \ell^2 / s$, and $r_n := \ell / \sqrt{N}$. Notice that $r_n^2 = t_n \sqrt{m/N}$.

For each fixed unit $x \in \mathbb{S}^{m-1}$, $x^\top W c = \langle T_x^* c, g \rangle$ is centered Gaussian with variance $\|T_x^* c\|_2^2 \le D$. A $1/2$-net of $\mathbb{S}^{m-1}$ of size $\le 5^m$ gives a summable tail at scale $\sqrt{m}$, so $\|W c\|_2 = O(\sqrt{m})$ almost surely. The centered Gram matrix of $V$ is
$$
\frac{V V^\top - h I_m}{s} = H + t_n a a^\top + E_n,
\qquad E_n := \frac{\ell}{s}\bigl((Wc)a^\top + a(Wc)^\top\bigr),
$$
with $\|E_n\|_{\mathrm{op}} \le 2\ell \|W c\|_2 / s = O(\ell / \sqrt{N}) = O(r_n)$ almost surely.

*Left overlap.* Let $x_n = \hat v_\ell$ be a leading unit left singular vector of $V$.
- Suppose $t_n \to t \in [0, \infty)$. Then $r_n \to 0$ and $\|E_n\|_{\mathrm{op}} = o(1)$, so $x_n^\top (H + t a a^\top) x_n = \lambda_{\max}(H + t a a^\top) + o(1)$. For any $\delta > 0$, testing $x_n$ against $H + (t \pm \delta) a a^\top$ gives
  $$
  \frac{\lambda_{\max}(H + t aa^\top) - \lambda_{\max}(H + (t-\delta)aa^\top)}{\delta} - \frac{o(1)}{\delta}
  \le |a^\top x_n|^2
  \le \frac{\lambda_{\max}(H + (t+\delta)aa^\top) - \lambda_{\max}(H + t aa^\top)}{\delta} + \frac{o(1)}{\delta}
  $$
  (for $t = 0$, use nonnegativity and the upper bound only). Applying Proposition 4 as $n \to \infty$ and then sending $\delta \downarrow 0$, since $f$ is continuously differentiable on $[0, \infty)$ — including at $t = 1$, where $f'(1^-) = f'(1^+) = 0$ — we obtain $|a^\top x_n|^2 \to f'(t) = (1 - t^{-2})^+$.
- Suppose $t_n \to \infty$. Then $\|H + E_n\|_{\mathrm{op}} / t_n = O(t_n^{-1} + \sqrt{m}/\ell) = o(1)$, so the Rayleigh principle immediately gives $|a^\top x_n|^2 \to 1$.

*Right overlap.* Let $s_1 = \|V\|_{\mathrm{op}}$ be the leading singular value of $V$. By the triangle inequality on $V V^\top = h I_m + \ell^2 a a^\top + s H + s E_n$,
$$
|s_1^2 - (h + \ell^2)| \le s\|H\|_{\mathrm{op}} + 2\ell \|W c\|_2 = O\bigl(\sqrt{mN} + \ell\sqrt{m}\bigr) = o(N + \ell^2)
$$
uniformly in $\ell \ge 0$, because $m/N \to 0$ and $h/N \to 1$. Hence $s_1^2 / (N + \ell^2) \to 1$ almost surely. For the paired right singular vector $y_n = V^\top x_n / s_1 = \hat v_r$,
$$
|c^\top y_n| = \frac{|\ell\, a^\top x_n + (W c)^\top x_n|}{s_1}.
$$
Here $|(W c)^\top x_n| / s_1 \le \|W c\|_2 / s_1 = O(\sqrt{m/(N+\ell^2)}) \to 0$. If $r_n \to 0$, then $\ell / s_1 \to 0$ and $|c^\top y_n|^2 \to 0$. If $r_n \to r \in (0, \infty]$, then $t_n = r_n^2 \sqrt{N/m} \to \infty$, so $|a^\top x_n| \to 1$ and $|c^\top y_n|^2 = (\ell^2 / (N + \ell^2))(1 + o(1)) = r_n^2 / (1 + r_n^2) + o(1)$.

*Matching constants and subsequence tracking.* From $m \sim n^k / k!$, $N \sim n^b / b!$, and $\sigma^2 = k!b!/(n(p-1)!)$, we have
$$
\sigma (mN)^{1/4} \sim \left(\frac{k!\,b!}{(p-1)!^2}\right)^{1/4} n^{(p-2)/4} = \gamma_{\ell,p,k}\, n^{(p-2)/4},
\qquad
\sigma \sqrt{N} \sim \left(\frac{k!}{(p-1)!}\right)^{1/2} n^{(b-1)/2} = \gamma_{r,p,k}\, n^{(p-k-1)/2},
$$
so $t_n \sim \tau_{\ell,n}^2$ and $r_n \sim \tau_{r,n}$. Passing to convergent subsequences in $[0, \infty]$ gives both fixed-strength limits and uniform tracking across varying strengths. $\square$

---

## 4. Verification and Attribution

Run the reproducible verification script from the project root:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B solutions/sym_tensor_unfolding/explore.py
```

With default seed `261105137`, the script first verifies compact versus directly symmetrized dense singular values, the exact centering identity (2), and the block-exchange expansion behind Lemma 3 on all nine $(p, k)$ pairs for $2 \le p \le 6$, $1 \le k \le \lfloor p/2 \rfloor$ (to tolerance $2 \times 10^{-13}$). It then simulates 432 independent GOTE tensors across 18 $(p, k, n)$ configurations (24 repetitions per configuration, 120 grid rows, 2,880 spiked fits; `L` fixes $\tau$ or $\tau_\ell$, `R` fixes $\tau_r$). Representative rows (means and Monte Carlo SEs in parentheses):

| $(p, k, n)$ | Scale & strength | Lifted / left squared overlap | Recursive / right squared overlap | Asymptotic limit |
|---|---|---:|---:|---|
| $(2, 1, 160)$ | $L = 2.0$ | $0.7360\ (0.0058)$ | $0.7360\ (0.0058)$ | $0.75$ / $0.75$ (refutes `conj:recursive`) |
| $(4, 2, 28)$ | $L = 0.6$ | $0.0072\ (0.0021)$ | $0.0394\ (0.0161)$ | $0$ / (open) |
| $(4, 2, 28)$ | $L = 2.0$ | $0.7035\ (0.0074)$ | $0.9688\ (0.0020)$ | $0.75$ / $1$ |
| $(6, 3, 12)$ | $L = 2.0$ | $0.3884\ (0.0269)$ | $0.9264\ (0.0184)$ | $0.75$ / $1$ |
| $(3, 1, 64)$ | $L = 0.6$ | $0.0191\ (0.0050)$ | $0.0019\ (0.0006)$ | $0$ / $0$ |
| $(3, 1, 64)$ | $L = 2.0$ | $0.9010\ (0.0031)$ | $0.3788\ (0.0055)$ | $0.9375$ / $0$ |
| $(3, 1, 64)$ | $R = 1.0$ | $0.9413\ (0.0018)$ | $0.4738\ (0.0048)$ | $1$ / $0.5$ |
| $(5, 2, 16)$ | $R = 2.0$ | $0.9084\ (0.0043)$ | $0.6822\ (0.0078)$ | $1$ / $0.8$ |

Finite-size effects are visible at small $n$ for high-order splits (e.g., $(5, 2, 16)$ at $R = 2$ gives $0.6822$ vs. limiting $0.8$); standard errors reflect only Monte Carlo uncertainty around the finite-$n$ mean.

**References.**
- T. Szwagier, X. Mai, and J. H. de M. Goulart, *A Random Matrix Analysis of Symmetric Tensor Unfolding*, [arXiv:2611.05137](https://arxiv.org/abs/2611.05137).
- A. Montanari and E. Richard, *A Statistical Model for Tensor PCA*, NeurIPS 2014, [arXiv:1411.1076](https://arxiv.org/abs/1411.1076).
- M. Capitaine, C. Donati-Martin, and D. Féral, *The largest eigenvalues of finite rank deformation of large Wigner matrices: Convergence and nonuniversality of the fluctuations*, Ann. Probab. 37(1):1–47, 2009, [arXiv:0706.0136](https://arxiv.org/abs/0706.0136).
- Y. Koike, *Notes on the dimension dependence in high-dimensional central limit theorems for hyperrectangles*, Jpn. J. Stat. Data Sci. 4:257–297, 2021, [arXiv:1709.00353](https://arxiv.org/abs/1709.00353).
- M. J. Feldman, *Spiked singular values and vectors under extreme aspect ratios*, J. Multivariate Anal. 196:105187, 2023.
