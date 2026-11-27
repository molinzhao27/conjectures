# Proof of the KernelSHAP-IQ Inverse and Split-SII Conjectures

**Target paper:** Fabian Fumagalli, Maximilian Muschalik, Patrick Kolpaczki, Eyke Hüllermeier, and Barbara Hammer, *KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions*, ICML 2024, [arXiv:2405.10852](https://arxiv.org/abs/2405.10852), Section 3.3, Conjectures 3.8 (`conj_inverse`) and 3.9 (`conj_sii`).  
**Companion files:** [`solutions/kernelshap_iq_inverse/explore.py`](explore.py), [`solutions/kernelshap_iq_inverse/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 3.8 (`conj_inverse`): PROVED** for every integer $k \ge 1$ and $n \ge 2k$. The limiting precision matrix $A_k = \lim_{M \to \infty}(X_k^\top W_k X_k)^{-1}$ equals $P / (n - 2k + 1)$, where $P$ is the orthogonal projector onto the top harmonic space $H$ of the Johnson scheme on $\binom{N}{k}$, with entries
  $$
  (A_k)_{S,U} = \frac{(-1)^{k - |S \cap U|}}{(n - k + 1)\binom{n - k}{k - |S \cap U|}} \qquad (|S| = |U| = k),
  $$
  and exact rank $\operatorname{rank}(A_k) = \operatorname{tr}(P) = \binom{n}{k} - \binom{n}{k-1}$.
- **Conjecture 3.9 (`conj_sii`): PROVED** for every integer $k \ge 1$ and $n \ge 2k$:
  $$
  \phi_k^{\mathrm{SII}}(\nu) = Q_k y_k^- + \lim_{M \to \infty} \operatorname*{argmin}_{b \in \mathbb{R}^{\binom{n}{k}}} \left\|W_k^{1/2}\bigl(y_k^+ - X_k b\bigr)\right\|_2^2.
  $$
  Every finite-weight interior coalition $T$ ($k \le |T| \le n - k$) satisfies the single-column projection identity $w_t A_k x_T = q_T$, recovering the exact column of Shapley Interaction Index (SII) weights.
- **Stronger scope:** Although [arXiv:2405.10852](https://arxiv.org/abs/2405.10852) assumes $\nu(\emptyset) = 0$, our proof establishes Conjecture 3.9 for *all* set functions $\nu : 2^N \to \mathbb{R}$ without requiring $\nu(\emptyset) = 0$, and covers all boundary cases including $n = 2k$ and $(n, k) = (2, 1)$.

---

## 1. Setup and Definitions

Let $N = \{1, \dots, n\}$ with $n \ge 2k$ and $k \ge 1$. Index vectors and matrices acting on order-$k$ interactions by the $\binom{n}{k}$-element family $\binom{N}{k}$ of $k$-subsets of $N$, equipped with the standard Euclidean inner product $\langle u, v \rangle = \sum_{|S|=k} u(S)v(S)$.

For a coalition $T \subseteq N$, write $t := |T|$ and let $x_T \in \mathbb{R}^{\binom{n}{k}}$ denote the transpose of row $T$ of the KernelSHAP-IQ design matrix $X_k \in \mathbb{R}^{2^n \times \binom{n}{k}}$. Adopting the paper's Bernoulli number convention $B_0 = 1$, $B_1 = -1/2$ (and $B_{2m+1} = 0$ for $m \ge 1$), the entries of $X_k$ are
$$
x_T(S) := \lambda(k, |T \cap S|), \qquad
\lambda(k, j) := \sum_{r=1}^j \binom{j}{r} B_{k-r} \qquad (0 \le j \le k),
$$
with the empty sum $\lambda(k, 0) = 0$. Partition $2^N$ into the *interior* and *boundary* coalition families
$$
I := \{T \subseteq N : k \le |T| \le n - k\}, \qquad E := 2^N \setminus I.
$$
For $M = \mu_\infty > 0$, let $W_k \in \mathbb{R}^{2^n \times 2^n}$ be the diagonal weight matrix with entries
$$
(W_k)_{TT} := \begin{cases}
w_t := \binom{n - 2k}{t - k}^{-1}, & T \in I, \\[4pt]
M, & T \in E,
\end{cases}
$$
and set $G(M) := X_k^\top W_k X_k$.

For $0 \le a \le n - k$, define the SII weight function
$$
m_k(a) := \frac{a!\,(n - k - a)!}{(n - k + 1)!} = \frac{1}{(n - k + 1)\binom{n - k}{a}},
$$
and for every $T \subseteq N$ define the SII column vector $q_T \in \mathbb{R}^{\binom{n}{k}}$ by
$$
q_T(S) := (-1)^{k - |T \cap S|} m_k\bigl(t - |T \cap S|\bigr) \qquad (|S| = k).
$$
Let $Q_k \in \mathbb{R}^{\binom{n}{k} \times 2^n}$ have column $q_T$ for $T \in E$ and zero for $T \in I$. Given any game $\nu : 2^N \to \mathbb{R}$, set $y_1 := \nu$ and, for $k \ge 2$, $y_k := \nu - \hat{\nu}_{k-1}$, where $\hat{\nu}_{k-1}(T) = \sum_{R \subseteq T,\, 1 \le |R| < k} \Phi_{k-1}(R)$ is any lower-order approximation (in particular, the paper's $k$-SII residual of Bordt & von Luxburg, 2023). Masking by $I$ and $E$ yields $y_k^+ := \mathbf{1}_I \odot y_k$ and $y_k^- := \mathbf{1}_E \odot y_k$.

---

## 2. The Boundary Row Space and Harmonic Decomposition

For any $R \subseteq N$, define the incidence vector $v_R \in \mathbb{R}^{\binom{n}{k}}$ by $v_R(S) := \mathbf{1}_{R \subseteq S}$. Define the *lower-order* subspace and its orthogonal complement (the top harmonic space of the Johnson scheme; Gottlieb, 1966; Delsarte, 1973) by
$$
L := \operatorname{span}_{\mathbb{R}}\{v_R : |R| = k - 1\}, \qquad H := L^\perp,
$$
and let $P$ be the orthogonal projector onto $H$.

**Lemma 1 (Exact boundary row span).** *For every $k \ge 1$ and $n \ge 2k$:*
1. *Every $v_R$ with $|R| = r < k$ belongs to $L$.*
2. *Writing $f_T(S) := \mathbf{1}_{S \subseteq T}$, every coalition $T \subseteq N$ satisfies*
   $$
   x_T = \sum_{r=1}^k B_{k-r} \sum_{\substack{R \subseteq T \\ |R| = r}} v_R,
   \qquad x_T - f_T \in L.
   \tag{1}
   $$
3. *The boundary rows $\{x_T : T \in E\}$ span precisely $L$. Consequently, $D_E := \sum_{T \in E} x_T x_T^\top$ vanishes on $H$ and is strictly positive definite on $L$.*

*Proof.* For $|R| = r < k$, every $k$-subset $S \supseteq R$ contains exactly $k - r$ subsets $D$ of size $k - 1$ with $R \subseteq D \subseteq S$, so
$$
\sum_{\substack{D \supseteq R \\ |D| = k-1}} v_D = (k - r)\,v_R.
$$
Since $k - r \ge 1$, $v_R \in L$, proving (1).

For (2), counting $r$-subsets $R \subseteq T \cap S$ gives $\binom{|T \cap S|}{r} = \sum_{R \subseteq T,\, |R|=r} v_R(S)$. Substituting into $x_T(S) = \sum_{r=1}^k B_{k-r}\binom{|T \cap S|}{r}$ yields the expansion in (1). Its $r = k$ term is $B_0 \sum_{R \subseteq T,\, |R|=k} v_R = f_T$ (because $B_0 = 1$ and $v_R(S) = \mathbf{1}_{R = S}$ for $|R| = k$), while all $1 \le r < k$ terms lie in $L$ by (1). Hence $x_T - f_T \in L$.

For (3), if $|T| < k$ then $f_T = 0$, so $x_T \in L$. If $|T| > n - k$ then $|N \setminus T| < k$, and inclusion–exclusion gives
$$
f_T(S) = \mathbf{1}_{S \cap (N \setminus T) = \emptyset}
= \sum_{R \subseteq N \setminus T} (-1)^{|R|} v_R(S) \in L
$$
by part (1), so $x_T = (x_T - f_T) + f_T \in L$. Thus $\operatorname{span}\{x_T : T \in E\} \subseteq L$.

Conversely, if $k = 1$, then $L = \operatorname{span}\{v_\emptyset\} = \operatorname{span}\{\mathbf{1}\}$ and $N \in E$ (since $n \ge 2$ implies $|N| = n > n - 1$), with $x_N = B_0 \mathbf{1} = \mathbf{1}$. If $k \ge 2$ and $|R| = k - 1$, then every $D \subseteq R$ satisfies $|D| \le k - 1 < k$, hence $D \in E$. Applying Boolean Möbius inversion to (1) over the lattice of subsets of $R$ yields
$$
\sum_{D \subseteq R} (-1)^{|R| - |D|} x_D
= \sum_{r=1}^{k-1} B_{k-r} \sum_{\substack{U \subseteq R \\ |U| = r}} v_U \sum_{U \subseteq D \subseteq R} (-1)^{|R| - |D|}
= B_1 v_R,
$$
because the inner alternating sum vanishes unless $U = R$ (which occurs only at $r = k - 1$). Since $B_1 = -1/2 \neq 0$, we obtain $v_R \in \operatorname{span}\{x_T : T \in E\}$ for all $|R| = k - 1$. Therefore $\operatorname{span}\{x_T : T \in E\} = L$, and $D_E = \sum_{T \in E} x_T x_T^\top$ has kernel $L^\perp = H$ and is positive definite on $L$. $\square$

---

## 3. Block-Diagonal Action of $G(M)$ and Proof of Conjecture 3.8

Any symmetric matrix $M_0$ indexed by $\binom{N}{k}$ whose entries depend only on $j = |S \cap U|$ is a unique linear combination of the Johnson incidence matrices
$$
K_r(S, U) := \binom{|S \cap U|}{r} = \sum_{|R| = r} v_R(S) v_R(U) \qquad (0 \le r \le k),
$$
because $\bigl\{\binom{j}{r}\bigr\}_{r=0}^k$ is a triangular basis of functions on $\{0, 1, \dots, k\}$. For $r < k$, every column of $K_r$ lies in $L$ and $K_r$ annihilates $H$; for $r = k$, $K_k = \mathrm{Id}$. Hence every $S_n$-equivariant matrix $M_0(S, U) = \psi(|S \cap U|)$ preserves the orthogonal decomposition $\mathbb{R}^{\binom{n}{k}} = H \oplus L$ and acts on $H$ as scalar multiplication by its top binomial coefficient $\sum_{j=0}^k (-1)^{k-j}\binom{k}{j}\psi(j)$.

**Lemma 2 (Interior Gram action and limiting inverse).** *Set $c := n - 2k + 1$ (the number of interior coalition sizes) and $G_I := \sum_{T \in I} w_t x_T x_T^\top$. Then $G(M) = G_I + M D_E$ preserves $H \oplus L$, acts on $H$ as $c\,\mathrm{Id}_H$, and is strictly positive definite for every $M > 0$. Moreover, in operator norm,*
$$
A_k := \lim_{M \to \infty} G(M)^{-1} = \frac{1}{c}\,P.
\tag{2}
$$

*Proof.* For each interior size $t \in \{k, \dots, n - k\}$, define $G_t := \sum_{|T|=t} x_T x_T^\top$ and $C_t := \sum_{|T|=t} f_T f_T^\top$. Both are $S_n$-equivariant, so both preserve $H \oplus L$. Since $P x_T = P f_T$ by (1), we have $P G_t P = P C_t P$, so $G_t|_H = C_t|_H$. For $k$-sets $S, U$ with $|S \cap U| = j$, we have $|S \cup U| = 2k - j$, and the number of $t$-sets $T \supseteq S \cup U$ is
$$
C_t(S, U) = \binom{n - 2k + j}{t - 2k + j} = \binom{n - 2k + j}{n - t}
= \sum_{r=0}^k \binom{j}{r} \binom{n - 2k}{n - t - r}
$$
by Vandermonde's convolution (valid for all $0 \le j \le k$ and $k \le t \le n - k$, with out-of-range lower indices equal to zero; note that if $t < 2k - j$, then $n - t > n - 2k + j$ and both sides equal $0$). Extracting the $r = k$ coefficient gives
$$
G_t|_H = C_t|_H = \binom{n - 2k}{n - t - k}\mathrm{Id}_H = \binom{n - 2k}{t - k}\mathrm{Id}_H = w_t^{-1}\mathrm{Id}_H.
$$
Multiplying by $w_t$ and summing over the $c = n - 2k + 1$ interior sizes $t = k, \dots, n - k$ gives $G_I|_H = c\,\mathrm{Id}_H$. Since $D_E|_H = 0$ by Lemma 1, we obtain $G(M)|_H = c\,\mathrm{Id}_H$ for every $M > 0$.

On the orthogonal invariant subspace $L$, $G_I|_L \succeq 0$ and $D_E|_L \succ 0$, so
$$
G(M)|_L = G_I|_L + M D_E|_L \succeq M \lambda_{\min}(D_E|_L)\,\mathrm{Id}_L \succ 0.
$$
Thus $G(M)$ is invertible for every $M > 0$, with exact block inverse $G(M)^{-1} = \frac{1}{c} P + (G_I|_L + M D_E|_L)^{-1} (I - P)$, whose $L$-block has operator norm at most $\bigl(M \lambda_{\min}(D_E|_L)\bigr)^{-1} \to 0$ as $M \to \infty$. This proves (2). $\square$

---

## 4. Interior Column Projection and Proof of Both Conjectures

**Theorem 1 (Interior column projection and Conjecture 3.8).** *For every $k \ge 1$, $n \ge 2k$, and every interior coalition $T \in I$ with $|T| = t$,*
$$
q_T = \frac{w_t}{c}\, P f_T = w_t A_k x_T.
\tag{3}
$$
*In particular, specializing to $|U| = k$ yields Conjecture 3.8 (`conj_inverse`):*
$$
(A_k)_{S,U} = q_U(S) = \frac{(-1)^{k - |S \cap U|}}{(n - k + 1)\binom{n - k}{k - |S \cap U|}} \qquad (|S| = |U| = k),
\tag{4}
$$
*and $\operatorname{rank}(A_k) = \operatorname{tr}(P) = \binom{n}{k} - \binom{n}{k-1}$.*

*Proof.* Fix $T \in I$ with $k \le t \le n - k$. First we verify $q_T \in H = L^\perp$. Take any $R \subseteq N$ with $|R| = k - 1$, and set $b := |R \cap T|$ and $a := t - b$. Because $0 \le b \le k - 1$ and $k \le t \le n - k$, we have $1 \le a \le n - k$. Every $k$-subset $S \supseteq R$ is formed by adjoining one element $i \notin R$: there are $a = |T \setminus R|$ choices in $T \setminus R$ (giving $|T \cap S| = b + 1$ and $t - |T \cap S| = a - 1 \ge 0$) and $(n - k + 1) - a = |N \setminus (T \cup R)|$ choices outside $T \cup R$ (giving $|T \cap S| = b$ and $t - |T \cap S| = a \le n - k$). Therefore,
$$
\langle v_R, q_T \rangle
= (-1)^{k - b - 1} \Bigl[ a\, m_k(a - 1) - (n - k + 1 - a)\, m_k(a) \Bigr] = 0,
$$
since both terms equal $a!\,(n - k + 1 - a)! / (n - k + 1)!$. Thus $q_T \in H$ and $P q_T = q_T$.

Next expand the function $q(j) := (-1)^{k - j} m_k(t - j)$ ($0 \le j \le k$) in the Newton basis $\bigl\{\binom{j}{r}\bigr\}_{r=0}^k$. By the Beta-integral identity $m_k(u) = \int_0^1 x^u (1 - x)^{n - k - u}\,dx$ (valid for $0 \le u \le n - k$), the top coefficient ($r = k$) is
$$
\sum_{j=0}^k (-1)^{k - j} \binom{k}{j} q(j)
= \sum_{j=0}^k \binom{k}{j} m_k(t - j)
= \int_0^1 x^{t - k} (1 - x)^{n - k - t} \sum_{j=0}^k \binom{k}{j} x^{k - j} (1 - x)^j\,dx.
$$
The inner sum is $(x + (1 - x))^k = 1$, leaving
$$
\int_0^1 x^{t - k} (1 - x)^{n - k - t}\,dx
= \frac{(t - k)!\,(n - k - t)!}{(n - 2k + 1)!}
= \frac{1}{(n - 2k + 1)\binom{n - 2k}{t - k}}
= \frac{w_t}{c}.
$$
Substituting $j = |S \cap T|$ into the Newton expansion of $q(j)$, each lower term $0 \le r < k$ equals $\sum_{R \subseteq T,\, |R|=r} v_R(S) \in L$, while the $r = k$ term is $(w_t / c) f_T(S)$. Hence $q_T - (w_t / c) f_T \in L$. Applying $P$ and invoking $P q_T = q_T$, (1), and (2) gives
$$
q_T = P q_T = \frac{w_t}{c}\, P f_T = w_t A_k f_T = w_t A_k x_T,
$$
establishing (3).

Taking $T = U$ with $|U| = k$ gives $w_k = \binom{n - 2k}{0}^{-1} = 1$ and $f_U = e_U$ (the coordinate basis vector at $U$). Thus column $U$ of $A_k$ equals $A_k e_U = \frac{1}{c} P f_U = q_U$, proving (4) and Conjecture 3.8. Finally, the diagonal entry of $P = c A_k$ is $c\, m_k(0) = \frac{n - 2k + 1}{n - k + 1}$, so
$$
\operatorname{rank}(A_k) = \operatorname{tr}(P) = \frac{n - 2k + 1}{n - k + 1}\binom{n}{k} = \binom{n}{k} - \binom{n}{k - 1}. \qquad \square
$$

**Theorem 2 (Split WLS representation; Conjecture 3.9).** *For every $k \ge 1$, $n \ge 2k$, and every set function $\nu : 2^N \to \mathbb{R}$ (not necessarily satisfying $\nu(\emptyset) = 0$),*
$$
\phi_k^{\mathrm{SII}}(\nu)
= Q_k y_k^- + \lim_{M \to \infty} \operatorname*{argmin}_{b \in \mathbb{R}^{\binom{n}{k}}} \left\|W_k^{1/2}\bigl(y_k^+ - X_k b\bigr)\right\|_2^2.
\tag{5}
$$

*Proof.* By Lemma 2, $G(M) = X_k^\top W_k X_k$ is positive definite for every $M > 0$, so the WLS problem in (5) has the unique minimizer
$$
b^*(M) = G(M)^{-1} X_k^\top W_k y_k^+ = G(M)^{-1} \sum_{T \in I} w_t x_T y_k(T),
$$
because $y_k^+$ vanishes identically on $E$. The right-hand vector $\sum_{T \in I} w_t x_T y_k(T)$ is independent of $M$. Taking $M \to \infty$ via Lemma 2 and applying Theorem 1 gives
$$
\lim_{M \to \infty} b^*(M) = \sum_{T \in I} w_t A_k x_T\, y_k(T) = \sum_{T \in I} q_T\, y_k(T).
$$
Adding the boundary term $Q_k y_k^- = \sum_{T \in E} q_T\, y_k(T)$ yields $\sum_{T \subseteq N} q_T\, y_k(T)$.

By definition of the order-$k$ Shapley Interaction Index (Grabisch & Roubens, 1999), for any set function $g : 2^N \to \mathbb{R}$ and $|S| = k$,
$$
\phi_k^{\mathrm{SII}}(g)(S)
= \sum_{V \subseteq N \setminus S} m_k(|V|) \sum_{J \subseteq S} (-1)^{k - |J|} g(V \cup J)
= \sum_{T \subseteq N} q_T(S)\, g(T),
$$
via the bijection $T \leftrightarrow (V, J) = (T \setminus S,\, T \cap S)$ with $|V| = t - |T \cap S|$. For $k = 1$, $y_1 = \nu$ gives (5) immediately. For $k \ge 2$, $\hat{\nu}_{k-1}$ is a linear combination of unanimity games $u_R(T) := \mathbf{1}_{R \subseteq T}$ with $1 \le |R| < k$ (in fact, even $R = \emptyset$ is annihilated). For any $|R| < k$ and $|S| = k$, pick an element $i \in S \setminus R$; pairing $J$ with $J \triangle \{i\}$ cancels every term in $\sum_{J \subseteq S} (-1)^{k - |J|} u_R(V \cup J)$ because $R \subseteq V \cup J \iff R \subseteq V \cup (J \triangle \{i\})$. Hence $\phi_k^{\mathrm{SII}}(\hat{\nu}_{k-1}) = 0$, and therefore $\phi_k^{\mathrm{SII}}(y_k) = \phi_k^{\mathrm{SII}}(\nu)$. $\square$

*Boundary case $n = 2k$:* When $n = 2k$, $I$ consists of the single size $t = k$, with $c = 1$ and $w_k = 1$. Every factorial and projection identity above holds verbatim, including $(n, k) = (2, 1)$ where $E = \{\emptyset, N\}$ and $x_N = \mathbf{1}$ spans $L$.

---

## 5. Reproducible Exact Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/kernelshap_iq_inverse/explore.py
```

The script [`explore.py`](explore.py) (which uses variable name `d` for $n$) performs exact rational arithmetic in SymPy (`bernoulli(k - r, 0)` ensuring $B_1 = -1/2$) with zero floating-point approximation:

- **Full-matrix verification (16 cases, $2 \le n \le 8$, $1 \le k \le \lfloor n/2 \rfloor$):** Constructs the full $2^n \times \binom{n}{k}$ design matrix $X_k$, computes a basis $Z$ of $\ker(X_{k,E})$, verifies $\dim \ker(X_{k,E}) = \binom{n}{k} - \binom{n}{k-1}$ and $G_I Z = (n - 2k + 1) Z$, forms $A_k = Z (Z^\top G_I Z)^{-1} Z^\top$, checks exact equality with (4), verifies every column of the split estimator $Q_k + A_k X_{k,I}^\top W_{k,I}$ against the SII operator on all $2^n$ point-mass games, and verifies exact vanishing on every unanimity game of order $< k$.
- **Johnson intersection and column-projection checks (256 cases, $2 \le n \le 32$, $1 \le k \le \lfloor n/2 \rfloor$, orders up to $k = 16$):** Directly checks the $k$-th forward-difference eigenvalue $\binom{n - 2k}{t - k}$ and the projected interior column identity $w_t A_k f_T = q_T$ for all $t \in [k, n - k]$ and overlaps $0 \le |S \cap T| \le k$.

All 16 full-matrix cases and 256 Johnson-scheme cases pass with exit status `0`.

---

## References

- S. Bordt and U. von Luxburg, *From Shapley Values to Generalized Additive Models and Back*, AISTATS 2023.
- I. Covert and S.-I. Lee, *Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression*, AISTATS 2021.
- P. Delsarte, *An Algebraic Approach to the Association Schemes of Coding Theory*, Philips Research Reports Supplements 10, 1973.
- F. Fumagalli, M. Muschalik, P. Kolpaczki, E. Hüllermeier, and B. Hammer, *KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions*, ICML 2024, [arXiv:2405.10852](https://arxiv.org/abs/2405.10852).
- D. H. Gottlieb, *A Certain Class of Incidence Matrices*, Proceedings of the American Mathematical Society 17(6):1233–1237, 1966.
- M. Grabisch and M. Roubens, *An Axiomatic Approach to the Concept of Interaction Among Players in Cooperative Games*, International Journal of Game Theory 28(4):547–565, 1999.
- S. M. Lundberg and S.-I. Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017.
