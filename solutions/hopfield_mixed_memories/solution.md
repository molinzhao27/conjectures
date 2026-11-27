# Refutation of Conjecture 1.4 (`1.conj.2`) — Mixed Memories in Hopfield Networks (arXiv:2504.04879)

- **Conjecture Source**: Véronique Gayrard, *"Mixed memories in Hopfield networks"*, [arXiv:2504.04879](https://arxiv.org/abs/2504.04879) (`Mix-ALL_v2.tex`, revised June 2026), Conjecture 1.4 (TeX label `1.conj.2`).
- **Domain**: Computational Neuroscience / Statistical Mechanics of Associative Memory (Classical, Dense, and General Hopfield Networks).
- **Verdict**: **FALSE** (the "only if" direction fails structurally; the "if" direction is Gayrard's Theorem `1.theo2.mix`) — refuted for **both** the classical Hopfield model $F(x) = \frac{1}{2}x^2$ (for **every odd $n \ge 5$**, while true for $n \in \{1, 3\}$) **and** dense Hopfield models $F(x) = \frac{1}{p}x^p$ (proved rigorously for $3 \le p \le 19$ and constructively verified via exact rational arithmetic for $20 \le p \le 500$, with $n_0(3) = n_0(4) = 7$ and lifting to all odd $n \ge n_0(p)$), as well as for general smooth $F$ with $F'(x) > 0$ on $(0, \infty)$ (at $n = 1, 3$ when $F'$ is not odd, and at $n = 5$ even for smooth odd strictly increasing $F'$ with $m \ge 0$).
- **Prior Art & Novelty Note**:
  - **Classical Hopfield ($p = 2$) prior art:** The $n = 5$ counterexample $(\frac{5}{8}, \frac{3}{8}, \frac{3}{8}, \frac{1}{8}, \frac{1}{8})$, the even-multiplicity obstruction (Lemma B), and the even-block lifting to all odd $n \ge 5$ were posted publicly in October 2026 (`the-omega-institute/trureturing#13571`, PR `#13650`, and `#17309`). In addition, Francesco Tosello announced a UBC seminar (Nov 18, 2026, *"Stable Mixed Memories in the Hopfield Model"*, preprint not yet public) addressing stable mixed memories and Gayrard's conjecture; that work should be consulted once publicly posted.
  - **Genuinely novel contributions here:** (i) the **dense Hopfield ($p \ge 3$)** refutation — minimal $n = 7$ witness $(\frac{11}{32}^3, \frac{9}{32}^4)$ for $p = 3, 4$ (certified in Lean 4), the general two-block perturbed-majority family $(2a+1, 2b)$ with exact Chow formulas $(\Phi_1, \Phi_2)$, exact rational proof for $p = 3..19$ and exact rational verification for $p = 20..500$, and the Parseval lifting inequality $\sum \tilde{m}_\tau^{p-1} \le 1 < 2$ (also certified in Lean 4); and (ii) the **general smooth $F$** counterexamples ($F(x) = e^x - x$ at $n = 1, 3$ and $F(x) = \sqrt{1 + 4x^2}$ at $n = 5$).

---

## 1. Foundational Characterization & Invariant

### Lemma A (Exact Characterization of Tie-Free $n$-Mixed Memories)
Let $F : \mathbb{R} \to \mathbb{R}$ be smooth with $F'(x) > 0$ for $x > 0$, let $n \in \mathbb{N}$ be independent of $N$, and let $m \in [-1,1]^M$ be deterministic with exactly $n$ non-zero entries $\hat{m} = (m_{\mu_1}, \dots, m_{\mu_n}) \in ([-1,1]\setminus\{0\})^n$. Set $w = (F'(\hat{m}_1), \dots, F'(\hat{m}_n)) \in \mathbb{R}^n$.
Suppose:
1. **(Tie-freeness)** $(r, w) := \sum_{\nu=1}^n r_\nu w_\nu \neq 0$ for every $r \in \{-1,1\}^n$; and
2. **(Chow fixed-point equation)**
   $$
   \hat{m}_\nu = \Phi_\nu(w) := \frac{1}{2^n} \sum_{r \in \{-1,1\}^n} r_\nu \operatorname{sign}\bigl((r, w)\bigr) \qquad \forall\, 1 \le \nu \le n. \tag{$\star$}
   $$

Then:
- **(a)** $\xi^{(N)}(m) \in \{-1,1\}^N = \Sigma_N$ deterministically for every $N \ge 1$, and $\xi^{(N)}(m)$ is an **$n$-mixed memory of type $F$** in the sense of Definition `1.def1.2` (by Kolmogorov's Strong Law of Large Numbers on i.i.d. bounded $\{-1,1\}$ random variables $\xi_i^\mu \xi_i(m)$).
- **(b)** Letting $C_F(m) := \min_{r \in \{-1,1\}^n} |(r, w)| > 0$, Gayrard's Section 4 Borel–Cantelli proofs (Lemmas `4.lem1`–`4.lem4bis`) depend only on $(\star)$ and $C_F(m) > 0$; thus $\xi^{(N)}(m)$ is an **asymptotically almost sure local minimum** (fixed point of $T^G$ and $T^{HK}$) whenever $M(N) \le \frac{C_F(m)^2 N}{2(2+\varepsilon)\ln N}$ (for $p=2$) or $M(N) \le \frac{C_F(m)^2 N^{p-1}}{2(2+\varepsilon)[(2p-3)!!]\ln N}$ (for $p \ge 3$). $\blacksquare$

### Lemma B (Multiplicity Invariant of $\mathcal{M}_n^{\mathrm{all}}$)
Let $n \in \mathbb{N}$ be odd and $m' \in \mathcal{M}_n^{\mathrm{all}}$ (in particular, any $m' \in \mathcal{M}_{n,F}$). Let $v_1 > v_2 > \dots > v_r > 0$ be the distinct positive magnitudes appearing in $|m'|$, with multiplicities $c_1, \dots, c_r$. Then:
$$
c_1, c_2, \dots, c_{r-1} \text{ are ALL EVEN, and } c_r \text{ is ODD.}
$$

*Proof.* By definition, $|m'|$ has values $\gamma^{(k)} = \prod_{l=1}^k \alpha^{(n_l)}$ with multiplicity $n_k$ ($1 \le k \le \ell$), where $n_1, \dots, n_{\ell-1} \ge 2$ are even and $n_\ell \ge 1$ is odd. Since $\alpha^{(2j)} = 2^{-2j}\binom{2j}{j} \le \frac{1}{2} < 1$ for all $j \ge 1$, $\gamma^{(1)} > \dots > \gamma^{(\ell-1)} > 0$. If $n_\ell = 1$ then $\alpha^{(1)} = 1$ merges $\gamma^{(\ell)} = \gamma^{(\ell-1)}$, giving $(c_1,\dots,c_r) = (n_1,\dots,n_{\ell-2}, n_{\ell-1}+1)$ (or $(1)$ if $\ell=1$); if $n_\ell \ge 3$ then $\alpha^{(n_\ell)} = \alpha^{(n_\ell-1)} \le \frac{1}{2} < 1$, giving $(c_1,\dots,c_r) = (n_1,\dots,n_{\ell-1}, n_\ell)$. In both cases $c_1,\dots,c_{r-1}$ are even and $c_r$ is odd. $\blacksquare$

---

## 2. Theorem 1: Refutation for Classical Hopfield $F(x) = \frac{1}{2}x^2$ (Every Odd $n \ge 5$)

**Theorem 1.** Let $F(x) = \frac{1}{2}x^2$ ($F'(x) = x$).
1. For $n \in \{1, 3\}$, every tie-free $n$-mixed memory of type $F$ satisfies $m \in \mathcal{M}_{n,F}$.
2. For **every integer $k \ge 0$** and $n = 2k + 5$ (all odd $n \ge 5$) and $M \ge n$,
   $$
   m = \alpha^{(2k)}\left(\underbrace{1,\dots,1}_{2k},\;\frac{5}{8},\;\frac{3}{8},\;\frac{3}{8},\;\frac{1}{8},\;\frac{1}{8},\;0,\dots,0\right) \in [-1,1]^M \qquad (\alpha^{(0)} := 1)
   $$
   is strictly tie-free with margin $C(m) = \frac{1}{8}\alpha^{(2k)} > 0$, defines a genuine $n$-mixed memory $\xi^{(N)}(m)$ of type $F$ (and an a.s. asymptotic local minimum of $E_{N,M}$), and satisfies **$m \notin \mathcal{M}_n^{\mathrm{all}}$** (hence $m \notin \mathcal{M}_{n,F}$).

### Proof
- **Base case $k = 0$ ($n = 5$):** For $\tilde{m} = (\frac{5}{8}, \frac{3}{8}, \frac{3}{8}, \frac{1}{8}, \frac{1}{8})$, $8(r, \tilde{m}) = 5r_1 + 3r_2 + 3r_3 + r_4 + r_5 \equiv 1 \pmod 2$, so $|(r, \tilde{m})| \ge \frac{1}{8} > 0$ for all $r \in \{-1,1\}^5$. Among the $16$ vectors with $r_1 = +1$, $8(r, \tilde{m}) < 0$ holds **only** on the $3$ vectors $B = \{(+1,-1,-1,+1,-1),\; (+1,-1,-1,-1,+1),\; (+1,-1,-1,-1,-1)\}$. By oddness of $\operatorname{sign}$,
  - $\Phi_1(\tilde{m}) = \frac{13 - 3}{16} = \frac{5}{8} = \tilde{m}_1$;
  - $\Phi_2(\tilde{m}) = \Phi_3(\tilde{m}) = \frac{0 - 2(-3)}{16} = \frac{3}{8} = \tilde{m}_2 = \tilde{m}_3$;
  - $\Phi_4(\tilde{m}) = \Phi_5(\tilde{m}) = \frac{0 - 2(-1)}{16} = \frac{1}{8} = \tilde{m}_4 = \tilde{m}_5$.
  Its strictly decreasing positive values $\frac{5}{8} > \frac{3}{8} > \frac{1}{8}$ have multiplicities $(1, 2, 2)$, violating Lemma B, so $\tilde{m} \notin \mathcal{M}_5^{\mathrm{all}}$. (Its limiting energy is $-\frac{45}{128}$, strictly below Gayrard's $(2,3)$ memory at $-\frac{44}{128}$ and equal to the symmetric $(3/8)^5$ memory.)
- **General case $k \ge 1$ ($n = 2k+5$):** Writing $r = (r', r'') \in \{-1,1\}^{2k} \times \{-1,1\}^5$ and $S_1(r') = \sum_{\nu=1}^{2k} r'_\nu$, since $|(r'', \tilde{m})| \le \frac{13}{8} < 2$, whenever $S_1(r') \neq 0$ we have $|S_1(r')| \ge 2$ and $\operatorname{sign}((r, \hat{m})) = \operatorname{sign}(S_1(r'))$ with margin $\ge \frac{3}{8}\alpha^{(2k)}$; when $S_1(r') = 0$, $\operatorname{sign}((r, \hat{m})) = \operatorname{sign}((r'', \tilde{m}))$ with margin $\ge \frac{1}{8}\alpha^{(2k)}$. Splitting $\sum_{r', r''}$ into $\{S_1 \neq 0\}$ and $\{S_1 = 0\}$ yields $\Phi_\nu(\hat{m}) = \alpha^{(2k)}$ ($1 \le \nu \le 2k$) and $\Phi_{2k+\tau}(\hat{m}) = 2^{-2k}\binom{2k}{k}\tilde{m}_\tau = \alpha^{(2k)}\tilde{m}_\tau$ ($1 \le \tau \le 5$). Its multiplicities $(2k, 1, 2, 2)$ violate Lemma B. $\blacksquare$

---

## 3. Theorem 2: Refutation for Dense Hopfield Models $F(x) = \frac{1}{p}x^p$ ($p \ge 3$)

**Theorem 2.** For every integer $p \in [3, 500]$ (proved analytically/symbolically for $3 \le p \le 19$ via six explicit $(a, b)$ pairs, and verified via exact rational arithmetic for $20 \le p \le 500$; numerically observed to cover all $p \lesssim 8.8 \times 10^{13}$), there exists an odd $n_0(p) \ge 7$ such that for **every odd $n \ge n_0(p)$** there is a strictly tie-free $n$-mixed memory $m \in [0,1]^M$ of type $F(x) = \frac{1}{p}x^p$ (an a.s. asymptotic local minimum for $T^G$ and $T^{HK}$) with $m \notin \mathcal{M}_n^{\mathrm{all}}$. For $p \in \{3, 4\}$, $n_0(p) = 7$ via
$$
\tilde{m} = \left(\frac{11}{32},\;\frac{11}{32},\;\frac{11}{32},\;\frac{9}{32},\;\frac{9}{32},\;\frac{9}{32},\;\frac{9}{32}\right) \notin \mathcal{M}_7^{\mathrm{all}}.
$$

### Proof Summary
1. **Two-block perturbed-majority family $(2a+1, 2b)$ ($a \ge 1$, $b \ge a+1$, $n_0 = 2a+2b+1$):** For weights $w = (A^{2a+1}, B^{2b})$ satisfying
   $$
   \frac{2a+2}{2a+1} < \frac{A}{B} < \frac{2a}{2a-1}, \tag{R}
   $$
   $(r, w) \neq 0$ for all $r \in \{-1,1\}^{n_0}$, and $\operatorname{sign}((r,w)) \neq \operatorname{maj}(r)$ iff $r \in E_+ \cup (-E_+)$ where $E_+ = \{S_1(r') = 2a+1,\ S_2(r'') = -(2a+2)\}$, $|E_+| = \binom{2b}{b-a-1}$. Thus the exact Chow parameters are $\Phi(w) = (\Phi_1^{2a+1}, \Phi_2^{2b})$ with
   $$
   \Phi_1(a,b) = \alpha^{(n_0)}(1 + x(a,b)), \qquad \Phi_2(a,b) = \alpha^{(n_0)}\left(1 - \frac{a+1}{b}x(a,b)\right), \qquad x(a,b) = \frac{2\binom{2b}{b-a-1}}{\binom{2a+2b}{a+b}}.
   $$
   Setting $q(a,b) := \Phi_1(a,b)/\Phi_2(a,b)$, whenever $\frac{2a+2}{2a+1} < q(a,b)^{p-1} < \frac{2a}{2a-1}$, the vector $\tilde{m}(a,b) = (\Phi_1^{2a+1}, \Phi_2^{2b})$ solves $(\star)$ for $F(x) = \frac{1}{p}x^p$ and has multiplicities $(2a+1, 2b)$ (odd, even), violating Lemma B.
2. **Universal Parseval Lifting ($n_0 \mapsto n_0 + 2k$ for all $k \ge 1$):** For any tie-free solution $\tilde{m} \in (0,1)^{n_0}$ of $(\star)$ and any $p \ge 3$, Parseval's identity for the Boolean function $g(r'') = \operatorname{sign}((r'', \tilde{m}^{p-1}))$ gives $\sum_{\tau=1}^{n_0} \tilde{m}_\tau^{p-1} \le \sum_{\tau=1}^{n_0} \tilde{m}_\tau^2 \le 1 < 2$. Hence $\hat{m} = \alpha^{(2k)}(1^{2k}, \tilde{m})$ is strictly tie-free, solves $(\star)$ for $F(x) = \frac{1}{p}x^p$, and violates Lemma B.
3. **Explicit coverage in $p$:**
   - **$3 \le p \le 19$ (proved via exact finite rational checks):** explicitly covered by $(a,b) \in \{(1,2), (2,6), (2,5), (2,4), (3,13), (2,3)\}$ (giving $q \in \{\frac{11}{9}, \frac{121}{115}, \frac{587}{563}, \frac{235}{228}, \frac{138977}{137107}, \frac{127}{125}\}$).
   - **$20 \le p \le 500$ (exact rational verification) and higher $p$ (numerical):** `verify_hopfield.py` constructs and verifies an exact rational $(a, b)$ satisfying (R) for every integer $p \in [20, 500]$, and numerical monotonicity/overlap checks of $q(a, b)$ cover $a = 4..25$ up to $p \approx 8.8 \times 10^{13}$ (an explicit analytic proof of the monotonicity and interval-overlap inequalities for all $a \ge 6$ is not yet written out). $\blacksquare$

---

## 4. General Smooth $F$ Counterexamples & Conceptual Diagnosis

- **Sign-parity defect at $n = 1, 3$:** For $F(x) = e^x - x$ ($F'(x) = e^x - 1 > 0$ on $(0,\infty)$, not odd), Eq. `(1.2.1.15)` restricts $\mathcal{M}_{n,F}$ to non-negative vectors, yet $m = (-1, 0, \dots)$ ($n=1$) and $m = (-\frac{1}{2}, -\frac{1}{2}, -\frac{1}{2}, 0, \dots)$ ($n=3$) are genuine tie-free mixed memories of type $F$.
- **Odd strictly increasing $F'$, $m \ge 0$, $n = 5$:** For $F(x) = \sqrt{1 + 4x^2}$, $\hat{m} = (\frac{7}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}, \frac{1}{8}) \notin \mathcal{M}_5^{\mathrm{all}}$ satisfies $2 < F'(7/8)/F'(1/8) = \sqrt{833/65} < 4$ and solves $(\star)$ with zero ties.
- **Why `1.conj.2` fails:** Tie-free $n$-mixed memories are fixed points $\hat{m} = \Phi(F'(\hat{m}))$ of the Chow-parameter map on odd linear threshold functions (LTFs). Gayrard's $\mathcal{M}_n^{\mathrm{all}}$ enumerates only *lexicographic* (hierarchical-majority) LTFs. All non-degenerate odd LTFs are lexicographic for $n = 1, 3$, but non-lexicographic odd LTFs appear at $n = 5$ (such as $(3,2,2,1,1)$ for $p=2$ and $(3,1,1,1,1)$) and at $n = 7$ (such as $(11^3, 9^4)$ for $p = 3, 4$), persisting for every polynomial $F(x) = x^p/p$.

Computational verification: [`verify_hopfield.py`](./verify_hopfield.py).
