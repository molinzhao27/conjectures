# Refutation of the Bounded-Fan-Out Mildly Context-Sensitive Contrastive Learnability Conjecture

**Target paper:** Rakesh Sengupta, *A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners*, [arXiv:2610.28747](https://arxiv.org/abs/2610.28747), Section 6.1, Conjecture `conj:mcs` (Bounded-fan-out MCS).  
**Companion files:** [`solutions/reverse_gold_contrastive/explore.py`](explore.py), [`solutions/reverse_gold_contrastive/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:mcs` (Bounded-fan-out MCS): FALSE** on both conclusions—the collision bound and the uniform polynomial capacity bound fail independently.
- **Quantifier collapse to pointwise orbit purity (Proposition 1):** Because the conjecture quantifies uniformly over all admissible input distributions $p_d$ with $\operatorname{supp}(p_d) \subseteq L$ and support orbit bound $M \le M_0$, point masses $\delta_x$ on eligible positives are admissible. Consequently, a uniform collision bound $\mathbb{E}_{p_d}[\log(1 + m^+(x))] \le c < D(B, M_0)$ is equivalent to pointwise orbit purity ($m^+(x) = 0$) on every eligible positive.
- **Failure of the collision bound (Theorems 2 and 3):**
  1. *Unary fan-out-2 MCFG (Theorem 2):* A linear, non-erasing fan-out-2 MCFG with language $L = \{a^3, a^5, a^7\}$ and a fixed universal recursive operator whose global closure on **all** of $\{a\}^*$ satisfies $|\mathrm{Cl}_R(w)| \le |w| + 1$ has a witness $a^7$ with one positive training neighbor ($a^5$), one negative training neighbor ($a^6$), one joint co-substitution ($a^3$), and closure size $4$. On $p_d = \delta_{a^7}$, $\mathbb{E}[\log(1+m^+(x))] = \log 2 > D(B, 3)$ for every finite score bound $B > 0$.
  2. *Infinite marked MCFG and TAG families (Theorem 3):* With overtly distinct linked endpoints $U_i, V_j$ ($i, j \in \mathbb{Z}/3$), both a fan-out-2 MCFG (with middles $a^n b^n c^n$; optionally $\epsilon$-free starting at $P(ab,c)$ for $n \ge 1$) and a TAG (with middles $a^n h b^n$) satisfy $m^+(x) = m^-(x) = 2$ at **every** positive. For any target word, $|\mathrm{Cl}_R(x)| = 9$ and $|\mathrm{Cl}_R(L_{\le K})| = O(K)$ (while off-target strings with $m$ independent links have closure size $9^m$, in contrast to the unary MCFG's global $n+1$ bound on $\Sigma^*$). Every full-support distribution gives collision expectation $\log 3 > D(B, M_0)$, and on any fixed-depth slice even the paper's *actual* approximation gap forces mean loss at least $\log 3$, making the theorem's numerical bound vacuous.
- **Independent failure of the capacity bound even with pure training orbits (Theorem 4):** Using the same marked operator and the family $L_d = \{U_i a^t V_{i+d_t} : 1 \le t \le N,\; i \in \mathbb{Z}/3\}$ over all $N \ge 1$ and $d \in \{0,1\}^N$ (fan-out $2$, pure training orbits $m^+ = 0$, $M_0 = 6$, $|\mathrm{Cl}_R(L_{d,\le K})| \le 9K$), saturated realizability forces the common scorer class $\mathcal{H}$ to shatter $N$ probes with scores $\pm B$, yielding $\kappa \ge B^2 N$ for every $N \ge 1$. This obstruction uses unbounded grammar size (permitted by the stated conjecture); a binary-middle variant further gives $\kappa \ge B^2 2^{K-2}$ at fixed string length $K$.

---

## 1. Setup and Pointwise Orbit Purity

Fix a finite alphabet $\Sigma$, a target language $L \subseteq \Sigma^*$, a recursive contrast operator $R : \Sigma^* \to 2^{\Sigma^*}$ (local, universal, and compatible with structural composition), and its dependency-breaking training restriction $R_{\mathrm{train}}(x) \subseteq R(x)$. For $x \in L$, write
$$
m^+(x) := |R_{\mathrm{train}}(x) \cap L|, \qquad
m^-(x) := |R_{\mathrm{train}}(x) \setminus L|, \qquad
D(B, M_0) := \log 2 - \log(1 + M_0 e^{-2B}).
$$
These count distinct **strings**; membership $y \in L$ means $y$ admits *some* derivation in the target grammar, not necessarily one respecting the edited derivation tree.

**Conjecture `conj:mcs` (arXiv:2610.28747, Section 6.1).** *For any class $\mathcal{C}$ in the mildly context-sensitive hierarchy whose closure under a fixed multi-component operator $R$ of bounded fan-out is polynomially bounded in input size, there is a hypothesis class $\mathcal{H}$ with capacity $\kappa$ polynomial in the relevant complexity parameter (fan-out for MCFGs, adjunct depth for TAGs, phase depth for minimalist grammars) such that for every $L \in \mathcal{C}$ and every admissible input distribution $p_d$ with $\operatorname{supp}(p_d) \subseteq L$ and support orbit bound $M := \sup_{x \in \operatorname{supp}(p_d)} |R_{\mathrm{train}}(x)| \le M_0$, the pair $(R_{\mathrm{train}}, \mathcal{H})$ is saturatedly realizable and $\mathbb{E}_{p_d}[\log(1 + m^+(x))] \le c$ for some constant $c < D(B, M_0)$.*

Here $\mathcal{H}$ has capacity $\kappa$ if its empirical Rademacher complexity satisfies $\widehat{\mathfrak{R}}_S(\mathcal{H}) \le \sqrt{\kappa/n}$ for every $n$ and every $S \in \mathcal{X}_R^n$, where $\mathcal{X}_R := \operatorname{supp}(p_d) \cup \bigcup_{x \in \operatorname{supp}(p_d)} R_{\mathrm{train}}(x)$, and $|s(x)| \le B$ for all $s \in \mathcal{H}$ and $x \in \Sigma^*$.

**Proposition 1 (Pointwise orbit purity).** *Fix $L \in \mathcal{C}$, $R_{\mathrm{train}}$, $B > 0$, and $M_0 \ge 1$ such that $D(B, M_0) > 0$. Call $x \in L$ eligible if $|R_{\mathrm{train}}(x)| \le M_0$ (and $m^-(x) \ge 1$ if admissibility includes boundary support). There exists $c < D(B, M_0)$ with $\mathbb{E}_{p_d}[\log(1+m^+(x))] \le c$ for all admissible $p_d$ with $M \le M_0$ if and only if $m^+(x) = 0$ for every eligible $x \in L$.*

*Proof.* If $m^+(x) = 0$ for every eligible $x$, then $\mathbb{E}_{p_d}[\log(1+m^+(x))] = 0 < D(B, M_0)$, so $c = 0$ works. Conversely, for any eligible $x \in L$, the point mass $p_d = \delta_x$ is admissible (supported on $\{x\} \subseteq L$, with $M = |R_{\mathrm{train}}(x)| \le M_0$ and boundary support if $m^-(x) \ge 1$). If $m^+(x) \ge 1$, then
$$
\mathbb{E}_{\delta_x}[\log(1 + m^+(x))] = \log(1 + m^+(x)) \ge \log 2 > \log 2 - \log(1 + M_0 e^{-2B}) = D(B, M_0),
$$
contradicting any bound $c < D(B, M_0)$. $\square$

---

## 2. Unary Fan-Out-2 MCFG Counterexample (Global Polynomial Closure on $\Sigma^*$)

Recall from Seki, Matsumura, Fujii, and Kasami (1991) that a multiple context-free grammar (MCFG) of fan-out $k$ manipulates tuples of up to $k$ strings via linear, non-deleting string-combining functions.

**Theorem 2 (Unary fan-out-2 MCFG refutation).** *There exist a linear, non-erasing fan-out-2 MCFG generating $L = \{a^3, a^5, a^7\} \subseteq \{a\}^*$ and a fixed, universal, local, recursive contrast operator $R$ on $\{a\}^*$ with global closure bound $|\mathrm{Cl}_R(w)| \le |w| + 1$ for every $w \in \{a\}^*$ (and $|\mathrm{Cl}_R(\{a\}^{\le n})| \le n+1$) such that on $p_d = \delta_{a^7}$ (with boundary support, $M_0 = 3$, and saturated realizability in a singleton scorer class of capacity $\kappa = 0$),*
$$
\mathbb{E}_{p_d}[\log(1 + m^+(x))] = \log 2 > D(B, 3) \qquad (\forall B > 0).
$$

*Proof.* Over $\Sigma = \{a\}$, define the fan-out-2 MCFG with start symbol $S$, two-component nonterminal $P$, terminal productions
$$
P(a, a), \qquad P(aa, aa), \qquad P(aaa, aaa),
$$
and start rule $S(u\,a\,v) \leftarrow P(u, v)$. The two components of $P$ are siblings separated by an inert $a$, and the language is $L = \{a^3, a^5, a^7\}$. Take the singleton class $\mathcal{C} = \{L\}$.

Define $R$ uniformly on all of $\{a\}^*$: at every occurrence of the constant-width template $[aaa]\,a\,[aaa]$ (i.e., substring $a^7$ bracketed into two linked $aaa$ sites around a central $a$),
- $R_{\mathrm{train}}$ shortens either bracketed component to $aa$ or $a$ while leaving its mate fixed;
- the joint co-substitution replaces $(aaa, aaa)$ simultaneously by $(a, a)$.

These are fixed, target-independent local substitutions. Every occurrence of $a^7$ inside $u$ or $v$ persists in $uv$, so $R(uv) \supseteq R(u)v \cup uR(v)$ for arbitrary splits (recursivity holds unconditionally). Every edit strictly decreases string length, so the iterated closure $\mathrm{Cl}_R(w)$ of any $w \in \{a\}^*$ of length $n$ contains at most one string per length in $\{0, \dots, n\}$, giving $|\mathrm{Cl}_R(w)| \le n + 1$ (and likewise $|\mathrm{Cl}_R(\{a\}^{\le n})| \le n + 1$).

At $x = a^7$, the full orbit is:

| Move | Edited derivation tuple | Yield | Membership |
|---|---|---|---|
| Shorten one end by $1$ ($R_{\mathrm{train}}$) | $(aa, aaa)$ or $(aaa, aa)$ | $a^6$ | $a^6 \notin L$ |
| Shorten one end by $2$ ($R_{\mathrm{train}}$) | $(a, aaa)$ or $(aaa, a)$ | $a^5$ | $a^5 \in L$ via $P(aa, aa)$ |
| Shorten both ends by $2$ (joint) | $(a, a)$ | $a^3$ | $a^3 \in L$ (closure only) |

Both training edits break equality of the original components, yet $a^5$ admits an alternative derivation $S(aa \cdot a \cdot aa)$ shifting the central terminal. All three outputs have length $< 7$, so $\mathrm{Cl}_R(a^7) = \{a^7, a^6, a^5, a^3\}$ has size $4$. Here $|R_{\mathrm{train}}(a^7)| = 2$, $|R(a^7)| = 3$, $m^+(a^7) = 1$, and $m^-(a^7) = 1$. Taking $M_0 = 3$ (covering both $|R_{\mathrm{train}}|$ and $|R|$), $p_d = \delta_{a^7}$, and the singleton scorer class $\mathcal{H} = \{B \mathbf{1}_L - B \mathbf{1}_{L^c}\}$ (which has $\kappa = 0$ and saturatedly realizes $L$), we get $\mathbb{E}_{p_d}[\log(1+m^+(x))] = \log 2 > \log 2 - \log(1 + 3e^{-2B}) = D(B, 3)$ for every $B > 0$. $\square$

---

## 3. Infinite Marked MCFG and TAG Counterexamples

The collision obstruction does not depend on unary alphabets, degenerate finite languages, or point-mass distributions; it persists with overtly distinct endpoints, arbitrarily long intervening spans, full-support distributions, and both MCFGs (Seki et al., 1991) and Tree Adjoining Grammars (Joshi, 1985).

Define the alphabet $\Sigma = \{U_0, U_1, U_2, V_0, V_1, V_2, a, b, c, h\}$ with indices in $\mathbb{Z}/3$. Define a target-independent factor operator $R_{\mathrm{mark}}$ on $\Sigma^*$: for every factor $U_i w V_j$ with $w \in \{a,b,c,h\}^*$,
- $R_{\mathrm{train}}$ replaces $U_i$ by $U_{i+r}$ (fixing $V_j$) or $V_j$ by $V_{j+r}$ (fixing $U_i$) for $r \in \{1, 2\} \pmod 3$;
- the joint co-substitution replaces $(U_i, V_j)$ simultaneously by $(U_{i+r}, V_{j+r})$ for $r \in \{1, 2\} \pmod 3$.

Because factors persist under concatenation, $R_{\mathrm{mark}}$ is universal, local (fan-out $2$), and recursive under every split.

**Theorem 3 (Infinite marked MCFG and TAG refutation).** *Under $R_{\mathrm{mark}}$:*
1. *(MCFG)* *Let $L_{\mathrm{MCFG}}$ be generated by the fan-out-2 MCFG with rules $E_d(U_i, V_{i+d})$ ($i \in \mathbb{Z}/3$, $d \in \{0, 1\}$), $P(\epsilon, \epsilon)$, $P(a x b, c y) \leftarrow P(x, y)$, and $S(u x y v) \leftarrow E_d(u, v), P(x, y)$ ($d \in \{0, 1\}$). Its language is $\{U_i a^n b^n c^n V_{i+d} : n \ge 0,\; i \in \mathbb{Z}/3,\; d \in \{0,1\}\}$. (If strict $\epsilon$-freeness is demanded, replacing $P(\epsilon,\epsilon)$ by $P(ab,c)$ restricts to $n \ge 1$ with identical orbit and closure properties.)*
2. *(TAG)* *Let $L_{\mathrm{TAG}}$ be generated by the TAG with six initial trees $S(U_i, T(h), V_{i+d})$ ($i \in \mathbb{Z}/3$, $d \in \{0,1\}$) and one auxiliary tree $T(a, T^*, b)$ (foot $T^*$). Its language is $\{U_i a^n h b^n V_{i+d} : n \ge 0,\; i \in \mathbb{Z}/3,\; d \in \{0,1\}\}$.*

*For both languages:*
- *Every positive $x \in L$ has exactly one linked factor, so $|\mathrm{Cl}_{R_{\mathrm{mark}}}(x)| = 9$ (all $9$ endpoint pairs with fixed middle) and $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_{\le K})| = O(K)$ on all target words of length at most $K$.*
- *Every $x \in L$ has $|R_{\mathrm{train}}(x)| = 4$, $|R_{\mathrm{mark}}(x)| = 6$, and $m^+(x) = m^-(x) = 2$. Hence for every distribution $p_d$ supported on $L$ (including full-support laws such as depth-$n$ mass $2^{-n-1}$ split uniformly across the six positives at depth $n$), $\mathbb{E}_{p_d}[\log(1 + m^+(x))] = \log 3 > D(B, M_0)$ for every $B > 0$ and $M_0 \ge 4$.*
- *Moreover, on any uniform six-string fixed-depth slice, **every** bounded scorer $s : \Sigma^* \to [-B, B]$ incurs mean InfoNCE loss at least $\log 3$; when the common loss floor $f = \log(1 + 4e^{-2B})$ satisfies $f < \log 2$, the actual approximation-gap ratio $(\log 3 - f)/(\log 2 - f)$ exceeds $1$, rendering the paper's misranking bound vacuous even without passing through the collision upper bound.*

*Proof.* Every word $x = U_i w V_{i+d} \in L$ has a unique offset $d \in \{0, 1\} \pmod 3$. All four training edits change the offset $j - i \pmod 3$ from $d$ to $d \pm 1$: two edits yield $1 - d \in \{0, 1\}$ (landing in $L$ via the other $E_{1-d}$ rule or initial tree) and two yield offset $2 \notin \{0,1\}$ (landing outside $L$). The two joint edits shift both indices by the same $r \in \{1, 2\}$, preserving $d$. Thus $m^+(x) = m^-(x) = 2$ everywhere on $L$, and iterated edits reach exactly the $9$ pairs $\{U_{i'} w V_{j'} : i', j' \in \mathbb{Z}/3\}$. Since $L_{\mathrm{MCFG}}$ has at most $6$ words of each length $3n+2$ (and $L_{\mathrm{TAG}}$ at most $6$ of each length $2n+3$), $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_{\le K})| \le 9(\lfloor(K-2)/3\rfloor + 1)$ for $L_{\mathrm{MCFG}}$ and $\le 9(\lfloor(K-3)/2\rfloor + 1)$ for $L_{\mathrm{TAG}}$, both $O(K)$.

For the direct loss lower bound on a fixed-depth six-string slice $S_n = \{U_i w_n V_{i+d} : i \in \mathbb{Z}/3,\; d \in \{0,1\}\}$, let $s_x := s(x)$ for any scorer $s$. Dropping the two strictly positive negative exponentials from the InfoNCE loss at $x \in S_n$ and applying the AM–GM inequality to the self term $1$ and the two in-language training collision terms $e^{s_y - s_x}$ ($y \in R_{\mathrm{train}}(x) \cap S_n$) gives
$$
\ell(s; x)
= \log\!\biggl(1 + \sum_{y \in R_{\mathrm{train}}(x)} e^{s_y - s_x}\biggr)
> \log\!\biggl(1 + \sum_{y \in R_{\mathrm{train}}(x) \cap S_n} e^{s_y - s_x}\biggr)
\ge \log 3 + \frac{1}{3} \sum_{y \in R_{\mathrm{train}}(x) \cap S_n} (s_y - s_x).
$$
The collision relation $y \in R_{\mathrm{train}}(x) \cap S_n$ is symmetric and $2$-regular on $S_n$ (each vertex flips its offset between $0$ and $1$ by changing either $U$ or $V$). Averaging over the six vertices of $S_n$ cancels all $s_y - s_x$ terms identically, yielding $\frac{1}{6}\sum_{x \in S_n} \ell(s; x) \ge \log 3$. Subtracting the per-example floor $f = \log(1 + 4e^{-2B})$ gives actual approximation gap $a_{\mathcal{H}} \ge \log 3 - f$, so for $f < \log 2$ the misranking-bound term $a_{\mathcal{H}}/(\log 2 - f) \ge (\log 3 - f)/(\log 2 - f) > 1$ is vacuous, even though the membership scorer $B\mathbf{1}_L - B\mathbf{1}_{L^c}$ achieves zero misranking error. $\square$

**Remark (Closure scope: target words vs. arbitrary strings in $\Sigma^*$).** For $R_{\mathrm{mark}}$, every target word has a single linked factor and closure size $9$, so $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_{\le K})| = O(K)$. However, off-target strings containing $m$ disjoint linked factors (e.g., $U_0 a V_0 U_1 b V_1$, with $m = 2$) have closure size $9^m$, which grows exponentially in string length over all of $\Sigma^*$. By contrast, the unary MCFG operator of Theorem 2 satisfies $|\mathrm{Cl}_R(w)| \le |w| + 1$ globally on **all** of $\Sigma^*$, showing that the collision obstruction holds even under the strictest global-closure reading.

---

## 4. Independent Capacity Obstruction Under Pure Training Orbits

Even when training orbits are completely pure ($m^+(x) = 0$ everywhere, so the collision condition holds with $c = 0$), the uniform polynomial capacity conclusion fails independently.

**Theorem 4 (Unbounded capacity with fan-out 2 and pure training orbits).** *Fix the marked operator $R_{\mathrm{mark}}$ and let*
$$
\mathcal{C}_{\mathrm{cap}} := \bigl\{L_d : N \ge 1,\; d = (d_1, \dots, d_N) \in \{0,1\}^N\bigr\}, \qquad
L_d := \bigl\{U_i a^t V_{i+d_t} : 1 \le t \le N,\; i \in \mathbb{Z}/3\bigr\}.
$$
*Every $L_d \in \mathcal{C}_{\mathrm{cap}}$ is generated by a fan-out-2 MCFG, has $|\mathrm{Cl}_{R_{\mathrm{mark}}}(x)| = 9$ for all $x \in L_d$ and $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_{d,\le K})| \le 9K$, has pure training orbits ($m^+(x) = 0$, $m^-(x) = 4$), and has membership-preserving co-substitutions ($|R_{\mathrm{mark}}(x)| = 6$). Yet any bounded scorer class $\mathcal{H} \subseteq [-B, B]^{\Sigma^*}$ that saturatedly realizes $(R_{\mathrm{train}}, L_d)$ for every $L_d \in \mathcal{C}_{\mathrm{cap}}$ and admissible $p_d$ (with fixed $M_0 = 6$) must satisfy*
$$
\kappa \ge B^2 N \qquad \text{for every } N \ge 1,
$$
*so no finite capacity $\kappa$—in particular, no polynomial in fan-out—can bound $\mathcal{H}$ uniformly over $\mathcal{C}_{\mathrm{cap}}$.*

*Proof.* Each $L_d$ is generated by the fan-out-2 MCFG with rules $E_0(U_i, V_i)$, $E_1(U_i, V_{i+1})$, and $S(u a^t v) \leftarrow E_{d_t}(u, v)$ ($1 \le t \le N$). At each length $t + 2$, only the single offset $d_t \in \{0,1\}$ is licensed, so every one-end edit lands outside $L_d$ ($m^+ = 0, m^- = 4$) while both joint edits preserve $d_t$.

Fix $N \ge 1$. For each $d \in \{0,1\}^N$, let $p_d$ be uniform over the $N$ support positives $x_t^{(d)} := U_0 a^t V_{d_t}$ ($1 \le t \le N$), and consider the fixed probe sample $S = (z_1, \dots, z_N)$ with $z_t := U_0 a^t V_0$. For each $t$:
- if $d_t = 0$, then $z_t = x_t^{(d)} \in \operatorname{supp}(p_d) \subseteq L_d$;
- if $d_t = 1$, then $z_t \in R_{\mathrm{train}}(x_t^{(d)}) \setminus L_d$ (obtained by shifting $V_1 \mapsto V_0$).

Thus saturated realizability of $L_d$ supplies a scorer $s_d \in \mathcal{H}$ with $s_d(z_t) = +B$ when $d_t = 0$ and $s_d(z_t) = -B$ when $d_t = 1$. Moreover, for the all-zero target $L_{\mathbf{0}}$, every $z_t$ lies in $\operatorname{supp}(p_{\mathbf{0}}) \subseteq \mathcal{X}_R$, so $S \in \mathcal{X}_R^N$ is a valid worst-case sample even for a single target's domain. For every realization of Rademacher signs $\sigma \in \{\pm 1\}^N$, choosing $d_t = \mathbf{1}_{\sigma_t = -1}$ yields $\sup_{s \in \mathcal{H}} \frac{1}{N}\sum_{t=1}^N \sigma_t s(z_t) = B$. Taking the expectation over $\sigma$ gives $\widehat{\mathfrak{R}}_S(\mathcal{H}) = B$, so Definition `def:hclass` ($\widehat{\mathfrak{R}}_S(\mathcal{H}) \le \sqrt{\kappa/N}$) forces $B \le \sqrt{\kappa/N}$, i.e., $\kappa \ge B^2 N$. $\square$

**Remark (Grammar-size caveat and binary-middle lift).** Theorem 4 uses grammars whose description size grows with $N$ while keeping fan-out fixed at $2$, which is allowed because Conjecture `conj:mcs` bounds $\kappa$ as a polynomial in fan-out alone and places no cap on grammar size. If one further replaces the unary middles $a^t$ by all $2^{K-2}$ binary strings $w \in \{a, b\}^{K-2}$ at a fixed word length $K \ge 2$ (with rules $S(u w v) \leftarrow E_{d_w}(u, v)$), the same shattering argument on the $2^{K-2}$ probes $U_0 w V_0$ forces $\kappa \ge B^2 2^{K-2}$ at fixed string length $K$ (at the cost of $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_d)| = 9 \cdot 2^{K-2}$, whereas the unary-middle family in Theorem 4 keeps $|\mathrm{Cl}_{R_{\mathrm{mark}}}(L_{d,\le K})| \le 9K$ linear in $K$). Note also that Theorem 4 remains valid even if "dependency-breaking" is read semantically to require global ungrammaticality, because every training neighbor of every word in $L_d$ is already out-of-language.

---

## 5. Verification and Literature Notes

- **Python verification** ([`solutions/reverse_gold_contrastive/explore.py`](explore.py), standard library only):
  - Verifies the unary MCFG witness (`train 2, in 1, out 1, joint 1, closure 4`) and checks the global unary closure bound $|\mathrm{Cl}_R(a^n)| \le n+1$ across all 65 lengths $n = 0, \dots, 64$.
  - Verifies all $246$ marked MCFG positives and $246$ marked TAG positives across depths $n = 0, \dots, 40$, confirming `train 4, in 2, out 2, joint 2, closure 9` at every word, and checks the two-link off-target string `AaXBbY` has closure size $81$.
  - Verifies all $510 = \sum_{N=1}^8 2^N$ binary assignments in the capacity construction for $N = 1, \dots, 8$, confirming orbit purity and exact $\pm B$ probe forcing.
  - Confirms at $B = 2$ that $\log 3 \approx 1.098612289$ exceeds $D(2, 4) \approx 0.622444054$ and $D(2, 6) \approx 0.588882815$.
- **Relation to proved results in arXiv:2610.28747:** The paper's general contrastive ranking bound (Theorem `thm:main`), collision bound (Proposition `prop:collision`), and strictly 2-local worked example (forbidding $bb$ under $b$-doubling, with capacity $5B^2(K+1)^2$) are unaffected. What fails in `conj:mcs` is inferring either low orbit collision or low uniform capacity from polynomial closure and bounded fan-out alone.
- **References:**
  - H. Seki, T. Matsumura, M. Fujii, and T. Kasami (1991), *On multiple context-free grammars*, Theoretical Computer Science 88(2):191–229.
  - A. K. Joshi (1985), *Tree adjoining grammars: How much context-sensitivity is required to provide reasonable structural descriptions?*, in *Natural Language Parsing*, Cambridge University Press, pp. 206–250.
