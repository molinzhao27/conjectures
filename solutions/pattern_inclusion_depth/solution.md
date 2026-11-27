# Partial Resolution of the Monotone-Measure Conjecture for Pattern Inclusion Depth

**Target paper:** Wei Luo, *Compute Inclusion Depth of a Pattern*, COLT 2005 (LNCS 3559, pp. 689–690) / [arXiv:2605.30389](https://arxiv.org/abs/2605.30389) (2026), Definition 1, Questions 1–2, and Conjecture `conj:monotone-measure`.  
**Companion files:** [`solutions/pattern_inclusion_depth/explore.py`](explore.py), [`solutions/pattern_inclusion_depth/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:monotone-measure` (non-erasing pattern languages over $|\Sigma| \ge 2$): OPEN in general**, and **PROVED** over every finite alphabet of size $|\Sigma| \ge 2$ in each of the following regimes:
  1. **Length gap at most one (Theorem 2):** Whenever $n(p) - n(q) \le 1$, at arbitrary pattern lengths and with arbitrary terminals and repeated variables.
  2. **All patterns of length at most $12$ (Theorem 3):** Whenever $n(p) \le 12$, establishing the exact depth formula $\operatorname{ID}_\Sigma(p) = 2n(p) - v(p) - 1$ for every pattern of length at most $12$ over every finite alphabet $|\Sigma| \ge 2$.
  3. **Morphic, low-multiplicity-excess, and small-defect upper patterns (Propositions 1, 2, and 4):** At arbitrary pattern lengths whenever inclusion is witnessed by a non-erasing morphism; whenever $q$'s minimum variable multiplicity $f$ and multiplicity excess $E(q)$ satisfy $E(q) < (2f - 1)(n(p) - n(q))$ (in particular, whenever all variables of $q$ share a common multiplicity $f$); or whenever $c(q) := n(q) - v(q) \le 2$.
- **Alphabet-reduction correction (Lemma 1):** Luo's remark (`arXiv:2605.30389`, Section 3) passes to a two-letter subalphabet $\Sigma' \subseteq \Sigma$, which fails when $p$ and $q$ together contain more than two distinct terminal symbols. Replacing restriction by a letter surjection $\pi : \Sigma \to \{0,1\}$ with a section lift repairs the reduction for all positive length gaps.
- **Unresolved scope:** Nonmorphic inclusions with length gap $d \ge 2$, longer length $n(p) \ge 13$, defect $c(q) \ge 3$, and $E(q) \ge (2f - 1)d$ remain open, as do general decidability and polynomial-time computability of $\operatorname{ID}_\Sigma(p)$ (Question 1; we prove finiteness and upper semicomputability).
- **Variant refutations (Proposition 6):** The **erasing** analogue is false over every nonempty alphabet (even without terminals, and via genuinely nonmorphic binary pairs), and erasing inclusion depth can be unbounded for a fixed pattern ($p = XX$). The **unary** non-erasing analogue ($|\Sigma| = 1$) is also false. Neither variant refutes the published conjecture.

*Prior-art note:* Luo ([arXiv:2605.30389](https://arxiv.org/abs/2605.30389)) verified lengths up to $7$. A later note (*A monotone measure for pattern-language inclusion: every morphic inclusion, every pattern of length at most eight, and the inclusion depth in that range*, [`ideosphere.ai/papers/patlang-depth`](https://ideosphere.ai/papers/patlang-depth)) proves the morphic case and checks binary length up to $8$. All proofs and computations below are independent and self-contained.

---

## 1. Definitions and Foundational Reductions

Fix a finite alphabet $\Sigma$ with $s := |\Sigma| \ge 2$ and a disjoint countable set of variables $X$. A *pattern* is a nonempty word $p \in (\Sigma \cup X)^+$. Its **non-erasing pattern language** $L_\Sigma(p)$ consists of all ground strings obtained by replacing each variable with a nonempty word in $\Sigma^+$ (uniformly across occurrences) while leaving terminals fixed. In the *erasing* variant $L_{\Sigma,\mathrm{E}}(p)$, variables may also map to the empty word $\varepsilon$. Write:
- $n(p)$ for the length (number of positions) of $p$,
- $v(p)$ for the number of distinct variables occurring in $p$,
- $t(p)$ for the number of terminal positions in $p$,
- $c(p) := n(p) - v(p)$ for the **defect** (terminal positions plus variable occurrences beyond the first), and
- $\mu(p) := 2n(p) - v(p) = n(p) + c(p)$ for Luo's measure.

**Conjecture `conj:monotone-measure`** asserts that for any patterns $p, q$ over $|\Sigma| \ge 2$,
$$
L_\Sigma(p) \subsetneq L_\Sigma(q) \quad \Longrightarrow \quad \mu(p) > \mu(q).
$$
Writing $d := n(p) - n(q)$, the inequality $\mu(p) > \mu(q)$ is equivalent to $v(p) < v(q) + 2d$, or $c(p) > c(q) - d$.

### Lemma 1 (Basic structural constraints and binary surjection descent)
*Let $|\Sigma| \ge 2$ and suppose $L_\Sigma(p) \subseteq L_\Sigma(q)$.*
1. ***Length and equal-length morphism (cf. Angluin, 1980).*** *We have $n(p) \ge n(q)$. If $n(p) = n(q)$, then $p$ is obtained from $q$ by a letter-to-token (length-preserving) substitution fixing terminals; hence $L_\Sigma(p) = L_\Sigma(q)$ if and only if $p$ and $q$ coincide up to bijective variable renaming, and $L_\Sigma(p) \subsetneq L_\Sigma(q)$ implies $v(p) < v(q)$ and $\mu(p) > \mu(q)$.*
2. ***Terminal lower bounds.*** *For every terminal $a \in \Sigma$, the number of occurrences $\#_a(p)$ of $a$ in $p$ satisfies $\#_a(p) \ge \#_a(q)$. In particular $t(p) \ge t(q)$, and if $q$ has no repeated variables then $c(q) = t(q) \le t(p) \le c(p)$, so $d > 0$ implies $\mu(p) > \mu(q)$.*
3. ***Binary surjection descent (repairing the subalphabet remark).*** *For any surjection $\pi : \Sigma \to \{0,1\}$ applied coordinatewise to terminals (leaving variables unchanged), $L_{\{0,1\}}(\pi p) \subseteq L_{\{0,1\}}(\pi q)$. Moreover, $n, v, t$ are preserved, and whenever $d = n(p) - n(q) > 0$ the inclusion $L_{\{0,1\}}(\pi p) \subsetneq L_{\{0,1\}}(\pi q)$ remains proper.*

*Proof.*
(1) The shortest words of $L_\Sigma(p)$ have length $n(p)$ and come from mapping each variable to a single letter in $\Sigma$; hence $n(p) \ge n(q)$. When $n(p) = n(q)$, every single-letter substitution for $p$ yields a word of length $n(q)$ in $L_\Sigma(q)$, which must arise from a single-letter substitution for $q$. Because $|\Sigma| \ge 2$, any $p$-variable can take two distinct letters, and any two distinct $p$-variables can take distinct letters independently. Consequently, each terminal $a$ at position $j$ of $q$ forces position $j$ of $p$ to be the same terminal $a$, and any two positions sharing a variable in $q$ must carry the same token in $p$. This defines a letter-to-token morphism $h$ with $h(q) = p$. Each variable of $p$ must appear in the image of at least one variable of $q$, so $v(p) \le v(q)$, with equality if and only if $h$ maps variables bijectively to variables (Angluin, 1980), in which case $L_\Sigma(p) = L_\Sigma(q)$.

(2) Fix $a \in \Sigma$ and choose $b \in \Sigma \setminus \{a\}$. Substituting $b$ for every variable of $p$ produces a word in $L_\Sigma(p) \subseteq L_\Sigma(q)$ containing exactly $\#_a(p)$ copies of $a$, which must be at least $\#_a(q)$.

(3) Choose a section $\iota : \{0,1\} \to \Sigma$ with $\pi \circ \iota = \mathrm{id}_{\{0,1\}}$ and extend both $\pi$ and $\iota$ to words by acting letterwise. For any non-erasing substitution $\sigma : X \to \{0,1\}^+$, the lifted substitution $\iota \circ \sigma$ gives a word $w = (\iota \circ \sigma)(p) \in L_\Sigma(p) \subseteq L_\Sigma(q)$, so $w = \tau(q)$ for some $\tau : X \to \Sigma^+$. Applying $\pi$ yields $\sigma(\pi p) = \pi(w) = (\pi \circ \tau)(\pi q) \in L_{\{0,1\}}(\pi q)$. Lengths, variable counts, and terminal counts are untouched by $\pi$, and if $n(p) > n(q)$ then $\pi q$ generates words of length $n(q) < n(p)$ absent from $L_{\{0,1\}}(\pi p)$. (Unlike restricting to a two-letter subalphabet $\Sigma' \subseteq \Sigma$, which fails when $p$ or $q$ contains terminals outside $\Sigma'$, the surjection $\pi$ works for arbitrary finite alphabets and arbitrary terminal sets.) $\square$

### Proposition 1 (Morphic inclusions and large alphabets)
*If $p = h(q)$ for a non-erasing, terminal-preserving pattern morphism $h : (\Sigma \cup X)^+ \to (\Sigma \cup X)^+$, then $\mu(p) - \mu(q) \ge n(p) - n(q)$, with strict inequality whenever $L_\Sigma(p) \subsetneq L_\Sigma(q)$. In particular, every inclusion holds over any alphabet with at least $v(p)$ letters unused as terminals in $p$ or $q$.*

*Proof.* Let $m_x \ge 1$ be the multiplicity of variable $x$ in $q$ and $\ell_x := |h(x)| \ge 1$. Then $d = n(p) - n(q) = \sum_{x \in \mathrm{var}(q)} m_x(\ell_x - 1)$ and $v(p) \le \sum_{x \in \mathrm{var}(q)} \ell_x = v(q) + \sum_{x}(\ell_x - 1) \le v(q) + d$. Hence $\mu(p) - \mu(q) = 2d - (v(p) - v(q)) \ge d$. For $d = 0$, strictness follows from Lemma 1(1). If $|\Sigma|$ has at least $v(p)$ letters outside the terminals of $p$ and $q$, substitute distinct fresh markers for the variables of $p$, match the resulting word with $q$, and decode each marker back to its variable to produce a non-erasing morphism $h$ with $h(q) = p$. $\square$

---

## 2. Upper Patterns with Low Multiplicity Excess

### Proposition 2 (Multiplicity-excess bound)
*Suppose $L_\Sigma(p) \subseteq L_\Sigma(q)$ over $|\Sigma| = s \ge 2$ with $v(q) \ge 1$. Let $f := \min_{x \in \mathrm{var}(q)} m_x(q)$ be the minimum variable multiplicity in $q$, and let $E(q) := n(q) - t(q) - f\,v(q) \ge 0$ be the total multiplicity excess of $q$. Then every variable of $p$ has multiplicity at least $f$, and*
$$
f\bigl(v(p) - v(q)\bigr) \le d + E(q), \qquad d := n(p) - n(q).
$$
*Consequently, whenever $d > 0$ and $E(q) < (2f - 1)d$—in particular, whenever all variables of $q$ have the same multiplicity $f$—we have $\mu(p) > \mu(q)$.*

*Proof.* Suppose for contradiction that some variable $y$ of $p$ occurs $e < f$ times. Fix all other variables of $p$ to single letters and substitute an arbitrary word $u \in \Sigma^k$ for $y$. The starting index of the first occurrence of $y$ is fixed, so distinct choices of $u$ produce $s^k$ distinct words in $L_\Sigma(p)$ of length $N = ek + C$ (where $C = n(p) - e$ is independent of $k$). On the other hand, any word in $L_\Sigma(q) \cap \Sigma^N$ is determined by the lengths $(\ell_x)_{x \in \mathrm{var}(q)}$ of the variable images—at most $(N+1)^{v(q)}$ choices—and the images themselves. Since $f \sum_x \ell_x \le \sum_x m_x(q) \ell_x \le N$, each length vector yields at most $s^{N/f}$ words. Thus
$$
s^k \le |L_\Sigma(p) \cap \Sigma^N| \le |L_\Sigma(q) \cap \Sigma^N| \le (N + 1)^{v(q)} s^{(ek + C)/f}.
$$
Because $e/f < 1$, the right-hand side is strictly smaller than $s^k$ for large $k$, a contradiction. Therefore every variable of $p$ has multiplicity at least $f$, so $n(p) \ge t(p) + f\,v(p)$. Subtracting $n(q) = t(q) + f\,v(q) + E(q)$ and using $t(p) \ge t(q)$ (Lemma 1(2)) gives $f(v(p) - v(q)) \le d + E(q)$. If $E(q) < (2f - 1)d$, then $v(p) - v(q) \le (d + E(q))/f < 2d$, i.e. $\mu(p) > \mu(q)$. (If $v(q) = 0$, $q$ is ground and admits no proper nonempty sublanguage.) $\square$

---

## 3. One-Symbol Length Gap at All Lengths

We first record the linear-matroid form of Rado's (1942) independent-representatives theorem, with a short self-contained proof.

### Lemma 2 (Linear independent representatives; Rado, 1942)
*Let $S_1, \dots, S_R$ be finite subsets of a vector space $V$ over a field $\mathbb{K}$. If $\dim \operatorname{span}\bigl(\bigcup_{i \in I} S_i\bigr) \ge |I|$ for every nonempty $I \subseteq \{1, \dots, R\}$, then there exist linearly independent vectors $u_1, \dots, u_R$ with $u_i \in S_i$.*

*Proof.* Induct on $R$; the case $R = 1$ is immediate. For $R \ge 2$, suppose first that some nonempty proper subset $J \subsetneq \{1, \dots, R\}$ is tight: $\dim W = |J|$ where $W := \operatorname{span}(\bigcup_{i \in J} S_i)$. By induction, select linearly independent $u_i \in S_i$ ($i \in J$), which form a basis of $W$. In the quotient $V / W$, every nonempty $I \subseteq \{1, \dots, R\} \setminus J$ satisfies
$$
\dim\operatorname{span}\Bigl(\bigcup_{i \in I} (S_i + W)\Bigr) = \dim\operatorname{span}\Bigl(\bigcup_{i \in I \cup J} S_i\Bigr) - \dim W \ge |I| + |J| - |J| = |I|,
$$
so induction selects vectors $u_i \in S_i$ ($i \notin J$) whose images in $V / W$ are linearly independent; together with $\{u_i\}_{i \in J}$ they are linearly independent in $V$. If no proper nonempty subset is tight, choose any nonzero $u_R \in S_R$, pass to $V / \operatorname{span}\{u_R\}$ (where every nonempty $I \subseteq \{1, \dots, R-1\}$ loses at most $1$ in dimension and hence retains dimension $\ge |I|$), and apply induction. $\square$

### Theorem 2 (All-lengths gap-one theorem)
*Let $|\Sigma| \ge 2$ and let $p, q$ be patterns with $n(p) = n(q) + 1$ and $L_\Sigma(p) \subseteq L_\Sigma(q)$. Then $v(p) \le v(q) + 1$; equivalently $c(p) \ge c(q)$, and hence $\mu(p) \ge \mu(q) + 1 > \mu(q)$.*

*Proof.* By Lemma 1(3) it suffices to prove the result over $\Sigma = \{0,1\}$. Set $n := n(p) = n(q) + 1$ and identify $\{0,1\}^n$ with $\mathbb{F}_2^n$. The shortest words of $L_{\{0,1\}}(p)$ form an affine subspace $P \subseteq \mathbb{F}_2^n$ of codimension $c(p)$, cut out by fixing one coordinate per terminal of $p$ and equating each repeated occurrence of a variable to its first occurrence; these $c(p)$ equations are linearly independent because each eliminates a distinct coordinate.

At length $n = n(q) + 1$, a word in $L_{\{0,1\}}(q)$ must lengthen exactly one variable of $q$ that occurs **once** (a *singleton variable*), from length $1$ to length $2$. Group maximal consecutive runs of singleton variables in $q$ from left to right as runs $1, \dots, R$. Within any single run, lengthening any member produces the same set of words—an arbitrary binary block one letter longer. Thus $L_{\{0,1\}}(q) \cap \mathbb{F}_2^n = \bigcup_{i=1}^R Q_i$, where each $Q_i$ is an affine subspace of codimension $r := c(q)$ obtained by inserting one free coordinate into run $i$. (If $R = 0$, $L_{\{0,1\}}(q) \cap \mathbb{F}_2^n = \varnothing$, impossible.)

For each $i \in \{1, \dots, R\}$, let $S_i \subset (\mathbb{F}_2^n)^*$ be the set of $r$ linearly independent linear functionals giving the linear parts of the defining equations of $Q_i$. We claim that for every nonempty subset $I = \{i_1 < i_2 < \cdots < i_k\} \subseteq \{1, \dots, R\}$,
$$
\dim \operatorname{span}\Bigl(\bigcup_{i \in I} S_i\Bigr) \ge r + k - 1.
$$
For $k = 1$, $\dim\operatorname{span}(S_{i_1}) = r$. For $t \in \{2, \dots, k\}$, let $b$ be the rightmost position in the *unexpanded* $q$ of run $i_{t-1}$, and let $h := b + 1$. Because run $i_{t-1}$ is maximal and followed by run $i_t$, position $h$ exists in $q$ and carries a *constrained* token—either a terminal or an occurrence of a repeated variable. Now compare coordinate $h$ in the expanded words of length $n$:
- For every earlier selected run $i_s$ with $s < t$: if $i_s = i_{t-1}$, coordinate $h = b + 1$ is the newly inserted free coordinate of run $i_{t-1}$; if $i_s < i_{t-1}$, inserting a coordinate earlier shifts position $b$ of run $i_{t-1}$ to coordinate $b + 1 = h$. In either case, coordinate $h$ is a completely free coordinate of $Q_{i_s}$, so **every** functional in $S_{i_s}$ has $0$ in coordinate $h$.
- For run $i_t$ (which lies strictly after position $h$), position $h$ is unshifted and still holds its constrained token. Hence **at least one** defining equation of $Q_{i_t}$ has coefficient $1$ in coordinate $h$ (either pinning that terminal, or equating a first/later occurrence of that repeated variable to another occurrence).

Therefore each successive $S_{i_t}$ ($t = 2, \dots, k$) adds at least one functional linearly independent of $\bigcup_{s < t} S_{i_s}$, establishing the rank lower bound $r + k - 1$.

Now suppose for contradiction that $v(p) \ge v(q) + 2$, i.e. $c(p) \le r - 1$. Let $V_P \subseteq \mathbb{F}_2^n$ be the direction space of $P$, of codimension $c(p)$. Restricting linear functionals from $\mathbb{F}_2^n$ to $V_P$ reduces the dimension of any subspace of $(\mathbb{F}_2^n)^*$ by at most $\dim(V_P^\perp) = c(p) \le r - 1$. Thus the restricted sets $\bar{S}_i := \{\varphi|_{V_P} : \varphi \in S_i\} \subset V_P^*$ satisfy
$$
\dim\operatorname{span}\Bigl(\bigcup_{i \in I} \bar{S}_i\Bigr) \ge (r + |I| - 1) - (r - 1) = |I|
$$
for every nonempty $I \subseteq \{1, \dots, R\}$. By Lemma 2, we can pick one affine defining equation $\ell_i(x) = \beta_i$ of each $Q_i$ ($i = 1, \dots, R$) whose linear parts $\ell_1|_{V_P}, \dots, \ell_R|_{V_P}$ are linearly independent on $V_P$. The affine evaluation map $x \mapsto (\ell_1(x), \dots, \ell_R(x))$ is therefore surjective from $P$ onto $\mathbb{F}_2^R$. Choosing $x \in P$ with $\ell_i(x) = \beta_i + 1 \pmod 2$ for every $i$ yields $x \in P \setminus \bigcup_{i=1}^R Q_i$, contradicting $L_{\{0,1\}}(p) \subseteq L_{\{0,1\}}(q)$. Note that this proof applies without assuming the inclusion is morphic. $\square$

---

## 4. Exact Finite Exclusion Through Length 12 and All Lengths with $c(q) \le 2$

### Theorem 3 (Exact exclusion for $n(p) \le 12$)
*Over every finite alphabet $|\Sigma| \ge 2$, Conjecture `conj:monotone-measure` holds whenever $n(p) \le 12$.*

*Proof.* Equal lengths ($d = 0$) and gap one ($d = 1$) are settled by Lemma 1(1) and Theorem 2. For $d = n(p) - n(q) \ge 2$, Lemma 1(3) reduces the claim to $\Sigma = \{0,1\}$. A counterexample $(p, q)$ must satisfy $v(p) \ge v(q) + 2d$. If $v(p) > v(q) + 2d$, fixing surplus variables of $p$ to terminal $0$ replaces $p$ by a pattern $p'$ of the same length $n = n(p)$ with $v(p') = v(q) + 2d$, whose shortest-word cube is a subcube of $p$'s shortest-word cube and still satisfies the terminal lower bounds $\#_a(p') \ge \#_a(q)$ of Lemma 1(2). Thus it suffices to show that no $q$ of length $m = n - d$ ($2 \le d \le \lfloor (n-1)/2 \rfloor$) has its exact length-$n$ slice $L_{\{0,1\}}(q) \cap \{0,1\}^n$ containing any shortest-word cube of a pattern $p$ of length $n$ with $v(p) = v(q) + 2d$, $\#_0(p) \ge \#_0(q)$, and $\#_1(p) \ge \#_1(q)$.

The script [`explore.py`](explore.py) performs this exact exclusion as follows:
1. **Symmetry reduction:** Enumerate canonical patterns $q$ (variables named in first-occurrence order) and retain one representative per orbit under the commuting $\mathbb{Z}_2 \times \mathbb{Z}_2$ action generated by bit complementation ($0 \leftrightarrow 1$) and string reversal, both of which preserve inclusion and all parameters.
2. **Exact allocation bound:** Group consecutive singleton variables of $q$ into maximal runs (cost $f = 1$ per extra letter) and give each repeated variable its multiplicity $f \ge 2$ as cost per extra letter. An expansion of $q$ to length $n = m + d$ allocates nonnegative increments $e_j$ with $\sum f_j e_j = d$, contributing a subcube of size $2^{v(q) + \sum e_j}$. Summing over allocations gives the valid upper bound
   $$
   \bigl|L_{\{0,1\}}(q) \cap \{0,1\}^n\bigr| \le 2^{v(q)} [z^d] \prod_j (1 - 2 z^{f_j})^{-1}.
   $$
   Whenever the bracketed coefficient is strictly less than $2^{2d} = 4^d$, $q$ cannot cover any cube of size $2^{v(q) + 2d}$ and is pruned immediately.
3. **Exact bitmask construction and containment:** For every surviving $q$, compute the exact $2^n$-bit mask of $L_{\{0,1\}}(q) \cap \{0,1\}^n$ as the bitwise OR of all allocation cubes. Discard $q$ if its popcount is $< 2^{v(q)+2d}$; otherwise test every admissible source cube $p$ via bitwise containment against the complement mask.

Every single candidate for $n \le 12$ fails containment, yielding a concrete length-$n$ ground nonmembership witness (establishing a finite theorem about the full infinite languages without any bounded-length truncation heuristic):

| Longer length $n$ | Exact upper-pattern slices built | Final source-cube comparisons |
|---:|---:|---:|
| 1–7 | 0 | 0 |
| 8 | 20 | 0 |
| 9 | 198 | 0 |
| 10 | 2,089 | 3,995 |
| 11 | 19,394 | 1,342,396 |
| 12 | 167,135 | 220,303,162 |

No source cube survives for any $n \le 12$. $\square$

### Proposition 4 (All lengths when $c(q) \le 2$)
*Over every finite alphabet $|\Sigma| \ge 2$, Conjecture `conj:monotone-measure` holds for all pattern lengths whenever $c(q) = n(q) - v(q) \le 2$.*

*Proof.* Cases $d \le 1$ are settled by Lemma 1(1) and Theorem 2. For $d \ge 2$, failure of $\mu(p) > \mu(q)$ requires $c(p) \le c(q) - d \le 2 - d \le 0$, forcing $d = c(q) = 2$ and $c(p) = 0$. By Lemma 1(3), work over $\{0,1\}$. Since $c(p) = 0$, $p$ consists of $n = n(q) + 2$ distinct singleton variables, so $L_{\{0,1\}}(p) \cap \{0,1\}^n = \{0,1\}^n$ and $t(q) \le t(p) = 0$ by Lemma 1(2). Thus $q$ is terminal-free with defect $2$, meaning its subsequence of repeated variables is (up to renaming) one of four cores: $AAA$, $AABB$, $ABAB$, or $ABBA$, separated and flanked by $4$ (for $AAA$) or $5$ (for the length-$4$ cores) maximal runs of singleton variables.

We claim that any singleton run of length $g \ge 3$ can be shortened by $1$ without losing full-cube coverage of $\{0,1\}^{n(q)+2}$. Indeed, let the run occupy positions $a, a+1, \dots, a+g-1$ in $q$, and consider coordinate $a+2$ of any ground word $w \in \{0,1\}^{n(q)+2}$. In any cost-$2$ expansion of $q$, at most $2$ extra letters are inserted before position $a$, so the expanded run starts at some position $\le a+2$ and ends at some position $\ge a+g-1 \ge a+2$. Thus coordinate $a+2$ always falls strictly inside that completely free block. Deleting coordinate $a+2$ from $w$ and shortening the singleton run in $q$ from $g$ to $g-1$ preserves membership for the same allocation; since projection from $\{0,1\}^{m+2}$ to $\{0,1\}^{m+1}$ maps the full cube onto the full cube, the shortened pattern also covers the full cube at gap $2$.

Iterating reduces every singleton run to length $0, 1$, or $2$, leaving exactly $3^4 + 3 \cdot 3^5 = 810$ reduced shapes. Exact bitmask verification in [`explore.py`](explore.py) confirms that none of the $810$ shapes covers the full cube $\{0,1\}^{n(q)+2}$. $\square$

---

## 5. Inclusion Depth, Semicomputability, and Variant Counterexamples

### Proposition 5 (Depth lower bound, exactness for $n \le 12$, and upper semicomputability)
*Let $|\Sigma| \ge 2$.*
1. *For every pattern $p$, $\operatorname{ID}_\Sigma(p) \ge 2n(p) - v(p) - 1 = \mu(p) - 1$.*
2. *For every pattern $p$ with $n(p) \le 12$, $\operatorname{ID}_\Sigma(p) = 2n(p) - v(p) - 1$.*
3. *For every pattern $p$, $\operatorname{ID}_\Sigma(p)$ is finite and upper semicomputable.*

*Proof.*
(1) Starting from the universal pattern $x_1$, split the rightmost variable into two fresh variables $n(p) - 1$ times to reach $x_1 x_2 \cdots x_{n(p)}$; each step strictly excludes words of the former minimum length. Then, at fixed length $n(p)$, pin each terminal position of $p$ one by one, and finally identify each repeated variable occurrence with its first occurrence one by one; each of these $c(p)$ steps strictly shrinks the shortest-word slice because $|\Sigma| \ge 2$. This yields a strict chain of length $(n(p) - 1) + c(p) = \mu(p) - 1$.

(2) In any strict inclusion chain $L_\Sigma(x_1) \supsetneq L_\Sigma(p_1) \supsetneq \cdots \supsetneq L_\Sigma(p_k) = L_\Sigma(p)$, lengths are nondecreasing by Lemma 1(1), so $n(p_i) \le n(p) \le 12$ for every $i$. By Theorem 3, $\mu$ strictly increases at every step from $\mu(x_1) = 1$ to $\mu(p)$, forcing $k \le \mu(p) - 1$. (Note also that a single counterexample $L_\Sigma(p) \subsetneq L_\Sigma(q)$ with $\mu(p) \le \mu(q)$ would refute $\operatorname{ID}_\Sigma(p) = \mu(p) - 1$ by appending $L_\Sigma(p) \subsetneq L_\Sigma(q)$ to the lower-bound chain for $q$.)

(3) Every intermediate pattern in a strict chain ending at $p$ has length $\le n(p)$, and by Lemma 1(1) equal languages correspond to identical canonical patterns up to variable renaming. Thus there are only finitely many canonical candidate patterns and finitely many candidate strict chains, so $\operatorname{ID}_\Sigma(p)$ is finite. Moreover, membership $w \in L_\Sigma(r)$ is decidable by bounded segmentation of $w$. Enumerating ground words $w \in \Sigma^+$ and discarding any candidate chain as soon as some $w$ falsifies one of its inclusions produces a nonincreasing sequence of upper bounds that eventually stabilizes at $\operatorname{ID}_\Sigma(p)$ (without supplying a computable stopping bound). $\square$

### Proposition 6 (Erasing, unary, and nonmorphic controls)
1. ***Erasing, terminal-free:*** *Over every alphabet $|\Sigma| \ge 1$, $L_{\Sigma,\mathrm{E}}(XX) \subsetneq L_{\Sigma,\mathrm{E}}(ABC)$ (squares vs. all words, separated by any single letter), with $\mu(XX) = 3 = \mu(ABC)$, while $L_{\Sigma,\mathrm{E}}(XX) \subsetneq L_{\Sigma,\mathrm{E}}(ABCD)$ gives $\mu(XX) = 3 < 4 = \mu(ABCD)$. Prefixing a common terminal $a \in \Sigma$ preserves strictness and adds $2$ to both measures.*
2. ***Erasing, binary, genuinely nonmorphic:*** *Over $\Sigma = \{0,1\}$, $L_{\{0,1\},\mathrm{E}}(0X1) \subsetneq L_{\{0,1\},\mathrm{E}}(A01B)$ (every binary word starting with $0$ and ending with $1$ contains an adjacent $01$ at its first $1$, while $101$ separates them), with $\mu(0X1) = 5 < 6 = \mu(A01B)$. No terminal-preserving morphism maps $A01B$ to $0X1$ because any image of $A01B$ contains a literal $01$.*
3. ***Unbounded erasing depth:*** *Over every $|\Sigma| \ge 1$, $p = XX$ and $q_j = XX Y^{3^j}$ ($j \ge 0$) satisfy $L_{\Sigma,\mathrm{E}}(q_0) = \Sigma^*$, $L_{\Sigma,\mathrm{E}}(q_j) \supsetneq L_{\Sigma,\mathrm{E}}(q_{j+1}) \supsetneq L_{\Sigma,\mathrm{E}}(XX)$ (mapping $Y \mapsto Y^3$ and $Y \mapsto \varepsilon$; separated by $a^{3^j}$, which is odd-length and shorter than $3^{j+1}$). Thus arbitrarily long finite strict erasing chains terminate at $XX$.*
4. ***Unary non-erasing:*** *Over $\Sigma = \{0\}$, $L_{\{0\}}(ABCD) = \{0^k : k \ge 4\} \subsetneq \{0^k : k \ge 3\} = L_{\{0\}}(XXY)$ has $\mu(ABCD) = 4 = \mu(XXY)$.*
5. ***Genuinely nonmorphic non-erasing family (cf. Jiang et al., 1995):*** *For every finite alphabet of size $s \ge 2$, let $p = X_1 \cdots X_{2s+3}$ and $q = ABCBD$. In any word of length $\ge 2s + 3$, among positions $2, 4, \dots, 2s + 2$ two carry the same letter by pigeonhole; assigning that letter to $B$ and the nonempty prefix, intervening block, and suffix to $A, C, D$ shows $L_\Sigma(p) \subsetneq L_\Sigma(q)$ (proper by length). No non-erasing morphism maps $q$ to $p$ since any nonempty image of $B$ would occur twice in $p$, whereas all variables of $p$ are distinct. Here $\mu(p) = 2s + 3 > 6 = \mu(q)$. Appending a common terminal yields a terminal-containing nonmorphic inclusion by right cancellation.*

---

## 6. Reproducible Verification

Run from the repository root (Python standard library only):

```sh
.venv/bin/python -B solutions/pattern_inclusion_depth/explore.py
```

The script [`explore.py`](explore.py) performs:
- **25,719 direct checks** cross-validating the bitmask layer routine against an independent backtracking substitution matcher on all length-$3$ binary patterns through length $5$, together with explicit checks of all five families in Proposition 6.
- **810 reduced defect-two checks** completing the proof of Proposition 4.
- **Bounded scans** of all canonical patterns of length $1$–$3$ over alphabets $s \in \{2, 3, 4, 5\}$ on all words up to length $4$, finding `0, 0, 0, 0` non-erasing and `110, 193, 296, 419` erasing strict-inclusion candidates with nonincreasing $\mu$.
- **Exact binary exclusion for $n = 1, \dots, 12$** (up to `167,135` exact target layers and `220,303,162` final source-cube comparisons at $n = 12$), completing Theorem 3.

All assertions pass with exit code `0`.

---

## References

- D. Angluin, *Finding patterns common to a set of strings*, Journal of Computer and System Sciences 21(1):46–62, 1980.
- Ideosphere preprint, *A monotone measure for pattern-language inclusion: every morphic inclusion, every pattern of length at most eight, and the inclusion depth in that range*, [`https://ideosphere.ai/papers/patlang-depth`](https://ideosphere.ai/papers/patlang-depth).
- T. Jiang, A. Salomaa, K. Salomaa, and S. Yu, *Decision problems for patterns*, Journal of Computer and System Sciences 50(1):53–63, 1995.
- W. Luo, *Compute Inclusion Depth of a Pattern*, in *COLT 2005*, LNCS 3559, pp. 689–690, 2005; author-prepared version [arXiv:2605.30389](https://arxiv.org/abs/2605.30389), 2026.
- R. Rado, *A theorem on independence relations*, Quarterly Journal of Mathematics 13(1):83–89, 1942.
