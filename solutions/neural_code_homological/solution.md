# Combinatorial Betti Formulas, Extremal Classification, and Alexander Duality for Polarized Neural Ideals

**Target paper:** Selvi Kara and Ellie Lew, *Neural codes via homological invariants of polarized neural ideals*, [arXiv:2602.16993](https://arxiv.org/abs/2602.16993) (February 2026), Section 5 (`q:classify_maxpdim`) and Section 10 (`ques:duality-principle`, `ques:region`, `ques:betti`).  
**Companion files:** [`solutions/neural_code_homological/explore.py`](explore.py), [`solutions/neural_code_homological/proof.pdf`](proof.pdf).

---

## Verdict

- **Question `q:classify_maxpdim`: RESOLVED.** For every $n \ge 3$ and every field $k$, a nonempty proper neural code $C \subseteq \{0, 1\}^n$ satisfies $\operatorname{pd}(I_C) = 2n - 3$ if and only if $C = \{a, \bar{a}\}$ is an antipodal pair (in which case $\beta_{2n-3, V}(I_C) = n - 1$ and $\operatorname{reg}(I_C) = 3$). For $n = 2$, singleton codes $\{a\}$ also attain $\operatorname{pd}(I_C) = 1 = 2n - 3$ with $\operatorname{reg}(I_C) = 1$, resolving a ghost-vertex edge case in Lemma 5.3 of Kara–Lew.
- **Question `ques:betti`: RESOLVED.** Two exact combinatorial formulas determine all multigraded Betti numbers $\beta_{i,W}(I_C)$ over any field $k$: a cube-incidence Hochster formula (Theorem 1) in terms of projections of $C$, and a relative Taylor formula (Theorem 2) on the maximal coordinate subcubes of $Q \setminus C$. In addition, the entire $H_1$-strand on any support $W$ containing $d(W) > 0$ doubled coordinates is given by the exact graph formula $\dim_k \widetilde{H}_1(K(W); k) = (d(W) - 1)(h(W) - 1)$ (Theorem 3).
- **Question `ques:duality-principle`: RESOLVED.** We characterize exactly when code complementation implements literal-reversed monomial Alexander duality $\iota(I_C^\vee) = I_{Q \setminus C}$ via a coordinate-slice lifting criterion (Theorem 5), on which complementation swaps $(\operatorname{pd}, \operatorname{reg}) = (p, r) \mapsto (r - 1, p + 1)$ via Terai's theorem (Terai, 1999). Outside this class, complementation does not even induce a function on $(\operatorname{pd}, \operatorname{reg})$: for every $s \ge 1$ and $n \ge 4s$, there exist codes sharing the pair $(2s - 1, 2s + 1)$ whose complements have pairs $(2s, 2s)$ and $(s, 3s)$, giving unbounded separation (Proposition 6).
- **Question `ques:region`: PARTIALLY RESOLVED (structural through $p + r \le 2n$, exact finite enumerative formula for all $n$).** We give a complete structural description of the achievable pairs $(p, r) \in \mathcal{R}_n(k)$ with $p + r \le 2n$ for every $n \ge 2$ and every field $k$ (Theorem 8), doubling the $p + r \le n$ diagonal of Kara–Lew, together with explicit families reaching $p + r = 3n - 3$ (Proposition 9). For general $n$, we derive an exact finite enumerative inclusion–exclusion formula in an avoidance algebra that computes the exact number of labeled codes at any $(p, r)$ and constructs a witness when positive (Theorem 7); its size grows doubly exponentially in $n$, so a compact inequality description for $n \ge 5$ remains open. Question `ques:restricted-classes` (convex or hyperplane codes) is left open.
- **Field conventions:** All structural theorems in Sections 1–3 and 5 hold over an arbitrary field $k$; the enumerative formula in Section 4 depends on $\operatorname{char} k$ via the independent-column families $\mathcal{I}_{t,j}$; the companion verification script [`explore.py`](explore.py) executes exact linear algebra over $\mathbb{F}_2$.

---

## 1. Setup and Multigraded Betti Formulas (`ques:betti`)

Write $Q := \{0, 1\}^n$ and $V := \{x_1, \dots, x_n, y_1, \dots, y_n\}$, viewing $x_i$ as the literal $i = 1$ and $y_i$ as the literal $i = 0$. Throughout we use the ideal-module grading $\beta_{i, W}(I) := \dim_k \operatorname{Tor}_i^S(I, k)_W$ of [arXiv:2602.16993](https://arxiv.org/abs/2602.16993) over an arbitrary field $k$, where $S = k[V]$ and $I_C := \mathcal{P}(J_C)$ is the polarized canonical-form neural ideal of $C \subseteq Q$.

The empty code $C = \varnothing$ gives $I_\varnothing = S$, contributing $(0, 0)$; the full code $C = Q$ gives $I_Q = (0)$, whose Betti table is empty and contributes no finite pair. We assume $\varnothing \neq C \subsetneq Q$ and set $N := Q \setminus C$.

A subset $A \subseteq V$ of literals is *consistent* if it contains at most one of $x_i, y_i$ for each $i \in [n]$; it specifies the coordinate subcube $Q_A \subseteq Q$ of words satisfying every literal in $A$.

**Lemma (Canonical form and maximal missing subcubes).** *The minimal monomial generators of $I_C$ are precisely the products $\prod_{u \in A} u$ over consistent literal sets $A$ such that $Q_A$ is a maximal coordinate subcube contained in $N = Q \setminus C$.*

*Proof.* The unpolarized pseudo-monomial corresponding to $A$ evaluates to $1$ on $Q_A$ and $0$ on $Q \setminus Q_A$. If $Q_A \subseteq N$, summing the nonword indicator polynomials over $Q_A$ yields this pseudo-monomial via $\prod_{j \text{ free}} (x_j + (1 - x_j)) = 1$, so it belongs to $J_C$. Conversely, if $Q_A \cap C \neq \varnothing$, evaluating at a word in the intersection shows it cannot lie in $J_C$. Minimal divisibility corresponds to inclusion-maximality of $Q_A \subseteq N$. $\square$

For any literal subset $F \subseteq V$, let $\mathcal{T}(F)$ denote the collection of all consistent literal sets obtained by picking one literal at each coordinate appearing in $F$ (choosing either $x_i$ or $y_i$ when both lie in $F$), with $\mathcal{T}(\varnothing) = \{\varnothing\}$. For any $W \subseteq V$, define the simplicial complex $K(W)$ on ground set $W$ by
$$
F \in K(W) \iff F \subseteq W \text{ and } Q_A \cap C \neq \varnothing \text{ for every } A \in \mathcal{T}(F).
$$

**Theorem 1 (Cube-incidence Hochster formula).** *Over any field $k$, for every $\varnothing \neq C \subsetneq Q$, every $i \ge 0$, and every squarefree multidegree $W \subseteq V$,*
$$
\beta_{i, W}(I_C) = \dim_k \widetilde{H}_{|W| - i - 2}(K(W); k),
\tag{1}
$$
*using augmented reduced homology (so $\{\varnothing\}$ has $\dim_k \widetilde{H}_{-1}(\{\varnothing\}; k) = 1$, and the void complex has zero homology in all degrees). All nonsquarefree multidegrees vanish.*

*Proof.* A literal set $F$ fails the face test if and only if some $A \in \mathcal{T}(F)$ satisfies $Q_A \subseteq N$, which happens if and only if $F$ contains a consistent set $A'$ with $Q_{A'} \subseteq N$ (every forbidden pattern inside $F$ extends to a choice in $\mathcal{T}(F)$). Thus $K(W)$ is the induced Stanley–Reisner complex of $I_C$ on $W$ (allowing ghost vertices, where $K(W) = \{\varnothing\}$ if every vertex of $W$ is a ghost). Hochster's formula for the ideal $I_C$ (shifted by one homological degree from $S / I_C$) gives (1). $\square$

There is also a dual description directly on the maximal missing subcubes. For $W \subseteq V$, let $G(W)$ be the set of minimal generators (maximal subcubes of $N$) whose literal supports lie in $W$. Define a simplicial complex $B(W)$ on vertex set $G(W)$ whose faces are collections of generators in $G(W)$ whose combined literal union is a *proper* subset of $W$.

**Theorem 2 (Relative Taylor formula on missing subcubes).** *Over any field $k$, if $W \subseteq V$ is nonempty and $G(W) \neq \varnothing$, then*
$$
\beta_{i, W}(I_C) = \dim_k \widetilde{H}_{i - 1}(B(W); k).
\tag{2}
$$
*If $G(W) = \varnothing$ (or $W$ is not a union of generator supports), then $\beta_{i, W}(I_C) = 0$ for all $i$.*

*Proof.* The Taylor resolution of $I_C$ has one basis element in homological degree $|A| - 1$ for each nonempty subset $A$ of minimal generators, in multidegree $\bigcup_{g \in A} \operatorname{supp}(g)$. In any multidegree $m$, its augmented strand is the chain complex of the full simplex on the generators dividing $m$ (hence exact when nonempty, and zero otherwise), so it resolves $I_C$. Tensoring with $k$, the multidegree-$W$ piece is precisely the relative augmented chain complex of the full simplex $\Delta^{G(W)}$ modulo $B(W)$ shifted by one degree, since a boundary deletion survives modulo the maximal ideal if and only if it preserves the literal union $W$. Taking homology of the exact pair $(\Delta^{G(W)}, B(W))$ with acyclic total space $\Delta^{G(W)}$ yields (2). $\square$

---

## 2. The $H_1$-Strand and Extremal Classification (`q:classify_maxpdim`)

For $W \subseteq V$, let $d = d(W)$ be the number of doubled coordinates (indices $j \in [n]$ with $\{x_j, y_j\} \subseteq W$). For each codeword $a \in C$, its *trace* $T_a \subseteq W$ is the set of literals in $W$ satisfied by $a$. Form the *trace-sharing graph* on $C$ by connecting $a, b \in C$ whenever $T_a \cap T_b \neq \varnothing$.

**Theorem 3 (Graph formula for the $H_1$-strand).** *Suppose $d = d(W) > 0$, and let $h = h(W)$ be the number of connected components of the trace-sharing graph on $C$. Then $h \le 2$, $K(W)$ is connected, and over every field $k$,*
$$
\dim_k \widetilde{H}_1(K(W); k) = (d - 1)(h - 1).
\tag{3}
$$

*Proof.* By the face test, each trace $T_a$ spans a simplex in $K(W)$, and the only remaining edges of $K(W)$ are the opposite pairs $\{x_j, y_j\}$ at doubled coordinates $j$ where both $0$ and $1$ occur in $C$. Fix one doubled coordinate $j_0$: every trace $T_a$ contains either $x_{j_0}$ or $y_{j_0}$, so every vertex of $C$ falls into at most two components (hence $h \le 2$), and if both literals occur the opposite edge $\{x_{j_0}, y_{j_0}\}$ joins them, proving $K(W)$ is connected.

- **Case $h = 2$:** Every doubled coordinate must be constant on each component and take opposite values on the two components (otherwise words in different components would share a literal). Thus the union of trace simplices in each component is a cone over a vertex in every doubled coordinate. The $d$ opposite edges join the two cones, and no opposite edge lies in a triangle (a third vertex would be a literal shared across both components). Every face without an opposite pair lies in a single trace simplex. Contracting the two contractible cones yields a homotopy equivalence $K(W) \simeq \bigvee_{d-1} S^1$, giving $\dim_k \widetilde{H}_1(K(W); k) = d - 1$ over every field $k$.

- **Case $h = 1$:** Mark the barycenter of each trace simplex $T_a$ (or the vertex itself if $|T_a| = 1$). For any doubled coordinate $j$ and any $a, b \in C$, define the path $P_j(a, b)$ from the marked point of $T_a$ to that of $T_b$ by moving to $a$'s $j$-literal, traversing the opposite edge $\{x_j, y_j\}$ if $a_j \neq b_j$, and moving to the marked point of $T_b$. For fixed $j$, the concatenation $P_j(a, b) * P_j(b, c)$ is endpoint-preservingly homotopic to $P_j(a, c)$ because the coordinate subcomplex on $\{x_j, y_j\} \cap K(W)$ is a point or an interval.

  If $a, b \in C$ share a literal $u \in T_a \cap T_b$, then $P_j(a, b)$ is homotopic to the path through $u$: when $a_j = b_j$ this holds inside the two trace simplices sharing $\{u, \text{the } j\text{-literal}\}$, and when $a_j \neq b_j$ the face test guarantees the triangle $\{u, x_j, y_j\} \in K(W)$. Thus for adjacent words in the trace-sharing graph, $P_j(a, b)$ is independent of $j$ up to homotopy. Since $h = 1$, decomposing along a path in the trace-sharing graph shows $P_j(a, b)$ is independent of $j$ for all $a, b \in C$.

  Every edge loop in $K(W)$ is homotopic to a concatenation of paths between marked points: replace each trace edge by a detour through a marked point of a trace containing it, represent each transition between traces at a shared literal by $P_{j_0}$, and replace each traversal of an opposite edge at coordinate $i$ by $P_i$ (using any incident traces at its endpoints; consecutive opposite edges backtrack). Replacing every $P_i$ by $P_{j_0}$ reduces the loop to a closed concatenation along the single coordinate $j_0$, which contracts. Hence $\pi_1(K(W)) = 1$, so $\widetilde{H}_1(K(W); k) = 0$ over every field $k$. $\square$

**Theorem 4 (Classification of maximal projective dimension, `q:classify_maxpdim`).** *Let $\varnothing \neq C \subsetneq \{0, 1\}^n$ and let $k$ be any field.*
1. *For every $n \ge 3$,*
   $$
   \operatorname{pd}(I_C) = 2n - 3 \iff C = \{a, \bar{a}\} \text{ is an antipodal pair.}
   $$
   *In this case $\beta_{2n-3, V}(I_C) = n - 1$ and $\operatorname{reg}(I_C) = 3$.*
2. *For $n = 2$, $\operatorname{pd}(I_C) = 1 = 2n - 3$ holds precisely for antipodal pairs (giving $(\operatorname{pd}, \operatorname{reg}) = (1, 3)$) and singleton codes $C = \{a\}$ (giving $(\operatorname{pd}, \operatorname{reg}) = (1, 1)$).*

*Proof.* By (1), $\beta_{i, W}(I_C) = \dim_k \widetilde{H}_{|W| - i - 2}(K(W); k)$.
- Homology in degree $-1$ requires $K(W) = \{\varnothing\}$, meaning every literal in $W$ is a ghost (absent from $C$), so $|W| \le n$ and $i = |W| - 1 \le n - 1$. For $n \ge 3$, $n - 1 < 2n - 3$; for $n = 2$, $n - 1 = 1 = 2n - 3$ is achieved precisely when $|W| = 2$ has two ghost literals, i.e., $C$ is a singleton $\{a\}$ (giving $I_C$ generated by two variables, hence $(1, 1)$).
- Homology in degree $0$ requires $K(W)$ to be disconnected, so $d(W) = 0$ by Theorem 3, giving $|W| \le n$ and $i = |W| - 2 \le n - 2 < 2n - 3$.
- Homology in degree $q \ge 1$ gives $i = |W| - q - 2 \le 2n - 3$, with equality if and only if $W = V$ ($d(V) = n$) and $q = 1$. By Theorem 3, $\widetilde{H}_1(K(V); k) \neq 0$ if and only if $h(V) = 2$, with dimension $n - 1$. In the trace-sharing graph on $V$, every word is adjacent to every word other than its antipode $\bar{a}$, so $h(V) = 2$ holds if and only if $C = \{a, \bar{a}\}$. For an antipodal pair, every induced subcomplex $K(W)$ consists of at most two simplices joined by disjoint edges with no mixed triangles, so $\widetilde{H}_q(K(W); k) = 0$ for all $q \ge 2$, giving $\operatorname{reg}(I_C) = 3$. $\square$

*Remark (Ghost-vertex edge case in Lemma 5.3 of Kara–Lew).* In [arXiv:2602.16993](https://arxiv.org/abs/2602.16993), Lemma 5.3 (`lem:pdim-max-implies-reg-ge-3`) restricts Hochster's formula to $W \subseteq V(\Delta)$ and therefore concludes $\operatorname{reg}(I_C) \ge 3$ whenever $\operatorname{pd}(I_C) = 2n - 3$. This holds for all $n \ge 3$, whereas for $n = 2$ singleton codes $C = \{a\}$ have two ghost vertices and achieve $(\operatorname{pd}, \operatorname{reg}) = (1, 1)$.

*Remark (Other extremal corners).* The same tools recover the remaining boundary characterizations cleanly:
- $\operatorname{reg}(I_C) = 2n - 1$ ($n \ge 2$) $\iff N = \{a, \bar{a}\}$ is an antipodal pair: since generators have degree $\le n$ and shifts are $\le 2n$, $\operatorname{reg} = 2n - 1$ requires $\beta_{1, V} \neq 0$, i.e., $B(V)$ disconnected by (2); two generators in different components cover $V$ and must be opposite full words, and any third generator would bridge them.
- $\operatorname{pd}(I_C) = 0 \iff N$ is a single coordinate subcube (principal ideal); $\operatorname{reg}(I_C) = 1 \iff C$ is a coordinate subcube (linear Koszul resolution of variables). If the subcube has codimension $t$, these yield $(0, t)$ and $(t - 1, 1)$ respectively.

---

## 3. Code Complementation and Monomial Alexander Duality (`ques:duality-principle`)

For a squarefree monomial ideal $I \subset S$, the minimal generators of its Alexander dual $I^\vee$ correspond to minimal transversal sets of literals meeting every generator of $I$. Let $\iota$ be the involution interchanging $x_i \leftrightarrow y_i$ for each $i \in [n]$.

**Theorem 5 (Exact criterion for complementation as Alexander duality).** *Let $\varnothing \neq C \subsetneq \{0, 1\}^n$. The consistent minimal generators of $\iota(I_C^\vee)$ generate $I_{Q \setminus C}$ unconditionally. Moreover, the full ideals coincide—that is, $\iota(I_C^\vee) = I_{Q \setminus C}$—if and only if for every partition $[n] = D \sqcup T \sqcup U$ and every assignment $t \in \{0, 1\}^T$,*
$$
\bigl[\,\forall u \in \{0, 1\}^U\;\exists d \in \{0, 1\}^D : (d, t, u) \in C\,\bigr]
\implies
\bigl[\,\exists d \in \{0, 1\}^D\;\forall u \in \{0, 1\}^U : (d, t, u) \in C\,\bigr].
\tag{4}
$$
*Geometrically, in every coordinate slice obtained by fixing $T$, every surjective projection onto $U$ must contain a full coordinate $U$-subcube with the remaining coordinates $D$ fixed.*

*Proof.* Let $H \subseteq V$ be any squarefree monomial support, with doubled coordinates $D$, single-literal coordinates $T$ (specifying values $t \in \{0, 1\}^T$), and unused coordinates $U$.
- $H \in \iota(I_C^\vee)$ iff $\iota(H)$ meets every minimal generator of $I_C$, i.e., no missing subcube $Q_A \subseteq Q \setminus C$ is disjoint from $\iota(H)$. Being disjoint from $\iota(H)$ means $A$ specifies no coordinate in $D$ and agrees with $t$ wherever it specifies a coordinate in $T$. Such a missing pattern exists iff there exists $u \in \{0, 1\}^U$ such that $(d, t, u) \notin C$ for all $d \in \{0, 1\}^D$ (extend a missing pattern to $T \cup U$, or take the full cylinder $(t, u)$). Negating, $H \in \iota(I_C^\vee)$ holds iff the left-hand side of (4) holds.
- On the other hand, $H \in I_{Q \setminus C}$ iff $H$ contains the literals of some coordinate subcube $Q_A \subseteq C$. Completing a consistent subset of $H$ on $D \cup T$ leaves precisely $U$ free, so $H \in I_{Q \setminus C}$ holds iff the right-hand side of (4) holds.

Since the right-hand side always implies the left-hand side, $I_{Q \setminus C} \subseteq \iota(I_C^\vee)$ always, with equality on consistent supports ($D = \varnothing$, where both sides say $\forall u,\, (t, u) \in C$) and full equality iff (4) holds everywhere. $\square$

By **Terai's theorem** (Terai, 1999), $\operatorname{pd}(I^\vee) = \operatorname{reg}(I) - 1$ for every nonzero proper squarefree ideal $I$ over any field $k$; hence whenever $C$ satisfies (4), passing from $C$ to $Q \setminus C$ swaps $(\operatorname{pd}, \operatorname{reg}) = (p, r) \mapsto (r - 1, p + 1)$.

*Self-contained proof of Terai's identity.* Let $\Delta$ be the Stanley–Reisner complex of $I$ and $\Delta^* = \{F \subseteq V : V \setminus F \notin \Delta\}$ the complex of $I^\vee$. For nonempty $W \subseteq V$, the Alexander dual on ground set $W$ of $\Delta^*|_W$ is $\operatorname{link}_\Delta(V \setminus W)$. On an ordered ground set of size $m$, complementation identifies augmented relative chains of $\Delta^m$ modulo $K$ with reduced cochains of $K^*$ in complementary degrees $a \leftrightarrow m - a - 2$ (with shuffle orientation signs), giving $\widetilde{H}_j(K; k) \cong \widetilde{H}^{m - j - 3}(K^*; k)$ and therefore by (1),
$$
\beta_{i, W}(I^\vee) = \dim_k \widetilde{H}^{i - 1}\bigl(\operatorname{link}_\Delta(V \setminus W); k\bigr).
$$
Over a field $k$, cohomology and homology dimensions agree, so it suffices to check that the largest reduced homology degree among links in $\Delta$ equals the largest among induced subcomplexes $\Delta_W$. From the Mayer–Vietoris sequence of a vertex deletion $\operatorname{del}_K(v)$ and the cone over $\operatorname{link}_K(v)$:
- $\widetilde{H}_q(\operatorname{link}_K(v)) \neq 0$ forces $\widetilde{H}_q(\operatorname{del}_K(v)) \neq 0$ or $\widetilde{H}_{q+1}(K) \neq 0$; restricting away from $v$ shows the maximum over induced subcomplexes bounds the maximum over links.
- $\widetilde{H}_q(\operatorname{del}_K(v)) \neq 0$ forces $\widetilde{H}_q(K) \neq 0$ or $\widetilde{H}_q(\operatorname{link}_K(v)) \neq 0$; applying this to each face link (which commutes with outer deletions; ghost deletions leave $K$ unchanged) shows the maximum over links bounds the maximum over induced subcomplexes.

Hence $\operatorname{pd}(I^\vee) = \operatorname{reg}(I) - 1$, and applying it twice yields the swap $(p, r) \mapsto (r - 1, p + 1)$ on codes satisfying (4). $\square$

**Proposition 6 (Unbounded non-functionality of complementation).** *For $n \ge 4$, code complementation does not induce a function on $(\operatorname{pd}, \operatorname{reg})$. Moreover, for every $s \ge 1$ and every $n \ge 4s$, there exist codes $C_1, C_2 \subseteq \{0, 1\}^n$ with identical invariants*
$$
\bigl(\operatorname{pd}(I_{C_1}), \operatorname{reg}(I_{C_1})\bigr)
= \bigl(\operatorname{pd}(I_{C_2}), \operatorname{reg}(I_{C_2})\bigr)
= (2s - 1,\, 2s + 1),
$$
*whose complements satisfy*
$$
\bigl(\operatorname{pd}(I_{Q \setminus C_1}), \operatorname{reg}(I_{Q \setminus C_1})\bigr) = (2s,\, 2s),
\qquad
\bigl(\operatorname{pd}(I_{Q \setminus C_2}), \operatorname{reg}(I_{Q \setminus C_2})\bigr) = (s,\, 3s).
$$

*Proof.* For $n = 4$:
- Let $A \subset \{0, 1\}^4$ consist of all words with fourth bit $0$ except $1110$. Then $I_A = (x_4, x_1 x_2 x_3)$ is a complete intersection giving $(1, 3)$, while $I_{Q \setminus A} = y_4(y_1, y_2, y_3)$ has shifted Koszul resolution giving $(2, 2)$.
- Let $B \subset \{0, 1\}^4$ consist of all words whose third and fourth bits differ. Then $I_B = (x_3 x_4, y_3 y_4)$ and $I_{Q \setminus B} = (x_3 y_4, y_3 x_4)$ are both complete intersections giving $(1, 3)$.

On disjoint coordinate blocks, $I_{C_1 \times C_2} = I_{C_1} + I_{C_2}$ (a coordinate box misses the product iff one factor misses its code), while $I_{\overline{C_1 \times C_2}} = I_{\bar{C}_1} I_{\bar{C}_2}$ (maximal boxes in a product are products of maximal boxes). Tensoring quotient resolutions in the first case and ideal resolutions in the second yields the pair operations
$$
(p_1 + p_2 + 1,\, r_1 + r_2 - 1) \qquad\text{and}\qquad (p'_1 + p'_2,\, r'_1 + r'_2),
$$
respectively. Taking $s$ disjoint copies of $A$ versus $B$ (and adding $n - 4s$ free coordinates, which leave the ideals unchanged) gives the stated pairs. $\square$

---

## 4. Exact All-Dimension Enumerative Formula (`ques:region`)

We give an exact finite algebraic formula that determines, for any $n$, $p$, $r$, and field $k$, the exact number $M_n(p, r)$ of labeled nonempty proper codes $C \subseteq \{0, 1\}^n$ with $(\operatorname{pd}(I_C), \operatorname{reg}(I_C)) = (p, r)$, and constructs a witness whenever $M_n(p, r) > 0$.

Let $\mathcal{A}_n := \mathbb{Q}[2^Q]$ be the finite-dimensional commutative semilattice algebra with basis $\{e_A : A \subseteq Q\}$, unit $e_\varnothing = 1$, and multiplication $e_A e_B = e_{A \cup B}$. Define the linear functional $L : \mathcal{A}_n \to \mathbb{Q}$ by $L(e_A) := 2^{2^n - |A|}$. For each literal set $F \subseteq V$, put
$$
f_F := \prod_{A \in \mathcal{T}(F)} (1 - e_{Q_A}) \in \mathcal{A}_n.
$$
For each nonempty $W \subseteq V$ and $0 \le t \le |W|$, let $D_t$ be the oriented boundary matrix of the full simplex on $W$ with columns indexed by $t$-element faces ($D_1$ is the augmentation row of ones, and $D_0$ is the zero matrix with one column). For $1 \le j \le \operatorname{rank}_k D_t$, let $\mathcal{I}_{t, j}$ be the family of $j$-element sets of columns of $D_t$ that are linearly independent over $k$. Suppressing $W$, define
$$
R_t := \sum_{j = 1}^{\operatorname{rank}_k D_t} \left[1 - \prod_{J \in \mathcal{I}_{t, j}} \left(1 - \prod_{F \in J} f_F\right)\right],
$$
and for $0 \le t < |W|$,
$$
b_{W, t} := \sum_{\substack{F \subseteq W \\ |F| = t}} f_F - R_t - R_{t+1},
\qquad
z_{W, t} := \prod_{a = 1}^{\binom{|W|}{t}} \left(1 - \frac{b_{W, t}}{a}\right).
$$
For integers $p, r$, set
$$
A(p, r) := L\!\left(\prod_{\substack{\varnothing \neq W \subseteq V,\; 0 \le t < |W| \\ |W| - t - 1 > p \text{ or } t + 1 > r}} z_{W, t}\right) - 2.
$$

**Theorem 7 (Exact enumerative region formula).** *For every $n \ge 1$, every field $k$, and all integers $p \ge 0$, $r \ge 1$,*
$$
(p, r) \in \mathcal{R}_n(k) \iff M_n(p, r) := A(p, r) - A(p - 1, r) - A(p, r - 1) + A(p - 1, r - 1) > 0,
\tag{5}
$$
*where $M_n(p, r)$ is the exact number of labeled nonempty proper codes $C \subseteq \{0, 1\}^n$ attaining $(\operatorname{pd}(I_C), \operatorname{reg}(I_C)) = (p, r)$. (The empty code contributes $(0, 0)$; for $n \ge 2$, all nonempty proper pairs lie in $0 \le p \le 2n - 3$, $1 \le r \le 2n - 1$.)*

*Proof.* For each code $C \subseteq Q$, define the algebra homomorphism $\operatorname{ev}_C : \mathcal{A}_n \to \mathbb{Q}$ by $\operatorname{ev}_C(e_A) := \mathbf{1}[A \cap C = \varnothing]$. Multiplication preserves this indicator because $(A \cup B) \cap C = \varnothing \iff A \cap C = \varnothing \text{ and } B \cap C = \varnothing$, and summing $\operatorname{ev}_C(e_A)$ over all $2^{2^n}$ codes gives $2^{2^n - |A|} = L(e_A)$.

By the face test in Section 1, $\operatorname{ev}_C(f_F) = \mathbf{1}[F \in K(W)]$. Each bracket in $R_t$ evaluates to $1$ iff some $k$-independent $j$-tuple of columns of $D_t$ is active, so $\operatorname{ev}_C(R_t)$ equals the rank over $k$ of the active columns of $D_t$. Since every boundary face of an active face is active, this equals the rank of the $t$-th augmented boundary map of $K(W)$. Hence $\operatorname{ev}_C(b_{W, t}) = \dim_k \widetilde{H}_{t-1}(K(W); k) \in \{0, 1, \dots, \binom{|W|}{t}\}$, and Lagrange interpolation gives $\operatorname{ev}_C(z_{W, t}) = \mathbf{1}[\widetilde{H}_{t-1}(K(W); k) = 0]$.

By (1), $\widetilde{H}_{t-1}(K(W); k)$ contributes to homological degree $i = |W| - t - 1$ and strand $|W| - i = t + 1$. Thus the product inside $A(p, r)$ evaluates to $1$ iff $\operatorname{pd}(I_C) \le p$ and $\operatorname{reg}(I_C) \le r$. Both $C = Q$ (acyclic simplices) and $C = \varnothing$ (every $f_F$ evaluates to $0$, including $f_\varnothing$) evaluate to $1$ for every $p, r$, so subtracting $2$ removes both degenerate codes (in particular $A(-1, r) = A(p, 0) = 0$). Inclusion–exclusion on the cumulative count $A(p, r)$ yields $M_n(p, r)$. $\square$

*Witness construction and complexity note.* Fixing disjoint word sets $P$ (forced present) and $E$ (forced absent), replacing $L(e_A)$ by $0$ if $A \cap P \neq \varnothing$ and $2^{2^n - |A \cup E \cup P|}$ otherwise counts witnesses extending $(P, E)$ in the mixed difference (where the constant $-2$ cancels). Iterating over $Q$ in lexicographic order constructs an explicit witness in $2^n$ evaluations whenever $M_n(p, r) > 0$. Because the ambient algebra has dimension $2^{2^n}$, Formula (5) is an exact finite *enumerative* characterization rather than a compact inequality description for $n \ge 5$.

---

## 5. Complete Structure Through $p + r \le 2n$ and Beyond

All results in this section hold over an arbitrary field $k$.

**Theorem 8 (Complete description of $\mathcal{R}_n(k)$ for $p + r \le 2n$).** *For every $n \ge 2$ and every field $k$, the nonempty proper pairs $(p, r) \in \mathcal{R}_n(k)$ with $p + r \le 2n$ are precisely the union of:*
1. $(0, r)$ *for $1 \le r \le n$;*
2. $(p, 1)$ *for $0 \le p \le n - 1$;*
3. $(p, 2)$ *for $0 \le p \le n - 2$;*
4. *all $(p, r)$ with $p \ge 1$, $r \ge 3$, and $p + r \le 2n$.*

*Proof.* Parts (1) and (2) were established in Section 2.

**Regularity $2$ (part (3)).** When $\operatorname{reg}(I_C) = 2$, only homology degrees $0$ and $-1$ can contribute to (1). In degree $0$, connectedness (Theorem 3) forces $d(W) = 0$, so $|W| \le n$ and $i = |W| - 2 \le n - 2$. In degree $-1$, reaching $i = n - 1$ would require $n$ ghost literals, fixing all $n$ coordinates of $C$ and forcing $\operatorname{reg}(I_C) = 1$, a contradiction. Thus $p \le n - 2$. Conversely, for $0 \le p \le n - 2$, omitting the upward-closed set of words satisfying $x_1 x_2, \dots, x_1 x_{p+2}$ yields $I_C = x_1(x_2, \dots, x_{p+2})$, which has pair $(p, 2)$ by Koszul.

**Private-literals family (lower half of diagonal).** For $2 \le k \le s \le n$, omit $k$ words whose first $k$ columns form the identity matrix $I_k$, whose next $s - k$ columns repeat the first column, and whose last $n - s$ columns are $0$. Any two omitted words have Hamming distance $\ge 2$, so the minimal generators of $I_C$ correspond to the $k$ individual words, each possessing a private literal in the first $k$ coordinates. Hence every boundary deletion in the Taylor resolution strictly shrinks the literal union, so the Taylor resolution is minimal. Every nonempty subset of generators contributes one basis element with shift equal to its literal union and homological degree its size minus $1$. Adding a generator increases shift by at least $1$, so the full set of $k$ generators (shift $n + s$) maximizes both $i$ and shift minus $i$, yielding
$$
(p, r) = (k - 1,\, n + s - k + 1).
$$

**Complementary diagonal family (upper half of diagonal).** Now take $C$ to be the $k$ omitted words of the construction above. Each of the $n - s$ constant coordinates adds a variable generator, raising $\operatorname{pd}$ by $1$ without changing $\operatorname{reg}$. On the $s$ active coordinates, $C$ consists of one word whose $1$s form a block of $b := s - k + 1$ coordinates (literals $a_i, b_i$) and $t := k - 1$ singleton words on the remaining coordinates (literals $x_j, y_j$). Setting $Y := \prod_{j=1}^t y_j$, the minimal generators on the active coordinates are
$$
a_i b_\ell\;(i \neq \ell), \qquad a_i x_j, \qquad x_j x_h\;(j < h), \qquad b_i Y\;(1 \le i \le b).
$$
Replacing $Y$ by a single variable $z$ produces a quadratic ideal on $M := 2b + t + 1$ variables whose Stanley–Reisner complex consists of a simplex on $\{a_i\} \cup \{z\}$ and a union of simplices $\{b_1, \dots, b_b\} \cup \{x_j\}$ ($1 \le j \le t$) joined by $b$ matching edges $a_i b_i$ and $t$ edges $z x_j$, all outside triangles. Both pieces are contractible, so the full complex has $\dim_k \widetilde{H}_1 = b + t - 1 > 0$; every induced subcomplex consists of at most one simplex and a cone/discrete union joined by isolated edges, giving $\widetilde{H}_q = 0$ for all $q \ge 2$. Since there are no ghosts and the full complex is connected, (1) yields $(p, r) = (M - 3, 3)$ for the auxiliary ideal, both witnessed at full support.

Because $k[y_1, \dots, y_t]$ is free over $k[Y]$ (on monomials with at least one exponent $0$), substituting $z \mapsto Y$ is flat and preserves minimality of the resolution: $\operatorname{pd}$ remains $M - 3 = 2s - k - 1$, while every squarefree shift containing $z$ increases in degree by $t - 1$ and all others keep their degree. The full-support witness gives $\operatorname{reg} = 3 + t - 1 = k + 1$. Restoring the $n - s$ constant coordinates gives
$$
(p, r) = (n + s - k - 1,\, k + 1).
$$

**Filling $p \ge 1$, $r \ge 3$, $p + r \le 2n$ (part (4)).**
- If $p + r = 2m$ is even ($m \le n$), setting $s = n' = m$ in the two families gives $(p, 2m - p)$ for $1 \le p \le m - 1$ (first family) and $(2m - r, r)$ for $3 \le r \le m + 1$ (second family), covering the entire diagonal; pad with $n - m$ free coordinates.
- If $p + r = 2m + 1$ is odd ($m + 1 \le n$) and $r \ge 4$, realize $(p, r - 1)$ on $m$ coordinates and restrict the missing set to one layer of a new coordinate (multiplying $I_C$ by one variable, raising $\operatorname{reg}$ by $1$); pad with free coordinates.
- If $p + r = 2m + 1$ is odd and $r = 3$, then $p \ge 2$; realize $(p - 1, 3)$ on $m$ coordinates and fix the new coordinate on $C$ (adding a disjoint variable generator, raising $\operatorname{pd}$ by $1$); pad with free coordinates. $\square$

**Proposition 9 (Explicit family reaching $p + r = 3n - 3$).** *For every $n \ge 3$ and every field $k$, every pair $(p, 2n - 2)$ with $1 \le p \le n - 1$ lies in $\mathcal{R}_n(k)$. In particular, $p + r$ attains $3n - 3$.*

*Proof.* Set $m := n - 1 \ge 2$, $P := x_1 \cdots x_m$, $Y := y_1 \cdots y_m$, $z := x_n$, $w := y_n$. For $1 \le t \le m$, omit the all-zero word and all words satisfying $P$ or any $z x_j$ ($1 \le j \le t$). The all-zero word is isolated since $m \ge 2$, and the upward-closed part has incomparable monotone generators, so
$$
I_C = (P,\, z x_1, \dots, z x_t,\, w Y).
$$
Starting from $z(x_1, \dots, x_t)$ with $(\operatorname{pd}, \operatorname{reg}) = (t - 1, 2)$, adjoining $P$ yields $(P) \cap z(x_1, \dots, x_t) = (zP)$, whose mapping cone is minimal (since $m \ge 2$, both coefficients $z$ and $P / x_1$ have positive degree) and has $(\operatorname{pd}, \operatorname{reg}) = (\max(1, t - 1), m)$. Since $wY$ is a disjoint degree-$n$ monomial, tensoring quotient resolutions adds $1$ to $\operatorname{pd}$ and $n - 1$ to $\operatorname{reg}$, giving $(\max(2, t), 2n - 2)$. Finally, omitting two words at Hamming distance $n - 1 \ge 2$ yields two degree-$n$ generators with union size $2n - 1$, giving $(1, 2n - 2)$. $\square$

---

## 6. Reproducible Exact Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/neural_code_homological/explore.py
```

The script [`explore.py`](explore.py) performs exact binary elimination over $\mathbb{F}_2$ across all Hamming-cube symmetry orbits (coordinate permutations and bit flips) for $1 \le n \le 4$:

| $n$ | Labeled codes | Cube orbits | Distinct pairs | Regularities at successive projective dimensions ($p = 0, 1, \dots$) |
|---:|---:|---:|---:|:---|
| 1 | 2 | 1 | 1 | $\{1\}$ |
| 2 | 14 | 4 | 4 | $\{1, 2\} \cdot \{1, 3\}$ |
| 3 | 254 | 20 | 12 | $1\text{--}3 \cdot 1\text{--}5 \cdot \{1, 3, 4\} \cdot \{3\}$ |
| 4 | 65,534 | 400 | 26 | $1\text{--}4 \cdot 1\text{--}7 \cdot 1\text{--}6 \cdot \{1, 3, 4, 5, 6\} \cdot 3\text{--}5 \cdot \{3\}$ |

For $n \le 3$, no pair exceeds $p + r = 2n$. For $n = 4$, exactly two pairs exceed it: $(3, 6)$ (from Proposition 9 with $t = 3$) and $(4, 5)$ (witnessed by words $\{0, 7, 9, 10, 12\}$ in binary, coordinate $1$ least significant). The script independently verifies:
- Agreement of Theorems 1 and 2 in every multidegree of all 425 orbit representatives.
- Theorem 3 at every doubled support of all 425 representatives, and Theorem 4 for $n = 3, 4$.
- Theorem 5 across all supports, finding exact literal-reversed Alexander duality for $2, 12, 126, 5{,}528$ labeled codes at $n = 1, 2, 3, 4$, respectively.
- The independent avoidance-algebra formula (5) against exact labeled multiplicities for $n = 1, 2$.
- Both diagonal families and the $p + r = 3n - 3$ family through $n = 6$, the complement counterexample $A, B$, and both degenerate codes.

All assertions passed. (Prior context: T. Chau, [arXiv:2511.19043](https://arxiv.org/abs/2511.19043), studies the broader class of squarefree monomial ideals without $x_i y_i$ generators, where canonical-form polar structure is absent.)

---

## References

- T. Chau, *Projective dimension and regularity of squarefree monomial ideals without $x_i y_i$*, [arXiv:2511.19043](https://arxiv.org/abs/2511.19043), 2025.
- S. Kara and E. Lew, *Neural codes via homological invariants of polarized neural ideals*, [arXiv:2602.16993](https://arxiv.org/abs/2602.16993), February 2026.
- N. Terai, *Alexander duality theorem and Stanley–Reisner rings*, in *Free Resolutions of Coordinate Rings of Projective Varieties and Related Topics*, RIMS Kôkyûroku 1078:174–184, 1999.
