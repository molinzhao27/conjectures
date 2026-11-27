# Counterexamples to Three Coded Block Retrieval Conjectures

**Target paper:** Daniella Bar-Lev, *Coded Information Retrieval for Block-Structured DNA-Based Data Storage*, [arXiv:2603.17154](https://arxiv.org/abs/2603.17154) (March 2026), Conjectures `conj:hyperbola`, `conj:monotone`, and `conj:limiting`.  
**Companion files:** [`solutions/coded_block_retrieval/explore.py`](explore.py), [`solutions/coded_block_retrieval/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:hyperbola` (Universal hyperbolic bound): FALSE.**
  - For $(s_1, s_2) = (2, 1)$ over $\mathbb{F}_{11}$ at length $n = 22$, an explicit rank-$3$ generator matrix achieves $(E_1, E_2) = (937/210,\, 25/14)$, giving
    $$
    \frac{2}{E_1} + \frac{1}{E_2} = \frac{23618}{23425} = 1 + \frac{193}{23425} > 1.
    $$
  - For the balanced partition $(s_1, s_2) = (2, 2)$ over $\mathbb{F}_{83}$ at length $n = 40$, an explicit rank-$4$ generator matrix with **40 pairwise non-collinear (projectively distinct) columns** achieves $(E_1, E_2) = (30729/9139,\, 134404/27417)$, giving
    $$
    \frac{2}{E_1} + \frac{2}{E_2} = \frac{2070815149}{2065050258} > 1.
    $$
    Thus `conj:hyperbola` fails even if repeated or collinear columns are prohibited.
- **Conjecture `conj:monotone` (Length monotonicity): FALSE over every finite field $\mathbb{F}_q$.** For $(s_1, s_2) = (2, 1)$, the length-$9$ matrix $[e_1, e_2, e_3^7]$ achieves $(27/2,\, 9/7)$ over every $\mathbb{F}_q$, whereas no rank-$3$ matrix of length $10$ over any finite field satisfies both $E_1' \le 27/2$ and $E_2' \le 9/7$. This refutes the expectation in Remark 4 of arXiv:2603.17154 that $\max\{s_1, s_2\} \ge 2$ eliminates monotonicity obstructions.
- **Conjecture `conj:limiting` (Limiting achievability region): FALSE.** By block repetition (Proposition 10 of arXiv:2603.17154), $(937/210,\, 25/14) \in \Lambda_{22m, 3}(2, 1)$ for every integer $m \ge 1$ over $\mathbb{F}_{11}$, and $(30729/9139,\, 134404/27417) \in \Lambda_{40m, 4}(2, 2)$ for every $m \ge 1$ over $\mathbb{F}_{83}$, so the limiting region strictly contains points outside $\{s_1/E_1 + s_2/E_2 \le 1\}$.

---

## 1. Model and Conjectures

Let $G \in \mathbb{F}_q^{k \times n}$ be a rank-$k$ generator matrix with columns $g_1, \dots, g_n \in \mathbb{F}_q^k$ (repeated columns are permitted by the model, as in Section II, Remarks 3–4, and Proposition 10 of arXiv:2603.17154). Fix a partition $s_1 + s_2 = k$ with $s_1, s_2 \ge 1$, and define the two file subspaces
$$
F_1 := \langle e_1, \dots, e_{s_1} \rangle, \qquad F_2 := \langle e_{s_1+1}, \dots, e_k \rangle.
$$
Column indices $I_1, I_2, \dots$ are sampled independently and uniformly from $\{1, \dots, n\}$ with replacement. Writing $V_t := \operatorname{span}\{g_{I_1}, \dots, g_{I_t}\}$, the retrieval times and their expectations are
$$
\tau_i := \min\{t \ge 1 : F_i \subseteq V_t\}, \qquad E_i(G) := \mathbb{E}[\tau_i] \qquad (i \in \{1, 2\}).
$$
The finite-length achievability region $\Lambda_{n, k}(s_1, s_2)$ is the set of pairs $(E_1(G), E_2(G))$ realized by rank-$k$ matrices $G \in \mathbb{F}_q^{k \times n}$.

The paper conjectures that whenever $\max\{s_1, s_2\} \ge 2$:
1. (`conj:hyperbola`) Every rank-$k$ matrix $G$ satisfies $s_1/E_1(G) + s_2/E_2(G) \le 1$.
2. (`conj:monotone`) For every $n \ge k$ and $(E_1, E_2) \in \Lambda_{n,k}(s_1, s_2)$, there exists $(E_1', E_2') \in \Lambda_{n+1, k}(s_1, s_2)$ with $E_1' \le E_1$ and $E_2' \le E_2$.
3. (`conj:limiting`) $\operatorname{cl}\!\bigl(\bigcup_{n \ge k} \Lambda_{n, k}(s_1, s_2)\bigr) = \{(E_1, E_2) \in \mathbb{R}_{>0}^2 : s_1/E_1 + s_2/E_2 \le 1,\, E_i \ge s_i\}$.

---

## 2. Refutation of `conj:hyperbola` and `conj:limiting` at $(s_1, s_2) = (2, 1)$

**Theorem 1.** *Let $(s_1, s_2) = (2, 1)$ over $\mathbb{F}_{11}$, and let $G \in \mathbb{F}_{11}^{3 \times 22}$ consist of $11$ copies of $z := e_3 = (0, 0, 1)^\top$ together with one copy of each column*
$$
v_t := (1, t, t^2)^\top \qquad (t \in \mathbb{F}_{11}).
$$
*Then $\operatorname{rank}(G) = 3$ and*
$$
E_1(G) = \frac{937}{210}, \qquad E_2(G) = \frac{25}{14}, \qquad \frac{2}{E_1(G)} + \frac{1}{E_2(G)} = \frac{23618}{23425} = 1 + \frac{193}{23425} > 1.
$$
*Moreover, for every integer $m \ge 1$, the $m$-fold concatenation $G^{(m)} = [G \mid \cdots \mid G] \in \mathbb{F}_{11}^{3 \times 22m}$ achieves the same pair $(937/210,\, 25/14)$, refuting both `conj:hyperbola` and `conj:limiting`.*

*Proof.* Call $v_0, \dots, v_{10}$ the *ordinary* types and $z$ the *special* type. Any three distinct ordinary columns $v_{t_1}, v_{t_2}, v_{t_3}$ are linearly independent because their determinant is the Vandermonde product $\prod_{1 \le a < b \le 3}(t_b - t_a) \neq 0$ in $\mathbb{F}_{11}$. Any two distinct ordinary columns $v_{t_1}, v_{t_2}$ together with $z = e_3$ have determinant $t_2 - t_1 \neq 0$ and are therefore linearly independent. Consequently, any three distinct column types span $\mathbb{F}_{11}^3$.

Next we identify the recovery events for $F_1 = \langle e_1, e_2 \rangle$ and $F_2 = \langle e_3 \rangle$:
- **File $F_1$:** Among all $12$ column types, only $v_0 = (1, 0, 0)^\top$ lies in $F_1$. Any two distinct column types either include $z$ (so their span projects to a $1$-dimensional subspace of $\mathbb{F}_{11}^3/F_2$, hence cannot contain the $2$-dimensional subspace $F_1$), or consist of two distinct ordinary columns $v_{t_1}, v_{t_2}$ whose third coordinates $(t_1^2, t_2^2)$ cannot both vanish, so their $2$-dimensional span is not contained in $F_1$ and therefore cannot equal $F_1$. Thus $F_1$ is recovered if and only if full rank $3$ is reached.
- **File $F_2$:** Any span of at most two ordinary types projects injectively onto the first two coordinates (since $(1, t_1)$ and $(1, t_2)$ are linearly independent for $t_1 \neq t_2$), so its intersection with $F_2 = \langle e_3 \rangle$ is $\{0\}$. Thus $F_2$ is recovered if and only if $z$ is drawn or full rank $3$ is reached.

Let $W_i(j, h)$ denote the expected remaining draws to recover $F_i$ after observing $j$ distinct ordinary types and indicator $h \in \{0, 1\}$ for whether $z$ has appeared. By the characterization above, $W_i(j, h) = 0$ whenever $j + h = 3$, and also $W_2(j, 1) = 0$. At every remaining state, there are $a := 11(1 - h)$ columns that flip $h$ from $0$ to $1$ and $b := 11 - j$ columns that increase $j$ by $1$, while the remaining $22 - a - b$ columns leave the state unchanged. Conditioning on the next draw and isolating the self-loop yields
$$
W_i(j, h) = \frac{22 + a\,W_i(j, 1) + b\,W_i(j + 1, h)}{a + b},
$$
with the $a$-term omitted when $a = 0$. Exact back substitution gives:

| State $(j, h)$ | $a$ | $b$ | $W_1(j, h)$ | $W_2(j, h)$ |
|---|---:|---:|---:|---:|
| $(1, 1)$ | $0$ | $10$ | $11/5$ | $0$ |
| $(2, 0)$ | $11$ | $9$ | $11/10$ | $11/10$ |
| $(0, 1)$ | $0$ | $11$ | $21/5$ | $0$ |
| $(1, 0)$ | $11$ | $10$ | $286/105$ | $11/7$ |
| $(0, 0)$ | $11$ | $11$ | $937/210$ | $25/14$ |

Thus $E_1(G) = W_1(0, 0) = 937/210$, $E_2(G) = W_2(0, 0) = 25/14$, and
$$
\frac{2}{E_1(G)} + \frac{1}{E_2(G)} = \frac{420}{937} + \frac{14}{25} = \frac{23618}{23425} > 1.
$$
Finally, concatenating $m$ identical copies of $G$ gives $G^{(m)} \in \mathbb{F}_{11}^{3 \times 22m}$ in which each column type retains the exact same sampling probability ($1/2$ for $z$, $1/22$ for each $v_t$), preserving the entire joint law of $(\tau_1, \tau_2)$ and hence $(E_1, E_2) = (937/210, 25/14) \in \Lambda_{22m, 3}(2, 1)$ for all $m \ge 1$ (cf. Proposition 10 of arXiv:2603.17154). Hence this point lies in $\operatorname{cl}\!\bigl(\bigcup_{n \ge 3} \Lambda_{n, 3}(2, 1)\bigr)$ and strictly outside the conjectured limiting region. $\square$

---

## 3. Refutation of `conj:monotone` Over Every Finite Field

**Theorem 2.** *For $(s_1, s_2) = (2, 1)$ and every finite field $\mathbb{F}_q$, the rank-$3$ matrix $G = [e_1, e_2, e_3, \dots, e_3] \in \mathbb{F}_q^{3 \times 9}$ (one copy each of $e_1, e_2$ and seven copies of $e_3$) achieves*
$$
(E_1(G), E_2(G)) = \left(\frac{27}{2},\, \frac{9}{7}\right) \in \Lambda_{9, 3}(2, 1),
$$
*whereas no rank-$3$ matrix $G' \in \mathbb{F}_q^{3 \times 10}$ satisfies both $E_1(G') \le 27/2$ and $E_2(G') \le 9/7$. Consequently $\Lambda_{9, 3}(2, 1) \not\subseteq \Lambda_{10, 3}(2, 1)$ under coordinatewise dominance.*

*Proof.* For $G$ at length $n = 9$, file $F_2 = \langle e_3 \rangle$ is recovered on the first draw of any of the $7$ copies of $e_3$, so $\tau_2 \sim \mathrm{Geom}(7/9)$ and $E_2(G) = 9/7$. File $F_1 = \langle e_1, e_2 \rangle$ requires drawing both $e_1$ and $e_2$; the expected wait for the first of the two is $9/2$ and for the remaining one is $9$, so $E_1(G) = 9/2 + 9 = 27/2$.

Now let $G' \in \mathbb{F}_q^{3 \times 10}$ be any rank-$3$ matrix over any finite field $\mathbb{F}_q$, and let $m$ be the number of columns of $G'$ that are **not** nonzero scalar multiples of $e_3$ (so $m$ counts all columns outside $F_2$, as well as any zero columns). Equivalently, $10 - m$ columns recover $F_2$ on the first draw, so $\mathbb{P}(\tau_2 > 1) = m/10$. Because $\tau_2 \ge 1 + \mathbf{1}_{\{\tau_2 > 1\}}$,
$$
E_2(G') \ge 1 + \frac{m}{10}.
$$
If $E_2(G') \le 9/7$, then $m/10 \le 2/7$, forcing $m \le 2$. On the other hand, since $\operatorname{rank}(G') = 3$, the images of the columns of $G'$ in the $2$-dimensional quotient space $\mathbb{F}_q^3 / F_2$ must span $\mathbb{F}_q^3 / F_2$. Every nonzero multiple of $e_3$ (and every zero column) projects to $0$ in this quotient, so at least $2$ columns must lie outside $F_2$ with linearly independent quotient images. Combined with $m \le 2$, this forces $m = 2$, with both exceptional columns having linearly independent projections onto $\mathbb{F}_q^3 / F_2$, and all other $8$ columns being nonzero multiples of $e_3$.

Because $F_1$ projects isomorphically onto the $2$-dimensional quotient $\mathbb{F}_q^3 / F_2$, any column subset spanning $F_1$ must project onto all of $\mathbb{F}_q^3 / F_2$, and therefore must contain **both** exceptional column indices. The expected time to observe both of two fixed indices under uniform sampling from $10$ columns is $10/2 + 10 = 15$. Hence
$$
E_1(G') \ge 15 > \frac{27}{2} = E_1(G),
$$
ruling out any dominating length-$10$ matrix over any finite field. $\square$

---

## 4. A Projectively Distinct Balanced Counterexample at $(s_1, s_2) = (2, 2)$

Although repeated columns are explicitly allowed in arXiv:2603.17154, the hyperbolic bound and limiting region also fail for balanced files $(s_1, s_2) = (2, 2)$ with **no repeated or collinear columns**.

**Theorem 3.** *Let $(s_1, s_2) = (2, 2)$ over $\mathbb{F}_{83}$, and let $G \in \mathbb{F}_{83}^{4 \times 40}$ have the $40$ columns*
$$
A_u := (1, u, 0, 0)^\top \quad (61 \le u \le 80), \qquad
B_t := (t^2, t^3, 1, t)^\top \quad (1 \le t \le 20).
$$
*Then all $40$ columns of $G$ are pairwise non-collinear, $\operatorname{rank}(G) = 4$, and*
$$
(E_1(G), E_2(G)) = \left(\frac{30729}{9139},\, \frac{134404}{27417}\right), \qquad
\frac{2}{E_1(G)} + \frac{2}{E_2(G)} = \frac{2070815149}{2065050258} > 1.
$$

*Proof.* **Linear independence.** We check the three types of four-column subsets with at most two $A$-columns:
1. *Four distinct $B$-columns* $B_{t_1}, \dots, B_{t_4}$: permuting coordinates to $(1, t, t^2, t^3)^\top$ yields a nonzero $4 \times 4$ Vandermonde determinant in $\mathbb{F}_{83}$.
2. *Two distinct $A$-columns and two distinct $B$-columns* $A_{u_1}, A_{u_2}, B_{t_1}, B_{t_2}$: the matrix is block triangular with $2 \times 2$ diagonal blocks of determinants $u_2 - u_1 \neq 0$ and $t_2 - t_1 \neq 0$ in $\mathbb{F}_{83}$.
3. *One $A$-column and three distinct $B$-columns* $A_u, B_{t_1}, B_{t_2}, B_{t_3}$: a linear functional $\varphi(x_1, x_2, x_3, x_4) = c_2 x_1 + c_3 x_2 + c_0 x_3 + c_1 x_4$ evaluates on $B_t$ as the cubic polynomial $p(t) = c_0 + c_1 t + c_2 t^2 + c_3 t^3$. If $\varphi \neq 0$ vanishes on $B_{t_1}, B_{t_2}, B_{t_3}$, then $p(t)$ is a nonzero scalar multiple of $(t - t_1)(t - t_2)(t - t_3) = t^3 - (t_1 + t_2 + t_3)t^2 + \cdots$, so $c_3 \neq 0$ and $c_2 = -(t_1 + t_2 + t_3)c_3$. Hence
   $$
   \varphi(A_u) = c_2 + u c_3 = c_3\bigl(u - (t_1 + t_2 + t_3)\bigr).
   $$
   Because $1 \le t_1 < t_2 < t_3 \le 20$, the integer sum $t_1 + t_2 + t_3$ lies in $[6, 57]$, whereas $u \in [61, 80]$, leaving a strict gap modulo $83$ on both sides ($57 < 61$ and $80 < 83 + 6 = 89$). Thus $\varphi(A_u) \neq 0$, proving linear independence.

In particular, every collection of $i \in \{0, 1, 2\}$ distinct $A$-columns and $j$ distinct $B$-columns with $i + j \le 4$ can be extended to one of the bases above and therefore has rank $i + j$, while any two distinct $A$-columns already span all $20$ columns in $F_1 = \langle e_1, e_2 \rangle$. Also, the $40$ projections onto the first two coordinates are $(1, u)$ ($61 \le u \le 80$) and $t^2(1, t)$ ($1 \le t \le 20$), which are all nonzero with pairwise distinct slopes in $[1, 20] \sqcup [61, 80] \subset \mathbb{F}_{83}$, confirming pairwise non-collinearity.

**Recovery criteria.** Write $i \in \{0, 1, 2\}$ for the number of distinct $A$-columns seen (capped at $2$) and $j$ for the number of distinct $B$-columns seen, and let $V$ be the current span.
- **File $F_1 = \langle e_1, e_2 \rangle$:** Recovered if and only if $i = 2$ or $i + j = 4$. Indeed, when $i < 2$ and $i + j \le 3$, the projection of $V$ onto the last two coordinates is spanned by the $j$ vectors $(1, t_r)$, which has rank $\min(j, 2)$; by rank-nullity,
  $$
  \dim(V \cap F_1) = (i + j) - \min(j, 2) \le 1 < 2.
  $$
- **File $F_2 = \langle e_3, e_4 \rangle$:** Recovered if and only if $i + j = 4$. Indeed, if $\dim V \in \{2, 3\}$, then $V$ contains at least two distinct column types whose projections onto the first two coordinates have distinct slopes, so the projection of $V$ onto $F_1$ has rank $2$; by rank-nullity, $\dim(V \cap F_2) = \dim V - 2 \le 1 < 2$.

**Exact recurrence.** For either file, let $T(i, j)$ be the expected remaining draws until recovery, with $T(i, j) = 0$ at recovery states. At every non-recovery state, $a := (20 - i)\mathbf{1}_{\{i < 2\}}$ columns increase $i$ by $1$ and $b := 20 - j$ columns increase $j$ by $1$, so
$$
T(i, j) = \frac{40 + a\,T(i + 1, j) + b\,T(i, j + 1)}{a + b},
$$
with the $a$-term omitted when $a = 0$. Exact rational back substitution gives $E_1(G) = 30729/9139$ and $E_2(G) = 134404/27417$, whence
$$
\frac{2}{E_1(G)} + \frac{2}{E_2(G)} = \frac{18278}{30729} + \frac{54834}{134404} = \frac{2070815149}{2065050258} > 1.
$$
Repeating $G$ $m$ times likewise refutes `conj:limiting` for $(s_1, s_2) = (2, 2)$ at every length $40m$. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/coded_block_retrieval/explore.py
```

The script [`explore.py`](explore.py) performs exact rational arithmetic (`fractions.Fraction`) directly on canonical reduced row-echelon spans over prime fields without relying on the reduced $(j, h)$ or $(i, j)$ state machines:
- Cross-checks the exact span DP against an independent uniform-$j$-subset average across **81 seeded full-rank matrices** over $\mathbb{F}_2, \mathbb{F}_3, \mathbb{F}_5$ for partitions $(2, 1), (2, 2), (3, 2)$.
- Verifies $(E_1, E_2) = (937/210, 25/14)$ and hyperbolic score $23618/23425 > 1$ for the $\mathbb{F}_{11}$, $n = 22$ matrix, and invariance under repetition factors $m \in \{2, 3, 7\}$.
- Verifies $(E_1, E_2) = (30729/9139, 134404/27417)$ and hyperbolic score $2070815149/2065050258 > 1$ for the projectively distinct $\mathbb{F}_{83}$, $n = 40$ matrix.
- Verifies the length $9 \to 10$ monotonicity obstruction over $\mathbb{F}_2, \mathbb{F}_3, \mathbb{F}_5$.

All assertions passed.

---

## References

- D. Bar-Lev, *Coded Information Retrieval for Block-Structured DNA-Based Data Storage*, [arXiv:2603.17154](https://arxiv.org/abs/2603.17154), March 2026.
