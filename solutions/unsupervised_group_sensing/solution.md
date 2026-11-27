# Refutation of the Compact-Group Sensing Conjecture

**Target paper:** Julián Tachella, Dongdong Chen, and Mike Davies, *Sampling Theorems for Unsupervised Learning in Linear Inverse Problems*, [arXiv:2203.12513](https://arxiv.org/abs/2203.12513) (published in *Journal of Machine Learning Research* 24(191):1–45, 2023, [PDF](https://www.jmlr.org/papers/volume24/22-0315/22-0315.pdf)), Section 4.3, Conjecture 10 (`conj: all groups`).  
**Companion files:** [`solutions/unsupervised_group_sensing/explore.py`](explore.py), [`solutions/unsupervised_group_sensing/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 10 (`conj: all groups`): DISPROVED**, including its single-operator $\mathcal{G}$-invariant clause, under both the unbounded conic branch A1(b) and the bounded branch A1(a) of Assumption A1.
- **Minimal explicit counterexample (Theorem 1):** A faithful diagonal sign action of $G = (\mathbb{Z}/2)^6 = \mathbb{F}_2^3 \times \mathbb{F}_2^3$ on $\mathbb{R}^{28}$ decomposes into $28$ distinct one-dimensional characters ($c_j = s_j = 1$, hence $C = \max_j c_j/s_j = 1$) with exact nonidentity trace $\operatorname{tr} T_{(u,w)} = -4(-1)^{u\cdot w}$. Every nonidentity element has $16$ diagonal entries of one sign and $12$ of the other. For the $G$-invariant set of $12$-sparse signals ($k = 12$), Conjecture 10 predicts unique generic identification for $m > 2k + C + 1 = 26$ (so $m = 27$), whereas the exact generic identification threshold is **$m = 28$**: for almost every $A \in \mathbb{R}^{27 \times 28}$, a full-support kernel vector yields a distinct $G$-invariant model of the same dimension with identical measurements.
- **Arbitrarily large gap (Corollary 2):** Replacing $\mathbb{F}_2^3 \times \mathbb{F}_2^3$ by $\mathbb{F}_2^r \times \mathbb{F}_2^r$ ($r \ge 3$) yields $n = (2^r - 1)2^{r-1}$, $C = 1$, and sparsity $k = (n - 2^{r-1})/2$, where the exact generic threshold $n$ exceeds the first integer allowed by Conjecture 10 by $2^{r-1} - 3 \to \infty$ while $C = 1$ stays fixed.
- **Comparison subspace models (Proposition 3):** For diagonal actions of $SO(2)$, $O(2)$, $SO(3)$, and $S_4$ on $W^6$ with $\mathcal{X} = W \oplus 0^5$, quotient Jacobian rank calculations give the exact generic identification thresholds $m = 7, 5, 5, 5$ respectively, illustrating how the linear span of the group representation governs identification beyond the normalized multiplicity $C$.

---

## 1. Statement and Prior Results

Let $\mathcal{G}$ be a compact group acting on $\mathbb{R}^n$ through linear operators $\{T_g\}_{g \in \mathcal{G}}$, and let $\mathcal{X} \subset \mathbb{R}^n$ be a signal set fixed before choosing a measurement operator $A \in \mathbb{R}^{m \times n}$. The observed data are the indexed measurement sets $\{A T_g \mathcal{X}\}_{g \in \mathcal{G}}$; when $\mathcal{X}$ is $\mathcal{G}$-invariant, all indexed views coincide with $A\mathcal{X}$. Unique identification of $\mathcal{X}$ requires that no distinct admissible model $\mathcal{X}'$ satisfy $A T_g \mathcal{X}' = A T_g \mathcal{X}$ for all $g \in \mathcal{G}$.

**Assumption A1** allows either:
- **A1(a) (bounded branch):** $\mathcal{X}$ is a bounded subset of $\mathbb{R}^n$ of upper box-counting dimension $k$; or
- **A1(b) (unbounded conic branch):** $\mathcal{X}$ is a cone whose intersection with the unit sphere $\mathbb{S}^{n-1}$ has upper box-counting dimension $k - 1$.

Over $\mathbb{C}$, the representation decomposes into $c_j$ copies of each occurring irreducible representation of complex dimension $s_j$, so $n = \sum_j c_j s_j$. Set $C := \max_j c_j / s_j$. Tachella, Chen, and Davies prove the necessary compact-group bound $m \ge C$ and the sufficient bound $m > 2k + C + 1$ for finite cyclic groups (and circle actions via restriction to cyclic subgroups preserving distinct weights). Applying that theorem to a general group via a cyclic subgroup requires the multiplicities of the *restricted* representation. For independent generic operators, a separate theorem gives $m > k + n/|\mathcal{G}|$.

**Conjecture 10 (`conj: all groups`, arXiv:2203.12513 / JMLR 2023, §4.3).** *Let $\mathcal{G}$ be any compact group acting on $\mathbb{R}^n$ and assume the signal set $\mathcal{X}$ verifies Assumption A1. For almost every $A \in \mathbb{R}^{m \times n}$ with respect to Lebesgue measure, $\mathcal{X}$ can be uniquely identified from $\{A T_g \mathcal{X}\}_{g \in \mathcal{G}}$ if $m > 2k + \max_j c_j/s_j + 1$. In particular, for almost any $A \in \mathbb{R}^{m \times n}$, a $\mathcal{G}$-invariant signal set can be identified from $A\mathcal{X}$ if $m > 2k + \max_j c_j/s_j + 1$.*

---

## 2. Sign Representation with Small Minority Eigenspaces

Let $G := \mathbb{F}_2^3 \times \mathbb{F}_2^3 \cong (\mathbb{Z}/2)^6$ under coordinatewise addition modulo $2$, and index the coordinates of $\mathbb{R}^{28}$ by the subset
$$
D := \{(a, b) \in \mathbb{F}_2^3 \times \mathbb{F}_2^3 : a \cdot b = 1\}.
$$
Each of the $7$ nonzero vectors $a \in \mathbb{F}_2^3$ has $4$ partners $b \in \mathbb{F}_2^3$ satisfying $a \cdot b = 1$ (mod $2$), so $n := |D| = 28$. For each $g = (u, w) \in G$, define the diagonal orthogonal operator $T_g \in O(28)$ by
$$
(T_{(u,w)})_{(a,b),(a,b)} := (-1)^{u \cdot a + w \cdot b} \qquad ((a,b) \in D).
$$
Because the pairs $(a, b) \in D$ are distinct elements of the dual group $\widehat{G} \cong \mathbb{F}_2^6$, these $28$ diagonal entries are pairwise distinct real sign characters, remaining distinct one-dimensional irreducibles after complexification. Hence $c_j = s_j = 1$ for every occurring irreducible, and $C = \max_j c_j / s_j = 1$.

**Lemma 1 (Exact trace formula and faithfulness).** *For every nonzero $g = (u, w) \in G \setminus \{(0,0)\}$,*
$$
\operatorname{tr} T_{(u,w)} = -4(-1)^{u \cdot w}.
$$
*In particular, every nonidentity $T_g$ has $16$ diagonal entries of one sign and $12$ of the opposite sign, so the action is faithful.*

*Proof.* Since $g = (u, w) \neq (0, 0)$, the character sum over all $(a, b) \in \mathbb{F}_2^3 \times \mathbb{F}_2^3$ vanishes. Using the indicator $\mathbf{1}_{a \cdot b = 1} = \frac{1}{2}\bigl(1 - (-1)^{a \cdot b}\bigr)$,
$$
\operatorname{tr} T_{(u,w)}
= \sum_{(a,b) \in G} (-1)^{u \cdot a + w \cdot b} \frac{1 - (-1)^{a \cdot b}}{2}
= -\frac{1}{2} \sum_{a \in \mathbb{F}_2^3} (-1)^{u \cdot a} \sum_{b \in \mathbb{F}_2^3} (-1)^{(a + w) \cdot b}.
$$
The inner sum equals $8$ when $a = w$ and $0$ otherwise, giving $\operatorname{tr} T_{(u,w)} = -4(-1)^{u \cdot w}$. Since $n = 28$ and $\operatorname{tr} T_g \in \{+4, -4\}$ for $g \neq 0$, the counts of $+1$ and $-1$ entries are $\{16, 12\}$ (specifically $28$ elements have trace $+4$, $35$ have trace $-4$, and the identity has trace $28$). As $T_g \neq I$ for $g \neq 0$, the representation is faithful. $\square$

For each $g \in G$, let $\epsilon_g \in \{+1, -1\}$ denote the majority sign of $T_g$ (with $\epsilon_0 := +1$). Then the diagonal operator $T_g - \epsilon_g I$ vanishes on the $16$ majority coordinates (all $28$ when $g = 0$), so its image is supported on at most $12$ coordinates.

---

## 3. Refutation of Conjecture 10 and Arbitrarily Large Gap

**Theorem 1 (Sharp counterexample in $\mathbb{R}^{28}$).** *Under the faithful $G = (\mathbb{Z}/2)^6$ action above ($n = 28$, $C = 1$), both the unbounded $G$-invariant cone $\mathcal{X} := \{x \in \mathbb{R}^{28} : \|x\|_0 \le 12\}$ (A1(b), $k = 12$) and its bounded truncation $\mathcal{X}_B := \mathcal{X} \cap B_1(0)$ (A1(a), $k = 12$) satisfy $2k + C + 1 = 26$, yet fail to be uniquely identified from $A\mathcal{X}$ (equivalently from $\{A T_g \mathcal{X}\}_{g \in G}$) for almost every $A \in \mathbb{R}^{m \times 28}$ whenever $m \le 27$. For $m \ge 28$, almost every $A$ uniquely identifies both models. Thus the exact generic threshold is $m = 28$.*

*Proof.* First consider the unbounded conic branch A1(b). The set $\mathcal{X} = \{x \in \mathbb{R}^{28} : \|x\|_0 \le 12\}$ is a closed $G$-invariant cone (since $T_g$ is diagonal with $\pm 1$ entries). Its intersection with $\mathbb{S}^{27}$ is the finite union of $\binom{28}{12}$ coordinate $11$-spheres, which has upper box-counting dimension $11 = k - 1$, giving $k = 12$.

Fix any $m \le 27$. For each coordinate $i \in \{1, \dots, 28\}$, deleting column $i$ from a generic $A \in \mathbb{R}^{m \times 28}$ leaves a rank-$m$ matrix (since rank deficiency is cut out by vanishing $m \times m$ minors and a full-rank example exists). Hence column $i$ lies in the span of the other columns, which means the $i$-th coordinate functional $v \mapsto v_i$ does not vanish identically on $\ker A$. Because a nonzero vector space cannot be covered by finitely many proper hyperplanes, $\ker A$ contains a vector $v$ with all $28$ coordinates nonzero for almost every $A \in \mathbb{R}^{m \times 28}$.

Given such a full-support $v \in \ker A$, define for each $g \in G$ the witness
$$
y_g := (T_g - \epsilon_g I) v.
$$
By Lemma 1, $\|y_g\|_0 \le 12$ ($12$ for $g \neq 0$, $0$ for $g = 0$), so $y_g \in \mathcal{X}$. Since $Av = 0$,
$$
A T_g v = A(T_g - \epsilon_g I) v + \epsilon_g A v = A y_g \in A\mathcal{X}.
$$
Now define the competitor model
$$
\mathcal{X}'_A := \mathcal{X} \cup \bigcup_{g \in G} \operatorname{span}_{\mathbb{R}}\{T_g v\}.
$$
Because $v$ has full support ($28 > 12$), $v \notin \mathcal{X}$, so $\mathcal{X}'_A \neq \mathcal{X}$. The added lines are permuted by $G$, so $\mathcal{X}'_A$ is a $G$-invariant cone whose spherical section adds at most $2|G| = 128$ isolated points to $\mathcal{X} \cap \mathbb{S}^{27}$, preserving upper box-counting dimension $k - 1 = 11$. For every $t \in \mathbb{R}$ and $g \in G$, $A(t T_g v) = A(t y_g) \in A\mathcal{X}$, while $\mathcal{X} \subset \mathcal{X}'_A$ gives the reverse inclusion. Thus $A T_h \mathcal{X}'_A = A\mathcal{X}'_A = A\mathcal{X} = A T_h \mathcal{X}$ for all $h \in G$. (Testing whether $A$ uniquely identifies the fixed model $\mathcal{X}$ naturally permits the indistinguishable competitor $\mathcal{X}'_A$ to depend on $A$.)

For the bounded branch A1(a), take $\mathcal{X}_B := \mathcal{X} \cap B_1(0)$, scale $v \in \ker A$ so that $\|v\|_2 \le 1/2$, and set $\mathcal{X}'_{B,A} := \mathcal{X}_B \cup \{T_g v : g \in G\}$. Every witness satisfies $\|y_g\|_2 = \|(T_g - \epsilon_g I)v\|_2 \le 2\|v\|_2 \le 1$, so $y_g \in \mathcal{X}_B$ and $A T_g v = A y_g \in A\mathcal{X}_B$. Adding finitely many points preserves upper box-counting dimension $12$, so both models are distinct bounded $G$-invariant A1(a) models with identical measurements.

Finally, for $m \ge 28$, almost every $A \in \mathbb{R}^{m \times 28}$ has full column rank $28$ and therefore admits a left inverse, recovering any set $\mathcal{X}$ directly from $A\mathcal{X}$. Hence the exact generic identification threshold is $m = 28$, contradicting Conjecture 10 at $m = 27$. $\square$

**Corollary 2 (Unbounded gap at $C = 1$).** *For every integer $r \ge 3$, consider $G_r := \mathbb{F}_2^r \times \mathbb{F}_2^r$ acting diagonally on $\mathbb{R}^{n_r}$ indexed by $D_r := \{(a,b) \in G_r : a \cdot b = 1\}$, where $n_r = (2^r - 1)2^{r-1}$. Then $C = 1$, $\operatorname{tr} T_{(u,w)} = -2^{r-1}(-1)^{u\cdot w}$ for $(u,w) \neq (0,0)$, and for sparsity $k_r := (n_r - 2^{r-1})/2$ the exact generic identification threshold is $n_r$, exceeding the first integer $2k_r + C + 2 = n_r - 2^{r-1} + 3$ allowed by Conjecture 10 by $2^{r-1} - 3$.*

*Proof.* Counting pairs $(a,b)$ with $a \cdot b = 1$ gives $(2^r - 1)$ nonzero $a$'s and $2^{r-1}$ partners $b$ for each, so $n_r = (2^r - 1)2^{r-1}$. The same character sum yields $\operatorname{tr} T_{(u,w)} = -\frac{1}{2} \cdot 2^r (-1)^{u\cdot w} = -2^{r-1}(-1)^{u\cdot w}$ for $(u,w) \neq (0,0)$, so every nonidentity element has minority-sign count $(n_r - 2^{r-1})/2 = k_r$. The proof of Theorem 1 applies verbatim, giving generic non-identification for all $m \le n_r - 1$ and exact threshold $n_r$. Meanwhile $2k_r + C + 2 = (n_r - 2^{r-1}) + 3$, leaving a gap of $2^{r-1} - 3$. $\square$

---

## 4. Comparison Subspace Models

To illustrate how the algebra spanned by a group representation controls generic identification beyond $C = \max_j c_j/s_j$, let $W$ be a $d$-dimensional irreducible real representation of a compact group $\mathcal{G}$, consider the diagonal action on $W^6$ (ambient dimension $6d$), and fix the $\mathcal{G}$-invariant conic model $\mathcal{X} := W \oplus 0^5$ of dimension $k = d$.

**Proposition 3 (Quotient Jacobian thresholds on $W^6$).** *Among $\mathcal{G}$-invariant A1 models of dimension $d$, the exact generic identification threshold for $\mathcal{X} = W \oplus 0^5$ is:*
- *$m = 7$ for $SO(2)$ acting on $\mathbb{R}^2$ ($d = 2$, $6d = 12$);*
- *$m = 5$ for $O(2)$ on $\mathbb{R}^2$ ($d = 2$, $6d = 12$), $SO(3)$ on $\mathbb{R}^3$ ($d = 3$, $6d = 18$), and $S_4$ on its standard sum-zero representation ($d = 3$, $6d = 18$).*

*Proof.* If $m < d$, generically $A\mathcal{X} = \mathbb{R}^m$, so adjoining any hidden copy of $W$ preserves $A\mathcal{X}$ and dimension $d$. Assume $m \ge d$. Generically $A|_{\mathcal{X}}$ is injective; quotienting $\mathbb{R}^m$ by $A\mathcal{X}$ yields $q := m - d$ residual linear rows $B \in \mathbb{R}^{q \times 5d}$ on the five hidden copies $W^5$. Conditional on any full-rank visible block, the map from the hidden columns of $A$ to $B$ is a linear surjection in fixed quotient coordinates, so generic rank properties of $B$ transfer directly.

A point $(x_0, x_{\text{hid}}) \in W \oplus W^5$ satisfies $A T_g(x_0, x_{\text{hid}}) \in A\mathcal{X}$ for all $g \in \mathcal{G}$ if and only if $x_{\text{hid}}$ lies in the common kernel $K := \bigcap_{g \in \mathcal{G}} \ker\bigl(B (I_5 \otimes T_g|_W)\bigr)$. Thus the largest $\mathcal{G}$-invariant set with image contained in $A\mathcal{X}$ is $\mathcal{X} \oplus K$. If $K = \{0\}$, every competitor lies in $\mathcal{X}$ and injectivity of $A|_{\mathcal{X}}$ forces equality. If $K \neq \{0\}$, $K$ is a nonzero $\mathcal{G}$-invariant subspace of $W^5$, so by complete reducibility it contains an isomorphic copy $W' \subset K$ of the irreducible $W$; adjoining $W'$ to $\mathcal{X}$ by union preserves $\mathcal{G}$-invariance, dimension $d$, and the image $A\mathcal{X}$. Hence generic identification holds if and only if the stacked quotient Jacobian $\{B (I_5 \otimes T_g|_W)\}_{g \in \mathcal{G}}$ has full rank $5d$.

- For $SO(2)$ on $W = \mathbb{R}^2$, $\operatorname{span}_{\mathbb{R}}\{T_g|_W\} = \operatorname{span}_{\mathbb{R}}\{I_2, J\}$ (where $J$ is the quarter-turn). Identifying $W^5 \simeq \mathbb{C}^5$, each row of $B$ together with its $J$-rotate spans a complex line in $(\mathbb{C}^5)^*$, giving generic real rank $2\min(q, 5)$. Full rank $10$ first occurs at $q = 5$, i.e., $m = d + 5 = 7$.
- For $O(2)$, $SO(3)$, and $S_4$ (sum-zero), the representation matrices span all of $M_d(\mathbb{R})$ (certified on finite subgroups/generating sets in [`explore.py`](explore.py)). Reshaping each row of $B$ into a $d \times 5$ matrix, its translates under $M_d(\mathbb{R})$ span all $d \times 5$ matrices whose rows lie in that probe's row space. The combined row spaces of $q$ generic probes in $\mathbb{R}^5$ have dimension $\min(dq, 5)$, so the stacked Jacobian has generic rank $d\min(dq, 5)$. Full rank $5d$ first occurs at $q = \lceil 5/d \rceil$: $q = 3$ ($m = 5$) for $O(2)$ and $q = 2$ ($m = 5$) for $SO(3)$ and $S_4$. $\square$

---

## 5. Exact Verification

Run from the repository root with Python 3.10+ and SymPy:

```sh
.venv/bin/python -B solutions/unsupervised_group_sensing/explore.py
```

The script performs exact integer/SymPy arithmetic with zero floating-point approximation:
- Verifies that the $28$ character columns of $G = (\mathbb{Z}/2)^6$ on $D$ satisfy $M^\top M = 64 I_{28}$ and enumerates all $64$ traces: $\{28: 1,\; 4: 28,\; -4: 35\}$.
- Constructs an explicit rank-$27$ integer sensor $A \in \mathbb{Z}^{27 \times 28}$ with full-support kernel vector $v = (1, 2, \dots, 27, 1)^\top$, checks all $64$ sparse witnesses $y_g = (T_g - \epsilon_g I)v$ (exact sparsity $12$ for $g \neq 0$, $0$ for $g = 0$) and $A T_g v = A y_g$, checks the consistency-residual Jacobian identity $A(T_g - (T_g - \epsilon_g I)) = \epsilon_g A$ (rank $27$), and confirms that appending one independent row yields rank $28$.
- Certifies the action-algebra span dimensions ($2$ for $SO(2)$, $d^2$ for $O(2)$, $SO(3)$, and $S_4$) using exact finite test sets (signed permutations for $O(2)$, determinant-$1$ signed permutations for $SO(3)$, all $24$ permutations on the sum-zero subspace for $S_4$) and verifies the exact quotient Jacobian ranks:

| Group and representation | Ambient dimension | Quotient Jacobian ranks ($q = 0, 1, \dots$) | First identifying $m$ |
|---|---:|---|---:|
| $SO(2)$, plane ($d=2$) | 12 | 0, 2, 4, 6, 8, 10, 10 | 7 |
| $O(2)$, plane ($d=2$) | 12 | 0, 4, 8, 10, 10 | 5 |
| $SO(3)$, standard ($d=3$) | 18 | 0, 9, 15, 15 | 5 |
| $S_4$, sum-zero ($d=3$) | 18 | 0, 9, 15, 15 | 5 |
