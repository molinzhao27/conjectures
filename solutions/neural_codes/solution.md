# Refutation of the Boolean Rank Redundancy Conjecture and Resolution of Question 2 in *Adjoints of Morphisms of Neural Codes* (arXiv:2603.10837)

**Target paper:** Juliann Geraci, Alexander B. Kunin, and Alexandra Seceleanu, *Adjoints of Morphisms of Neural Codes*, [arXiv:2603.10837](https://arxiv.org/abs/2603.10837) (March 10, 2026).

---

## Executive Summary

> **Novelty Status Note:** As of Nov 26, 2026, arXiv:2603.10837 remains v1 with no published resolution found on arXiv/Semantic Scholar. Note however that Juliann Geraci's UNL PhD dissertation (*Boolean Rank via Monomial Ideals and Neural Ideals*, April 2026, [digitalcommons.unl.edu/dissunl/474](https://digitalcommons.unl.edu/dissunl/474/)) could not be fetched automatically (HTTP 403) and should be checked manually before asserting final publication priority.

We settle the open **Conjecture in Section 7.3** (lines 1118–1120) and **Question 2 in Section 7.2** (lines 1105–1108) of arXiv:2603.10837:

1. **The Conjecture in Section 7.3 is FALSE in BOTH directions:**
   - **Strict inequality $\operatorname{brank}(\mathbf{C}) < \operatorname{brank}(\rho(\mathbf{C})) + \ell$ (main structural refutation, valid under every reading):** Even for binary matrices with *no* trivial columns, *no* duplicate columns, *no* all-$1$s columns (so $k = \ell$ unambiguously), distinct rows, and $\ell = 1$, the conjecture fails.
     - The **provably smallest** such counterexample is a $3 \times 4$ matrix (and $4 \times 4$ if the empty codeword $\varnothing$ is required), with $k = \ell = 1$ and $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) = 3 < 4$.
     - We also construct a $5 \times 4$ counterexample with $\varnothing \in \mathcal{C}$ and $m_{\mathrm{nz}} = n = 4 > 3 = \operatorname{brank}(\mathbf{C})$, as well as an explicit infinite family $\mathbf{C}_m$ ($m \geq 2$) where $\operatorname{brank}(\mathbf{C}_m) = \operatorname{brank}(\rho(\mathbf{C}_m)) = m + 1$ while $k = \ell = 2^m - m - 1$, showing the gap $(\operatorname{brank}(\rho(\mathbf{C})) + \ell) - \operatorname{brank}(\mathbf{C})$ grows **exponentially** in $\operatorname{brank}(\rho(\mathbf{C}))$ and $\lambda(\mathcal{C})$.
     - A second family based on $J_n - I_n$ ($n \geq 3$) achieves $\operatorname{brank}(\rho(\mathbf{C})) = \Theta(\log n) < \operatorname{brank}(\mathbf{C}) = n \ll \operatorname{brank}(\rho(\mathbf{C})) + \ell = 2^n - n + O(\log n)$.
   - **Strict inequality $\operatorname{brank}(\mathbf{C}) > \operatorname{brank}(\rho(\mathbf{C})) + \ell$ (refuting even the conjectured upper bound):**
     - **All-$1$s columns (refuting the claim on line 1112 unconditionally whenever $\varnothing \in \mathcal{C}$):** Adding an all-$1$s column *increases* $\operatorname{brank}$ by $1$ if and only if the matrix has an all-zero row ($\varnothing \in \mathcal{C}$). For $\mathbf{C} = \bigl[\begin{smallmatrix} 0 & 1 \\ 1 & 1 \end{smallmatrix}\bigr]$, $k = 1$, $\ell = 0$, yet $\operatorname{brank}(\mathbf{C}) = 2 > 1 = \operatorname{brank}(\rho(\mathbf{C})) + \ell$.
     - **Duplicated non-simple redundancies (under simultaneous redundancy counting):** Under the literal simultaneous definition of line 1114, duplicating a non-simply redundant column makes *both* copies redundant to a singleton (each other), hence simply redundant ($k = 2, \ell = 0$) while $\operatorname{brank} = 3 > 2$. *(Note: under sequential deduplication-first counting, one copy is removed first and the remaining copy counts as $\ell = 1$, restoring equality here; thus the unconditional reverse-direction refutation rests on $\mathbf{C}_{\mathrm{all1}}$ above.)*
2. **Corrected Sharp Bounds & Positive Theorem:**
   - Defining the deduplicated effective redundancy count $\ell_{\mathrm{eff}}(\mathbf{C})$ properly restores the sharp two-sided bound
     $$\operatorname{brank}(\rho(\mathbf{C})) \;\leq\; \operatorname{brank}(\mathbf{C}) \;\leq\; \operatorname{brank}(\rho(\mathbf{C})) + \ell_{\mathrm{eff}}(\mathbf{C}).$$
   - We characterize exactly when adding a single non-simply redundant column leaves $\operatorname{brank}$ unchanged.
   - **Positive Theorem:** For every **intersection-complete** code $\mathcal{C}$ represented by $\mathbf{C}$, $\operatorname{brank}(\mathbf{C})$ equals $u(\mathbf{C})$, the number of distinct nonzero columns of $\mathbf{C}$. Consequently, $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) + \ell_{\mathrm{eff}}(\mathbf{C})$ holds with **exact equality** for all intersection-complete codes — in particular, the authors' conjecture **is true** for intersection-complete codes without trivial or simply redundant neurons. (This also sharpens Theorem `thm:intcompletefullrank` of the paper.)
3. **Complete Resolution of Question 2 (Section 7.2):**
   - While the paper's empirical figure (lines 1104–1108) already observed attainment at $\lambda = 3$ ($\lambda(\mathcal{C}^{(i)}) = 4$), we prove that the upper bound $\lambda(\mathcal{C}^{(i)}) \leq 2\lambda(\mathcal{C}) - 2$ is **attained for every integer $n = \lambda(\mathcal{C}) \geq 1$**.
   - In particular, for every $n \geq 3$, the reduced code $\mathcal{C}_n = \{\varnothing\} \cup \{\{j\} : 2 \leq j \leq n\} \cup \{\{1, j\} : 2 \leq j \leq n\}$ (as well as its intersection completion $\widehat{\mathcal{C}}_n = \mathcal{C}_n \cup \{\{1\}\}$, which works for all $n \geq 1$) has $\lambda(\mathcal{C}_n) = n$ and $\lambda(\mathcal{C}_n^{(1)}) = 2n - 2$. Moreover, for $\mathcal{C}_n$, neuron $1$ is free, so $f_1: \mathcal{C}_n \to \mathcal{C}_n^{(1)}$ is a BMF covering map.
   - Consequently, $\sup_{\mathcal{C} \in \mathbf{Code}} \{\lambda(\mathcal{C}^{(i)}) - \lambda(\mathcal{C})\} = +\infty$ (and assuming the paper's bound $\lambda(\mathcal{C}^{(i)}) \leq 2\lambda(\mathcal{C}) - 2$ for fixed $\lambda(\mathcal{C}) = n$, the maximum is $n - 2$ for every $n \geq 1$).

---

## 1. Exact Definitions from arXiv:2603.10837

We record the exact definitions from `BMFandNeuralRings.tex` (arXiv:2603.10837) to ensure zero ambiguity:

- **Combinatorial code and matrix representation** (Section 2, lines 189–200): A combinatorial code is a collection $\mathcal{C} \subseteq 2^{[n]}$ of subsets (codewords) of $[n] := \{1, \dots, n\}$ (neurons). Writing each codeword $\sigma_i \in \mathcal{C}$ as a row gives a binary matrix $\mathbf{C} = [c_{ij}] \in \{0,1\}^{m \times n}$ with $c_{ij} = 1 \iff j \in \sigma_i$.
- **Trunks and roots** (Section 2.2, lines 251–255; Section 3.1, Definition `def:root`, lines 414–420): For $\tau \subseteq [n]$, the trunk is $\operatorname{Tk}_{\mathcal{C}}(\tau) = \{\sigma \in \mathcal{C} : \sigma \supseteq \tau\}$ (with $\operatorname{Tk}_{\mathcal{C}}(\varnothing) = \mathcal{C}$). For a nonempty trunk $T$, its root is $\sqrt{T} = \bigcap_{\sigma \in T} \sigma$, and $\sqrt[\mathcal{C}]{\tau} := \sqrt{\operatorname{Tk}_{\mathcal{C}}(\tau)}$.
- **Trivial, redundant, and reduced codes** (Section 2.3, lines 280–306; Section 6, Lemma `lem:redundancylemma`, lines 1015–1018):
  - Neuron $i \in [n]$ is *trivial* if $\operatorname{Tk}_{\mathcal{C}}(i) = \varnothing$ (column $i$ of $\mathbf{C}$ is all $0$s).
  - Neuron $i \in [n]$ is *redundant to $\sigma$* if $\sigma \subseteq [n] \setminus \{i\}$ and $\operatorname{Tk}_{\mathcal{C}}(i) = \operatorname{Tk}_{\mathcal{C}}(\sigma)$ (equivalently, column $i$ is the entrywise product $\bigwedge_{j \in \sigma} \mathbf{c}_j$, with the empty product for $\sigma = \varnothing$ equal to the all-$1$s vector $\mathbf{1}$). Neuron $i$ is *redundant* if such a $\sigma$ exists. By the Redundancy Lemma (`lem:redundancylemma`), a non-trivial neuron $i$ is redundant iff $\operatorname{Tk}_{\mathcal{C}}(i) = \operatorname{Tk}_{\mathcal{C}}(\sqrt[\mathcal{C}]{i} \setminus \{i\})$.
  - A code $\mathcal{C}$ is *reduced* if it has no trivial or redundant neurons. Deleting all trivial neurons and then deleting redundant neurons one at a time yields a reduced representative $\rho(\mathcal{C})$ (represented by $\rho(\mathbf{C})$), unique up to column permutation (Remark `rmk:reductionmorphism`).
  - The *minimum neuron number* $\lambda(\mathcal{C})$ (Section 7.2, line 1102) is the smallest $n$ such that $\mathcal{C}$ is isomorphic to a code in $2^{[n]}$; by Remark `rmk:reductionmorphism`, $\lambda(\mathcal{C})$ equals the number of columns of $\rho(\mathbf{C})$.
- **Boolean rank** (Section 6, lines 981–982): For $\mathbf{C} \in \mathbb{B}^{m \times n}$ over the Boolean semiring $\mathbb{B} = (\{0,1\}, \vee, \wedge)$, $\operatorname{brank}(\mathbf{C})$ is the minimum $r \geq 0$ for which there exist $\mathbf{V} \in \mathbb{B}^{m \times r}$ and $\mathbf{H} \in \mathbb{B}^{r \times n}$ with $\mathbf{C} = \mathbf{V}\mathbf{H}$. Equivalently, $\operatorname{brank}(\mathbf{C})$ is the minimum number of all-$1$ submatrices (rectangles $R_t \times C_t$) whose union is $\{(i,j) : C_{ij} = 1\}$.
  - Lemma `lem:reducedminimumrank` (line 1002): $\operatorname{brank}(\rho(\mathbf{C})) \leq \operatorname{brank}(\mathbf{C})$.
  - Lemma `lem:redundancybound` (line 1028): If $\mathbf{C}$ has $k$ redundant neurons (and no trivial neurons), $\operatorname{brank}(\mathbf{C}) \leq \operatorname{brank}(\rho(\mathbf{C})) + k$.
- **Simple vs. non-simple redundancy and the Conjecture** (Section 7.3, lines 1114–1120):
  - A neuron $i$ is *simply redundant* if it is redundant to a singleton set $\{j\}$ ($j \neq i$) or the empty set $\varnothing$ (i.e., column $i$ duplicates another column or is all $1$s).
  - A neuron $i$ is *not simply redundant* (a *non-simple redundancy*) if it is redundant and not simply redundant.
  - **Conjecture (Section 7.3):** *Let $\mathbf{C}$ be a binary matrix with $k$ redundant neurons of which $\ell \leq k$ are not simply redundant. Then $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) + \ell$.*

---

## 2. Refutation of the Conjecture in Section 7.3

We call a binary matrix $\mathbf{C}$ **clean** if it has no trivial (all-$0$) columns, no duplicate columns, no all-$1$s columns, and no duplicate rows. For every clean matrix, there are no simply redundant neurons, so $k = \ell$ unambiguously counts the redundant neurons.

### 2.1. Provably Smallest Clean Counterexamples ($\operatorname{brank}(\mathbf{C}) < \operatorname{brank}(\rho(\mathbf{C})) + \ell$)

**Theorem 2.1 (Minimal Clean Counterexample).** *The $3 \times 4$ clean binary matrix*
$$\mathbf{C} = \begin{bmatrix} 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \end{bmatrix} \quad (\mathcal{C} = \{124, 13, 23\} \subseteq 2^{[4]})$$
*has $k = \ell = 1$ non-simply redundant neuron, reduced matrix $\rho(\mathbf{C}) = J_3 - I_3$, and satisfies*
$$\operatorname{brank}(\mathbf{C}) = 3 = \operatorname{brank}(\rho(\mathbf{C})) < 4 = \operatorname{brank}(\rho(\mathbf{C})) + \ell.$$
*Moreover, $\mathbf{C}$ is minimal in both dimensions: every clean counterexample has at least $3$ rows and at least $4$ columns. Appending the empty codeword $(0,0,0,0)$ yields a $4 \times 4$ clean counterexample $\mathbf{C}_0$ with $\varnothing \in \mathcal{C}_0$, which is minimal among clean counterexamples containing $\varnothing$.*

*Proof.*
1. **Redundancy structure.** The columns of $\mathbf{C}$ are $\mathbf{c}_1 = (1,1,0)^\top$, $\mathbf{c}_2 = (1,0,1)^\top$, $\mathbf{c}_3 = (0,1,1)^\top$, $\mathbf{c}_4 = (1,0,0)^\top$, which are pairwise distinct, nonzero, and non-all-$1$s. Thus $\mathbf{C}$ is clean (no trivial or simply redundant neurons). Computing roots:
   - $\operatorname{Tk}_{\mathcal{C}}(1) = \{124, 13\}$, $\sqrt[\mathcal{C}]{1} = \{1\}$, $\operatorname{Tk}_{\mathcal{C}}(\varnothing) = \mathcal{C} \neq \operatorname{Tk}_{\mathcal{C}}(1)$;
   - $\operatorname{Tk}_{\mathcal{C}}(2) = \{124, 23\}$, $\sqrt[\mathcal{C}]{2} = \{2\}$, $\operatorname{Tk}_{\mathcal{C}}(\varnothing) = \mathcal{C} \neq \operatorname{Tk}_{\mathcal{C}}(2)$;
   - $\operatorname{Tk}_{\mathcal{C}}(3) = \{13, 23\}$, $\sqrt[\mathcal{C}]{3} = \{3\}$, $\operatorname{Tk}_{\mathcal{C}}(\varnothing) = \mathcal{C} \neq \operatorname{Tk}_{\mathcal{C}}(3)$;
   - $\operatorname{Tk}_{\mathcal{C}}(4) = \{124\}$, $\sqrt[\mathcal{C}]{4} = \{1,2,4\}$, $\operatorname{Tk}_{\mathcal{C}}(12) = \{124\} = \operatorname{Tk}_{\mathcal{C}}(4)$.

   By the Redundancy Lemma (`lem:redundancylemma`), neuron $4$ is the unique redundant neuron, and it is not simply redundant. Hence $k = \ell = 1$ and $\rho(\mathbf{C}) = [\mathbf{c}_1\ \mathbf{c}_2\ \mathbf{c}_3] = J_3 - I_3$.

2. **Exact value $\operatorname{brank}(\rho(\mathbf{C})) = 3$.** Since $\rho(\mathbf{C})$ has $3$ columns, $\operatorname{brank}(\rho(\mathbf{C})) \leq 3$. Suppose for contradiction that $\rho(\mathbf{C}) = \mathbf{V}\mathbf{H}$ with $\mathbf{H} \in \mathbb{B}^{2 \times 3}$ having rows $\mathbf{h}_1, \mathbf{h}_2$. Then every row of $\rho(\mathbf{C})$ lies in $\{\mathbf{0}, \mathbf{h}_1, \mathbf{h}_2, \mathbf{h}_1 \vee \mathbf{h}_2\}$. In this set, $\mathbf{0}$ is $\leq$ every element and $\mathbf{h}_1 \vee \mathbf{h}_2$ is $\geq$ every element, so any antichain has size at most $2$ (at most $\{\mathbf{h}_1, \mathbf{h}_2\}$). Yet the three rows $(1,1,0), (1,0,1), (0,1,1)$ of $\rho(\mathbf{C})$ form an antichain of size $3$, contradiction. Thus $\operatorname{brank}(\rho(\mathbf{C})) = 3$.

3. **Exact value $\operatorname{brank}(\mathbf{C}) = 3$.** By Lemma `lem:reducedminimumrank`, $\operatorname{brank}(\mathbf{C}) \geq \operatorname{brank}(\rho(\mathbf{C})) = 3$. Conversely, $\mathbf{C}$ has only $3$ rows, so $\mathbf{C} = I_3 \mathbf{C}$ is a rank-$3$ Boolean factorization. Hence $\operatorname{brank}(\mathbf{C}) = 3 < 3 + 1 = \operatorname{brank}(\rho(\mathbf{C})) + \ell$.

4. **Empty codeword extension $\mathbf{C}_0$.** Appending row $(0,0,0,0)$ does not alter the trunks of $1,2,3,4$, keeps all columns clean, leaves $k = \ell = 1$, and preserves $\operatorname{brank}(\rho(\mathbf{C}_0)) = \operatorname{brank}(\mathbf{C}_0) = 3$.

5. **Minimality.** Let $\mathbf{D} \in \mathbb{B}^{m \times n}$ be any clean counterexample, with $\rho(\mathbf{D}) \in \mathbb{B}^{m \times n_0}$ and $\ell = n - n_0 \geq 1$.
   - Clearly $n_0 \geq 2$ (a non-simply redundant column is a product of $\geq 2$ columns of $\rho(\mathbf{D})$).
   - Suppose $n_0 = 2$, with columns $\mathbf{b}_1, \mathbf{b}_2$ of $\rho(\mathbf{D})$. The only possible non-simply redundant column is $\mathbf{b}_3 = \mathbf{b}_1 \wedge \mathbf{b}_2$, so $\ell = 1$ and $n = 3$. Since $\mathbf{b}_3 \neq \mathbf{0}$ and $\mathbf{b}_3 \notin \{\mathbf{b}_1, \mathbf{b}_2\}$, there must exist rows $r_{10}, r_{01}, r_{11}$ of $\mathbf{D}$ equal to $(1,0,0), (0,1,0), (1,1,1)$ respectively. This $3 \times 3$ submatrix is lower-triangular with $1$'s on the diagonal, which has Boolean rank $3$ (see Lemma 3.3 below), while $\operatorname{brank}(\rho(\mathbf{D})) = 2$. Hence $\operatorname{brank}(\mathbf{D}) = 3 = \operatorname{brank}(\rho(\mathbf{D})) + 1$, so no clean counterexample exists with $n_0 = 2$.
   - Thus $n_0 \geq 3$ and $n = n_0 + \ell \geq 4$.
   - Since there are only $2^m - 2$ distinct nonzero, non-all-$1$ vectors in $\{0,1\}^m$, having $n \geq 4$ requires $2^m - 2 \geq 4$, i.e. $m \geq 3$. If $\varnothing \in \mathcal{C}$, one row is all-$0$, so the remaining $m - 1$ rows must support $n \geq 4$ distinct nonzero columns (an all-$1$s pattern on the remaining $m-1$ rows is now permitted since the $\varnothing$ row is $0$), requiring $2^{m-1} - 1 \geq 4$, i.e. $m \geq 4$. $\square$

**Remark 2.2 (Counterexamples with $m_{\mathrm{nz}} \geq n$, not caused by row scarcity).** Appending both $(0,0,0,0)$ and $(1,1,1,1)$ to $\mathbf{C}$ yields the $5 \times 4$ clean matrix
$$\mathbf{C}_1 = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 0 & 0 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \end{bmatrix},$$
which has $m_{\mathrm{nz}} = 4$ nonzero rows and $n = 4$ columns, $k = \ell = 1$, $\operatorname{brank}(\rho(\mathbf{C}_1)) = 3$, and $\operatorname{brank}(\mathbf{C}_1) = 3 < \min(m_{\mathrm{nz}}, n) = 4$. Taking the $t$-fold Cartesian product code $\mathcal{C}_1^t \subseteq 2^{[4t]}$ produces clean matrices with $5^t$ distinct codewords (exponentially more rows than columns), $4t$ columns, $k = \ell = t$, and $\operatorname{brank}(\mathcal{C}_1^t) = \operatorname{brank}(\rho(\mathcal{C}_1^t)) = 3t$ (since $\mathbf{H} = \operatorname{diag}(\mathbf{H}_1, \dots, \mathbf{H}_1)$ gives rank $\leq 3t$, while the submatrix on codewords active in a single block is $\operatorname{diag}(J_3 - I_3, \dots, J_3 - I_3)$; any all-$1$ rectangle in a block-diagonal matrix can meet at most one diagonal block, so $\operatorname{brank}$ is additive over diagonal blocks and equals $3t$).

---

### 2.2. Exponential-Gap Infinite Families

The discrepancy between $\operatorname{brank}(\mathbf{C})$ and $\operatorname{brank}(\rho(\mathbf{C})) + \ell$ can be **exponentially large**, even with $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C}))$.

**Theorem 2.3 (Exponential Gap with Zero Rank Increase).** *For every integer $m \geq 2$, there exists a clean $(m+1) \times (2^m + m - 1)$ binary matrix $\mathbf{C}_m$ (and a clean $(m+2) \times (2^m + m - 1)$ matrix containing $\varnothing$) such that:*
1. *$\rho(\mathbf{C}_m)$ has $2m$ columns and $\operatorname{brank}(\rho(\mathbf{C}_m)) = m + 1$;*
2. *$\mathbf{C}_m$ has $k = \ell = 2^m - m - 1$ non-simply redundant neurons (and no trivial or simply redundant neurons);*
3. *$\operatorname{brank}(\mathbf{C}_m) = m + 1 = \operatorname{brank}(\rho(\mathbf{C}_m))$, whereas the Conjecture predicts $\operatorname{brank}(\rho(\mathbf{C}_m)) + \ell = 2^m$.*

*Proof.* Define the $(m+1) \times 2m$ binary matrix
$$\mathbf{B}_m := \begin{bmatrix} I_m & J_m - I_m \\ \mathbf{0}_{1 \times m} & \mathbf{1}_{1 \times m} \end{bmatrix},$$
with rows indexed by $[m+1]$ and columns $\mathbf{b}_1, \dots, \mathbf{b}_{2m}$. Their column supports are $\operatorname{supp}(\mathbf{b}_a) = \{a\}$ for $a \in [m]$ and $\operatorname{supp}(\mathbf{b}_{m+b}) = ([m] \setminus \{b\}) \cup \{m+1\}$ for $b \in [m]$.

Define $\mathbf{C}_m$ by appending to $\mathbf{B}_m$, for every subset $S \subseteq [m]$ with $|S| \geq 2$, the column
$$\mathbf{c}_S := \bigwedge_{b \in S} \mathbf{b}_{m+b}, \qquad \operatorname{supp}(\mathbf{c}_S) = ([m] \setminus S) \cup \{m+1\}.$$
There are $\sum_{j=2}^m \binom{m}{j} = 2^m - m - 1$ appended columns.

- **Cleanliness:** Every $\mathbf{c}_S$ contains $m+1$ in its support (hence is nonzero and distinct from every $\mathbf{b}_a$, $a \in [m]$) and omits $|S| \geq 2$ elements of $[m]$ (hence is not $\mathbf{1}$ and is distinct from every $\mathbf{b}_{m+b}$, which omits only $1$ element of $[m]$). Distinct subsets $S_1 \neq S_2$ yield distinct supports $([m] \setminus S_1) \cup \{m+1\} \neq ([m] \setminus S_2) \cup \{m+1\}$. Hence all $2^m + m - 1$ columns of $\mathbf{C}_m$ are pairwise distinct, nonzero, and non-all-$1$s.
- **Exact redundancy classification:** Each appended column $\mathbf{c}_S$ is redundant to $\{m+b : b \in S\}$ and (by cleanliness) is non-simply redundant. Conversely, no column of $\mathbf{B}_m$ is redundant in $\mathbf{C}_m$:
  - For $a \in [m]$, any intersection of columns in $\mathbf{C}_m \setminus \{\mathbf{b}_a\}$ that includes some $\mathbf{b}_{a'}$ ($a' \in [m] \setminus \{a\}$) omits $a$; any intersection using only columns from $\{\mathbf{b}_{m+1}, \dots, \mathbf{b}_{2m}\} \cup \{\mathbf{c}_S\}$ (or the empty intersection) contains $m+1 \notin \{a\}$.
  - For $b \in [m]$, any intersection involving some $\mathbf{b}_a$ ($a \in [m]$) omits $m+1 \in \operatorname{supp}(\mathbf{b}_{m+b})$; any intersection involving some $\mathbf{c}_S$ omits at least $|S| \geq 2$ elements of $[m]$ whereas $\mathbf{b}_{m+b}$ omits only $b$; and any intersection of a subset of $\{\mathbf{b}_{m+b'} : b' \neq b\}$ (including the empty intersection) contains $b \notin \operatorname{supp}(\mathbf{b}_{m+b})$.
  
  Therefore $\rho(\mathbf{C}_m) = \mathbf{B}_m$ and $k = \ell = 2^m - m - 1$.
- **Exact Boolean ranks:** Clearly $\operatorname{brank}(\mathbf{B}_m) \leq \operatorname{brank}(\mathbf{C}_m) \leq m + 1$ (since $\mathbf{C}_m$ has $m + 1$ rows). For the lower bound $\operatorname{brank}(\mathbf{B}_m) \geq m + 1$, let $\mathbf{B}_m = \mathbf{V}\mathbf{H}$ be any rank-$r$ factorization, corresponding to all-$1$ rectangles $R_1 \times C_1, \dots, R_r \times C_r$. For each $a \in [m]$, the entry $(a, a) = 1$ must lie in some rectangle $R_{t_a} \times C_{t_a}$. Since column $a$ of $\mathbf{B}_m$ has its sole $1$ in row $a$, we must have $R_{t_a} = \{a\}$. Thus $t_1, \dots, t_m$ are $m$ distinct indices, and none covers any $1$-entry in row $m+1$. Since row $m+1$ is nonzero, at least one further rectangle is required, so $r \geq m + 1$. Thus $\operatorname{brank}(\mathbf{B}_m) = \operatorname{brank}(\mathbf{C}_m) = m + 1$.

Appending the all-$0$ row preserves cleanliness, redundancy, and Boolean rank. $\square$

**Proposition 2.4 (Strict Intermediate Rank with Exponential Gap).** *For $n \geq 3$, let $\mathbf{B} = J_n - I_n \in \mathbb{B}^{n \times n}$ and let $\mathbf{C} \in \mathbb{B}^{n \times (2^n - 2)}$ be formed by appending all $\sum_{j=2}^{n-1} \binom{n}{j} = 2^n - n - 2$ proper intersections $\mathbf{c}_S = \bigwedge_{i \in S} \mathbf{b}_i$ ($2 \leq |S| \leq n-1$). Then $\mathbf{C}$ is clean, $\rho(\mathbf{C}) = J_n - I_n$, $k = \ell = 2^n - n - 2$,*
$$\operatorname{brank}(\rho(\mathbf{C})) = \sigma(n) := \min\left\{r : \binom{r}{\lfloor r/2 \rfloor} \geq n\right\} = \Theta(\log n), \qquad \operatorname{brank}(\mathbf{C}) = n.$$
*In particular, for $n \geq 5$ (e.g., $n = 5$: $4 < 5 < 4 + 25 = 29$), we have the strict double inequality $\operatorname{brank}(\rho(\mathbf{C})) < \operatorname{brank}(\mathbf{C}) < \operatorname{brank}(\rho(\mathbf{C})) + \ell$.*

*Proof.* Cleanliness and $\rho(\mathbf{C}) = J_n - I_n$ follow exactly as above (column $i$ of $J_n - I_n$ omits only $i$, while every $\mathbf{c}_S$ omits $|S| \in [2, n-1]$ entries). Since the columns with $|S| = n - 1$ form the $n \times n$ identity matrix $I_n$, we have $n = \operatorname{brank}(I_n) \leq \operatorname{brank}(\mathbf{C}) \leq n$. Finally, $\operatorname{brank}(J_n - I_n) = \sigma(n)$ is a classical consequence of Sperner's theorem: in any rank-$r$ factorization $J_n - I_n = \mathbf{V}\mathbf{H}$ with row supports $S_i = \operatorname{supp}(\mathbf{V}_{i,\cdot})$ and column supports $T_j = \operatorname{supp}(\mathbf{H}_{\cdot,j})$ in $[r]$, we have $S_i \cap T_i = \varnothing$ and $S_i \cap T_j \neq \varnothing$ for $i \neq j$, forcing $S_i \not\subseteq S_j$ for all $i \neq j$; hence $\{S_1, \dots, S_n\}$ is an antichain in $2^{[r]}$, so $n \leq \binom{r}{\lfloor r/2 \rfloor}$ and $r \geq \sigma(n)$. Conversely, taking $n$ subsets $S_1, \dots, S_n \subseteq [\sigma(n)]$ of size $\lfloor \sigma(n)/2 \rfloor$ and $T_j = [\sigma(n)] \setminus S_j$ realizes $J_n - I_n = \mathbf{V}\mathbf{H}$ of rank $\sigma(n)$. $\square$

---

### 2.3. Failure in the Reverse Direction ($\operatorname{brank}(\mathbf{C}) > \operatorname{brank}(\rho(\mathbf{C})) + \ell$)

Under the exact definitions in Section 7.3 (lines 1112–1120), even the upper bound $\operatorname{brank}(\mathbf{C}) \leq \operatorname{brank}(\rho(\mathbf{C})) + \ell$ fails in two distinct ways.

#### Failure Mode A: All-$1$s columns (and refutation of line 1112)

Section 7.3 (line 1112) states: *"Adding a repeated column or a column of all $1$s likewise does not change the Boolean rank, but it does introduce a redundant neuron to the code."* This statement is **false** for a column of all $1$s whenever the matrix has an all-zero row.

**Proposition 2.5 (Exact Effect of an All-$1$s Column).** *For any binary matrix $\mathbf{M} \in \mathbb{B}^{m \times n}$,*
$$\operatorname{brank}([\mathbf{M} \mid \mathbf{1}]) = \operatorname{brank}(\mathbf{M}) + \mathbf{1}[\mathbf{0} \text{ is a row of } \mathbf{M}].$$

*Proof.*
- If $\mathbf{M}$ has no all-zero row, let $\mathbf{M} = \mathbf{V}\mathbf{H}$ with $r = \operatorname{brank}(\mathbf{M})$. Every row of $\mathbf{V}$ must have at least one $1$ (otherwise the corresponding row of $\mathbf{M}$ would be $\mathbf{0}$). Thus $\mathbf{V}[\mathbf{H} \mid \mathbf{1}_{r \times 1}] = [\mathbf{M} \mid \mathbf{1}]$, so $\operatorname{brank}([\mathbf{M} \mid \mathbf{1}]) \leq r$ (and equality holds since $\mathbf{M}$ is a submatrix).
- If $\mathbf{M}$ has an all-zero row $i_0$, then row $i_0$ of $[\mathbf{M} \mid \mathbf{1}]$ is $\mathbf{e}_{n+1}^\top = (0, \dots, 0, 1)$. In any rank-$r'$ factorization $[\mathbf{M} \mid \mathbf{1}] = \mathbf{V}'\mathbf{H}'$, some row $\mathbf{h}'_t$ of $\mathbf{H}'$ with $V'_{i_0, t} = 1$ must be nonzero and satisfy $\mathbf{h}'_t \leq \mathbf{e}_{n+1}^\top$, forcing $\mathbf{h}'_t = \mathbf{e}_{n+1}^\top$. Removing factor $t$ and column $n+1$ gives a rank-$(r'-1)$ factorization of $\mathbf{M}$, so $r' \geq \operatorname{brank}(\mathbf{M}) + 1$. Conversely, appending factor $\mathbf{1}_{m \times 1} \mathbf{e}_{n+1}^\top$ gives $r' \leq \operatorname{brank}(\mathbf{M}) + 1$. $\square$

**Counterexample 2.6.** Take
$$\mathbf{C} = \begin{bmatrix} 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad \rho(\mathbf{C}) = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.$$
Neuron $2$ is redundant to $\varnothing$, hence *simply redundant*, so $k = 1$ and $\ell = 0$. Yet $\operatorname{brank}(\rho(\mathbf{C})) = 1$ while $\operatorname{brank}(\mathbf{C}) = 2 > 1 + 0$.

#### Failure Mode B: Duplicated non-simply redundant columns (no all-$1$s column, $\varnothing \in \mathcal{C}$)

**Counterexample 2.7.** Duplicate column $3$ of $\mathbf{C}'$ in Example `ex:isomorphicnotbrank`:
$$\mathbf{C} = \begin{bmatrix} 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 \end{bmatrix}, \qquad \rho(\mathbf{C}) = \begin{bmatrix} 0 & 0 \\ 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}.$$
In $\mathbf{C}$, there are no trivial columns and no all-$1$s columns, and $\varnothing \in \mathcal{C}$. Columns $3$ and $4$ are identical: $\operatorname{Tk}_{\mathcal{C}}(3) = \operatorname{Tk}_{\mathcal{C}}(4) = \{1234\}$. Thus neuron $3$ is redundant to the singleton $\{4\}$, and neuron $4$ is redundant to the singleton $\{3\}$ — by definition (line 1114), **both** are *simply redundant*. Hence $k = 2$ and $\ell = 0$. Yet deleting a duplicate column preserves Boolean rank, so
$$\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\mathbf{C}') = 3 > 2 = \operatorname{brank}(\rho(\mathbf{C})) + 0 = \operatorname{brank}(\rho(\mathbf{C})) + \ell.$$
Moreover, taking the $t$-fold block-diagonal sum $\operatorname{diag}(\mathbf{C}, \dots, \mathbf{C})$ gives a matrix with no trivial or all-$1$s columns, $\varnothing \in \mathcal{C}$, $k = 2t$, $\ell = 0$, and $\operatorname{brank}(\mathbf{C}^{\oplus t}) - (\operatorname{brank}(\rho(\mathbf{C}^{\oplus t})) + \ell) = 3t - 2t = t \to \infty$.

---

## 3. Corrected Sharp Bounds, Exact Criterion, and Positive Theorem

### 3.1. Corrected Redundancy Upper Bound

Let $\mathbf{C} \in \mathbb{B}^{m \times n}$ be any binary matrix. Let $s(\mathbf{C})$ be the number of **distinct** nonzero, non-all-$1$ columns of $\mathbf{C}$ that do not appear as columns of $\rho(\mathbf{C})$, and define the **effective redundancy count**
$$\ell_{\mathrm{eff}}(\mathbf{C}) := s(\mathbf{C}) + \mathbf{1}\bigl[\mathbf{1} \text{ is a column of } \mathbf{C} \text{ and } \mathbf{0} \text{ is a row of } \rho(\mathbf{C})\bigr].$$

**Theorem 3.1 (Corrected Redundancy Bound).** *For every binary matrix $\mathbf{C}$,*
$$\operatorname{brank}(\rho(\mathbf{C})) \;\leq\; \operatorname{brank}(\mathbf{C}) \;\leq\; \operatorname{brank}(\rho(\mathbf{C})) + \ell_{\mathrm{eff}}(\mathbf{C}).$$

*Proof.* Deleting all-$0$ columns and deduplicating repeated columns yields a matrix $\tilde{\mathbf{C}}$ with $\operatorname{brank}(\tilde{\mathbf{C}}) = \operatorname{brank}(\mathbf{C})$ and $\rho(\tilde{\mathbf{C}}) = \rho(\mathbf{C})$. Let $\tilde{\mathbf{C}}_{\neq \mathbf{1}}$ be $\tilde{\mathbf{C}}$ with any all-$1$s column removed. Notice that every column of $\tilde{\mathbf{C}}_{\neq \mathbf{1}}$ not in $\rho(\mathbf{C})$ is a nonempty entrywise product of columns of $\rho(\mathbf{C})$, so it has $0$ in every row where $\rho(\mathbf{C})$ is all-$0$. Hence $\mathbf{0}$ is a row of $\tilde{\mathbf{C}}_{\neq \mathbf{1}}$ if and only if $\mathbf{0}$ is a row of $\rho(\mathbf{C})$. By Proposition 2.5, if $\mathbf{1}$ is a column of $\mathbf{C}$, then
$$\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\tilde{\mathbf{C}}_{\neq \mathbf{1}}) + \mathbf{1}[\mathbf{0} \text{ is a row of } \rho(\mathbf{C})].$$
Finally, $\tilde{\mathbf{C}}_{\neq \mathbf{1}}$ consists of $\rho(\mathbf{C})$ together with $s(\mathbf{C})$ extra columns; by the block construction of Lemma `lem:redundancybound`, $\operatorname{brank}(\tilde{\mathbf{C}}_{\neq \mathbf{1}}) \leq \operatorname{brank}(\rho(\mathbf{C})) + s(\mathbf{C})$. Combining gives the upper bound. The lower bound is Lemma `lem:reducedminimumrank`. $\square$

### 3.2. Exact Criterion for Adding One Non-Simply Redundant Column

**Proposition 3.2.** *Let $\mathbf{C}_0 \in \mathbb{B}^{m \times n_0}$ and let $\mathbf{c} \in \mathbb{B}^{m \times 1}$ be a non-simply redundant column (entrywise product of $\geq 2$ columns of $\mathbf{C}_0$). Then $\operatorname{brank}([\mathbf{C}_0 \mid \mathbf{c}]) \in \{\operatorname{brank}(\mathbf{C}_0),\, \operatorname{brank}(\mathbf{C}_0) + 1\}$, and*
$$\operatorname{brank}([\mathbf{C}_0 \mid \mathbf{c}]) = \operatorname{brank}(\mathbf{C}_0) \iff \exists \text{ optimal } \mathbf{C}_0 = \mathbf{V}\mathbf{H} \text{ of rank } \operatorname{brank}(\mathbf{C}_0) \text{ with } \mathbf{c} \in \operatorname{colspan}_{\mathbb{B}}(\mathbf{V}),$$
*where $\operatorname{colspan}_{\mathbb{B}}(\mathbf{V}) = \{\mathbf{V}\mathbf{h} : \mathbf{h} \in \mathbb{B}^{r \times 1}\}$ is the Boolean union-span of the columns of $\mathbf{V}$.*

*Proof.* Immediate: $[\mathbf{C}_0 \mid \mathbf{c}] = \mathbf{V}[\mathbf{H} \mid \mathbf{h}]$ holds iff $\mathbf{C}_0 = \mathbf{V}\mathbf{H}$ and $\mathbf{c} = \mathbf{V}\mathbf{h}$. $\square$

### 3.3. Positive Theorem: The Conjecture Holds for All Intersection-Complete Codes

Remarkably, the exact equality sought by the authors **does hold** across the entire class of intersection-complete codes! We first record a short lemma.

**Lemma 3.3.** *Every $u \times u$ lower-triangular binary matrix $\mathbf{M} \in \mathbb{B}^{u \times u}$ with $M_{ii} = 1$ for all $i \in [u]$ and $M_{ij} = 0$ for all $j > i$ satisfies $\operatorname{brank}(\mathbf{M}) = u$.*

*Proof.* Clearly $\operatorname{brank}(\mathbf{M}) \leq u$. Let $\mathbf{M} = \mathbf{V}\mathbf{H}$ be any rank-$r$ Boolean factorization with rows $\mathbf{h}_1, \dots, \mathbf{h}_r \in \mathbb{B}^u$ of $\mathbf{H}$. For each $i \in [u]$, since $M_{ii} = 1$, there exists $f(i) \in [r]$ with $V_{i, f(i)} = 1$ and $(\mathbf{h}_{f(i)})_i = 1$. Since $V_{i, f(i)} = 1$ and $\mathbf{V}\mathbf{H} = \mathbf{M}$, we have $\mathbf{h}_{f(i)} \leq \mathbf{M}_{i,\cdot}$, so $(\mathbf{h}_{f(i)})_j = 0$ for all $j > i$. Thus $\max \operatorname{supp}(\mathbf{h}_{f(i)}) = i$, showing $f: [u] \to [r]$ is injective and hence $r \geq u$. $\square$

**Theorem 3.4 (Exact Boolean Rank of Intersection-Complete Codes).** *Let $\mathbf{C} \in \mathbb{B}^{m \times n}$ represent an intersection-complete code $\mathcal{C} \subseteq 2^{[n]}$, and let $u(\mathbf{C})$ be the number of distinct nonzero columns of $\mathbf{C}$. Then*
$$\operatorname{brank}(\mathbf{C}) = u(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) + \ell_{\mathrm{eff}}(\mathbf{C}).$$
*In particular, if $\mathbf{C}$ is an intersection-complete code with no trivial or simply redundant neurons, then $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) + \ell$.*

*Proof.* Let $\tilde{\mathbf{C}} \in \mathbb{B}^{m \times u}$ ($u = u(\mathbf{C})$) be obtained from $\mathbf{C}$ by removing all-$0$ columns and deduplicating identical columns, representing $\tilde{\mathcal{C}} \subseteq 2^{[u]}$. Then $\operatorname{brank}(\tilde{\mathbf{C}}) = \operatorname{brank}(\mathbf{C}) \leq u$, and $\tilde{\mathcal{C}}$ remains intersection-complete (since coordinate projection commutes with bitwise AND).

For each $j \in [u]$, $\operatorname{Tk}_{\tilde{\mathcal{C}}}(j) \neq \varnothing$ because column $j$ is nonzero. Since $\tilde{\mathcal{C}}$ is finite and intersection-complete, the root
$$\sigma_j := \sqrt[\tilde{\mathcal{C}}]{j} = \bigcap_{\tau \in \operatorname{Tk}_{\tilde{\mathcal{C}}}(j)} \tau$$
is a codeword in $\tilde{\mathcal{C}}$. Moreover, $\sigma_1, \dots, \sigma_u$ are pairwise distinct: if $\sigma_i = \sigma_j$, then by Lemma `lem:misctrunks`(2), $\operatorname{Tk}_{\tilde{\mathcal{C}}}(i) = \operatorname{Tk}_{\tilde{\mathcal{C}}}(\sigma_i) = \operatorname{Tk}_{\tilde{\mathcal{C}}}(\sigma_j) = \operatorname{Tk}_{\tilde{\mathcal{C}}}(j)$, contradicting that the columns of $\tilde{\mathbf{C}}$ are distinct.

Reindex $1, \dots, u$ so that $|\sigma_1| \leq |\sigma_2| \leq \dots \leq |\sigma_u|$, and let $\mathbf{M} \in \mathbb{B}^{u \times u}$ be the submatrix of $\tilde{\mathbf{C}}$ on rows $\sigma_1, \dots, \sigma_u$ and columns $1, \dots, u$. Then:
- $M_{ii} = 1$ for all $i \in [u]$, as $i \in \sqrt[\tilde{\mathcal{C}}]{i} = \sigma_i$;
- $M_{ij} = 0$ for all $1 \leq i < j \leq u$: if $M_{ij} = 1$, then $j \in \sigma_i$, so $\sigma_i \in \operatorname{Tk}_{\tilde{\mathcal{C}}}(j)$, forcing $\sigma_j = \sqrt[\tilde{\mathcal{C}}]{j} \subseteq \sigma_i$; together with $|\sigma_j| \geq |\sigma_i|$ this yields $\sigma_j = \sigma_i$, a contradiction.

By Lemma 3.3, $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\tilde{\mathbf{C}}) \geq \operatorname{brank}(\mathbf{M}) = u$, so $\operatorname{brank}(\mathbf{C}) = u(\mathbf{C})$.

Finally, $\rho(\mathcal{C})$ is also intersection-complete and reduced (all $n_0 = \lambda(\mathcal{C})$ columns are distinct and nonzero), so $\operatorname{brank}(\rho(\mathbf{C})) = n_0$. Furthermore, the global intersection $\sigma_{\min} = \bigcap_{\sigma \in \mathcal{C}} \sigma \in \mathcal{C}$ consists precisely of the all-$1$s neurons of $\mathcal{C}$, all of which are deleted in $\rho(\mathcal{C})$, so $\varnothing \in \rho(\mathcal{C})$ always (i.e., $\mathbf{0}$ is always a row of $\rho(\mathbf{C})$). Hence $\ell_{\mathrm{eff}}(\mathbf{C}) = u(\mathbf{C}) - n_0$, giving $\operatorname{brank}(\mathbf{C}) = \operatorname{brank}(\rho(\mathbf{C})) + \ell_{\mathrm{eff}}(\mathbf{C})$. If $\mathbf{C}$ has no trivial or simply redundant neurons, then all columns are distinct, nonzero, and non-all-$1$s, so $\rho$ deletes precisely the $\ell$ redundant columns (each column not in $\rho(\mathbf{C})$ is non-simply redundant, and no column of $\rho(\mathbf{C})$ is redundant in $\mathbf{C}$ since a column of $\rho(\mathbf{C})$ cannot be an intersection of other columns of $\mathbf{C}$, each of which is itself an intersection of columns of $\rho(\mathbf{C})$); hence $\ell_{\mathrm{eff}}(\mathbf{C}) = \ell$, proving the last claim. $\square$

*Remark 3.5 (Refinement of Theorem `thm:intcompletefullrank`).* Theorem `thm:intcompletefullrank` (line 1046) states that if $\mathcal{C} = \{\sigma_1, \dots, \sigma_m\} \subseteq 2^{[n]}$ is intersection-complete, then $\operatorname{brank}(\mathcal{C}) = \min(m, n)$. Without excluding trivial or duplicate columns, this can fail (e.g., $\mathcal{C} = \{\varnothing, 12\} \subseteq 2^{[2]}$ has $m = n = 2$ but $\operatorname{brank}(\mathcal{C}) = 1$). Theorem 3.4 gives the exact unrestricted formula $\operatorname{brank}(\mathcal{C}) = u(\mathbf{C})$ and shows that $m \geq u(\mathbf{C})$ always (via the $u$ distinct roots $\sigma_1, \dots, \sigma_u \in \mathcal{C}$), so $\operatorname{brank}(\mathcal{C}) = n$ holds precisely when $\mathcal{C}$ has no trivial or duplicate columns (which in particular implies $n = u(\mathbf{C}) \le m$, so $\operatorname{brank}(\mathcal{C}) = \min(m, n)$).

---

## 4. Complete Resolution of Question 2 (Section 7.2)

Section 7.2 (lines 1105–1108) asks:
> **Question 2.** *What is $\max_{\mathcal{C} \in \mathbf{Code}} \{\lambda(\mathcal{C}^{(i)}) - \lambda(\mathcal{C})\}$? From Definition `def:excessivecovering` we see that $\lambda(\mathcal{C}^{(i)}) \leq 2\lambda(\mathcal{C}) - 2$. Does any code achieve this bound?*

We prove that the upper bound $\lambda(\mathcal{C}^{(i)}) \leq 2\lambda(\mathcal{C}) - 2$ is **sharp for every $\lambda(\mathcal{C}) \geq 1$**, and consequently $\lambda(\mathcal{C}^{(i)}) - \lambda(\mathcal{C})$ is **unbounded above**.

**Theorem 4.1 (Sharpness of $\lambda(\mathcal{C}^{(i)}) \leq 2\lambda(\mathcal{C}) - 2$).**
1. *For every integer $n \geq 3$, the code*
   $$\mathcal{C}_n := \{\varnothing\} \cup \bigl\{\{j\} : 2 \leq j \leq n\bigr\} \cup \bigl\{\{1, j\} : 2 \leq j \leq n\bigr\} \subseteq 2^{[n]}$$
   *is reduced ($\lambda(\mathcal{C}_n) = n$), neuron $1$ is free (so the covering map $f_1: \mathcal{C}_n \to \mathcal{C}_n^{(1)}$ is a BMF by Theorem `thm:coveringBMF`), and*
   $$\lambda(\mathcal{C}_n^{(1)}) = 2n - 2 = 2\lambda(\mathcal{C}_n) - 2.$$
2. *For every integer $n \geq 1$, the intersection-complete code*
   $$\widehat{\mathcal{C}}_n := \{\varnothing, \{1\}\} \cup \bigl\{\{j\} : 2 \leq j \leq n\bigr\} \cup \bigl\{\{1, j\} : 2 \leq j \leq n\bigr\} \subseteq 2^{[n]}$$
   *is reduced ($\lambda(\widehat{\mathcal{C}}_n) = n$) and satisfies $\lambda(\widehat{\mathcal{C}}_n^{(1)}) = 2n - 2 = 2\lambda(\widehat{\mathcal{C}}_n) - 2$.*
3. *Therefore, for fixed $\lambda(\mathcal{C}) = n \geq 1$,*
   $$\max_{\substack{\mathcal{C} \in \mathbf{Code},\, \lambda(\mathcal{C}) = n \\ i \in [n]}} \bigl(\lambda(\mathcal{C}^{(i)}) - \lambda(\mathcal{C})\bigr) = n - 2,$$
   *and globally $\sup_{\mathcal{C} \in \mathbf{Code},\, i} \{\lambda(\mathcal{C}^{(i)}) - \lambda(\mathcal{C})\} = +\infty$.*

*Proof.*
- **Case $n = 1$:** $\widehat{\mathcal{C}}_1 = \{\varnothing, \{1\}\}$ is reduced ($\lambda = 1$). By Definition `def:excessivecovering`, $T_1 = T_2 = \varnothing$, so $\widehat{\mathcal{C}}_1^{(1)} = \{\varnothing\}$ has $\lambda = 0 = 2(1) - 2$.
- **Case $n = 2$:** $\widehat{\mathcal{C}}_2 = \{\varnothing, \{1\}, \{2\}, \{1,2\}\}$ is reduced ($\lambda = 2$). For $i = 1$, $T_1 = T_3 = \varnothing$, $T_2 = \operatorname{Tk}(2) = \{\{2\}, \{1,2\}\}$, and $T_4 = \operatorname{Tk}(2) \cap \operatorname{Tk}(1) = \{\{1,2\}\} \neq \operatorname{Tk}(1)$. Thus $f_1(\varnothing) = f_1(\{1\}) = \varnothing$, $f_1(\{2\}) = \{2\}$, $f_1(\{1,2\}) = \{2,4\}$. Dropping trivial neurons $1,3$ gives $\{\varnothing, \{2\}, \{2,4\}\}$, which is reduced on $2 = 2(2) - 2$ neurons.
- **General case $n \geq 3$ for $\mathcal{C}_n$ (and $\widehat{\mathcal{C}}_n$):**
  1. *$\mathcal{C}_n$ is reduced (hence $\lambda(\mathcal{C}_n) = n$):*
     The simple trunks are $\operatorname{Tk}_{\mathcal{C}_n}(1) = \{\{1,j\} : 2 \leq j \leq n\}$ (size $n - 1 \geq 2$) and $\operatorname{Tk}_{\mathcal{C}_n}(j) = \{\{j\}, \{1,j\}\}$ (size $2$) for $2 \leq j \leq n$. None is empty. Since $n \geq 3$, $\{1,2\}, \{1,3\} \in \operatorname{Tk}_{\mathcal{C}_n}(1)$, so $\sqrt[\mathcal{C}_n]{1} = \{1\}$ (in particular $\sqrt[\mathcal{C}_n]{1} \notin \mathcal{C}_n$, so neuron $1$ is free). For $2 \leq j \leq n$, $\sqrt[\mathcal{C}_n]{j} = \{j\} \cap \{1,j\} = \{j\}$. For every $a \in [n]$, $\sqrt[\mathcal{C}_n]{a} \setminus \{a\} = \varnothing$, and $\operatorname{Tk}_{\mathcal{C}_n}(\varnothing) = \mathcal{C}_n$ has size $2n - 1 > \max(n-1, 2)$, so $\operatorname{Tk}_{\mathcal{C}_n}(a) \neq \operatorname{Tk}_{\mathcal{C}_n}(\sqrt[\mathcal{C}_n]{a} \setminus \{a\})$. By Lemma `lem:redundancylemma`, $\mathcal{C}_n$ is reduced. (In $\widehat{\mathcal{C}}_n$, the exact same roots $\{1\}$ and $\{j\}$ hold, and $|\widehat{\mathcal{C}}_n| = 2n > \max(n, 2)$, so $\widehat{\mathcal{C}}_n$ is also reduced.)
  2. *Trunks of $f_1$ (Definition `def:excessivecovering`):*
     - $T_1 = \varnothing$ and $T_{n+1} = \varnothing$.
     - For $j \in \{2, \dots, n\}$: $\{j\} \in \operatorname{Tk}(j) \setminus \operatorname{Tk}(1)$, so $T_j = \operatorname{Tk}(j) = \{\{j\}, \{1,j\}\}$.
     - For $j \in \{2, \dots, n\}$: $\operatorname{Tk}(j) \cap \operatorname{Tk}(1) = \{\{1,j\}\} \neq \operatorname{Tk}(1)$ (since $|\operatorname{Tk}(1)| \geq n - 1 \geq 2 > 1$), so $T_{n+j} = \{\{1,j\}\}$.
     
     Evaluating $f_1(\sigma) = \{a \in [2n] : \sigma \in T_a\}$ gives $f_1(\varnothing) = \varnothing$ (and $f_1(\{1\}) = \varnothing$ in $\widehat{\mathcal{C}}_n$), $f_1(\{j\}) = \{j\}$, and $f_1(\{1,j\}) = \{j, n+j\}$ for $2 \leq j \leq n$.
  3. *$\mathcal{C}_n^{(1)}$ reduces to $2n - 2$ neurons:*
     Deleting the two trivial neurons $1$ and $n+1$ yields the code $\mathcal{D}_n \subseteq 2^N$ on the $(2n - 2)$-element neuron set $N := \{2, \dots, n\} \cup \{n+2, \dots, 2n\}$:
     $$\mathcal{D}_n = \{\varnothing\} \cup \bigl\{\{j\} : 2 \leq j \leq n\bigr\} \cup \bigl\{\{j, n+j\} : 2 \leq j \leq n\bigr\}.$$
     We verify $\mathcal{D}_n$ is reduced via Lemma `lem:redundancylemma`:
     - For $j \in \{2, \dots, n\}$: $\operatorname{Tk}_{\mathcal{D}_n}(j) = \{\{j\}, \{j, n+j\}\}$ (nonempty), $\sqrt[\mathcal{D}_n]{j} = \{j\}$, and $\operatorname{Tk}_{\mathcal{D}_n}(\varnothing) = \mathcal{D}_n$ has size $2n - 1 > 2$, so $j$ is not redundant.
     - For $n + j$ ($2 \leq j \leq n$): $\operatorname{Tk}_{\mathcal{D}_n}(n+j) = \{\{j, n+j\}\}$ (nonempty), $\sqrt[\mathcal{D}_n]{n+j} = \{j, n+j\}$, and $\operatorname{Tk}_{\mathcal{D}_n}(\sqrt[\mathcal{D}_n]{n+j} \setminus \{n+j\}) = \operatorname{Tk}_{\mathcal{D}_n}(j) = \{\{j\}, \{j, n+j\}\} \neq \operatorname{Tk}_{\mathcal{D}_n}(n+j)$, so $n + j$ is not redundant.
     
     Hence $\mathcal{D}_n = \rho(\mathcal{C}_n^{(1)}) = \rho(\widehat{\mathcal{C}}_n^{(1)})$, giving $\lambda(\mathcal{C}_n^{(1)}) = \lambda(\widehat{\mathcal{C}}_n^{(1)}) = |N| = 2n - 2$. $\square$

---

## 5. Verification Code

A self-contained Python verification script implementing exact Boolean rank (via both maximal all-$1$ rectangle MILP set cover and independent brute-force Luce–Plemmons residual search $(\mathbf{C}:\mathbf{H})\mathbf{H} = \mathbf{C}$), exact redundancy classification, exhaustive search for minimal clean counterexamples, verification of Families 1–2, verification of Theorem 3.4 across all intersection-complete codes on $n \leq 3$ neurons and random samples on $n \leq 6$, and verification of Theorem 4.1 for $n = 3, \dots, 8$ has been saved at:

- [`verify_conjecture.py`](./verify_conjecture.py)
