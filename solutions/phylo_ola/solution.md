# Sharp Deterministic Refinement of the OLA–SPR Distance Bound (arXiv:2503.10169 / Bull. Math. Biol. 2026)

- **Source**: David Harry Richman, Cheng Zhang, Frederick A. Matsen IV, *"Vector encoding of phylogenetic trees by ordered leaf attachment"*, [arXiv:2503.10169](https://arxiv.org/abs/2503.10169) (v1/v2 Mar 2025, Conjecture 11 [`conj:spr-move-bound`]); published journal version in *Bulletin of Mathematical Biology* **88**:58 (Mar 2026, [PMC12999618](https://pmc.ncbi.nlm.nih.gov/articles/PMC12999618/)).
- **Domain**: Mathematical Biology / Phylogenetics & Machine Learning on Tree Spaces.
- **Novelty & Attribution Note**: While arXiv:2503.10169 v1/v2 left the bound as Conjecture 11, the published *Bull. Math. Biol.* version (Mar 2026) upgraded it to **Theorem 6** ($d_{\mathrm{OLA}}(T, T') \le 2\,\mathrm{ht}_\rho(T)$) using a referee's tripod sketch. Our independent proof gives **strictly sharper and exact results** not in the published paper:
  1. The finer local bound $d_{\mathrm{OLA}}(T, T') \le |A_T(p) \cup A_T(v_{\mathrm{top}})| - 1 \le 2\,\mathrm{ht}_r(T) - 2 = 2\,\mathrm{ht}_\rho(T) - 4$ (strictly improving Theorem 6's constant by 4 in the $\mathrm{ht}_\rho$ convention);
  2. **Exact worst-case sharpness** ($d_{\mathrm{OLA}}(T, T') = 2h - 2$ attained for every $h \ge 2$ and every $2h \le n \le 2^h$);
  3. **Asymptotic tightness in expectation** ($\mathbb{E}[d_{\mathrm{OLA}}(T_n, T')] \ge \frac{n}{24} - O(1) = \Omega(\mathrm{ht}(T_n))$ on caterpillars);
  4. **Lean 4 formalization** of the combinatorial core (`PhyloOLA.lean`).

---

## 1. Exact Definitions

Let $\widehat{\mathcal{RBT}}_n$ ($n \ge 2$) be the set of rooted binary phylogenetic trees on leaf set $L(T) = \{0, 1, \dots, n-1\}$ with a degree-$1$ root $\rho$ and $n - 1$ bifurcating internal vertices $V_{\mathrm{int}}(T)$, whose topmost internal vertex (the unique child of $\rho$) is $r(T)$. Write:
- $\mathrm{ht}_r(T) := \max_{\ell \in L(T)} \mathrm{dist}_T(r(T), \ell) \ge 1$ (height from the topmost bifurcating node),
- $\mathrm{ht}_\rho(T) := \max_{\ell \in L(T)} \mathrm{dist}_T(\rho, \ell) = \mathrm{ht}_r(T) + 1$ (height from $\rho$).

For each $v \in V(T) \setminus \{\rho\}$, let $T_v$ be the rooted leaf-labeled subtree rooted at $v$ and $L_T(v) \subseteq \{0, \dots, n-1\}$ its leaf set.
- **Clade-founder & clade-splitter** (Algorithm 1 of arXiv:2503.10169):
  $$\mathrm{CF}_T(v) := \min L_T(v), \qquad \mathrm{CS}_T(v) := \max\bigl(\mathrm{CF}_T(c_1), \mathrm{CF}_T(c_2)\bigr) \quad (v \in V_{\mathrm{int}}(T)),$$
  where $c_1, c_2$ are the children of $v$. Canonical labels are $\mathrm{label}_T(v) := -\mathrm{CS}_T(v)$ for $v \in V_{\mathrm{int}}(T)$ and $\mathrm{label}_T(\ell) := \ell$ for leaves $\ell$. By Proposition 4 of arXiv:2503.10169, $\mathrm{CS}_T : V_{\mathrm{int}}(T) \to \{1, \dots, n-1\}$ is a bijection.
- **OLA encoding** (Algorithm 2 & Proposition 6): For each $i \in \{1, \dots, n-1\}$, let $T|_i$ be the induced rooted binary tree on leaf set $\{0, 1, \dots, i\}$ (deleting leaves $> i$ and suppressing degree-$2$ vertices). Then the $i$-th entry $a_i(T)$ of $\Phi(T) = (a_1(T), \dots, a_{n-1}(T))$ is the canonical label in $T|_i$ (equivalently in $T$, by Proposition 4(a)) of the sister of leaf $i$ in $T|_i$.
- **OLA distance**: $d_{\mathrm{OLA}}(T, T') := |\{i \in \{1, \dots, n-1\} : a_i(T) \ne a_i(T')\}|$ (Hamming distance).
- **Rooted SPR (rSPR) move**: Choose $u \in V(T) \setminus \{\rho, r(T)\}$ with parent $p = \mathrm{parent}_T(u) \in V_{\mathrm{int}}(T)$; prune $S = T_u$ by deleting $(p, u)$ and suppressing $p$ to obtain $T \setminus S$; choose an edge $e = (v_{\mathrm{top}}, v_{\mathrm{bot}})$ of $T \setminus S$ (directed away from $\rho$, possibly with $v_{\mathrm{top}} = \rho$); subdivide $e$ by a new internal node $q$ and attach $u$ as its second child to obtain $T'$.

---

## 2. Proof of the Main Theorem

### Lemma 1 (Subtree Locality of OLA Entries)
*Let $T, T' \in \widehat{\mathcal{RBT}}_n$. Suppose $w \in V_{\mathrm{int}}(T)$ and $w' \in V_{\mathrm{int}}(T')$ have isomorphic rooted leaf-labeled subtrees $T_w \cong T'_{w'}$. Then $\mathrm{CS}_T(w) = \mathrm{CS}_{T'}(w')$, and setting $i := \mathrm{CS}_T(w)$, we have $a_i(T) = a_i(T')$.*

**Proof.** Let $\varphi : V(T_w) \to V(T'_{w'})$ be the rooted leaf-label-preserving isomorphism with $\varphi(w) = w'$. For every $x \in V(T_w)$, $L_{T'}(\varphi(x)) = L_T(x)$, so $\mathrm{CF}_{T'}(\varphi(x)) = \mathrm{CF}_T(x)$ and $\mathrm{label}_{T'}(\varphi(x)) = \mathrm{label}_T(x)$. In particular, $i := \mathrm{CS}_T(w) = \mathrm{CS}_{T'}(w') \in \{1, \dots, n-1\}$.

Order the children $c_1, c_2$ of $w$ in $T$ so that $\mathrm{CF}_T(c_1) < \mathrm{CF}_T(c_2) = i$, and set $A := \{0, 1, \dots, i\}$. Then $L_T(c_2) \cap A = \{i\}$ and $L_T(c_1) \cap A \neq \emptyset$. A vertex $x \in V(T) \setminus \{\rho\}$ survives in $T|_i$ iff $x \in A$ or $x$ is an internal node both of whose children have clades intersecting $A$. Thus $w$ survives in $T|_i$: its child on the $c_2$-side in $T|_i$ is leaf $i$, and its child on the $c_1$-side in $T|_i$ is the highest surviving vertex in $T_{c_1}$, namely
$$z := \operatorname{lca}_{T_{c_1}}\!\bigl(L_T(c_1) \cap A\bigr).$$
By canonical-label stability under restriction (Proposition 4(a) & Proposition 6), $a_i(T) = \mathrm{label}_{T|_i}(z) = \mathrm{label}_T(z)$. Applying the same argument to $w'$ in $T'$ gives the sister of leaf $i$ in $T'|_i$ as $z' = \operatorname{lca}_{T'_{\varphi(c_1)}}(L_{T'}(\varphi(c_1)) \cap A) = \varphi(z)$, hence
$$a_i(T') = \mathrm{label}_{T'}(\varphi(z)) = \mathrm{label}_T(z) = a_i(T). \qquad \square$$

### Lemma 2 (Upward-Closed Change Set Bound)
*Let $T, T' \in \widehat{\mathcal{RBT}}_n$. Suppose $B \subseteq V_{\mathrm{int}}(T)$ is a nonempty upward-closed subset of internal nodes of $T$ (if $w \in B$ and $v \in V_{\mathrm{int}}(T)$ is an ancestor of $w$, then $v \in B$) such that for every $w \in V_{\mathrm{int}}(T) \setminus B$, there exists $w' \in V_{\mathrm{int}}(T')$ with $T'_{w'} \cong T_w$. Then*
$$d_{\mathrm{OLA}}(T, T') \le |B| - 1.$$

**Proof.** Let $J := \{\mathrm{CS}_T(w) : w \in B\} \subseteq \{1, \dots, n-1\}$. By Proposition 4, $|J| = |B| \ge 1$. For every $i \in \{1, \dots, n-1\} \setminus J$, the unique $w \in V_{\mathrm{int}}(T)$ with $\mathrm{CS}_T(w) = i$ lies outside $B$, so Lemma 1 gives $a_i(T) = a_i(T')$.

Let $j^* := \min J$. We show that $a_{j^*}(T) = a_{j^*}(T')$ as well:
- If $j^* = 1$, then $a_1(T) = 0 = a_1(T')$.
- If $j^* \ge 2$, let $v^* := \operatorname{lca}_T(\{0, 1, \dots, j^* - 1\}) \in V_{\mathrm{int}}(T)$. Both children of $v^*$ intersect $\{0, \dots, j^* - 1\}$, so $\mathrm{CS}_T(v^*) \le j^* - 1 < j^* = \min J$, forcing $v^* \notin B$. Hence there exists $v^{*\prime} \in V_{\mathrm{int}}(T')$ with $T'_{v^{*\prime}} \cong T_{v^*}$.

  We claim $j^* \notin L_T(v^*)$. Otherwise, let $w^* \in B$ be the unique internal node with $\mathrm{CS}_T(w^*) = j^*$; then $j^* \in L_T(w^*)$, so $v^*$ and $w^*$ are comparable in the ancestor order:
  - $v^*$ cannot be an ancestor of $w^*$, since $w^* \in B$ and $B$ upward-closed would imply $v^* \in B$, contradicting $v^* \notin B$.
  - $w^*$ cannot be a proper ancestor of $v^*$, since then the child $c$ of $w^*$ containing $v^*$ would satisfy $L_T(c) \supseteq L_T(v^*) \supseteq \{0, \dots, j^*\}$, forcing $\mathrm{CF}_T(c) = 0$ and $\mathrm{CF}_T(c') \ge j^* + 1$ for the other child $c'$, contradicting $\mathrm{CS}_T(w^*) = j^*$.

  Hence $j^* \notin L_T(v^*) = L_{T'}(v^{*\prime})$. Therefore $L_T(v^*) \cap \{0, \dots, j^*\} = \{0, \dots, j^* - 1\}$, so the two children of the topmost node of $T|_{j^*}$ are $v^*$ and leaf $j^*$, giving $a_{j^*}(T) = \mathrm{label}_T(v^*) = -\mathrm{CS}_T(v^*)$. Since $T'_{v^{*\prime}} \cong T_{v^*}$, the same holds in $T'|_{j^*}$, yielding $a_{j^*}(T') = -\mathrm{CS}_{T'}(v^{*\prime}) = -\mathrm{CS}_T(v^*) = a_{j^*}(T)$.

Thus $a_i(T) = a_i(T')$ for all $i \notin J \setminus \{j^*\}$, so $d_{\mathrm{OLA}}(T, T') \le |J| - 1 = |B| - 1$. $\square$

### Theorem 3 (Sharp Deterministic rSPR Bound; Proof of Conjecture 11)
*Let $T \in \widehat{\mathcal{RBT}}_n$ ($n \ge 2$), and let $T'$ be obtained from $T$ by a single rSPR move pruning $S = T_u$ ($p = \mathrm{parent}_T(u) \in V_{\mathrm{int}}(T)$) and regrafting onto edge $e = (v_{\mathrm{top}}, v_{\mathrm{bot}})$ of $T \setminus S$. For $x \in V_{\mathrm{int}}(T)$ let $A_T(x)$ be the set of internal ancestors of $x$ in $T$ (including $x$), and set $A_T(\rho) := \emptyset$. Then*
$$d_{\mathrm{OLA}}(T, T') \le \bigl|A_T(p) \cup A_T(v_{\mathrm{top}})\bigr| - 1 \le 2\,\mathrm{ht}_r(T) - 2 = 2\,\mathrm{ht}_\rho(T) - 4.$$
*In particular, by symmetry of rSPR, $d_{\mathrm{OLA}}(T, T') \le 2\min(\mathrm{ht}_r(T), \mathrm{ht}_r(T')) - 2$, and for any probability distribution on single-SPR neighbors $T'$ of $T$,*
$$\mathbb{E}\bigl[d_{\mathrm{OLA}}(T, T')\bigr] \le 2\,\mathrm{ht}_r(T) - 2 = O\bigl(\mathrm{ht}(T)\bigr).$$

**Proof.** Note that $v_{\mathrm{top}} \in \{\rho\} \cup (V_{\mathrm{int}}(T) \setminus \{p\})$. Set $B := A_T(p) \cup A_T(v_{\mathrm{top}}) \subseteq V_{\mathrm{int}}(T)$, which is nonempty ($p \in B$) and upward-closed in $T$. For every $w \in V_{\mathrm{int}}(T) \setminus B$:
- $w \neq p$, so $w$ is not suppressed when forming $T \setminus S$.
- If $w \in V_{\mathrm{int}}(S)$, then $T'_u \cong T_u$, hence $T'_w \cong T_w$.
- If $w \notin V_{\mathrm{int}}(S)$, then $w \notin A_T(p)$ implies $T_w$ is disjoint from $\{p\} \cup V(S)$, so $(T \setminus S)_w \cong T_w$. Since the internal ancestors of the new node $q$ in $T'$ (other than $q$) are $A_T(v_{\mathrm{top}}) \setminus \{p\} \subseteq B$, $w$ is not an ancestor of $q$ in $T'$, so subdividing $e$ leaves the subtree below $w$ untouched: $T'_w \cong (T \setminus S)_w \cong T_w$.

By Lemma 2, $d_{\mathrm{OLA}}(T, T') \le |B| - 1$. Every $x \in V_{\mathrm{int}}(T)$ has a leaf strictly below it, so $|A_T(x)| = \mathrm{depth}_r(x) + 1 \le \mathrm{ht}_r(T)$.
- If $v_{\mathrm{top}} = \rho$, $|B| - 1 = |A_T(p)| - 1 \le \mathrm{ht}_r(T) - 1 \le 2\,\mathrm{ht}_r(T) - 2$.
- If $v_{\mathrm{top}} \in V_{\mathrm{int}}(T) \setminus \{p\}$, then $r(T) \in A_T(p) \cap A_T(v_{\mathrm{top}})$, so $|B| \le |A_T(p)| + |A_T(v_{\mathrm{top}})| - 1 \le 2\,\mathrm{ht}_r(T) - 1$, hence $|B| - 1 \le 2\,\mathrm{ht}_r(T) - 2$. $\blacksquare$

---

## 3. Exact Sharpness & Corollaries

### Proposition 4 (Worst-Case Exact Sharpness)
*For every $h \ge 2$ and every $n$ satisfying $2h \le n \le 2^h$, there exists $T \in \widehat{\mathcal{RBT}}_n$ with $\mathrm{ht}_r(T) = h$ and a single rSPR neighbor $T'$ with $d_{\mathrm{OLA}}(T, T') = 2h - 2$.*

**Proof.** Note that any rooted binary tree of height $h$ has at most $2^h$ leaves, so $n \le 2^h$ is necessary. For $n = 2h$, let $T$ have topmost node $r$ with two height-$(h-1)$ caterpillar children: left spine $L_1, \dots, L_{h-1}$ attaching leaves $2, 3, \dots, h$ with bottom leaf $0$, and right spine $R_1, \dots, R_{h-1}$ attaching leaves $h+1, \dots, 2h-1$ with bottom leaf $1$. Then $\Phi(T) = (0^h, 1^{h-1})$. Pruning leaf $0$ and regrafting it onto the parent edge of leaf $1$ gives $T'$ with $\Phi(T') = (0, -1, 2, 3, \dots, h-1, (-1)^{h-1})$, which differs from $\Phi(T)$ in all $2h - 2$ coordinates $i = 2, \dots, 2h-1$. For $2h < n \le 2^h$, expanding side leaves $\{2, \dots, 2h-1\}$ into binary subtrees on the extra labels $\{2h, \dots, n-1\}$ while keeping height $\le h$ (and leaving leaves $0, 1$ and the spine structure intact) preserves all $2h - 2$ changed entries at coordinates $2, \dots, 2h-1$, while every new internal node lies outside $B$ and has unchanged entry by Lemma 1, giving $d_{\mathrm{OLA}}(T, T') = 2h - 2$. $\square$

### Proposition 5 (Asymptotic Tightness in Expectation)
*For the caterpillar $T_n = \Psi(0, \dots, 0) \in \widehat{\mathcal{RBT}}_n$ ($\mathrm{ht}_r(T_n) = n - 1$) and $T'$ uniform on $\mathcal{N}_{\mathrm{SPR}}(T_n)$,*
$$\mathbb{E}\bigl[d_{\mathrm{OLA}}(T_n, T')\bigr] \ge \frac{n}{24} - O(1) = \Omega\bigl(\mathrm{ht}(T_n)\bigr).$$

**Proof.** For $1 \le j \le k - 2 \le n - 4$, pruning leaf $j$ and regrafting onto the parent edge of internal node $-(k+1)$ produces pairwise distinct caterpillars $T'(j, k)$ (uniquely identified by their single descent $(k, j)$ in the top-to-bottom leaf peel sequence) with $a_m(T'(j, k)) = -j \neq 0 = a_m(T_n)$ for all $m \in \{j+1, \dots, k\}$, so $d_{\mathrm{OLA}}(T_n, T'(j, k)) \ge k - j$. Since $|\mathcal{N}_{\mathrm{SPR}}(T_n)| < 4n^2$, summing over $d = k - j \in \{2, \dots, n-3\}$ yields
$$\mathbb{E}\bigl[d_{\mathrm{OLA}}(T_n, T')\bigr] \ge \frac{1}{4n^2} \sum_{d=2}^{n-3} d(n - 2 - d) = \frac{n}{24} - O(1). \qquad \square$$

### Corollary (Sharp Worst-Case NNI Bound)
For an NNI move on internal edge $(v_1, v_2)$, $A_T(p) \cup A_T(v_{\mathrm{top}}) = A_T(v_2)$, so $d_{\mathrm{OLA}}(T, T') \le \mathrm{depth}_r(v_2) \le \mathrm{ht}_r(T) - 1$.
