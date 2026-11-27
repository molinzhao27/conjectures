# Refutation of Conjectures 3.9 and 3.13 on Open and Closed Convex Neural Codes

**Target paper:** Brianna Gambacini, R. Amzi Jeffs, Sam Macdonald, and Anne Shiu, *Non-monotonicity of closed convexity in neural codes*, [arXiv:1912.00963v3](https://arxiv.org/abs/1912.00963) (published in *Vietnam Journal of Mathematics* **50**:359–373, 2022), Section 3, Conjecture 3.9 (`conj3`) and Conjecture 3.13 (`conj:open-closed`).  
**Companion files:** [`solutions/neural_code_closed_convex/explore.py`](explore.py), [`solutions/neural_code_closed_convex/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 3.9 (`conj3`, originally Goldrup–Phillipson Conjecture 4.3): FALSE (sharp 5-neuron 1D counterexample).** Gambacini, Jeffs, Macdonald, and Shiu already disproved Conjecture 3.9 on 6 neurons using 2D rectangles (Theorem 3.10). We strengthen this to a **5-neuron, 8-word code realized on the real line $\mathbb{R}^1$** by both open and closed intervals, and prove by exhaustive enumeration over all 167 simplicial complexes on $\le 4$ labels that $n = 5$ is minimal among locally good codes containing $\varnothing$.
- **Conjecture 3.13 (`conj:open-closed`): FALSE (7-neuron counterexample; genuinely open since 2019/2022).** Conjecture 3.13 asserts that every locally good code on at most 7 neurons is either open convex or closed convex (Theorem 3.11 of Gambacini et al. constructed an 8-neuron counterexample). We construct an explicit **7-neuron, 14-word locally good code $C$** that admits **neither an open convex nor a closed convex realization in any dimension**, proved by taking neuron restrictions onto both the 5-neuron non-open-convex code $A$ (Lienkaemper–Shiu–Woodstock 2017, Theorem 3.1) and the 5-neuron non-closed-convex code $C_6$ (Goldrup–Phillipson 2020, Theorem 4.1).
- **Remaining open question ($n \le 6$):** For $n \le 4$, every locally good code is open convex (Curto et al. 2017). Whether a locally good code on $n \in \{5, 6\}$ neurons can be neither open convex nor closed convex remains **open**; an exhaustive search over all gluings of $A$ and $C_6$ finds $0$ qualifying gluings on 6 neurons and $100$ labeled gluings on 7 neurons, leaving non-gluing constructions on $5$ or $6$ neurons unresolved.

---

## 1. Preliminaries and Conjecture Statements

Write a subset $\{1, 2, 3\} \subseteq [n]$ compactly as $123$. Given a collection $\mathcal{U} = \{U_1, \dots, U_n\}$ of receptive fields in a stimulus space $X = \mathbb{R}^d$, the **neural code** of $\mathcal{U}$ is the set of exact firing patterns $\mathcal{C}(\mathcal{U}) := \{\{i \in [n] : x \in U_i\} : x \in \mathbb{R}^d\}$. A code is **open convex** (resp. **closed convex**) if it is realized by open (resp. closed) convex sets in some $\mathbb{R}^d$. Its **simplicial complex** $\Delta(\mathcal{C})$ is the downward closure of $\mathcal{C}$, whose facets are the maximal codewords of $\mathcal{C}$. A code is **max-intersection incomplete** if some nonempty intersection of maximal codewords is missing from $\mathcal{C}$. A face $\sigma \in \Delta(\mathcal{C})$ is **mandatory** if its link $\operatorname{Lk}_\sigma(\Delta(\mathcal{C})) := \{\tau \in \Delta(\mathcal{C}) : \tau \cap \sigma = \varnothing,\; \tau \cup \sigma \in \Delta(\mathcal{C})\}$ is noncontractible (Curto et al. 2017); a code is **locally good** if it contains every mandatory codeword (equivalently, $\operatorname{Lk}_\sigma(\Delta(\mathcal{C}))$ is contractible for every nonempty $\sigma \in \Delta(\mathcal{C}) \setminus \mathcal{C}$, which need only be checked on missing intersections of facets). Both open convex and closed convex codes are locally good (Curto et al. 2017; Cruz et al. 2019).

**Conjecture 3.9 (`conj3`, Goldrup–Phillipson Conjecture 4.3; arXiv:1912.00963v3).** *Let $\mathcal{C}$ be a max-intersection incomplete open convex code, where $\Delta(\mathcal{C})$ has at least two non-mandatory codewords not contained in $\mathcal{C}$. Suppose $\mathcal{C}$ has at least three maximal codewords $M_1, M_2, M_3$, and there is $\sigma \subsetneq M_1$ with $\sigma \in \mathcal{C}$ such that $\sigma \cap M_2 \notin \mathcal{C}$. Then $\mathcal{C}$ is not closed convex.*

**Conjecture 3.13 (`conj:open-closed`, Gambacini–Jeffs–Macdonald–Shiu 2022; arXiv:1912.00963v3).** *Every locally good code on at most 7 neurons is open convex or closed convex.*

---

## 2. Sharp 5-Neuron 1D Counterexample to Conjecture 3.9

**Theorem 1 (5-neuron interval counterexample and minimality).**
1. *The 5-neuron, 8-word code $\mathcal{C}_5 := \{\varnothing, 2, 3, 12, 13, 125, 123, 134\}$ satisfies all hypotheses of Conjecture 3.9 and is simultaneously open convex and closed convex in $\mathbb{R}^1$.*
2. *No locally good code containing $\varnothing$ on $n \le 4$ neurons is max-intersection incomplete; hence $n = 5$ is minimal.*

*Proof.* For (1), take the five intervals in $\mathbb{R}^1$ (in neuron order $1, \dots, 5$):
$$
I_1 = (1, 8), \qquad I_2 = (0, 5), \qquad I_3 = (4, 9), \qquad I_4 = (6, 7), \qquad I_5 = (2, 3).
$$
From left to right, the words on the two exterior rays and nine open intervals between consecutive endpoints $0 < 1 < 2 < 3 < 4 < 5 < 6 < 7 < 8 < 9$ are
$$
\varnothing,\; 2,\; 12,\; 125,\; 12,\; 123,\; 13,\; 134,\; 13,\; 3,\; \varnothing.
$$
Because all ten endpoints are pairwise distinct, crossing any endpoint changes membership in exactly one interval; hence the word at each endpoint equals the word of one of its two adjacent open intervals, whether every interval is taken open or every interval is taken closed. Thus both open and closed intervals realize the exact same 8-word code $\mathcal{C}_5$.

We check the hypotheses of Conjecture 3.9:
- The maximal codewords (facets) are $M_1 = 125$, $M_2 = 134$, and $M_3 = 123$ (three facets).
- The facet intersection $M_1 \cap M_2 = 125 \cap 134 = 1$ is missing, so $\mathcal{C}_5$ is max-intersection incomplete.
- Both $1$ and $5$ lie in $\Delta(\mathcal{C}_5) \setminus \mathcal{C}_5$ and are non-mandatory: $\operatorname{Lk}_{1}(\Delta(\mathcal{C}_5))$ is the contractible path $5 - 2 - 3 - 4$, and $\operatorname{Lk}_{5}(\Delta(\mathcal{C}_5))$ is the contractible edge $1 - 2$.
- Taking $\sigma = 12 \in \mathcal{C}_5$ gives $\sigma \subsetneq M_1$ and $\sigma \cap M_2 = 1 \notin \mathcal{C}_5$.

Thus $\mathcal{C}_5$ refutes Conjecture 3.9 with 5 neurons in $\mathbb{R}^1$.

For (2), exhaustive enumeration over all $167$ labeled simplicial complexes on subsets of $\{1, 2, 3, 4\}$ containing $\varnothing$ yields $266$ nonempty intersections of two or more distinct facets; their links have Euler characteristics $\chi \in \{0, 2, 3\}$ ($8$, $222$, and $36$ occurrences respectively) and never $\chi = 1$, so none are contractible. Hence every locally good code on $n \le 4$ neurons containing $\varnothing$ is max-intersection complete. $\square$

---

## 3. Seven-Neuron Counterexample to Conjecture 3.13

**Theorem 2 (Refutation of Conjecture 3.13).** *The 7-neuron, 14-word code*
$$
C := \{\varnothing,\, 3,\, 4,\, 34,\, 45,\, 123,\, 137,\, 147,\, 236,\, 1236,\, 1237,\, 1347,\, 1457,\, 23456\}
$$
*is locally good, yet is neither open convex nor closed convex in any Euclidean dimension.*

*Proof.* We verify local goodness and both non-convexity obstructions separately.

**Step 1: $C$ is locally good.** The facets of $\Delta(C)$ are the five maximal codewords $1236, 1237, 1347, 1457, 23456$. For any missing nonempty face $\sigma \in \Delta(C) \setminus C$, let $\bar{\sigma}$ be the intersection of all facets of $\Delta(C)$ containing $\sigma$. If $\bar{\sigma} \supsetneq \sigma$, then every vertex in $\bar{\sigma} \setminus \sigma$ belongs to every facet of $\operatorname{Lk}_\sigma(\Delta(C))$, making the link a cone and hence contractible. It therefore suffices to check the missing nonempty faces equal to intersections of facets. Computing all intersections of the five facets yields exactly four missing faces:

| Missing face $\sigma$ | Facets of $\operatorname{Lk}_\sigma(\Delta(C))$ in attachment order | Topology |
|---|---|---|
| $1$ | $236,\; 237,\; 347,\; 457$ | Four 2-simplices glued sequentially along edges $23, 37, 47$ |
| $13$ | $26,\; 27,\; 47$ | Path $6 - 2 - 7 - 4$ |
| $17$ | $23,\; 34,\; 45$ | Path $2 - 3 - 4 - 5$ |
| $23$ | $456,\; 16,\; 17$ | 2-simplex $456$ with path $6 - 1 - 7$ attached at vertex $6$ |

In every row, each simplex after the first intersects the union of all preceding simplices in a single nonempty face (an edge or vertex), so it deformation retracts onto that face while fixing the predecessor union. By induction, all four links deformation retract onto their first simplex and are contractible. (Independently, `explore.py` verifies explicit sequences of elementary collapses to a single vertex for all $46$ faces in $\Delta(C) \setminus C$.) Thus $C$ is locally good.

**Step 2: $C$ is neither open convex nor closed convex.** Restricting a realization $\mathcal{U} = \{U_1, \dots, U_7\}$ in $\mathbb{R}^d$ to a subset $S \subseteq [7]$ of neurons replaces each codeword $w$ by $w \cap S$, preserving both open convexity and closed convexity in the same ambient space $\mathbb{R}^d$.
- *Not open convex:* Restricting $C$ to $S_1 = 12345$ gives the exact 12-word image
  $$
  C|_{12345} = \{\varnothing,\, 3,\, 4,\, 13,\, 14,\, 23,\, 34,\, 45,\, 123,\, 134,\, 145,\, 2345\} = A,
  $$
  which is the 5-neuron non-open-convex code of Lienkaemper, Shiu, and Woodstock (2017, Theorem 3.1). Hence $C$ has no open convex realization in any dimension.
- *Not closed convex:* Restricting $C$ to $S_2 = 12467$ gives the exact 9-word image
  $$
  C|_{12467} = \{\varnothing,\, 4,\, 12,\, 17,\, 26,\, 126,\, 127,\, 147,\, 246\}.
  $$
  Relabeling $(1, 2, 6, 4, 7) \mapsto (1, 2, 3, 4, 5)$ produces
  $$
  B = \{\varnothing,\, 4,\, 12,\, 15,\, 23,\, 123,\, 125,\, 145,\, 234\},
  $$
  which is the code $C_6$ of Goldrup and Phillipson (2020, Table 1 and Theorem 4.1), proved to admit no closed convex realization in any dimension. Hence $C$ has no closed convex realization in any dimension. $\square$

*Construction note:* $C$ is the maximal common extension (fiber product) obtained by identifying neurons $1, 2, 4$ of $A$ and $B$ and sending the remaining two neurons $(3, 5)$ of $B$ to $(6, 7)$, keeping every 7-neuron word whose projections lie in $A$ and relabeled $B$.

---

## 4. Reproducible Verification and References

Run from the repository root (Python 3.10+ standard library only):

```sh
.venv/bin/python -B solutions/neural_code_closed_convex/explore.py
```

The script verifies using exact bitmasks and rational cell sampling:
- $n \le 4$ minimality: all $167$ simplicial complexes containing $\varnothing$ and all $266$ nonempty facet-intersection links ($\chi \in \{0, 2, 3\}$, never $1$).
- Exact open and closed box-realization verification for both our 5-neuron 1D interval code (8 words) and Gambacini et al.'s 6-neuron 2D rectangle code (12 words), checking every hypothesis of Conjecture 3.9.
- Exact projection identities $C|_{12345} = A$ and relabeled $C|_{12467} = C_6$, plus explicit elementary-collapse certificates to a point for all $46$ missing-face links of $C$.
- Exhaustive gluing search for $(A, C_6)$: $0$ qualifying gluings on $6$ neurons and $100$ labeled gluings on $7$ neurons.

### References
- J. Cruz, C. Giusti, V. Itskov, and B. Kronholm, *On open and closed convex codes*, Discrete & Computational Geometry **61**(2):247–270, 2019.
- C. Curto, E. Gross, J. Jeffries, K. Morrison, M. Omar, Z. Rosen, A. Shiu, and N. Youngs, *What makes a neural code convex?*, SIAM Journal on Applied Algebra and Geometry **1**(1):222–238, 2017.
- B. Gambacini, R. A. Jeffs, S. Macdonald, and A. Shiu, *Non-monotonicity of closed convexity in neural codes*, Vietnam Journal of Mathematics **50**:359–373, 2022 ([arXiv:1912.00963v3](https://arxiv.org/abs/1912.00963)).
- S. A. Goldrup and K. Phillipson, *Classification of open and closed convex codes on five neurons*, Advances in Applied Mathematics **112**:101948, 2020 ([arXiv:1909.09004](https://arxiv.org/abs/1909.09004)).
- C. Lienkaemper, A. Shiu, and Z. Woodstock, *Obstructions to convexity in neural codes*, Advances in Applied Mathematics **85**:31–59, 2017.
