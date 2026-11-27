# Refutation of Global Identifiability for Deep Polynomial Neural Networks

**Target paper:** Konstantin Usevich, Ricardo Borsoi, Clara Dérand, and Marianne Clausel, *Identifiability of Deep Polynomial Neural Networks*, [arXiv:2506.17093](https://arxiv.org/abs/2506.17093) (v3, NeurIPS 2025 camera-ready update), Section 4.3, Conjecture `conj:global_identifiability`.  
**Companion files:** [`solutions/deep_poly_ident/explore.py`](explore.py), [`solutions/deep_poly_ident/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:global_identifiability`: FALSE** over both $\mathbb{R}$ and $\mathbb{C}$.
- **Exact shallow counterexample (Theorem 1):** A generic two-layer homogeneous polynomial neural network (hPNN) of width tuple $(4, 7, 7)$ and activation degree $r_1 = 2$ has **exactly $8$** inequivalent representations modulo hidden-neuron permutation and scaling, over both $\mathbb{C}$ and $\mathbb{R}$. The underlying mechanism is the classical Cayley octad: three independent quadrics through $7$ general points in $\mathbb{P}^3$ determine a unique $8$th base point, yielding $\binom{8}{7} = 8$ square bases for the $7$-dimensional output quadric space.
- **Lift to arbitrary depth $L \ge 3$ (Theorem 2):** In the $L$-layer architecture with widths $(4, 7, \dots, 7)$ and all activation degrees $r_j = 2$, every reduced two-layer block in the Localization Theorem (`thm:localization_theorem`) has format $((4, 7, 7), 2)$, which is finitely identifiable by Theorem 1. Hence the hypotheses of the Localization Theorem hold, the generic deep fiber is **finite**, and every generic network in $(4, 7, \dots, 7)$ admits **at least $8$** inequivalent representations over both $\mathbb{R}$ and $\mathbb{C}$ (exact count $8$ proved for $L = 2$; finite and $\ge 8$ for $L \ge 3$).
- **Scope note:** A stronger variant of the conjecture assuming *global* (rather than finite) identifiability of every reduced two-layer block lies outside this counterexample. We also verify the width-two formats $(2,2,2)$ and $(2,2,2,2)$ at degree $2$ (both globally identifiable) and record a minor typo in Appendix E of arXiv:2506.17093 v3, which prints $\mathbf{d} = (2,2,2), \mathbf{r} = (2,2)$ in place of $\mathbf{d} = (2,2,2,2), \mathbf{r} = (2,2)$ for Example `ex:no_localization_uniqueness`.

---

## 1. Problem Statement and Reduction of the Two-Layer Fiber

An $L$-layer bias-free homogeneous polynomial neural network (hPNN) of architecture $(\mathbf{d}, \mathbf{r}) = ((d_0, \dots, d_L), (r_1, \dots, r_{L-1}))$ computes
$$
F_W = W_L \circ \rho_{r_{L-1}} \circ W_{L-1} \circ \cdots \circ \rho_{r_1} \circ W_1,
$$
where $W_j \in \mathbb{K}^{d_j \times d_{j-1}}$ ($\mathbb{K} \in \{\mathbb{R}, \mathbb{C}\}$) and $\rho_r$ raises coordinates to the $r$th power. Two weight tuples are equivalent if they differ by hidden-neuron permutations and coordinatewise rescalings multiplying incoming row $i$ of $W_j$ by $s \neq 0$ and outgoing column $i$ of $W_{j+1}$ by $s^{-r_j}$. An architecture is *finitely identifiable* if a generic weight tuple has finitely many equivalence classes in its fiber, and *globally identifiable* if a generic weight tuple has a unique equivalence class.

The paper's **Localization Theorem** (`thm:localization_theorem`) proves that if each reduced two-layer architecture $((\tilde{d}_{\ell-1}, d_\ell, d_{\ell+1}), r_\ell)$ with $\tilde{d}_{\ell-1} := \min_{0 \le i < \ell} d_i$ is finitely identifiable for all $\ell = 1, \dots, L-1$, then the $L$-layer architecture $(\mathbf{d}, \mathbf{r})$ is finitely identifiable. **Conjecture `conj:global_identifiability`** proposes that under the exact same assumptions, the $L$-layer hPNN is *globally* identifiable. (The paper's separate Kruskal criterion, Proposition `cor:gen_generic`, requires $r \ge 7/3$ on $(4,7,7)$, so finite identifiability at $r = 2$ must be proved directly.)

For a two-layer $(4, 7, 7)$ network of degree $2$, write $F(x) = B(\ell_1(x)^2, \dots, \ell_7(x)^2)^\top$, where $\ell_i(x) = a_i \cdot x$ ($a_i \in \mathbb{K}^4$) and $B \in \mathbb{K}^{7 \times 7}$. When $\det B \neq 0$ and the seven squares $\ell_1^2, \dots, \ell_7^2$ are linearly independent, the span $S \subset \operatorname{Sym}^2(\mathbb{K}^4)$ of the seven output quadrics has dimension $7$. Any other representation $F(x) = \tilde{B}(\tilde{\ell}_1(x)^2, \dots, \tilde{\ell}_7(x)^2)^\top$ (even if initially allowed to have degenerate weights) must have $\operatorname{span}\{\tilde{\ell}_1^2, \dots, \tilde{\ell}_7^2\} \supseteq S$ and dimension $\le 7$, hence must form a basis of $S$ consisting of seven squares of linear forms. Once the ordered linear forms $\tilde{\ell}_1, \dots, \tilde{\ell}_7$ are fixed, $\tilde{B}$ is uniquely determined as the change-of-basis matrix. Hence the fiber modulo equivalence is in bijection with **unordered $7$-element sets of projective points $[a] \in \mathbb{P}^3$ whose squares $\ell_a^2$ form a basis of $S$**.

Represent $\ell_a^2$ in the $10$-dimensional monomial basis $\{x_i^2, 2x_i x_j : 0 \le i < j \le 3\}$ by its quadratic Veronese vector $v(a) = (a_i a_j)_{0 \le i \le j \le 3} \in \mathbb{K}^{10}$. A linear functional on $\mathbb{K}^{10}$ evaluates on $v(a)$ as a homogeneous quadric $Q(a)$ in $a$. Therefore,
$$
\ell_a^2 \in S \quad \Longleftrightarrow \quad Q(a) = 0 \text{ for every quadric } Q \text{ vanishing on } [a_1], \dots, [a_7].
$$

---

## 2. Exact Two-Layer Fiber Count via Cayley Octads

By classical algebraic geometry (Plaumann, Sturmfels, and Vinzant, 2011, Section 7), the base locus of the net of quadrics through $7$ general points in $\mathbb{P}^3$ consists of $8$ points—a *Cayley octad*. We give a complete, self-contained proof of the rank, transversality, genericity, and reality properties needed for the exact fiber count.

**Theorem 1 (Exact fiber size $8$ for $(4, 7, 7)$ at degree $2$).** *There is a nonempty Zariski-open subset $\mathcal{U}$ of the weight space for architecture $((4, 7, 7), 2)$ (whose complement has real Lebesgue measure zero) on which every network has **exactly $8$** inequivalent representations over $\mathbb{C}$. Moreover, for every real weight tuple in $\mathcal{U} \cap \mathbb{R}^{4 \times 7 + 7 \times 7}$, all $8$ equivalence classes are realized by real weight tuples.*

*Proof.* **Step 1 (Cube witness).** Consider the $8$ cube vertices $p_\epsilon = [1 : \epsilon_1 : \epsilon_2 : \epsilon_3] \in \mathbb{P}^3$ for $\epsilon = (\epsilon_1, \epsilon_2, \epsilon_3) \in \{\pm 1\}^3$. On these $8$ points, the $10$ columns of the Veronese matrix $V$ (rows $v(p_\epsilon)$) evaluate to $1$ (on all four diagonal entries $a_k^2$) and to the six nonconstant sign characters $\epsilon_1, \epsilon_2, \epsilon_3, \epsilon_1\epsilon_2, \epsilon_1\epsilon_3, \epsilon_2\epsilon_3$. These $7$ characters are mutually orthogonal in $\mathbb{R}^8$, so $\operatorname{rank}(V) = 7$ and its left kernel is spanned by the single missing character $\epsilon_1\epsilon_2\epsilon_3$, giving the exact identity
$$
\sum_{\epsilon \in \{\pm 1\}^3} \epsilon_1\epsilon_2\epsilon_3\, (x_0 + \epsilon_1 x_1 + \epsilon_2 x_2 + \epsilon_3 x_3)^2 = 0.
$$
Because every coefficient $\epsilon_1\epsilon_2\epsilon_3 \in \{\pm 1\}$ is nonzero, deleting *any* one of the $8$ vertices leaves $7$ linearly independent Veronese rows. Hence the space of quadrics vanishing on any $7$ cube vertices has dimension $10 - 7 = 3$, spanned by $Q_j(a) = a_j^2 - a_0^2$ ($j = 1, 2, 3$). Their common projective zero set consists precisely of the $8$ cube vertices, all in the affine chart $a_0 = 1$, where the Jacobian $\partial(Q_1, Q_2, Q_3)/\partial(a_1, a_2, a_3) = \operatorname{diag}(2\epsilon_1, 2\epsilon_2, 2\epsilon_3)$ has full rank $3$ at all $8$ points.

**Step 2 (Genericity over $\mathbb{C}$).** Work on the irreducible parameter space $(\mathbb{P}^3)^7$. The condition that the $7 \times 10$ Veronese matrix $V(a_1, \dots, a_7)$ has rank $7$ is Zariski-open and nonempty (witnessed by any $7$ cube vertices). On each $7 \times 7$ minor chart containing the witness, linear elimination gives a basis $(Q_1, Q_2, Q_3)$ of the annihilating quadrics with regular coefficients. Inside the incidence variety $\mathcal{Z} = \{((a_1, \dots, a_7), [a]) : Q_1(a) = Q_2(a) = Q_3(a) = 0\} \subset U_0 \times \mathbb{P}^3$, the locus where the $3 \times 4$ Jacobian of $(Q_1, Q_2, Q_3)$ in $a$ has rank $< 3$ is Zariski-closed; since $\mathbb{P}^3$ is projective, its projection to $U_0$ is Zariski-closed. Removing this closed locus leaves a nonempty Zariski-open set $U_1 \subset (\mathbb{P}^3)^7$ containing the witness, on which every common zero of $(Q_1, Q_2, Q_3)$ is transverse and isolated. By projective Bézout's theorem, every fiber of $\mathcal{Z}|_{U_1} \to U_1$ consists of exactly $2^3 = 8$ distinct points.

Because $\mathcal{Z}|_{U_1} \to U_1$ is projective with finite fibers and has full relative Jacobian rank everywhere, it is a finite étale cover of degree $8$. The $7$ tautological sections $s_i(a_1, \dots, a_7) = [a_i]$ are disjoint open-and-closed images in $\mathcal{Z}|_{U_1}$. Their complement is therefore a degree-$1$ finite étale cover of $U_1$, i.e., an algebraic section $s_8 : U_1 \to \mathbb{P}^3$ supplying the residual $8$th point $[a_8]$. For each $k \in \{1, \dots, 8\}$, the condition that the $7$ Veronese rows $\{v(s_j)\}_{j \neq k}$ have rank $7$ is open on $U_1$ and holds at the cube witness; intersecting these $8$ open sets yields a nonempty Zariski-open set $U_2 \subset (\mathbb{P}^3)^7$. Pulling $U_2$ back to nonzero first-layer rows and intersecting with $\{\det B \neq 0\}$ gives the nonempty Zariski-open set $\mathcal{U}$ in weight space.

**Step 3 (Exact count over $\mathbb{C}$ and $\mathbb{R}$).** On $\mathcal{U}$, there are precisely $8$ projective points $[a_1], \dots, [a_8]$ with $\ell_{a_j}^2 \in S$, and every $7$-element subset of them spans $S$. Distinct omitted indices yield distinct unordered $7$-sets of projective first-layer rows, which are invariant under hidden-neuron permutations and rescalings; conversely, fixing a $7$-set fixes the network uniquely up to those exact symmetries. Hence the complex fiber has size exactly $\binom{8}{7} = 8$.

If the starting weights $(W_1, B)$ lie in $\mathcal{U}$ and are real, complex conjugation preserves the real quadrics $Q_1, Q_2, Q_3$ and hence permutes the $8$-point base locus $\{[a_1], \dots, [a_8]\}$. Because it fixes $[a_1], \dots, [a_7]$, it must also fix the $8$th point $[a_8]$. Thus $[a_8]$ admits a real homogeneous representative $a_8 \in \mathbb{R}^4 \setminus \{0\}$, and the unique change-of-basis matrices between the $8$ real square bases of $S$ are real. Finally, the complement of $\mathcal{U}$ is a proper algebraic subset of real weight space and hence has Lebesgue measure zero. $\square$

---

## 3. Lift to Arbitrary Depth $L \ge 3$

**Theorem 2 (Counterexample at every depth $L \ge 3$).** *For every $L \ge 3$, consider the $L$-layer hPNN architecture with width tuple $\mathbf{d} = (4, 7, \dots, 7) \in \mathbb{Z}^{L+1}$ and degree tuple $\mathbf{r} = (2, \dots, 2) \in \mathbb{Z}^{L-1}$.*
1. *Every reduced two-layer block $((\tilde{d}_{\ell-1}, d_\ell, d_{\ell+1}), r_\ell)$ ($\ell = 1, \dots, L-1$) equals $((4, 7, 7), 2)$, which is finitely identifiable by Theorem 1. Hence the architecture satisfies all hypotheses of the Localization Theorem (`thm:localization_theorem`), and its generic fiber is **finite**.*
2. *Over both $\mathbb{R}$ and $\mathbb{C}$, every generic weight tuple $(W_1, W_2, \dots, W_L)$ in this architecture admits **at least $8$** inequivalent representations. In particular, the architecture is not globally identifiable.*

*Proof.* Since $d_0 = 4$ and $d_1 = \cdots = d_L = 7$, we have $\tilde{d}_{\ell-1} = \min_{0 \le i < \ell} d_i = 4$ for every $\ell = 1, \dots, L-1$, so every reduced block is $((4, 7, 7), 2)$, which is finitely identifiable by Theorem 1. The Localization Theorem therefore guarantees finite identifiability of the $L$-layer network.

Now let $(W_1, W_2, \dots, W_L)$ be any weight tuple whose first two layers $(W_1, W_2)$ lie in the Zariski-open set $\mathcal{U}$ of Theorem 1. By Theorem 1, there exist $8$ pairs $(W_1^{(k)}, W_2^{(k)})$ ($k = 1, \dots, 8$, all real when $W_1, W_2$ are real) with pairwise distinct unordered sets of projective rows of $W_1^{(k)}$ such that
$$
W_2^{(k)} \rho_2(W_1^{(k)} x) = W_2 \rho_2(W_1 x) \qquad (\forall x \in \mathbb{K}^4).
$$
Because these $8$ networks already agree identically on the seven preactivation quadrics entering the second activation, composing with $W_L \circ \rho_2 \circ \cdots \circ W_3 \circ \rho_2$ yields $8$ representations $(W_1^{(k)}, W_2^{(k)}, W_3, \dots, W_L)$ of the same $L$-layer map $F_W$. Equivalence of deep hPNNs preserves the unordered set of projective rows of the first weight matrix $W_1$, so these $8$ representations lie in pairwise distinct equivalence classes. $\square$

*(Remark: Theorem 1 establishes the **exact** fiber count $8$ for $L = 2$, while Theorem 2 establishes a **finite** generic fiber of cardinality **at least $8$** for every $L \ge 3$. Note also that recent work linking finite and global identifiability for additive/shallow decompositions—Casarotti and Mella (2022) and Massarenti and Mella (`arXiv:2511.19703`)—explicitly excludes exceptional non-identifiable families such as the $7$-quadric case in $\mathbb{P}^3$.)*

---

## 4. Width-Two Control Checks and Appendix E Typo

To separate the generic obstruction in $(4,7,7)$ from lower-dimensional special-locus examples in the paper, we also record the exact behavior of width-two quadratic networks:

- **Format $((2, 2, 2), 2)$:** Let $u, v$ be independent first-layer linear forms. In $\operatorname{span}\{u^2, v^2\}$, any square $(\alpha u + \beta v)^2 = \alpha^2 u^2 + 2\alpha\beta uv + \beta^2 v^2$ must satisfy $\alpha\beta = 0$, leaving only $u^2$ and $v^2$ up to scale. Thus $((2, 2, 2), 2)$ is globally identifiable.
- **Format $((2, 2, 2, 2), (2, 2))$:** Let $q_1 = a u^2 + b v^2$ and $q_2 = c u^2 + d v^2$ with invertible weight matrices, so the output quartics span $\langle q_1^2, q_2^2 \rangle$. Any square $(Au^2 + Buv + Cv^2)^2$ in this span has vanishing $u^3 v$ and $uv^3$ coefficients, forcing $AB = BC = 0$. If $B = 0$, treating $u^2, v^2$ as independent variables shows $(Au^2 + Cv^2)^2 \in \langle q_1^2, q_2^2 \rangle$ holds only for scalar multiples of $q_1$ or $q_2$. The only remaining candidate is $(uv)^2$, which lies in $\langle q_1^2, q_2^2 \rangle$ if and only if $a^2 d^2 - b^2 c^2 = 0$. Generically $a^2 d^2 - b^2 c^2 \neq 0$, so $q_1, q_2$ and hence $u, v$ are uniquely recovered up to equivalence: $((2,2,2,2), (2,2))$ is globally identifiable.
- **Appendix E note:** Example `ex:no_localization_uniqueness` in arXiv:2506.17093 v3 uses $(a, b, c, d) = (1, 1, 1, -1)$, which sits on the exceptional locus $a^2 d^2 - b^2 c^2 = 0$ where $(uv)^2$ enters the pencil. In Appendix E (`app-new-E-proof-update.tex`), the text introducing this example prints $\mathbf{d} = (2,2,2), \mathbf{r} = (2,2)$ as a minor typographical error for the three-layer architecture $\mathbf{d} = (2,2,2,2), \mathbf{r} = (2,2)$.

---

## 5. Verification and References

- **Exact SymPy verification** ([`solutions/deep_poly_ident/explore.py`](explore.py), rational arithmetic over $\mathbb{Q}$):
  - Starting directly from the output coefficient matrix, computes annihilating quadrics and solves them by lexicographic Gröbner elimination across disjoint affine charts covering $\mathbb{P}^3$.
  - On the cube witness and four seeded integer $(4, 7, 7)$ configurations, finds all $8$ rational projective square directions in every case, reconstructs all $\binom{8}{7} = 8$ exact bases per configuration, and checks all $40$ reconstructed coefficient matrices against their targets.
  - Independently expands the full quartic output of a $(4, 7, 7, 7)$ cube network and verifies exact polynomial equality across all $8$ deep representations.
  - Verifies the width-two controls: $2$ square directions in $(2,2,2)$, $2$ square members in the generic $(2,2,2,2)$ quartic pencil $(1,2,3,4)$, and $3$ square members in the exceptional pencil $(1,1,1,-1)$.
- **References:**
  - D. Plaumann, B. Sturmfels, and C. Vinzant (2011), *Quartic curves and their bitangents*, Journal of Symbolic Computation 46(6):712–733 (Section 7).
  - A. Casarotti and M. Mella (2022), *From non-defectivity to identifiability*, Journal of the European Mathematical Society 25(3):931–945.
  - A. Massarenti and M. Mella (2025), *Bronowski's conjecture and the identifiability of projective varieties*, [arXiv:2511.19703](https://arxiv.org/abs/2511.19703).
