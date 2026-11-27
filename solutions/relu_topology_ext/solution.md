# ReLU Bottleneck Topology at Arbitrary Output Width and for Tori

**Context:** Extends our proof of Beshkov's Tight Topological Expressivity Conjecture ([arXiv:2610.13931](https://arxiv.org/abs/2610.13931); [`solutions/relu_topology/solution.md`](../relu_topology/solution.md)) from $(d, d, d+1)$ to arbitrary output width $(d, d, m)$ and to $k$-tori $T^k$ ($k \ge 2$).  
**Companion files:** [`lean_proofs/Conjectures/ReluTopology.lean`](../../lean_proofs/Conjectures/ReluTopology.lean), [`solutions/relu_topology_ext/explore.py`](explore.py), [`solutions/relu_topology_ext/proof.pdf`](proof.pdf).

---

## Summary of Results

Write $g(t) := (Wt + b)_+$ for an affine ReLU readout with $W \in \mathbb{R}^{m \times d}$ and $b \in \mathbb{R}^m$.

1. **Theorem A1 (Contractibility of clipped polyhedra, arbitrary output width $m$):**
   For every nonempty convex polyhedron $D \subseteq \mathbb{R}^d$ (compact or unbounded) and every $W \in \mathbb{R}^{m \times d}, b \in \mathbb{R}^m$, the image $g(D)$ is **contractible**.
   - In particular, for any one-hidden-layer $(d, d, m)$ network on input $\mathbb{R}^d$ with invertible $W_1$, $h(\mathbb{R}^d) = \mathbb{R}_{\ge 0}^d$ is a convex polyhedron, so the output $g(\mathbb{R}_{\ge 0}^d)$ is contractible for every $m$ — closing Remark 1 of [`solutions/relu_topology/solution.md`](../relu_topology/solution.md). (Rank-deficient $W_1$ can produce lower-dimensional spheres such as $S^1$ for $(2,2,3)$; compact domains $X$ are not covered by A1 because $h(X)$ need not be convex.)

2. **Theorem A2 (Bi-Lipschitz linear embedding back through the bottleneck, arbitrary $m$):**
   For every $W \in \mathbb{R}^{m \times d}$ and $b \in \mathbb{R}^m$, the single linear map $W^\top : \mathbb{R}^m \to \mathbb{R}^d$ restricts to a **bi-Lipschitz homeomorphism** from the entire clipped affine $d$-plane $g(\mathbb{R}^d)$ onto the explicit closed convex polyhedral cone $W^\top \mathbb{R}_{\ge 0}^m \subseteq \mathbb{R}^d$ (independently of $b$).
   Consequently, for **any** topological space $X$, **any** hidden map $h : X \to \mathbb{R}^d$, and **any** output width $m$, setting $Y = g(h(X))$:
   - $\check{H}^d(Y; G) = 0$ for every abelian group $G$ ($d \ge 1$).
   - $Y$ cannot contain an embedded $S^d$, cannot be homotopy equivalent to $S^d$ or $T^d$, and cannot factor a nonzero-degree map $S^d \to Y \to S^d$. This subsumes Part B of our first paper at *every* output width $m$ and closes its Remark 3.
   - If $h$ is a ReLU network on $\mathbb{R}^d$ or on any compact convex domain $X \subseteq \mathbb{R}^d$, then $W^\top(Y) \subseteq \mathbb{R}^d$ is an ENR, so singular homology also vanishes: $H_d(Y; G) = 0$.
   - *Sharpness for singular homology (wild non-contractible domain):* Taking $X$ to be the Barratt–Milnor shrinking bouquet of $2$-spheres translated into the positive octant $\mathbb{R}_{>0}^3$, with $h(x) = (I_3 x)_+ = x$ (a genuine $(3,3,m)$ ReLU network for any $m \ge 3$), yields a compact output $Y$ with $\check{H}^3(Y) = 0$ but nonzero singular $H_3(Y)$ (Barratt–Milnor 1962). Note that $X$ is not contractible, so Beshkov's compact-contractible domain setting remains unaffected.

3. **Theorem B1 (Three hidden units and three outputs exclude $T^2$, on $\mathbb{R}^2$ and on any compact-image domain):**
   For any $k \ge 2$ and any one-hidden-layer $(k, k+1, k+1)$ ReLU network $f$ (arbitrary weights, possibly degenerate), if either the domain is $\mathbb{R}^k$ or the image $Y = f(X)$ is compact (including every compact contractible domain $X$ in Beshkov's framework), then $Y$ cannot have the homotopy type of $T^k$ (or any aspherical space with nonzero $H_k$ or $\check{H}^k$). In particular, $(2, 3, 3)$ cannot realize $T^2$.

4. **Theorem B2 (Arbitrary output width $(k, k+1, m)$ when constant outside a compact set):**
   For any $k \ge 2$ and any $m \ge 1$, a one-hidden-layer $(k, k+1, m)$ ReLU network $f : \mathbb{R}^k \to \mathbb{R}^m$ that is constant outside a compact set cannot have image homotopy equivalent to $T^k$ (or any aspherical space with nonzero $H_k$).

5. **Open boundary for $T^k$:** Combining A2 (lower bound $k+1$, valid for any earlier layers and any domain) with the product of $k$ copies of our minimal $(1,2,2)$ circle network (upper bound $2k$, improving Beshkov's $(2,16,7)$ and $(2,8,4)$ bounds for $T^2$ to $(2,4,4)$) gives $k+1 \le w_{\min}(T^k) \le 2k$. For $T^2$ with one hidden layer, B1 rules out $(2,3,3)$ (on $\mathbb{R}^2$ and on all compact domains) and B2 rules out compactly supported $(2,3,m)$ for all $m$; **unrestricted $(2, 3, m)$ with $m \ge 4$** (nonconstant outside compact sets, e.g., on unbounded strips) remains open.

---

## 1. Proof of Theorem A1 (Contractibility of Clipped Polyhedra)

Let $D \subseteq \mathbb{R}^d$ be a nonempty convex polyhedron and $g(t) = (Wt + b)_+$.

- **Case 1 ($D$ compact):** $g : D \to g(D)$ is a proper surjective piecewise-linear map between compact polyhedra. For every $y \in g(D)$, the fiber
  $$
  g^{-1}(y) = \bigl\{t \in D : (Wt+b)_i = y_i \text{ if } y_i > 0,\ (Wt+b)_i \le 0 \text{ if } y_i = 0\bigr\}
  $$
  is a nonempty compact convex polytope, hence $\mathrm{LC}^\infty$ and $\infty$-connected. By the Vietoris–Smale theorem (Smale, *Proc. AMS* 8 (1957), 604–610), $g_* : \pi_r(D) \to \pi_r(g(D))$ is an isomorphism for all $r \ge 0$, so $g(D)$ is weakly contractible. Since $g(D)$ is a compact polyhedron (hence a CW complex), Whitehead's theorem yields contractibility.
- **Case 2 ($D$ unbounded):** Partition $D$ into finitely many closed convex polyhedral activation cells $C_\sigma = \{t \in D : \sigma_i (Wt+b)_i \ge 0\}$ ($\sigma \in \{-,+\}^m$). Given any compact $K \subseteq g(D)$, pick a closed box $Q \subset \mathbb{R}^m$ containing $K$. Because each $g|_{C_\sigma}$ is affine, $g(C_\sigma)$ is a closed convex polyhedron (by Minkowski–Weyl), so $g(C_\sigma) \cap Q$ is a compact convex polytope. For each nonempty $g(C_\sigma) \cap Q$, pick one preimage vertex in $C_\sigma$ for each vertex of $g(C_\sigma) \cap Q$, and let $D' \subseteq D$ be the convex hull of all chosen lifts across all $\sigma$. Then $D'$ is a compact convex polytope with $K \subseteq g(D') \subseteq g(D)$. By Case 1, $g(D')$ is contractible, so every map $S^r \to g(D)$ is nullhomotopic in $g(D)$ (its image is compact). Since $g(D) = \bigcup_\sigma g(C_\sigma)$ is a connected finite union of closed polyhedra, it admits a locally finite triangulation (hence is a CW complex), so Whitehead's theorem implies $g(D)$ is contractible. $\square$

*(For $D = \mathbb{R}^d$, Theorem A2 gives a one-line alternative: $g(\mathbb{R}^d)$ is homeomorphic to the convex cone $W^\top \mathbb{R}_{\ge 0}^m$.)*

---

## 2. Proof of Theorem A2 (Bi-Lipschitz Embedding Through the Bottleneck)

**Step 1 (Firm nonexpansiveness and global injectivity of $W^\top$).**
Scalar ReLU $r(u) = \max(0, u)$ is $1$-Lipschitz and monotone, hence firmly nonexpansive: $(r(u) - r(v))^2 \le (r(u) - r(v))(u - v)$ for all $u, v \in \mathbb{R}$. Summing over coordinates for $y = g(s)$ and $z = g(t)$ gives
$$
\|y - z\|_2^2 \le \langle y - z,\, W(s - t) \rangle = \langle W^\top(y - z),\, s - t \rangle.
$$
In particular, $W^\top y = W^\top z \implies \|y - z\|_2^2 \le 0 \implies y = z$, so $W^\top$ is globally injective on $g(\mathbb{R}^d)$.

**Step 2 (Exact cone image $W^\top g(\mathbb{R}^d) = W^\top \mathbb{R}_{\ge 0}^m$, independent of $b$).**
Since $g(\mathbb{R}^d) \subseteq \mathbb{R}_{\ge 0}^m$, we immediately have $W^\top g(\mathbb{R}^d) \subseteq W^\top \mathbb{R}_{\ge 0}^m$. Conversely, fix any $q \in W^\top \mathbb{R}_{\ge 0}^m$. The feasible set $\{z \in \mathbb{R}_{\ge 0}^m : W^\top z = q\}$ is nonempty, closed, and convex, so the strictly convex quadratic program
$$
\min_{z \in \mathbb{R}^m} \tfrac{1}{2}\|z - b\|_2^2 \quad \text{subject to } z \ge 0,\ W^\top z = q
$$
has a unique minimizer $z^*$. Because all constraints are linear, KKT conditions hold without constraint qualification: there exist multipliers $t \in \mathbb{R}^d$ and $u \in \mathbb{R}_{\ge 0}^m$ such that $z^* - b = Wt + u$ and $u_i z^*_i = 0$ for all $i$. For every $i$: if $z^*_i > 0$ then $u_i = 0$ so $(Wt+b)_i = z^*_i > 0$; if $z^*_i = 0$ then $(Wt+b)_i = -u_i \le 0$. Thus $z^* = (Wt+b)_+ = g(t)$ and $W^\top g(t) = W^\top z^* = q$.

**Step 3 (Bi-Lipschitz inverse).**
On each closed activation cell $C_\sigma$, $g(t) = D_\sigma(Wt + b)$ where $D_\sigma = \operatorname{diag}(\mathbf{1}_{\sigma_i = +})$ satisfies $D_\sigma^2 = D_\sigma$. Because $W^\top D_\sigma W = (D_\sigma W)^\top (D_\sigma W)$, we have $\ker(W^\top) \cap \operatorname{range}(D_\sigma W) = \{0\}$. Hence on each closed polyhedral piece $P_\sigma = g(C_\sigma)$, $W^\top$ is an injective affine map with a positive minimum singular value $s_\sigma > 0$ on the direction space $\operatorname{range}(D_\sigma W)$, so $W^\top(P_\sigma)$ is a closed convex polyhedron and $(W^\top|_{P_\sigma})^{-1}$ is $(1/s_\sigma)$-Lipschitz. Because $W^\top$ is globally injective on $g(\mathbb{R}^d)$, these finitely many piecewise-affine inverses agree on overlaps and paste (by the finite closed pasting lemma) to a continuous piecewise-affine map on the convex cone $W^\top \mathbb{R}_{\ge 0}^m$, which is therefore globally Lipschitz with constant at most $\max_\sigma s_\sigma^{-1}$. Thus $W^\top : g(\mathbb{R}^d) \to W^\top \mathbb{R}_{\ge 0}^m$ is a bi-Lipschitz homeomorphism, and restricts to an embedding on every subset $Y = g(h(X))$.

**Step 4 (Cohomological and homological consequences).**
For any subset $Y \subseteq g(\mathbb{R}^d)$, embed $Y \cong Y' := W^\top(Y) \subseteq \mathbb{R}^d$.
- Every subset of a metric space is taut for Alexander–Spanier / Čech cohomology (Spanier, *Pacific J. Math.* 75 (1978), p. 562, Theorem 1(3)), so $\check{H}^d(Y; G) \cong \varinjlim_{U \supseteq Y'} H^d(U; G)$ over open neighborhoods $U \subseteq \mathbb{R}^d$. Each connected component of $U$ is an orientable noncompact $d$-manifold, whose degree-$d$ cohomology vanishes for every $G$; hence $\check{H}^d(Y; G) = 0$. (For compact $Y$, Alexander duality in $S^d$ directly gives $\check{H}^d(Y; G) = 0$.)
- By Čech functoriality, $Y$ admits no embedded $S^d$, no homotopy equivalence to $S^d$ or $T^d$, and no nonzero-degree factorization $S^d \to Y \to S^d$.
- If $h$ is a ReLU network on $\mathbb{R}^d$ or on a compact convex domain $X \subseteq \mathbb{R}^d$, then $Y'$ is a finite union of closed convex sets (polyhedra on $\mathbb{R}^d$, compact convex sets on compact convex $X$), hence closed and locally star-shaped in $\mathbb{R}^d$, hence a Euclidean neighborhood retract (ENR). Its singular homology $H_d(Y; G)$ injects into $H_d(U; G) = 0$ for a retract neighborhood $U \subseteq \mathbb{R}^d$. $\square$

---

## 3. Proof of Theorem B1 (Exclusion of $T^k$ for $(k, k+1, k+1)$)

Let $k \ge 2$, $n := k + 1$, and let $f(x) = (W_2 (W_1 x + b_1)_+ + b_2)_+$ be a one-hidden-layer $(k, n, n)$ ReLU network on a domain $X$, with image $Y = f(X)$. Suppose either $X = \mathbb{R}^k$ or $Y$ is compact (e.g., $X$ compact).

- **Degenerate $W_2$ ($\operatorname{rank} W_2 \le k$):** Factor $W_2 = \widetilde{W}_2 A$ with $\widetilde{W}_2 \in \mathbb{R}^{n \times k}$. By Theorem A2, $Y$ embeds into $\mathbb{R}^k$, so $\check{H}^k(Y; \mathbb{Z}) = 0$ (and $H_k(Y; \mathbb{Z}) = 0$ on $\mathbb{R}^k$), excluding $T^k$. (If the output layer is linear instead of ReLU, $W_2 h(X) + b_2$ either lies in a $k$-plane or is homeomorphic to $h(X) \subseteq R(H)$, which is contractible on $\mathbb{R}^k$ by A1 and embeds in $\mathbb{R}^k$ by A2.)
- **Invertible $W_2 \in \mathrm{GL}_n(\mathbb{R})$:** Since $W_1 \in \mathbb{R}^{n \times k}$ has rank at most $k$, the first-layer preactivations lie in an affine hyperplane $H = \{u \in \mathbb{R}^n : \nu \cdot u = \alpha\}$ ($\nu \neq 0$). Thus $h(X) \subseteq H_+ := R(H) \subseteq \mathbb{R}_{\ge 0}^n$, and $Y \subseteq (W_2 \mathbb{R}_{\ge 0}^n + b_2)_+$.
  - **Existence of a bounded complementary component $B$:** Suppose $Y \simeq T^k$ (more generally, $Y$ is aspherical with $H_k(Y; \mathbb{Z}) \neq 0$ on $\mathbb{R}^k$, or $\check{H}^k(Y; \mathbb{Z}) \neq 0$ when $Y$ is compact). When $Y$ is compact, Alexander duality in $S^n$ gives $\tilde{H}_0(S^n \setminus Y; \mathbb{Z}) \cong \check{H}^k(Y; \mathbb{Z}) \neq 0$, yielding a bounded connected component $B$ of $\mathbb{R}^n \setminus Y$. When $X = \mathbb{R}^k$ (possibly unbounded $Y$), $Y$ is a closed polyhedron of dimension $\le k$; in a compatible triangulation of $\mathbb{R}^n$, any nonzero finite simplicial $k$-cycle in $Y$ bounds a finite $n$-chain whose oriented multiplicity is locally constant off $Y$, vanishes outside a compact set, and is nonzero on some component $B$ of $\mathbb{R}^n \setminus Y$, which must therefore be bounded.
  - **Hidden-coordinate escape rays ($B \subseteq C := \{y > 0,\, a(y) > 0\}$):** Set $a(y) := W_2^{-1}(y - b_2)$. Every point $y \in \mathbb{R}^n$ with some $y_j \le 0$ escapes to $\infty$ in $\mathbb{R}^n \setminus Y$ along $y - t e_j$ ($t \ge 0$), so $B \subseteq \mathbb{R}_{>0}^n$. Now suppose $y \in \mathbb{R}_{>0}^n$ has $a_i(y) < 0$ for some $i$, and let $c^\top$ be row $i$ of $W_2^{-1}$:
    1. If some $c_j \le 0$, the ray $y + t e_j$ ($t \ge 0$) stays in $\mathbb{R}_{>0}^n$ (where ReLU has unique preimage $u = y + t e_j$) with $a_i(y + t e_j) = a_i(y) + t c_j < 0$, so it misses $(W_2 \mathbb{R}_{\ge 0}^n + b_2)_+ \supseteq Y$.
    2. If $c_j > 0$ for all $j$, pick any $j$ and take the ray $y - t e_j$ ($t \ge 0$). For $0 \le t < y_j$, $y - t e_j > 0$ has unique preimage with $a_i(y - t e_j) < a_i(y) < 0$. At $t = y_j$, every ReLU preimage $u$ of $y' = y - y_j e_j$ satisfies $u \le y'$ coordinatewise; since $c > 0$, $c^\top(u - b_2) \le c^\top(y' - b_2) = a_i(y) - y_j c_j < 0$, so $W_2^{-1}(u - b_2) \notin \mathbb{R}_{\ge 0}^n$. For $t > y_j$, the $j$-th coordinate is negative. Thus the entire ray misses $(W_2 \mathbb{R}_{\ge 0}^n + b_2)_+ \supseteq Y$.
    Finally, $a_i(y) = 0$ cannot hold inside open $B$ because $a_i$ is nonconstant affine, so any neighborhood would contain points with $a_i < 0$. Hence $B \subseteq C := \{y \in \mathbb{R}^n : y > 0,\ a(y) > 0\}$.
  - **Convexity of a chamber boundary in $Y$:** On $C$, both ReLUs are strictly active, so $Z := Y \cap C \subseteq S := \{y \in C : \nu \cdot a(y) = \alpha\}$. Write $C \setminus S = C_+ \sqcup C_-$ (two open convex sets).
    - If $S = \varnothing$, then $B = C$ is a bounded open convex polytope with $\partial B \subseteq Y$.
    - If $Z = S$ (in particular, whenever $X = \mathbb{R}^k$ with $\operatorname{rank} W_1 = k$), then $B$ equals $C_+$ or $C_-$, a bounded open convex polytope with $\partial B \subseteq Y$.
    - If $Z \subsetneq S$ (possible only on proper subsets $X \subsetneq \mathbb{R}^k$), since $Z$ is relatively closed in $C$, any point $p \in S \setminus Z$ has a ball in $C \setminus Z$ bridging $C_+$ and $C_-$, so $C \setminus Z$ is connected. Because $B$ is a connected component of $\mathbb{R}^n \setminus Y$ inside $C \setminus Z$, we have $B = C \setminus Z$. Since $S$ is nowhere dense in $C$, $C$ itself must be bounded (otherwise $C \setminus S \subseteq B$ would be unbounded), and every $q \in \partial C$ lies in $\overline{B} \setminus B = \partial B \subseteq Y$.
  - **Asphericity contradiction:** In every case, there is a bounded open convex polytope $P \supseteq B$ (either $B$ or $C$) with $\partial P \subseteq Y$. Pick $p_0 \in B \subset P$. Radial projection from $p_0$ gives a continuous map $\rho : Y \to S^k$ whose restriction to $\partial P \cong S^k$ is a homeomorphism of degree $\pm 1$. Hence the sphere inclusion $S^k \cong \partial P \hookrightarrow Y$ induces a nonzero map on $H_k(-; \mathbb{Z})$. For $k \ge 2$, any aspherical space (such as $T^k$, which has $\pi_k(T^k) = 0$) has all spherical classes in $H_k$ equal to zero — contradiction. $\square$

---

## 4. Proof of Theorem B2 (Arbitrary Output Width, Constant Outside a Compact Set)

Let $k \ge 2$, $m \ge 1$, and let $f(x) = (W_2 (W_1 x + b_1)_+ + b_2)_+ : \mathbb{R}^k \to \mathbb{R}^m$ be a one-hidden-layer $(k, k+1, m)$ ReLU network that is constant outside a compact set (arbitrary weights, possibly degenerate). Let $Y = f(\mathbb{R}^k)$.

For $y \in Y$, let $r(y)$ be the rank of the rows of $W_2$ indexed by $\{i : y_i > 0\}$, and set $L := \{y \in Y : r(y) \le k\}$. Since $r$ is lower semicontinuous, $L$ is a closed subpolyhedron of $Y$. Under the embedding $W_2^\top : Y \hookrightarrow \mathbb{R}^{k+1}$ of Theorem A2, every $y \in L$ satisfies $W_2^\top y = \sum_{i : y_i > 0} y_i w_i$, which lies in a finite union of linear subspaces of dimension $\le k$ in $\mathbb{R}^{k+1}$. That union $U_L$ is a radially contractible polyhedron of dimension $\le k$, so $H_k(W_2^\top L; \mathbb{Z}) = 0$ (since top-dimensional homology injects under subcomplex inclusion into a $k$-dimensional complex, using a common triangulation of $U_L$ and $W_2^\top L$). Hence $H_k(L; \mathbb{Z}) = 0$.

Suppose $H_k(Y; \mathbb{Z}) \neq 0$. Because $H_k(L; \mathbb{Z}) = 0$, a nonzero simplicial $k$-cycle in $Y$ must contain a $k$-simplex $\Delta$ whose interior lies in $Y \setminus L$. Pick a generic point $p \in \operatorname{int}(\Delta)$ avoiding the constant value at infinity and the images of all lower-dimensional activation faces and rank-deficient affine pieces of $f$ (finitely many closed polyhedra of dimension $\le k-1$).
- Because $p \notin L$, the active rows $\{i : p_i > 0\}$ of $W_2$ have full rank $k + 1$, so the hidden vector $h \in \mathbb{R}^{k+1}$ with $g(h) = p$ is unique.
- At any input preimage $x \in f^{-1}(p)$, $x$ lies in the interior of a full-rank affine piece of $f$ (by genericity), so the active rows of $W_1$ at $x$ have rank $k$. Since the unique hidden vector $h$ fixes those active coordinates, $x$ is unique!
- Thus $f^{-1}(p) = \{x_0\}$ is a singleton, and $f$ maps a neighborhood of $x_0$ affinely and homeomorphically onto a neighborhood of $p$ in $\operatorname{int}(\Delta)$.

Extending $f$ by its constant value at infinity gives a continuous map $\bar{f} : S^k \to Y$. By excision at the unique regular preimage $x_0$, $\bar{f}_* : H_k(S^k; \mathbb{Z}) \to H_k(Y, Y \setminus \{p\}; \mathbb{Z})$ maps the fundamental class $[S^k]$ nontrivially; hence $\bar{f}_*[S^k] \neq 0$ already in $H_k(Y; \mathbb{Z})$. This rules out the homotopy type of $T^k$ (or any aspherical space) for $k \ge 2$. $\square$

---

## 5. Verification, Formalization, and Attribution

- **Lean 4 formalization** ([`lean_proofs/Conjectures/ReluTopology.lean`](../../lean_proofs/Conjectures/ReluTopology.lean), Lean `v4.24.0` + Mathlib, zero `sorry`, standard axioms only): machine-checks ReLU firm nonexpansiveness (`relu_firmly_nonexpansive`, `reluVec_firmly_nonexpansive`), global injectivity of $W^\top$ on $g(\mathbb{R}^d)$ for arbitrary $d, m$ (`transpose_injective_on_relu_affine`), piece-wise injectivity via $W^\top D_\sigma W = (D_\sigma W)^\top (D_\sigma W)$ (`transpose_ker_inter_active_range`), KKT cone sufficiency (`kkt_implies_relu_preimage`), and the B1 hidden-coordinate escape ray theorem (`b1_hidden_escape_ray`).
- **Python verification** ([`solutions/relu_topology_ext/explore.py`](explore.py)): verifies 4 full-domain LP nerves for A1 (all returning $\beta = (1,0,0)$), 5,000 firm-nonexpansiveness pairs, 60 cone-surjectivity KKT/LP checks, 15 bi-Lipschitz constant checks for A2, $\ge 100$ LP-certified B1 escape rays, and the $(2,4,4)$ product torus triangulation ($\beta = (1,2,1)$).
- **Literature & attribution note:** Firm nonexpansiveness of ReLU (Combettes–Pesquet, [arXiv:1808.07526](https://arxiv.org/abs/1808.07526)) and parametric-QP Lipschitz continuity are classical; their topological application to arbitrary-width $(d,d,m)$ embeddings and torus impossibility was not found in Beshkov ([arXiv:2610.13931](https://arxiv.org/abs/2610.13931)), Ren–Lim ([arXiv:2606.31856](https://arxiv.org/abs/2606.31856)), or preregistration `trureturing#25155` (which proposed, without proof, contractibility of $\operatorname{ReLU}(H)$ in the hyperplane $m = d+1$ case of A1).
