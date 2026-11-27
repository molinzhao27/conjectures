# Partial Resolution of the Oriented-Matroid Characterization of Convex Neural Codes and Refutation of the Closed Variant

**Target paper:** Trong-Thuc Trang and R. Amzi Jeffs, *Covering Relations in the Poset of Combinatorial Neural Codes*, [arXiv:2512.04241](https://arxiv.org/abs/2512.04241) (Dec 2025; *Proc. Machine Learning Research*, NeurReps 2025, PMLR 282:225–242, 2026), Conjecture `conj:convex-char` (citing Alexander Kunin, Caitlin Lienkaemper, and Zvi Rosen, *Oriented Matroids and Combinatorial Neural Codes*, [arXiv:2002.03542](https://arxiv.org/abs/2002.03542), *Combinatorial Theory*, 2023).  
**Companion files:** [`solutions/neural_code_oriented_matroid/explore.py`](explore.py), [`solutions/neural_code_oriented_matroid/proof.pdf`](proof.pdf).

---

## Verdict

Let $\mathbf{P}_{\mathrm{Code}}$ be the poset of isomorphism classes of combinatorial neural codes (containing $\varnothing$) ordered by surjective trunk morphisms, and for a representable oriented matroid $\mathcal{M}$ realized by vectors $(v_e)_{e \in E}$ in $\mathbb{R}^r$, let $L^+(\mathcal{M}) = \{\{e \in E : v_e \cdot z > 0\} : z \in \mathbb{R}^r\}$ be its positive covector code. Conjecture `conj:convex-char` asserts that a neural code $\mathcal{C}$ is open convex if and only if $\mathcal{C} \le L^+(\mathcal{M})$ in $\mathbf{P}_{\mathrm{Code}}$ for some representable oriented matroid $\mathcal{M}$ (equivalently, by Kunin–Lienkaemper–Rosen, $\mathcal{C}$ is realizable by open convex polytopes in some Euclidean space).

- **Open-convex conjecture (`conj:convex-char`): STRONG PARTIAL POSITIVE RESOLUTION; general $d \ge 3$ converse remains OPEN.**
  1. **Reverse implication ($\mathcal{C} \le L^+(\mathcal{M}) \implies$ open convex):** Holds in every dimension, via pullback of strict linear halfspaces (Theorem 1).
  2. **Common convex function polyhedralization (Theorem 2 — new):** For any finite convex function $q : \mathbb{R}^d \to \mathbb{R}$, affine thresholds $\ell_1, \dots, \ell_N$, and auxiliary strict affine tests $g_1, \dots, g_M$, there exists a convex piecewise-affine minorant $p(x) = \max_{1 \le r \le R} s_r(x)$ ($R \le 2^{N+M}$) whose sublevel sets $\{p < \ell_i\}$ and $\{g_j > 0\}$ produce the exact same code in $\mathbb{R}^d$ (preserving all finite intersections simultaneously). Consequently, **every open Euclidean ball code** and **every homothetic ellipsoid code** $(x - c_i)^\top Q (x - c_i) < \rho_i$ with a common symmetric positive semidefinite matrix $Q \succeq 0$—including arbitrary tangencies, coincident fields, and finite intersections—admits a **same-dimensional open polytope realization** in every dimension $d \ge 1$, hence lies below a representable oriented matroid in $\mathbf{P}_{\mathrm{Code}}$ (Corollary 3).
  3. **Intermediate-word and transverse $C^1$ smooth sufficient conditions (Theorem 4 & Corollary 5):** If an open convex realization $\mathcal{U}$ in $\mathbb{R}^d$ satisfies $I(x) \subseteq \sigma \subseteq J(x) \implies \sigma \in \mathcal{C}$ at every $x \in \mathbb{R}^d$ (where $I(x) = \{i : x \in U_i\}$ and $J(x) = \{i : x \in \overline{U_i}\}$), then clipped inner polyhedral exhaustions give a same-dimensional open polytope realization. This covers all transverse $C^1$ smooth convex families (including arbitrarily oriented ellipsoids with distinct quadratic parts) and requires no smoothness. We also prove the exact logical relation to non-degeneracy of Cruz–Giusti–Itskov–Kronholm ([arXiv:1609.03502](https://arxiv.org/abs/1609.03502), *DCG* 2019): the intermediate-word condition implies CGIK non-degeneracy and is strictly stronger as a cover property, which is necessary to prevent extraneous intermediate words under independent inner polyhedral approximations.
  4. **Max-intersection-complete codes and 3-neuron classification (Theorem 6 & Corollary 7):** Every max-intersection-complete code admits an explicit open-polytope certificate in an open simplex, placing it below a representable oriented matroid. On $n = 3$ neurons, all $128$ labeled codes containing $\varnothing$ split cleanly into **109** open-polytope codes and **19** codes with a disconnected-field obstruction.
  5. **Remaining open frontier:** Whether every arbitrary open convex realization in dimension $d \ge 3$ admits an open polytope replacement (in the same or a higher dimension, up to $|\mathcal{C}| - 1$) remains open ($d = 1$ is immediate from intervals; $d = 2$ was proved by Bukh–Jeffs, [arXiv:2207.06290](https://arxiv.org/abs/2207.06290)).

- **Closed-convex reinterpretation: REFUTED IN BOTH DIRECTIONS (Theorem 8).**
  1. **Closed-convex $\not\implies \mathcal{C} \le L^+(\mathcal{M})$:** The 5-neuron, 12-word closed triangular fan $\mathcal{C}_5$ of Lienkaemper–Shiu–Woodstock ([arXiv:1509.03328](https://arxiv.org/abs/1509.03328), *Adv. Appl. Math.* 2017) is closed convex in $\mathbb{R}^2$ but admits no open convex realization in any dimension (hence cannot lie below any $L^+(\mathcal{M})$).
  2. **$\mathcal{C} \le L^+(\mathcal{M}) \not\implies$ closed-convex:** The 6-neuron, 13-word central positive-halfplane code $\mathcal{C}_6 = L^+(\mathcal{M}_6)$ (code (2) of Cruz–Giusti–Itskov–Kronholm, [arXiv:1609.03502](https://arxiv.org/abs/1609.03502), Lemma 2.9) is itself the positive covector code of a rank-2 representable oriented matroid, yet admits no closed convex realization in any dimension.

---

## 1. The Poset $\mathbf{P}_{\mathrm{Code}}$ and Equivalence with Open Polytopes

A *combinatorial neural code* on $[n] := \{1, \dots, n\}$ is a subset $\mathcal{C} \subseteq 2^{[n]}$ with $\varnothing \in \mathcal{C}$. Given a family $\mathcal{U} = (U_1, \dots, U_n)$ of subsets of $\mathbb{R}^d$, the codeword at $x \in \mathbb{R}^d$ is $c_{\mathcal{U}}(x) := \{i \in [n] : x \in U_i\}$, and $\operatorname{code}(\mathcal{U}) := \{c_{\mathcal{U}}(x) : x \in \mathbb{R}^d\}$ (including labels attained on lower-dimensional subsets). A code is *open convex* (resp. *closed convex*, *open polyhedral*, *open polytope*) if it equals $\operatorname{code}(\mathcal{U})$ for open convex (resp. closed convex, finite intersections of strict affine halfspaces, interiors of bounded convex polytopes) sets $U_1, \dots, U_n \subseteq \mathbb{R}^d$ for some $d \ge 1$. Because $\varnothing \in \mathcal{C}$, intersecting every $U_i$ with a single sufficiently large open box $(-R, R)^d$ (or closed box $[-R, R]^d$ for closed fields) containing one witness per codeword preserves $\operatorname{code}(\mathcal{U})$ and makes all fields bounded; thus open polyhedral and bounded open polytope realizability are identical.

For $\sigma \subseteq [n]$, the *trunk* of $\sigma$ in $\mathcal{C}$ is $T_{\mathcal{C}}(\sigma) := \{c \in \mathcal{C} : \sigma \subseteq c\}$. A trunk is *proper* if $\varnothing \subsetneq T_{\mathcal{C}}(\sigma) \subsetneq \mathcal{C}$. A function $f : \mathcal{C} \to \mathcal{D}$ is a *morphism* if $f^{-1}(T)$ is a proper trunk of $\mathcal{C}$ for every proper trunk $T$ of $\mathcal{D}$; in $\mathbf{P}_{\mathrm{Code}}$, $\mathcal{D} \le \mathcal{C}$ if there exists a surjective morphism $f : \mathcal{C} \to \mathcal{D}$. Since $\varnothing \in \mathcal{D}$, no neuron of $\mathcal{D}$ is always active; each active neuron $j$ has a proper simple trunk $T_{\mathcal{D}}(\{j\})$ pulling back to $T_{\mathcal{C}}(\tau_j)$ for some nonempty $\tau_j$, so output bit $j$ is the conjunction of input bits in $\tau_j$ (while silent neurons are identically $0$). Consequently $f(\varnothing) = \varnothing$, and intersecting pullbacks of simple trunks gives pullbacks of all proper trunks.

For a real vector configuration $(v_e)_{e \in E}$ in $\mathbb{R}^r$ representing an oriented matroid $\mathcal{M}$, the code in Conjecture `conj:convex-char` is
$$
L^+(\mathcal{M}) := \bigl\{\{e \in E : v_e \cdot z > 0\} : z \in \mathbb{R}^r\bigr\},
$$
consisting of the positive supports of **all covectors** of $\mathcal{M}$ (not merely topes/chambers; e.g., two opposite nonzero forms attain $\varnothing$ only on their common kernel).

**Theorem 1 (Kunin–Lienkaemper–Rosen, [arXiv:2002.03542](https://arxiv.org/abs/2002.03542), Observation 2.5 & Theorem 1).** *A neural code $\mathcal{C}$ (with $\varnothing \in \mathcal{C}$) satisfies $\mathcal{C} \le L^+(\mathcal{M})$ in $\mathbf{P}_{\mathrm{Code}}$ for some representable oriented matroid $\mathcal{M}$ if and only if $\mathcal{C}$ admits an open polytope realization in some $\mathbb{R}^d$. In particular, $\mathcal{C} \le L^+(\mathcal{M})$ always implies that $\mathcal{C}$ is open convex.*

*Proof.* ($\Rightarrow$) Let $f : L^+(\mathcal{M}) \to \mathcal{C}$ be a surjective morphism where $\mathcal{M}$ is realized by $(v_e)_{e \in E}$ in $\mathbb{R}^r$. For each active output neuron $j$, let $f^{-1}(T_{\mathcal{C}}(\{j\})) = T_{L^+(\mathcal{M})}(\tau_j)$ and set $U_j := \{z \in \mathbb{R}^r : v_e \cdot z > 0 \text{ for all } e \in \tau_j\}$; for silent $j$, set $U_j := \varnothing$. Then $c_{\mathcal{U}}(z) = f(\{e : v_e \cdot z > 0\})$, so $\operatorname{code}(\mathcal{U}) = \mathcal{C}$. Truncating by a large open box $(-R, R)^r$ yields bounded open polytopes.

($\Leftarrow$) Suppose $\mathcal{C} = \operatorname{code}(\mathcal{U})$ where each active field is $U_j = \{x \in \mathbb{R}^d : a_{j,k} \cdot x + b_{j,k} > 0 \text{ for } k = 1, \dots, m_j\}$ and silent fields are empty. In $\mathbb{R}^{d+1}$ with coordinates $(y, t)$, form the vector configuration consisting of all $(a_{j,k}, b_{j,k})$ together with the gate vector $e_{d+1} = (0, \dots, 0, 1)$. For each active neuron $j$, define $\tau_j$ to be all $(a_{j,k}, b_{j,k})$ for $j$ together with $e_{d+1}$. At any $(y, t)$ with $t > 0$, homogeneity gives $a_{j,k} \cdot y + b_{j,k} t > 0 \iff a_{j,k} \cdot (y/t) + b_{j,k} > 0$, producing $c_{\mathcal{U}}(y/t)$; at $t \le 0$, the gate fails and every conjunction outputs $0$, yielding $\varnothing \in \mathcal{C}$. Since each active $U_j$ is nonempty (witnessed at $(x, 1)$) and $t = 0$ at $z = 0$, every $T_{L^+(\mathcal{M})}(\tau_j)$ is proper, defining a surjective morphism $L^+(\mathcal{M}) \to \mathcal{C}$. $\square$

**Elementary dimension bound.** If $\mathcal{C}$ has $k$ codewords and admits an open convex (resp. open polytope) realization in $\mathbb{R}^D$, restricting all fields to the affine span of $k$ chosen witnesses preserves relative openness, convexity, finite strict-halfspace descriptions, and $\operatorname{code}(\mathcal{U})$, yielding a realization in dimension at most $k - 1$. In $d = 1$, bounded open convex sets are already open intervals; in $d = 2$, Bukh and Jeffs ([arXiv:2207.06290](https://arxiv.org/abs/2207.06290), Theorem 1) proved that every planar open convex code admits a same-dimensional open polygon realization with an explicit vertex bound. By contrast, the local covering rules of Trang–Jeffs ([arXiv:2512.04241](https://arxiv.org/abs/2512.04241)) drop the number of nonempty trunks by one per step and do not by themselves supply a termination argument above an arbitrary open convex code in $d \ge 3$.

---

## 2. Polyhedralization for a Common Convex Function

**Theorem 2 (Common convex function polyhedralization).** *Let $q : \mathbb{R}^d \to \mathbb{R}$ be finite-valued and convex, let $\ell_1, \dots, \ell_N : \mathbb{R}^d \to \mathbb{R}$ be affine functions, and let $g_1, \dots, g_M : \mathbb{R}^d \to \mathbb{R}$ be any (possibly empty) finite list of auxiliary affine functions. There exists a convex piecewise-affine function $p(x) = \max_{1 \le r \le R} s_r(x)$ of $R \le 2^{N+M}$ affine minorants $s_r \le q$ such that the families*
$$
\mathcal{U}_q = \bigl(\{x \in \mathbb{R}^d : q(x) < \ell_i(x)\}_{i=1}^N,\; \{x \in \mathbb{R}^d : g_j(x) > 0\}_{j=1}^M\bigr)
$$
*and*
$$
\mathcal{U}_p = \bigl(\{x \in \mathbb{R}^d : p(x) < \ell_i(x)\}_{i=1}^N,\; \{x \in \mathbb{R}^d : g_j(x) > 0\}_{j=1}^M\bigr)
$$
*have the exact same combinatorial neural code in $\mathbb{R}^d$. The replacement simultaneously preserves the code of every family obtained by taking fixed finite intersections (conjunctions) of the fields of $\mathcal{U}_q$, and each resulting replacement field is an intersection of at most $R \le 2^{N+M}$ strict affine halfspaces per $\ell_i$-threshold (before common box truncation).*

*Proof.* Work in $\mathbb{R}^{d+1}$ with coordinates $(x, t) \in \mathbb{R}^d \times \mathbb{R}$. Define the signature $\sigma(x, t) \in \{0, 1\}^{N+M}$ by recording the truth values of $t < \ell_i(x)$ ($1 \le i \le N$) and $g_j(x) > 0$ ($1 \le j \le M$). Each signature region $S_\sigma := \{(x, t) : \sigma(x, t) = \sigma\}$ is an intersection of finitely many open and closed halfspaces in $\mathbb{R}^{d+1}$, hence convex.

*Upward-shift property.* For every $(x, t) \in S_\sigma$, there exists $\varepsilon > 0$ such that $(x, t + \varepsilon) \in S_\sigma$: every active threshold satisfies $t < \ell_i(x)$ strictly (so $t + \varepsilon < \ell_i(x)$ for small $\varepsilon > 0$), every inactive threshold satisfies $t \ge \ell_i(x)$ (so $t + \varepsilon > \ell_i(x)$ for all $\varepsilon > 0$, even at ties $t = \ell_i(x)$), and $g_j(x)$ does not depend on $t$.

Let $\Gamma_q := \{(x, q(x)) : x \in \mathbb{R}^d\}$ and $\operatorname{epi}(q) := \{(x, t) : t \ge q(x)\}$. A signature $\sigma$ belongs to $\operatorname{code}(\mathcal{U}_q)$ if and only if $S_\sigma \cap \Gamma_q \neq \varnothing$. We select at most one affine minorant $s_\sigma \le q$ for each signature $\sigma \in \{0, 1\}^{N+M}$:

1. *Attained signatures ($\sigma \in \mathcal{A}$, i.e., $S_\sigma \cap \Gamma_q \neq \varnothing$).* Pick one witness $(x_\sigma, q(x_\sigma)) \in S_\sigma \cap \Gamma_q$. Because $q$ is finite and convex on all of $\mathbb{R}^d$, its subdifferential $\partial q(x_\sigma)$ is nonempty; choose $v_\sigma \in \partial q(x_\sigma)$ and set $s_\sigma(x) := q(x_\sigma) + v_\sigma \cdot (x - x_\sigma)$. Then $s_\sigma \le q$ everywhere and $s_\sigma(x_\sigma) = q(x_\sigma)$. Note $\mathcal{A} \neq \varnothing$ since $\mathbb{R}^d \neq \varnothing$.
2. *Nonempty missed signatures ($S_\sigma \neq \varnothing$, $S_\sigma \cap \Gamma_q = \varnothing$).* Since $q$ is continuous on $\mathbb{R}^d$, the continuous function $(x, t) \mapsto t - q(x)$ never vanishes on the convex (hence connected) set $S_\sigma$, so it has constant strict sign on $S_\sigma$:
   - If $t > q(x)$ on $S_\sigma$, select no minorant.
   - If $t < q(x)$ on $S_\sigma$ ($\sigma \in \mathcal{B}_2$), then $S_\sigma$ and $\operatorname{epi}(q)$ are disjoint nonempty convex sets in $\mathbb{R}^{d+1}$. By weak hyperplane separation, there exist $(a, b) \in \mathbb{R}^d \times \mathbb{R} \setminus \{(0, 0)\}$ and $h \in \mathbb{R}$ with
     $$
     a \cdot x + b t \le h \quad (\forall (x, t) \in S_\sigma), \qquad a \cdot x + b t \ge h \quad (\forall (x, t) \in \operatorname{epi}(q)).
     $$
     Because $\operatorname{epi}(q)$ contains upward rays $\{(x, q(x) + \lambda) : \lambda \ge 0\}$, we have $b \ge 0$. If $b = 0$, then $a \cdot x \ge h$ for all $x \in \mathbb{R}^d$, forcing $a = 0$, a contradiction. Thus $b > 0$. By the upward-shift property, for every $(x, t) \in S_\sigma$ there is $\varepsilon > 0$ with $(x, t + \varepsilon) \in S_\sigma$, whence $a \cdot x + bt < a \cdot x + b(t + \varepsilon) \le h$. Setting $s_\sigma(x) := (h - a \cdot x)/b$ yields $s_\sigma \le q$ on $\mathbb{R}^d$ and $t < s_\sigma(x)$ on $S_\sigma$.

Define $p(x) := \max_{\sigma \in \mathcal{A} \cup \mathcal{B}_2} s_\sigma(x)$, using $R = |\mathcal{A} \cup \mathcal{B}_2| \le 2^{N+M}$ affine minorants. Clearly $p \le q$ everywhere. For every $\sigma \in \mathcal{A}$, $q(x_\sigma) = s_\sigma(x_\sigma) \le p(x_\sigma) \le q(x_\sigma)$, so $(x_\sigma, p(x_\sigma)) \in S_\sigma \cap \Gamma_p$. For every missed nonempty signature $\sigma \notin \mathcal{A}$: if $t > q(x)$ on $S_\sigma$, then $t > q(x) \ge p(x)$ on $S_\sigma$; if $\sigma \in \mathcal{B}_2$, then $t < s_\sigma(x) \le p(x)$ on $S_\sigma$. Hence $\Gamma_p$ meets exactly the same signature regions as $\Gamma_q$, so $\operatorname{code}(\mathcal{U}_p) = \operatorname{code}(\mathcal{U}_q)$. Finally, $\{x : p(x) < \ell_i(x)\} = \bigcap_{r=1}^R \{x : s_r(x) < \ell_i(x)\}$ is a finite intersection of strict affine halfspaces, and applying any fixed Boolean conjunctions to signatures preserves equality of the resulting codes. $\square$

**Corollary 3 (Open balls and homothetic ellipsoids in every dimension).** *Let $Q \in \mathbb{R}^{d \times d}$ be symmetric positive semidefinite ($Q \succeq 0$). Every neural code realized in $\mathbb{R}^d$ by open homothetic ellipsoids (or elliptic cylinders) $U_i = \{x \in \mathbb{R}^d : (x - c_i)^\top Q (x - c_i) < \rho_i\}$ with arbitrary centers $c_i \in \mathbb{R}^d$ and thresholds $\rho_i \in \mathbb{R}$—in particular, every open Euclidean ball code ($Q = I_d$) of arbitrary radii, including tangencies, coincident fields, finite intersections, and auxiliary strict linear halfspaces—admits a same-dimensional open polytope realization in $\mathbb{R}^d$, and hence lies below a representable oriented matroid in $\mathbf{P}_{\mathrm{Code}}$.*

*Proof.* Take $q(x) := x^\top Q x$ (finite and convex on $\mathbb{R}^d$ since $Q \succeq 0$) and $\ell_i(x) := 2 c_i^\top Q x + \rho_i - c_i^\top Q c_i$. Then $(x - c_i)^\top Q (x - c_i) < \rho_i \iff q(x) < \ell_i(x)$. Apply Theorem 2, truncate by a common open box, and apply Theorem 1. $\square$

*(Scope note: arbitrarily oriented ellipsoids have different quadratic parts $Q_i$, so a single shared $q$ does not cover them directly; transverse families of differently oriented ellipsoids are instead covered by Corollary 5 below.)*

---

## 3. Intermediate-Word and Transverse $C^1$ Smooth Sufficient Conditions

For a family $\mathcal{U} = (U_1, \dots, U_n)$ of open convex sets in $\mathbb{R}^d$ realizing $\mathcal{C}$ (with $\varnothing \in \mathcal{C}$), define at each $x \in \mathbb{R}^d$:
$$
I(x) := \{i \in [n] : x \in U_i\}, \qquad J(x) := \{i \in [n] : x \in \overline{U_i}\}.
$$

**Theorem 4 (Intermediate-word polyhedralization).** *Suppose $\mathcal{U} = (U_1, \dots, U_n)$ is an open convex realization of $\mathcal{C}$ in $\mathbb{R}^d$ satisfying the intermediate-word condition*
$$
\forall x \in \mathbb{R}^d,\quad \forall \sigma \subseteq [n],\qquad I(x) \subseteq \sigma \subseteq J(x) \implies \sigma \in \mathcal{C}.
$$
*Then $\mathcal{C}$ admits a same-dimensional open polytope realization in $\mathbb{R}^d$, and hence lies below a representable oriented matroid in $\mathbf{P}_{\mathrm{Code}}$.*

*Proof.* For each nonempty $U_i$, fix a countable dense subset $\{q_{i,1}, q_{i,2}, \dots\} \subseteq U_i$ and set $P_i^{(m)} := \operatorname{int}(\operatorname{conv}\{q_{i,1}, \dots, q_{i,m}\}) \subseteq U_i$ (with $P_i^{(m)} := \varnothing$ if $U_i = \varnothing$). Every $x \in U_i$ has an open $d$-simplex $\operatorname{int}(\operatorname{conv}\{v_0, \dots, v_d\}) \subset U_i$ containing $x$; perturbing each $v_k$ into the dense set yields an index $m_0$ such that $P_i^{(m)}$ contains a fixed open neighborhood of $x$ for all $m \ge m_0$.

Choose a bounded open box $B = (-R, R)^d$ containing one witness $w_\tau$ for each $\tau \in \mathcal{C}$, and set $Q_i^{(m)} := P_i^{(m)} \cap B$. Each $Q_i^{(m)}$ is an open polytope contained in $U_i \cap B$, and every point outside $B$ receives label $\varnothing \in \mathcal{C}$. Since every $w_\tau \in B$ eventually enters $P_i^{(m)}$ for all $i \in \tau$ and never enters $P_i^{(m)}$ for $i \notin \tau$, we have $\mathcal{C} \subseteq \operatorname{code}(\mathcal{Q}^{(m)})$ for all sufficiently large $m$.

Suppose for contradiction that for arbitrarily large $m$ there exists $\sigma_m \in \operatorname{code}(\mathcal{Q}^{(m)}) \setminus \mathcal{C}$, witnessed by $x_m \in \mathbb{R}^d$. Since $\varnothing \in \mathcal{C}$, $\sigma_m \neq \varnothing$, forcing $x_m \in B$. By finiteness of $2^{[n]}$ and compactness of $\overline{B}$, pass to a subsequence along which $\sigma_m = \sigma \notin \mathcal{C}$ is constant and $x_m \to x \in \overline{B}$. For each $i \in \sigma$, $x_m \in Q_i^{(m)} \subseteq U_i$ implies $x \in \overline{U_i}$, so $\sigma \subseteq J(x)$. For each $i \in I(x)$, $x \in U_i$, so there is an open neighborhood $V$ of $x$ with $V \subseteq P_i^{(m)}$ for all $m \ge m_0$; since $x_m \in V \cap B$ for large $m$ in the subsequence, $x_m \in P_i^{(m)} \cap B = Q_i^{(m)}$, giving $i \in \sigma$. Thus $I(x) \subseteq \sigma \subseteq J(x)$, whence $\sigma \in \mathcal{C}$ by hypothesis, a contradiction. Therefore $\operatorname{code}(\mathcal{Q}^{(m)}) = \mathcal{C}$ for all sufficiently large $m$. $\square$

**Corollary 5 (Transverse $C^1$ smooth convex fields).** *Suppose $\mathcal{U} = (U_1, \dots, U_n)$ consists of open convex sets in $\mathbb{R}^d$ such that near every $x \in \partial U_i$ there is a $C^1$ defining function $f_i$ with $\nabla f_i(x) \neq 0$ and $U_i = \{f_i < 0\}$ locally. If at every $x \in \mathbb{R}^d$ the gradients $\{\nabla f_i(x) : i \in J(x) \setminus I(x)\}$ are linearly independent, then $\mathcal{C} = \operatorname{code}(\mathcal{U})$ admits a same-dimensional open polytope realization.*

*Proof.* At any $x \in \mathbb{R}^d$, let $B(x) := J(x) \setminus I(x) = \{i : x \in \partial U_i\}$. The $C^1$ map $y \mapsto (f_i(y))_{i \in B(x)}$ vanishes at $x$ and has surjective differential there, so by the submersion theorem every strict sign assignment on $B(x)$ is attained in every sufficiently small neighborhood $W \subseteq \bigcap_{i \in I(x)} U_i \setminus \bigcup_{k \notin J(x)} \overline{U_k}$ of $x$. Hence every $\sigma$ with $I(x) \subseteq \sigma \subseteq J(x)$ lies in $\mathcal{C}$, and Theorem 4 applies. $\square$

**Remark (Relation to CGIK non-degeneracy and the remaining gap).** Theorem 4 strictly generalizes Corollary 5: it needs no smoothness and admits tangencies (for example, two tangent open ellipses $\{x^2/4 + (y-1)^2 < 1\}$ and $\{x^2 + (y-2)^2/4 < 1\}$ realize $\{\varnothing, 1, 2, 12\}$ and satisfy the intermediate-word condition everywhere, including at their tangency $(0, 0)$). Compared with the *non-degeneracy* condition of Cruz–Giusti–Itskov–Kronholm ([arXiv:1609.03502](https://arxiv.org/abs/1609.03502), *DCG* 2019, Definition 2.11 and Theorem 2.12)—which requires (i) top-dimensionality of every nonempty atom and (ii) $\bigcap_{i \in \sigma} \partial U_i \subseteq \partial(\bigcap_{i \in \sigma} U_i)$ for all nonempty $\sigma$, guaranteeing $\operatorname{code}(\overline{\mathcal{U}}) = \operatorname{code}(\mathcal{U})$—our intermediate-word condition **implies** CGIK non-degeneracy and is **strictly stronger**:
- *Implication:* If $x \in \bigcap_{i \in \sigma} \partial U_i$, then $I(x) \cup \sigma \in \mathcal{C}$, so $\bigcap_{i \in \sigma} U_i \neq \varnothing$; convexity of open sets then gives $(1-t)y + tx \in \bigcap_{i \in \sigma} U_i$ for any $y \in \bigcap_{i \in \sigma} U_i$ and $t \in [0, 1)$, proving $x \in \partial(\bigcap_{i \in \sigma} U_i)$. For (i), if an atom $A_\tau$ had empty interior in a small ball $B' \subseteq \bigcap_{i \in \tau} U_i \setminus \bigcup_{k \notin J(x)} \overline{U_k}$ around $x \in A_\tau$, then $\bigcup_{j \in J(x) \setminus \tau} \overline{U_j} \supseteq B'$; replacing each $U_j$ ($j \in J(x) \setminus \tau$) by a supporting open halfspace $H_j^+ = \{z : v_j \cdot (z - x) > 0\}$ gives $\bigcup_{j \in J(x) \setminus \tau} \{w : v_j \cdot w \ge 0\} = \mathbb{R}^d$, forcing $\bigcap_{j \in J(x) \setminus \tau} H_j^+ = \varnothing$ and hence $J(x) \notin \mathcal{C}$, a contradiction.
- *Strictness:* The three open halfplanes $U_1 = \{y > 0\}$, $U_2 = \{x > 0\}$, $U_3 = \{x + y > 0\}$ in $\mathbb{R}^2$ satisfy CGIK non-degeneracy (with $\operatorname{code}(\mathcal{U}) = \operatorname{code}(\overline{\mathcal{U}}) = \{\varnothing, 1, 2, 13, 23, 123\}$), yet fail the intermediate-word condition at $(0, 0)$ because $\{3\} \notin \mathcal{C}$. Indeed, uniform inner shrinking $\{y > \varepsilon\}, \{x > \varepsilon\}, \{x + y > \varepsilon\}$ creates the extraneous word $\{3\}$ at $(\tfrac{2}{3}\varepsilon, \tfrac{2}{3}\varepsilon)$.

This illustrates the core obstruction for arbitrary open convex families in $d \ge 3$: preserving *forbidden atoms*. Even for two identical open disks (code $\{\varnothing, 12\}$), independent inscribed polygons can introduce a singleton word. Theorem 2 controls forbidden atoms simultaneously via separation in one extra coordinate, whereas general open convex covers carry no shared-function structure.

---

## 4. Refutation of the Closed-Convex Reinterpretation in Both Directions

If "open convex" in Conjecture `conj:convex-char` is replaced by "closed convex" while keeping $\mathcal{C} \le L^+(\mathcal{M})$ on the right-hand side, **both implications fail**.

**Theorem 8 (Two-sided refutation of the closed-convex variant).**
1. *(Closed convex $\not\implies \mathcal{C} \le L^+(\mathcal{M})$)* The 5-neuron, 12-word code of Lienkaemper–Shiu–Woodstock ([arXiv:1509.03328](https://arxiv.org/abs/1509.03328), Theorem 3.1),
   $$
   \mathcal{C}_5 = \{\varnothing, 2345, 123, 134, 145, 13, 14, 23, 34, 45, 3, 4\},
   $$
   admits a closed convex realization in $\mathbb{R}^2$, but admits no open convex realization in any $\mathbb{R}^d$ and hence does not lie below $L^+(\mathcal{M})$ for any representable oriented matroid $\mathcal{M}$.
2. *($\mathcal{C} \le L^+(\mathcal{M}) \not\implies$ closed convex)* The 6-neuron, 13-word code (code (2) of Cruz–Giusti–Itskov–Kronholm, [arXiv:1609.03502](https://arxiv.org/abs/1609.03502), Lemma 2.9),
   $$
   \mathcal{C}_6 = \{\varnothing, 123, 234, 345, 456, 561, 612, 12, 23, 34, 45, 56, 61\},
   $$
   equals $L^+(\mathcal{M}_6)$ for a rank-2 representable oriented matroid $\mathcal{M}_6$, yet admits no closed convex realization in any $\mathbb{R}^d$.

*Proof of (1).* **Closed realization in $\mathbb{R}^2$:** Set $O = (0,0)$, $A = (-1,1)$, $B = (0,1)$, $C = (1,1)$ and take the five closed convex sets $U_1 = [A, C]$, $U_2 = [O, A]$, $U_3 = \operatorname{conv}(O, A, B)$, $U_4 = \operatorname{conv}(O, B, C)$, and $U_5 = [O, C]$. Then $O$ has label $2345$; moving left to right along $[A, C]$ yields $123, 13, 134, 14, 145$; below $[A, C]$ and above $O$ the three open segments and two open triangles yield $23, 3, 34, 4, 45$; and the exterior gives $\varnothing$.

**No open convex realization in any $\mathbb{R}^d$:** Suppose open convex sets $U_1, \dots, U_5 \subseteq \mathbb{R}^d$ realize $\mathcal{C}_5$. Let $A_1 := U_2 \cap U_3$, $A_2 := U_3 \cap U_4$, $A_3 := U_4 \cap U_5$, and $Z := U_2 \cap U_3 \cap U_4 \cap U_5 \neq \varnothing$. Because $2345$ is the unique codeword containing $234$ or $345$, all pairwise intersections coincide: $A_1 \cap A_2 = A_2 \cap A_3 = A_1 \cap A_3 = Z$. Pick $p_1 \in U_1 \cap A_1$ (label $123$) and $p_3 \in U_1 \cap A_3$ (label $145$). The closed segment $[p_1, p_3]$ lies in $U_1 \subseteq U_3 \cup U_4$, so connectedness yields $p_2 \in (p_1, p_3) \cap U_3 \cap U_4 \subseteq U_1 \cap A_2$ (label $134$). Let $\ell$ be the line through $p_1, p_2, p_3$. We have $\ell \cap Z = \varnothing$: on $[p_1, p_3] \subseteq U_1$ this follows from $U_1 \cap Z = \varnothing$; a point $z \in Z$ beyond $p_1$ would put $p_1 \in (z, p_2) \subseteq U_4$ (contradicting label $123$); and $z \in Z$ beyond $p_3$ would put $p_3 \in (p_2, z) \subseteq U_3$ (contradicting label $145$).

This contradicts the three-petal open convex sunflower lemma (Lienkaemper–Shiu–Woodstock, Lemma 3.2): pick $z_0 \in Z$, restrict to the 2-plane $\Pi = \operatorname{aff}(\ell, z_0)$, and choose affine coordinates $(x, y)$ on $\Pi$ with $\ell = \{y = 0\}$, $p_1 = (a_1, 0)$, $p_2 = (0, 0)$, $p_3 = (a_3, 0)$ ($a_1 < 0 < a_3$), and $z_0 = (0, y_0)$ ($y_0 > 0$). Then $Z \cap \Pi \subseteq \{y > 0\}$. Set $y_* := \inf\{y > 0 : (0, y) \in Z\} \in [0, y_0)$ and $w := (0, y_*) \in [p_2, z_0) \subset A_2 \cap \Pi$. Since $Z \cap \Pi$ is open and convex, $w \in \partial(Z \cap \Pi)$ and $z_\delta := (0, y_* + \delta) \in Z \cap \Pi$ for every $\delta \in (0, y_0 - y_*)$. A supporting line $L$ to $Z \cap \Pi$ at $w$ cannot be $\{x = 0\}$ (as $z_0 \in Z \cap \Pi$), so $L = \{y = y_* + mx\}$ with $f(x, y) := y - y_* - mx > 0$ on $Z \cap \Pi$. Without loss of generality $m a_3 \ge 0$ (otherwise $m a_1 \ge 0$ and use $p_1, A_1$). Choose $r > 0$ with $B_r(w) \subset A_2 \cap \Pi$ and $\eta > 0$ with $p'_3 := (a_3, -\eta) \in A_3 \cap \Pi$, so $f(p'_3) = -\eta - y_* - m a_3 \le -\eta < 0$. Fix $t \in (0, 1)$ with $t \|p'_3 - w\| < r/2$ and $\delta \in (0, y_0 - y_*)$ with $\delta < \min(r/2,\, \frac{t}{1-t}\eta)$. Then $u := (1 - t)z_\delta + t p'_3 \in A_3$ satisfies $\|u - w\| < r$ (so $u \in A_2$) and $f(u) \le (1 - t)\delta - t\eta < 0$ (so $u \notin Z$), contradicting $A_2 \cap A_3 = Z$. By Theorem 1, $\mathcal{C}_5 \not\le L^+(\mathcal{M})$ for every representable $\mathcal{M}$.

*Proof of (2).* **Oriented matroid realization:** In $\mathbb{R}^2$, let $\mathcal{M}_6$ be the rank-2 oriented matroid of the six cyclic normals
$$
v_1 = (1, 0),\quad v_2 = (1, 1),\quad v_3 = (0, 1),\quad v_4 = (-1, 0),\quad v_5 = (-1, -1),\quad v_6 = (0, -1).
$$
The six open sectors give the six cyclic triples, the six open boundary rays give the six consecutive pairs, and the origin gives $\varnothing$, so $L^+(\mathcal{M}_6) = \mathcal{C}_6$.

**No closed convex realization in any $\mathbb{R}^d$ (after CGIK, Lemma 2.9):** Suppose closed convex sets $U_1, \dots, U_6 \subseteq \mathbb{R}^d$ realize $\mathcal{C}_6$. Truncating all $U_i$ by a large closed box containing one witness per codeword, we may assume every $U_i$ is compact. The maximal codewords $123, 345, 561, 612, 234$ correspond to nonempty compact convex triple intersections $K_{ijk} := U_i \cap U_j \cap U_k$. Note that opposite fields are disjoint ($U_1 \cap U_4 = U_2 \cap U_5 = U_3 \cap U_6 = \varnothing$) and every codeword containing neuron $i$ contains $i - 1$ or $i + 1$ (mod $6$), so $U_i \subseteq U_{i-1} \cup U_{i+1}$.

Fix $x_{345} \in K_{345}$ and $x_{561} \in K_{561}$, and let $M := [x_{345}, x_{561}] \subseteq U_5$. Since $K_{123} \subseteq U_2$ and $U_2 \cap U_5 = \varnothing$, $K_{123}$ is disjoint from $M$. By compactness, choose $x_{123} \in K_{123}$ minimizing $\delta := \operatorname{dist}(a, M) > 0$ over $a \in K_{123}$, and let $q_M \in M$ satisfy $\|x_{123} - q_M\| = \delta$.
1. The segment $L_1 := [x_{123}, x_{561}] \subseteq U_1 \subseteq U_2 \cup U_6$ is covered by the closed sets $L_1 \cap U_2$ (containing $x_{123} \notin U_6$) and $L_1 \cap U_6$ (containing $x_{561} \notin U_2$). By connectedness, there exists $y_1 \in (x_{123}, x_{561}) \cap U_1 \cap U_2 \cap U_6 = (x_{123}, x_{561}) \cap K_{612}$, so $y_1 = (1 - \alpha)x_{123} + \alpha x_{561}$ with $\alpha \in (0, 1)$.
2. Symmetrically, $L_3 := [x_{123}, x_{345}] \subseteq U_3 \subseteq U_2 \cup U_4$ yields $y_3 \in (x_{123}, x_{345}) \cap K_{234}$, so $y_3 = (1 - \beta)x_{123} + \beta x_{345}$ with $\beta \in (0, 1)$.
3. Finally, $N := [y_1, y_3] \subseteq U_2 \subseteq U_1 \cup U_3$ (with $y_1 \in U_1 \setminus U_3$ and $y_3 \in U_3 \setminus U_1$) yields $y_{123} \in (y_1, y_3) \cap K_{123}$, so $y_{123} = (1 - \gamma)y_1 + \gamma y_3$ with $\gamma \in (0, 1)$.

Expanding gives $y_{123} = \lambda_0 x_{123} + \lambda_1 x_{561} + \lambda_2 x_{345}$ where $\lambda_1 = (1-\gamma)\alpha > 0$, $\lambda_2 = \gamma\beta > 0$, and $\lambda_0 = (1-\gamma)(1-\alpha) + \gamma(1-\beta) \in (0, 1)$ sum to $1$. Setting $p_M := (\lambda_1 x_{561} + \lambda_2 x_{345}) / (\lambda_1 + \lambda_2) \in M$ and $r_M := \lambda_0 q_M + (1 - \lambda_0) p_M \in M$ (by convexity of $M$), we have $y_{123} - r_M = \lambda_0 (x_{123} - q_M)$, whence
$$
\operatorname{dist}(y_{123}, M) \le \|y_{123} - r_M\| = \lambda_0 \|x_{123} - q_M\| = \lambda_0 \delta < \delta,
$$
contradicting the minimality of $\delta$. $\square$

---

## 5. Max-Intersection-Complete Open-Simplex Certificate, Three-Neuron Classification, and Exact Verification

A code $\mathcal{C}$ (with $\varnothing \in \mathcal{C}$) is *max-intersection-complete* if it contains every intersection of a nonempty subfamily of its maximal codewords $M_1, \dots, M_m$. Cruz–Giusti–Itskov–Kronholm ([arXiv:1609.03502](https://arxiv.org/abs/1609.03502), *DCG* 2019) proved that every max-intersection-complete code is both open and closed convex; the following explicit construction realizes it directly by open polytopes inside an open simplex.

**Theorem 6 (Open-simplex polytope certificate).** *Every max-intersection-complete code $\mathcal{C} \subseteq 2^{[n]}$ with $\varnothing \in \mathcal{C}$ admits an open polytope realization (hence lies below a representable oriented matroid in $\mathbf{P}_{\mathrm{Code}}$).*

*Proof.* If $\mathcal{C} = \{\varnothing\}$, take all fields empty in $\mathbb{R}^1$. Otherwise let $M_1, \dots, M_m$ ($m \ge 1$) be the maximal codewords of $\mathcal{C}$, let $\mathcal{B} := \{\varnothing\} \cup \{\bigcap_{j \in S} M_j : \varnothing \neq S \subseteq [m]\} \subseteq \mathcal{C}$, and let $\mathcal{E} := \mathcal{C} \setminus \mathcal{B}$. For each $e \in \mathcal{E}$, fix a parent index $p(e) \in [m]$ with $e \subsetneq M_{p(e)}$. Let $V := [m] \sqcup \mathcal{E}$. If $|V| = 1$ (so $\mathcal{C} = \{\varnothing, M_1\}$), take $U_i = (0, 1)$ for $i \in M_1$ and $U_i = \varnothing$ otherwise.

For $|V| \ge 2$, work in the open simplex $\Delta^\circ = \{\lambda \in \mathbb{R}_{>0}^V : \sum_{v \in V} \lambda_v = 1\} \subset \mathbb{R}^{|V|-1}$. Define group masses $t_j(\lambda) := \lambda_j + \sum_{e \in \mathcal{E} : p(e) = j} \lambda_e$ ($j \in [m]$), so $\sum_{j=1}^m t_j(\lambda) = 1$, and fix $\epsilon \in (0, \min(\tfrac{1}{2}, \tfrac{1}{m}))$. Define the open polytopes
$$
U_i := \Bigl\{\lambda \in \Delta^\circ :\; t_j(\lambda) < \tfrac{1}{m}\ \forall j \text{ with } i \notin M_j,\quad \text{and}\quad \lambda_e < 1 - \epsilon\ \forall e \in \mathcal{E} \text{ with } i \notin e\Bigr\}.
$$
Points outside $\Delta^\circ$ give $\varnothing$. Inside $\Delta^\circ$, $S(\lambda) := \{j \in [m] : t_j(\lambda) \ge 1/m\}$ is always nonempty, and neuron $i$ satisfies the first family of inequalities iff $i \in \bigcap_{j \in S(\lambda)} M_j$. The caps $C_e := \{\lambda \in \Delta^\circ : \lambda_e \ge 1 - \epsilon\}$ ($e \in \mathcal{E}$) are nonempty and pairwise disjoint (since $\epsilon < 1/2$). On $C_e$, $t_{p(e)}(\lambda) \ge 1 - \epsilon \ge 1/m$ while $t_j(\lambda) \le \epsilon < 1/m$ for $j \neq p(e)$, so $S(\lambda) = \{p(e)\}$; the second family removes precisely $i \notin e$, yielding exact label $M_{p(e)} \cap e = e$. Outside all caps, every $\lambda_e < 1 - \epsilon$, so the label is $\bigcap_{j \in S(\lambda)} M_j \in \mathcal{B}$. Every nonempty $S \subseteq [m]$ is attained outside all caps: set $t_j = 1/m$ if $S = [m]$, or pick $0 < \eta < 1/m$ and set $t_j = \eta$ ($j \notin S$) and $t_j = (1 - (m-|S|)\eta)/|S| > 1/m$ ($j \in S$), choosing extra coordinates $\lambda_e$ arbitrarily small inside each group. Thus $\operatorname{code}(\mathcal{U}) = \mathcal{C}$. $\square$

**Corollary 7 (Exact 3-neuron classification).** *Among all $2^7 = 128$ labeled 3-neuron codes containing $\varnothing$, exactly 109 are max-intersection-complete (hence open-polytope realizable and below $L^+(\mathcal{M})$), and the remaining 19 admit no open convex realization in any dimension.*

*Proof.* On $n = 3$, any missing nonempty intersection of maximal words must be a singleton $\{i\} = \{i, j\} \cap \{i, k\}$ with $\{1,2,3\} \notin \mathcal{C}$ and $\{i\} \notin \mathcal{C}$. In any open realization, $U_i = (U_i \cap U_j) \sqcup (U_i \cap U_k)$ would disconnect the nonempty convex set $U_i$ into two disjoint nonempty relatively open subsets. All remaining 109 codes are max-intersection-complete and hence open-polytope realizable by Theorem 6. $\square$

### Exact Computational Verification

The standalone standard-library script [`solutions/neural_code_oriented_matroid/explore.py`](explore.py) tests affine-system feasibility via exact integer Fourier–Motzkin elimination (propagating strictness accurately across lower-dimensional rays, edges, and vertices). Running `.venv/bin/python -B solutions/neural_code_oriented_matroid/explore.py` verifies all checks below with zero failures:

| Verification check | Exact result |
|---|---:|
| Labeled 3-neuron codes containing $\varnothing$ | 109 open-polytope certificates, 19 disconnected-field obstructions |
| Trunk maps generated by subsets of distinct proper trunks over all 128 3-neuron sources | 3,399 passed all proper-trunk preimage checks |
| Families of 1–6 distinct 1D open intervals with endpoints in $\{0, 1, 2, 3, 4\}$ | All 847 direct codes equal homogenized arrangement images |
| 5-neuron closed triangular fan $\mathcal{C}_5$ and 6-neuron open central arrangement $\mathcal{C}_6$ | Exactly 12 and 13 codewords |
| Six tangent 3D open balls centred at $\pm r e_j$ ($r = 1, \dots, 6$) vs. coordinate halfspaces | Exactly 27 codewords in each (all $3^3$ signed axis witnesses $v/4$) |
| Worked downward cover in Trang–Jeffs ([arXiv:2512.04241](https://arxiv.org/abs/2512.04241)) | 8 nonempty trunks to 7 |

---

## Literature and Attribution

- **Target conjecture:** T.-T. Trang and R. A. Jeffs, *Covering Relations in the Poset of Combinatorial Neural Codes*, [arXiv:2512.04241](https://arxiv.org/abs/2512.04241) (PMLR 282:225–242, NeurReps 2025/2026), Conjecture `conj:convex-char`, citing A. Kunin, C. Lienkaemper, and Z. Rosen, *Oriented Matroids and Combinatorial Neural Codes*, [arXiv:2002.03542](https://arxiv.org/abs/2002.03542) (*Combinatorial Theory*, 2023).
- **Planar case:** B. Bukh and R. A. Jeffs, *Planar convex codes are decidable*, [arXiv:2207.06290](https://arxiv.org/abs/2207.06290) (2022).
- **Open vs. closed obstructions and non-degeneracy:** C. Lienkaemper, A. Shiu, and Z. Woodstock, *Obstructions to convexity in neural codes*, *Adv. Appl. Math.* 85:31–59, 2017 ([arXiv:1509.03328](https://arxiv.org/abs/1509.03328)); J. Cruz, C. Giusti, V. Itskov, and B. Kronholm, *On open and closed convex codes*, *Discrete & Computational Geometry* 61(2):247–270, 2019 ([arXiv:1609.03502](https://arxiv.org/abs/1609.03502)).
- **New contributions here:** The common-convex-function polyhedralization theorem (Theorem 2) and its consequence for open Euclidean balls and homothetic ellipsoids ($Q \succeq 0$) with arbitrary tangencies and coincident fields in every dimension (Corollary 3); the intermediate-word inner-exhaustion theorem (Theorem 4) and its exact comparison with CGIK non-degeneracy; the unified open-simplex polytope certificate (Theorem 6); and exact Fourier–Motzkin verification of the finite checks and closed-convex counterexamples.
