# Weak Identifiability of Softmax Multi-Head Attention Networks

**Target paper:** Vahid Shahverdi, Giovanni Luca Marchetti, Georg Bökman, and Kathlén Kohn, *Identifiable Equivariant Networks are Layerwise Equivariant*, [arXiv:2601.21645](https://arxiv.org/abs/2601.21645) (v1, v2, and ICML 2026 camera-ready), Section 5.2 (MHA weak identifiability).  
**Companion files:** [`solutions/equivariant_mha_ident/explore.py`](explore.py), [`solutions/equivariant_mha_ident/proof.pdf`](proof.pdf).

---

## Verdict

- **Expanding-width pure-attention stacks: FALSE generically (both fixed-parameter and representative versions).** Both the fixed-parameter MHA weak-identifiability conjecture (in the camera-ready source's disabled block) and the weaker published expectation in §5.2 ("up to replacing $\theta$ with another parameter defining the same function") are **disproved** for the paper's class of pure-attention stacks. Specifically:
  - **Output-span rank-multiset obstruction (Theorem 1):** Every depth-$2$ pure-attention function with scalar input tokens ($d_0 = 1$) whose output token rows span a subspace of dimension at least $2$ has **no identifiable realization in any admissible depth-$2$ MHA submodel** (of any hidden head count and width), because the last-layer per-head block rank multiset can be changed without altering the end-to-end map.
  - **Open dense counterexample:** At $n \ge 2$, $(d_0, d_1, d_2) = (1, 3, 3)$, and $(h_0, h_1, h_2) = (1, 2, 1)$, an explicit maximal-rank parameter with distinct attention scores realizes a 2-dimensional output span and fails on an **open, dense, full-measure** subset of the architecture.
  - **Equivariance-lift refutation (Proposition 2):** The same parameter is end-to-end equivariant under $g = \operatorname{diag}(1, 1, 2)$ on outputs (trivial on inputs), yet admits **no invertible token-wise hidden lift** (and in particular no lift in $K_1$). The same determinant obstruction ($(\det R)^2 = 1$ vs. $\det R = 2$) holds generically in $(1, 3, 3)$.
- **Constant-width stacks ($d_0 = \dots = d_L = d \ge 1$, $n \ge 2$): PROVED generically at every finite depth $L \ge 1$ (Theorem 3).** On an open dense set with Lebesgue-null complement in the parameter space, every same-architecture competitor (including exceptional ones) realizing the same end-to-end function is related by a **unique** latent symmetry sequence in $K_1 \times \dots \times K_{L-1}$ (for the paper's layer functions; intra-layer factorization freedom persists). An explicit $n = 2$ opposite-score example shows that exceptional square invertible parameters with distinct scores can still fail.

---

## 1. Exact Architecture and Identifiability Definitions

In Section 5.2 of [arXiv:2601.21645](https://arxiv.org/abs/2601.21645), write token sequences as rows. Layer $i \in \{1, \dots, L\}$ takes a flattened matrix $X \in \mathbb{R}^{n \times (h_{i-1} d_{i-1})}$, aggregates across incoming heads via $Z = X O_i$ with $O_i \in \mathbb{R}^{(h_{i-1} d_{i-1}) \times d_{i-1}}$, and outputs the concatenated heads
$$
H_{ij} = T_{A_{ij}}(Z) V_{ij}, \qquad
T_A(Z) := \operatorname{softmax}_{\mathrm{row}}(Z A Z^\top) Z \qquad (j = 1, \dots, h_i),
$$
where $A_{ij} \in \mathbb{R}^{d_{i-1} \times d_{i-1}}$ and $V_{ij} \in \mathbb{R}^{d_{i-1} \times d_i}$ are unrestricted, with endpoint head counts $h_0 = h_L = 1$ and no biases, masks, skip connections, or normalizations. Note that head aggregation occurs at the *start* of the next layer with aggregate width $d_i$.

At each internal boundary $i \in \{1, \dots, L-1\}$, the latent symmetry group $K_i = \mathrm{GL}(d_i)^{h_i} \rtimes S_{h_i}$ permutes the $h_i$ heads and applies an invertible feature map within each head, identically on all $n$ tokens. Endpoint groups $K_0, K_L$ are trivial.

- **Identifiability (Definition 3.4):** $\theta \in \Theta$ is identifiable if for every $\theta' \in \Theta$ with the same end-to-end function, there is a unique sequence $(k_1, \dots, k_{L-1}) \in K_1 \times \dots \times K_{L-1}$ such that $f'_i = k_i \circ f_i \circ k_{i-1}^{-1}$ on the entire input space of each layer $i$.
- **Weak identifiability (Definition 3.5):** $\theta \in \Theta$ is weakly identifiable if it is the image of an identifiable parameter $\tilde{\theta}$ of a same-depth, same-class MHA submodel with linear embeddings $\alpha_i$ ($\alpha_i^* \alpha_i = I$), injective group homomorphisms, trivial endpoint maps, and $f_i(\cdot; \beta_i(\tilde{\theta}_i)) = \alpha_i \tilde{f}_i(\alpha_{i-1}^* \cdot; \tilde{\theta}_i)$. In particular, any admissible submodel realizing $\theta$ must realize the same end-to-end function.

---

## 2. Refutation on Expanding-Width Architectures

### 2.1 Output-span rank-multiset obstruction

**Theorem 1 (Output-span obstruction).** *Fix $L = 2$ and scalar input tokens ($d_0 = 1$). If the rows of the outputs of a depth-$2$ MHA network, taken over all inputs and token positions, span a subspace of dimension at least $2$, then that end-to-end function has no identifiable realization in any depth-$2$ MHA architecture of this class (for any hidden head count $h$ and hidden width $d$). Consequently, no representative of the function is weakly identifiable.*

*Proof.* Let any depth-$2$ realization have hidden head count $h$ and hidden width $d$. Because $d_0 = 1$, each first-layer value matrix $V_{1j} \in \mathbb{R}^{1 \times d}$ has rank at most $1$, so all reachable token rows in hidden head $j$ lie in a subspace $U_j \subseteq \mathbb{R}^d$ of dimension at most $1$. Write $O_2$ in vertical blocks $P_1, \dots, P_h \in \mathbb{R}^{d \times d}$ and set $V := V_{21} \in \mathbb{R}^{d \times d_2}$. Since every output row lies in the row space of $V$, the output-span hypothesis implies $\operatorname{rank}(V) \ge 2$ and $d \ge 2$.
1. Replace each $P_j$ by $P_j^0 := \Pi_j P_j$, where $\Pi_j$ is the orthogonal projector onto $U_j$ ($0$ if $U_j = 0$). Every reachable hidden array $H$ satisfies $H_j \Pi_j = H_j$, so $Z = \sum_j H_j P_j^0 = \sum_j H_j P_j$ is unchanged and the end-to-end function is preserved. Every product $P_j^0 V$ now has rank at most $1$.
2. Some $P_s^0 V$ is nonzero (else the end-to-end map is zero). Choose a unit column $u \in U_s^\perp \subseteq \mathbb{R}^d$ (existing since $\dim U_s \le 1 < 2 \le d$) and a row $w \in \mathbb{R}^{1 \times d}$ such that $wV$ lies outside the $1$-dimensional row space of $P_s^0 V$ (existing since $\operatorname{rank}(V) \ge 2$). Replace $P_s^0$ by $P_s^1 := P_s^0 + u w$, leaving all other blocks $P_j^0$ unchanged. Because $H_s u = 0$ on reachable inputs, the end-to-end function is again preserved. Yet $P_s^1 V = P_s^0 V + u(wV)$ has rank $2$: it agrees with $P_s^0 V \neq 0$ on $U_s$ and maps $u^\top$ to $wV$ outside the row space of $P_s^0 V$.
3. Feed layer $2$ arbitrary constant-token inputs with only head $j$ nonzero (each row equal to $x \in \mathbb{R}^{1 \times d}$ in head $j$ and $0$ elsewhere). Row-wise softmax of constant rows leaves every row unchanged, so layer $2$ outputs $x P_j V$ in every row. A hidden symmetry $k_1 \in K_1 = \mathrm{GL}(d)^h \rtimes S_h$ conjugates these $h$ linear maps by invertible feature changes and permutes them, preserving the multiset $\{\operatorname{rank}(P_j V)\}_{j=1}^h$. Because the two equal-function realizations have different rank multisets (max rank $\le 1$ vs. max rank $2$), they lie in distinct orbits.

By transitivity of the group action, no member of the fiber is identifiable. Because any candidate submodel has the same scalar input, output, depth, and MHA layer rule, applying this argument inside the submodel rules out weak identifiability for every representative. $\square$

### 2.2 Explicit open dense counterexample and equivariance-lift failure

Fix $n \ge 2$, $(d_0, d_1, d_2) = (1, 3, 3)$, $(h_0, h_1, h_2) = (1, 2, 1)$, and let $e_j$ denote the $j$-th standard basis row in $\mathbb{R}^{1 \times 3}$. Set
$$
O_1 = [1], \quad A_{11} = [1], \quad A_{12} = [2], \quad V_{11} = e_1, \quad V_{12} = e_2,
\qquad
O_2 = \begin{bmatrix} I_3 \\ I_3 \end{bmatrix}, \quad A_{21} = V_{21} = I_3.
$$
- **2D output span:** On the all-ones input $\mathbf{1}$, every output row is $(1, 1, 0)$. On $(1, 0, \dots, 0)^\top$, the projected special row is $(p_1, p_2, 0)$ with $p_a := e^a / (n - 1 + e^a)$ and $p_2 > p_1$, while every other projected row is $(1/n, 1/n, 0)$. Since softmax assigns strictly positive weight to the special row, every output row has strictly larger second coordinate than first, so the two probe outputs are linearly independent. Replacing the two blocks of $O_2$ by $e_1^\top e_1$ and $e_2^\top e_2$ preserves the end-to-end function while changing the layer-2 rank multiset from $\{3, 3\}$ to $\{1, 1\}$.
- **Generic failure:** The $2 \times 2$ determinant of the first two coordinates of output row $1$ across these two fixed inputs is real-analytic in all weights and nonzero at the example, hence nonzero on an open, dense, full-measure subset. Intersecting with maximal-rank weights and distinct first-layer scores yields generic failure of weak identifiability in $(1, 3, 3)$.

**Proposition 2 (Failure of layerwise equivariance lift).** *The explicit parameter above is end-to-end equivariant under the output action $g = \operatorname{diag}(1, 1, 2)$ (with trivial input action), yet admits no invertible token-wise hidden lift $M \in \mathrm{GL}(6)$ (in particular no lift in $K_1$). The same determinant obstruction holds generically in $(1, 3, 3)$.*

*Proof.* Every output row of the explicit network has third coordinate $0$, so $g = \operatorname{diag}(1, 1, 2)$ fixes every end-to-end output (and postcomposition by $g$ is absorbed into $V_{21}$ in every MHA submodel). If an invertible $M \in \mathrm{GL}(6)$ satisfied $f_2(HM) = f_2(H)g$ for all hidden arrays $H \in \mathbb{R}^{n \times 6}$, evaluating on constant-token hidden sequences forces $M O_2 = O_2 g$. Because $O_2$ is surjective, $Z = H O_2$ ranges over all of $\mathbb{R}^{n \times 3}$, forcing $T_{I_3}(Z g) = T_{I_3}(Z) g$ for all $Z$. Taking $Z$ with first row $e_3$ and all other rows $0$ yields third coordinate $2 p_4$ on the left and $2 p_1$ on the right, a contradiction since $p_4 \neq p_1$. (Equivalently, changing $V_{21}$ from $I_3$ to $g$ produces an equal-function competitor outside the $K_1$-orbit.)

Generically in $(1, 3, 3)$, the two incoming value lines mapped through $O_2 V$ span a $2$-dimensional plane; let $g$ fix that plane pointwise and double its orthogonal complement ($\det g = 2$). With $O_2$ surjective and $A = A_{21}, V = V_{21}$ invertible, constant tokens force $M O_2 = O_2 R$ with $R = V g V^{-1}$ ($\det R = 2$), and the two-type single-head test below forces $R A R^\top = A$, contradicting $(\det R)^2 = 1$. $\square$

---

## 3. Parameter Freedoms, Single-Head Test, and Exact Orbit Criterion

- **Intra-layer freedom:** Factoring $A = Q K^\top$ introduces $(Q, K) \mapsto (QS, K S^{-\top})$ without changing layer maps. Even in the paper's unfactored parametrization, within any layer the transformation $(O_i, A_{ij}, V_{ij}) \mapsto (O_i R, R^{-1} A_{ij} R^{-\top}, R^{-1} V_{ij})$ ($R \in \mathrm{GL}(d_{i-1})$) leaves every layer function identically unchanged.
- **Single-head determination (two-type probe):** Write a single head on the flattened input as $T_E(X) D$ with $E = O A O^\top$ and $D = O V$. Constant-token inputs determine $D$. For $n \ge 2$, taking $n - 1$ copies of a row $x$ and one row $x + v$, each common row outputs
  $$
  x D + \sigma_n(x E v^\top)\, v D, \qquad \sigma_n(s) := \frac{1}{1 + (n - 1) e^{-s}}.
  $$
  If $D \neq 0$, then $\{v : v D \neq 0\}$ is dense and strict monotonicity of $\sigma_n$ uniquely determines $E$. If $D = 0$, every $E$ gives the zero map; if $n = 1$, every attention weight is $1$ and $E$ drops out.
- **Exact orbit test:** A transporter $(M_1, \dots, M_{L-1}) \in K_1 \times \dots \times K_{L-1}$ (block permutations $M_i$ with invertible blocks $D_{ij}$ mapping head $j$ to $\pi_i(j)$, and $M_0 = M_L = I$) relates two layer-function tuples with effective parameters $(E_{ij}, C_{ij}) = (O_i A_{ij} O_i^\top, O_i V_{ij})$ if and only if, for every layer $i$ and head $j$,
  $$
  C'_{i, \pi_i(j)} = M_{i-1}^{-1} C_{ij} D_{ij}, \qquad
  E'_{i, \pi_i(j)} = M_{i-1}^{-1} E_{ij} M_{i-1}^{-\top}
  $$
  (omitting the $E$-equation when $C_{ij} = 0$ or $n = 1$).
- **Degeneracies and the $n = 2$ opposite-score exception:** Tied scores allow redistributing $\sum_j V_{ij} P_{ij}$; zero outgoing blocks allow arbitrary upstream changes. Moreover, at $n = 2$, $T_A(X) + T_{-A}(X)$ outputs the row sum on both rows. A depth-$2$, width-$d$ network with first-layer heads $(I, I), (-I, I)$, $O_1 = I$, $O_2 = [I; I]$, and $(A_{21}, V_{21}) = (I, I)$ maps $X$ to twice its row mean and intertwines $X \mapsto 2X$ with output doubling, yet admits no $K_1$-lift (constant tokens force doubling in both heads, which scales effective scores from $\pm I$ to $4I$). Thus even square invertible weights with distinct scores can fail at exceptional parameters.

---

## 4. Generic Identifiability at Constant Width

**Theorem 3 (Generic constant-width identifiability).** *Fix sequence length $n \ge 2$, depth $L \ge 1$, arbitrary internal head counts $h_1, \dots, h_{L-1} \ge 1$ ($h_0 = h_L = 1$), and constant width $d_0 = \dots = d_L = d \ge 1$. There is an open dense subset of the parameter space with Lebesgue-null complement on which every parameter $\theta$ is identifiable in the sense of Definition 3.4: every same-architecture competitor $\theta'$ (even exceptional) realizing the same end-to-end function is related to $\theta$ by a unique sequence in $K_1 \times \dots \times K_{L-1}$.*

*Proof.* We establish the result in five steps.

1. **Output-coordinate normalization.** Partition $O_{i+1}$ into vertical blocks $P_{ij} \in \mathbb{R}^{d \times d}$ ($i < L$), set $S_i := \sum_j V_{ij} P_{ij}$ ($i < L$), $S_L := V_{L1}$, and $U_i := S_i \cdots S_L$. On constant tokens, the end-to-end network multiplies every row by $C := O_1 U_1$. Generically $C$ is invertible, and since any equal-function competitor shares $C$, its square factors $O'_1$ and $S'_i$ are automatically invertible too. Setting $Y = Z U_i$ puts every aggregated input in output coordinates: the stack starts at $Y = X C$ and applies $L$ normalized blocks
   $$
   B_i(Y) = \sum_{j=1}^{h_i} T_{a_{ij}}(Y)\, b_{ij}, \qquad \sum_{j=1}^{h_i} b_{ij} = I_d,
   $$
   where $a_{ij} := U_i^{-1} A_{ij} U_i^{-\top}$, $D_{ij} := U_i^{-1} V_{ij}$, and $b_{ij} := D_{ij} P_{ij} U_{i+1}$ for $i < L$ ($D_{L1} = b_{L1} = I_d$). On an open dense set with null complement (defined by nonvanishing polynomials), every $b_{ij}$ is invertible, every $a_{ij} \neq 0$, within each layer no two scores satisfy $a_{ij} = \pm a_{ik}$ ($j \neq k$), and for $n = 2$ every nonempty subset sum satisfies $\sum_{j \in J} b_{ij} \neq 0$ (attainable with positive scalar multiples of $I_d$).

2. **Asymptotic extraction of the first block.** Feed the normalized stack a two-type input with $n - 1$ rows equal to $m/t$ and one special row $m/t + t v$ ($t \downarrow 0$). In block $1$, head $j$ assigns the special token weight $\sigma_n(m a_j v^\top)$ on common rows and $\sigma_n(m a_j v^\top + t^2 v a_j v^\top)$ on the special row. Hence every row exiting block $1$ has the expansion
   $$
   \frac{m}{t} + t\, R_1(m, v) + O(t^3), \qquad
   R_1(m, v) := \sum_{j=1}^{h_1} \sigma_n(m a_j v^\top)\, v b_j.
   $$
   Each subsequent block forms convex combinations of rows within $O(t^3)$ of a common vector and sums feature maps to $I_d$, preserving this expansion verbatim. Thus the end-to-end map determines $R_1(m, v)$ identically as $\lim_{t \downarrow 0} t^{-1}(\text{output row} - m/t)$.

3. **Unordered score/coefficient recovery.** Compare $R_1$ for $\theta$ and a competitor $\theta'$. Excluding the zero loci of finitely many nonzero bilinear forms gives a dense open set of pairs $(m, v)$ on which the scalar frequencies $q_a = m a v^\top$ over the combined score lists have no coincidences (or sign coincidences) except those forced by equal (or opposite) score matrices. Scaling $m \mapsto s m$ yields a one-variable identity in $\sigma_n(s q_a)$ with row coefficients $v b$.
   - For $n > 2$, the meromorphic continuations of $\sigma_n(s q)$ for distinct nonzero $q \in \mathbb{R}$ have simple poles on the distinct vertical lines $\operatorname{Re} s = \log(n - 1) / q$, so they and the constant $1$ are linearly independent. Each of the $h_1$ distinct nonzero scores $a_j$ (with invertible $b_j$, hence $v b_j \neq 0$ for generic $v$) must therefore appear in $\theta'$ with matching coefficient; since both stacks have $h_1$ heads, the matching is bijective, and varying $v$ recovers every $b_j$.
   - For $n = 2$, use $\sigma_2(-s) = 1 - \sigma_2(s)$. The functions $\sigma_2(qs)$ ($q > 0$) and $1$ are linearly independent: the nearest upper-half-plane pole $i\pi / q_{\max}$ isolates the largest frequency, and iterating peels off all terms. Each original score $a_j$ matches a unique competitor score $\pm a_j$, with coefficient $b_j$ (if $+$) or $-b_j$ (if $-$). Collecting the constant terms produced by a flipped subset $J$ yields $v \sum_{j \in J} b_j = 0$ for generic $v$, so $\sum_{j \in J} b_j = 0$; the subset-sum condition forces $J = \varnothing$, again giving exact recovery.

4. **Peeling subsequent blocks.** Two-type sequences are parametrized by row pairs in $\mathbb{R}^{2d}$, and every block preserves this family. In raw coordinates, the unnormalized block $Z \mapsto \sum_j T_{A_{ij}}(Z) V_{ij} P_{ij}$ evaluated at $A_{ij} = I_d$, $V_{ij} P_{ij} = I_d / h_i$, and the row pair $(r e_1, -r e_1)$ has cross-type attention probability $O_n(e^{-2r^2})$, so its $2d \times 2d$ Jacobian converges to $I_{2d}$ as $r \to \infty$ (differentiation and value differences contribute $O(r^2)$). Fixing finite $r$ where the Jacobian determinant is nonzero gives a real-analytic function of the raw weights that is nonzero generically. By the inverse function theorem, each nonfinal block's two-type map contains an open subset of $\mathbb{R}^{2d}$ in its image. Because $B_1 = B'_1$, the remaining tails agree on this open set; by real-analyticity on the connected space $\mathbb{R}^{2d}$, they agree everywhere on two-type inputs. Iterating steps 2–4 recovers all normalized blocks $(a_{ij}, b_{ij})$ up to layerwise head permutations.

5. **Unique lift to $K_1 \times \dots \times K_{L-1}$.** After indexing matched heads identically in each layer $i < L$, invertibility of $b_{ij} = D_{ij} P_{ij} U_{i+1} = D'_{ij} P'_{ij} U'_{i+1}$ implies invertibility of $D_{ij}, D'_{ij}, P_{ij}, P'_{ij}$. Right-multiplying head $j$ by $D_{ij}^{-1} D'_{ij} \in \mathrm{GL}(d)$ transports head $j$ of $\theta$ to head $j$ of $\theta'$ in common normalized input coordinates and satisfies
   $$
   (D_{ij}^{-1} D'_{ij}) P'_{ij} U'_{i+1} = P_{ij} U_{i+1},
   $$
   so the next layer receives identical normalized input on *all* hidden arrays. Since $X C$ is shared and $D_{L1} = I_d$, this gives a valid sequence in $K_1 \times \dots \times K_{L-1}$ with trivial endpoints. For uniqueness, induct from $i = 1$: once $k_{i-1}$ is fixed, both layers receive the same full-range normalized input $Y$ (as $O_1$ is invertible and each $O_i$ is surjective); constant tokens force any head match $j \mapsto k$ to use feature map $D_{ij}^{-1} D'_{ik}$, and the single-head two-type test forces $a_{ij} = a'_{ik}$, which fixes the permutation uniquely by score distinctness. $\square$

---

## 5. Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/equivariant_mha_ident/explore.py
```

Using NumPy with seed `21645`, all assertions pass:
- **Expanding $(1, 3, 3)$ witness ($n = 3$):** output-span $2 \times 2$ minor `0.095121516`; last-layer block ranks `(3, 3)` vs. `(1, 1)`; exact agreement on 64 random inputs with both competitors; forced equivariance lift for $g = \operatorname{diag}(1, 1, 2)$ fails by `0.777092542` on the $e_3$ hidden probe; all 32 random maximal-rank expanding models pass the rank and determinant checks.
- **Degeneracies:** tied-head value redistribution, zero-value score invisibility, unused-head variation, and the $n = 2$ opposite-score non-liftable equivariance all pass.
- **Constant-width $(2, 2, 2, 2)$, head counts $(2, 2, 1)$:** gauge-fixed complex-step Jacobian over all 32 normalized parameters achieves full numerical rank `32/32` at both $n = 2$ (singular-value ratio $4.25 \times 10^{-5}$) and $n = 3$ ($1.30 \times 10^{-3}$), with first-block asymptotic recovery error at $t = 0.002$ equal to $4.50 \times 10^{-6}$ ($n = 2$) and $1.32 \times 10^{-7}$ ($n = 3$).
