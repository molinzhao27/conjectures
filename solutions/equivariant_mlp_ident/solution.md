# Resolution of the Biased-MLP Weak Identifiability Conjecture

**Target paper:** Vahid Shahverdi, Giovanni Luca Marchetti, Georg Bökman, and Kathlén Kohn, *Identifiable Equivariant Networks are Layerwise Equivariant*, [arXiv:2601.21645](https://arxiv.org/abs/2601.21645) (ICML 2026 camera-ready), Section 5.1, Conjecture 1 (`conj:main`).  
**Companion files:** [`solutions/equivariant_mlp_ident/explore.py`](explore.py), [`solutions/equivariant_mlp_ident/proof.pdf`](proof.pdf).

---

## Verdict

- **Positive hidden widths ($d_i \ge 1$): FALSE for every origin-passing activation $\sigma$.** For depth $L \ge 2$ and scalar input/output, any constant map $x \mapsto c$ has zero-hidden-map realizations whose readout layers $z \mapsto c$ and $z \mapsto c + z_1$ lie in different latent-symmetry orbits inside every positive-width submodel (Theorem 1). Already the minimal ambient architecture $(1,1,1)$ refutes the universal quantifier over all realized functions.
- **ReLU, exact GELU, and all identity-realizing activations: FALSE under either width convention ($d_i \ge 0$ or $d_i \ge 1$).** Whenever an MLP with activation $\sigma$ realizes $f(x) = x$ at depth $L \ge 2$, the input/output translation family $(b_1, c) \mapsto (b_1 - t W_1, c + t)$ produces a continuum of realizations of $f(x) = x$ with distinct output biases $c + t$. Because linear latent symmetries fixing $0$ leave the readout bias invariant, no submodel realizing $f(x) = x$ has an identifiable parameter for it (Theorem 2). For ReLU and exact $\operatorname{GELU}(z) = z\Phi(z)$, the identity
  $$
  x = \sigma(x - t) - \sigma(t - x) + t
  $$
  refutes weak identifiability at the minimal hidden width $2$, with continuous minimal-width fibers of dimensions $3$ (ReLU) and $2$ (GELU). The same obstruction applies to every nonconstant polynomial activation of degree $d \ge 1$ at hidden width $d$ via the $(d-1)$st forward difference, showing that failures are dense in the compact-open topology on continuous origin-passing activations (Corollary 3).
- **Affirmative classification with zero hidden widths allowed ($d_i \ge 0$): PROVED for $\tanh$, centered logistic, and a dense sigmoidal family.**
  - For $\sigma = \tanh$ (and fixed nonzero input/output rescalings such as centered logistic $s(x) - \tfrac{1}{2} = \tfrac{1}{2}\tanh(x/2)$), every realized function admits a weakly identifiable representative in every finite architecture via reduction to a Vlačić–Bölcskei regular network (for nonconstant maps, citing Vlačić & Bölcskei, 2021) or the all-zero-width submodel (for constants) (Theorem 4). Plain logistic $s(x) = (1 + e^{-x})^{-1}$ violates $\sigma(0) = 0$ and fails under linear latent actions via $s(x) = 1 - s(-x)$.
  - For every origin-passing piecewise $C^1$ target $\rho$ with $\rho' \in BV(\mathbb{R})$ and every $\varepsilon > 0$, there exists an origin-passing activation $\sigma$ with $\|\sigma - \rho\|_\infty < \varepsilon$ satisfying Conjecture 1 at all finite depths and widths with $K_i = S_{d_i}$, derived by transferring Theorem 1 of Vlačić & Bölcskei (2022) to biased hidden layers, affine vector readouts, and unique symmetry sequences (Theorem 5). Thus the activation-specific counterexamples do not refute an existential interpretation over a broad sigmoidal family once zero widths are admitted.

---

## 1. Model, Latent Symmetries, and Weak Identifiability

Fix depth $L \ge 1$, endpoint dimensions $d_0, d_L$, hidden widths $d_1, \dots, d_{L-1} \in \mathbb{N}$ (the source does not specify whether $0 \in \mathbb{N}$), and a coordinatewise origin-passing activation $\sigma : \mathbb{R} \to \mathbb{R}$ with $\sigma(0) = 0$.
Each hidden layer $i \in \{1, \dots, L-1\}$ has full affine parameter space $(W_i, b_i) \in \mathbb{R}^{d_i \times d_{i-1}} \times \mathbb{R}^{d_i}$ with layer map $f_i(z; W_i, b_i) = \sigma(W_i z + b_i)$, and the output layer $L$ is affine with free bias: $f_L(z; W_L, c) = W_L z + c$.
The hidden symmetry groups $K_i \subseteq \mathrm{GL}(d_i)$ are the linear intertwiner groups satisfying $\sigma(A x) = B \sigma(x)$ for invertible $A, B$; for all obstructions below it suffices only that every $k_i \in K_i$ acts linearly (hence fixes $0$). Endpoint groups $K_0, K_L$ are trivial.

Under the source definitions (`def:submodel`, `defn:identii`, `def:weak_ident`, `conj:main` in arXiv:2601.21645):
- A parameter $\theta \in \Theta$ is **identifiable** if for every $\theta' \in \Theta$ with the same end-to-end function $f(\cdot; \theta') = f(\cdot; \theta)$, there is a **unique** tuple $(k_1, \dots, k_{L-1}) \in K_1 \times \dots \times K_{L-1}$ such that
  $$
  f_i(\cdot; \theta'_i) = k_i \circ f_i(\cdot; \theta_i) \circ k_{i-1}^{-1} \qquad (i = 1, \dots, L)
  $$
  on the entire ambient layer domain $\mathbb{R}^{d_{i-1}}$ (with $k_0 = I_{d_0}$, $k_L = I_{d_L}$). In particular, if a functional fiber contains two tuples of layer maps in different $(K_1 \times \dots \times K_{L-1})$-orbits, no member of that fiber is identifiable (by transitivity of the group action).
- A parameter $\theta' \in \Theta$ is **weakly identifiable** if it is the image of an identifiable parameter $\tilde{\theta}$ of a same-depth, same-activation MLP submodel $\tilde{\Theta}$ (with full affine layer spaces, same endpoints, linear inclusions $\alpha_i$ with left inverses $\alpha_i^* \alpha_i = I$, injective symmetry homomorphisms, and $f_i(x; \beta_i(\tilde{\theta}_i)) = \alpha_i \tilde{f}_i(\alpha_{i-1}^* x; \tilde{\theta}_i)$ for all $x \in \mathbb{R}^{d_{i-1}}$). Arbitrary frozen-weight singletons are excluded.
- **Coordinate padding:** Because $\sigma(0) = 0$, embedding a smaller-width MLP $(\tilde{d}_1, \dots, \tilde{d}_{L-1})$ with $\tilde{d}_i \le d_i$ by zero-padding weights and biases, projecting via $\alpha_i^*$, and extending symmetries by the identity on the padded coordinates defines a valid submodel.

**Conjecture 1 (`conj:main`, arXiv:2601.21645).** *For a broad class of activation functions $\sigma$, every function realized by an MLP with activation $\sigma$ admits a weakly identifiable representative: for every $\theta \in \Theta$, there exists $\theta' \in \Theta$ with $f(\cdot; \theta) = f(\cdot; \theta')$ such that $\theta'$ is weakly identifiable.*

---

## 2. Constant-Function Refutation for Positive Hidden Widths

**Theorem 1 (Universal failure under $d_i \ge 1$).** *Assume every admissible MLP submodel has positive hidden widths. For every origin-passing activation $\sigma$, depth $L = 2$, and scalar endpoints $(d_0, d_2) = (1, 1)$, no constant function $x \mapsto c$ admits a weakly identifiable representative.*

*Proof.* Let $\tilde{\Theta}$ be any depth-$2$ MLP submodel with hidden width $m \ge 1$ realizing $x \mapsto c$. Consider two tuples in $\tilde{\Theta}$ with identical zero hidden layer $(W_1, b_1) = (0, 0)$ (so $f_1(x) \equiv \sigma(0) = 0$):
1. $(W_2, c_2) = (0, c)$, giving readout $z \mapsto c$;
2. $(W'_2, c'_2) = (e_1^\top, c)$, giving readout $z \mapsto c + z_1$.

Both realize $x \mapsto c$ end-to-end. However, any $k_1 \in \tilde{K}_1 \subseteq \mathrm{GL}(m)$ is invertible, so precomposing a constant readout $z \mapsto c$ by $k_1^{-1}$ stays constant and can never equal $z \mapsto c + z_1$ on $\mathbb{R}^m$. Thus the two tuples lie in distinct layer-function orbits, so no element of the fiber of $x \mapsto c$ in $\tilde{\Theta}$ is identifiable. Because this holds for every positive-width submodel $\tilde{\Theta}$, no representative in the ambient architecture (already $(1, 1, 1)$) is weakly identifiable. $\square$

*Remark (Zero widths and reducible nonconstant maps).* If hidden width $0$ is admitted, the all-zero-width submodel $(1, 0, \dots, 0, 1)$ has only the free output bias $c$ and trivial hidden groups, making $x \mapsto c$ identifiable; coordinate padding supplies a weakly identifiable representative in any ambient architecture. At depth $L = 1$, the single layer is the end-to-end map and every parameter is identifiable. For nonconstant maps, a reducible ambient parameter can still admit a weakly identifiable representative: e.g., $f(x) = \tanh x$ in $(1, 2, 1)$ has a large ambient fiber (unused or cloned units), yet zero-padding the identifiable width-$1$ tuple $(w, b, a, c) = (1, 0, 1, 0)$ provides a valid witness.

---

## 3. Translation-Bias Obstruction: ReLU, Exact GELU, and Polynomials

**Lemma (Output-bias invariance).** *Under any linear latent symmetry $k_{L-1} \in K_{L-1}$, the last-layer relation $W'_L z + c' = W_L k_{L-1}^{-1} z + c$ for all $z \in \mathbb{R}^{d_{L-1}}$ evaluates at $z = 0$ to $c' = c$ (whether or not $0$ lies in the image of layer $L-1$).*

**Theorem 2 (Identity-realization obstruction).** *Let $\sigma$ be any activation and suppose a depth $L \ge 2$ MLP with scalar endpoints $(d_0, d_L) = (1, 1)$ realizes the identity $f(x) = x$. Then $f(x) = x$ admits no weakly identifiable representative under either width convention ($d_i \ge 0$ or $d_i \ge 1$).*

*Proof.* Let $\tilde{\Theta}$ be any same-depth, same-activation MLP submodel realizing $f(x) = x$ via some parameter $\tilde{\theta}$ with first-layer parameters $(W_1, b_1)$ and readout bias $c$. (Its hidden widths cannot all be zero, since width $0$ realizes only constants.) For every $t \in \mathbb{R}$, replacing $(b_1, c)$ by $(b_1 - t W_1, c + t)$ preserves the same architecture and realizes
$$
\tilde{f}_t(x) = f(x - t) + t = (x - t) + t = x.
$$
By the Lemma, parameters with distinct readout biases $c + t$ lie in distinct layer-function orbits. Thus the fiber of $f(x) = x$ inside every candidate submodel $\tilde{\Theta}$ contains a continuum of distinct orbits, so no parameter in $\tilde{\Theta}$ is identifiable. $\square$

**Application to ReLU and exact GELU.** Both $\sigma(z) = \max(z, 0)$ and $\sigma(z) = z\Phi(z)$ (where $\Phi$ is the standard normal CDF) satisfy $\sigma(z) - \sigma(-z) = z$, yielding the explicit depth-$2$, width-$2$ family
$$
f_t(x) = \sigma(x - t) - \sigma(t - x) + t = x \qquad (t \in \mathbb{R}).
$$
Width $0$ yields only constants, and width $1$ cannot realize $x$ because $a\sigma(wx + b) + c = x$ would require $aw \neq 0$ and force $\sigma$ to be affine on $\mathbb{R}$. Hence width $2$ is minimal. While arXiv:2601.21645 notes the identity $\sigma(z) - \sigma(-z) = z$ as a caution for ReLU, Theorem 2 proves the stronger negation over **all** potential representatives and submodels.

- *Exact minimal-width fibers:* Ordering the two hidden slopes as $w_+ > 0 > w_-$ (both signs are required because $\sigma(z) \to 0$ as $z \to -\infty$ and a constant plus one varying unit cannot give $x$), comparing asymptotics $\sigma(z) = z + o(1)$ as $z \to +\infty$ in $\sum_{i=1}^2 a_i \sigma(w_i x + b_i) + c = x$ forces $a_i = 1/w_i$ and $b_i = -c w_i$. For ReLU, every triple $(c, w_+, w_-) \in \mathbb{R} \times (0, \infty) \times (-\infty, 0)$ works (3-dimensional fiber). For exact GELU, setting $u = x - c$ and dividing by $u \neq 0$ leaves $\Phi(w_+ u) + \Phi(w_- u) = 1$, which by strict monotonicity and reflection of $\Phi$ forces $w_- = -w_+$ (2-dimensional fiber).

**Corollary 3 (Polynomials and compact-open density of failures).** *For every nonconstant polynomial activation $p$ of degree $d \ge 1$ with leading coefficient $a_d$, the $(d-1)$st forward difference $\Delta^{d-1} p(x) = \sum_{j=0}^{d-1} (-1)^{d-1-j} \binom{d-1}{j} p(x + j)$ is affine of slope $d! a_d \neq 0$, so $f(x) = x$ is realized at depth $2$, width $d$, and Theorem 2 refutes Conjecture 1 for that architecture. In particular, even the biased identity activation $\sigma(z) = z$ fails at width $1$ under linear latent groups. Since origin-passing nonconstant polynomials are dense in the compact-open topology on $C_0(\mathbb{R}) := \{\sigma \in C(\mathbb{R}) : \sigma(0) = 0\}$, no nonempty compact-open subset of $C_0(\mathbb{R})$ satisfies Conjecture 1 across all architectures.*

*(Scope note: for a fixed architecture, high-degree monomials $z^m$ with $m$ large relative to the widths remain identifiable, consistent with the paper's list of known cases; Corollary 3 applies at width $d = \deg p$.)*

---

## 4. Affirmative Classification for $\tanh$ and a Broad Sigmoidal Family (with Zero Widths)

**Theorem 4 ($\tanh$ and centered logistic).** *Let $\sigma = \tanh$ (or any fixed nonzero input/output rescaling thereof, such as the centered logistic $s(x) - \tfrac{1}{2} = \tfrac{1}{2}\tanh(x/2)$), with $K_i = \{\pm 1\}^{d_i} \rtimes S_{d_i}$. If zero hidden widths are allowed, every function realized by an MLP of any finite depth and width admits a weakly identifiable representative. If hidden widths must be positive, the same holds for every nonconstant function.*

*Proof.* Depth $L = 1$ is immediate, and constants are handled by the all-zero-width submodel when $d_i = 0$ is allowed. For a nonconstant realization at depth $L \ge 2$, apply three function-preserving neuron deletions until none remain: (i) delete any neuron with zero outgoing weight column; (ii) fold any neuron with zero incoming weight row into the next layer's bias; (iii) merge any pair of neurons in the same layer with equal or opposite incoming affine forms $(w, b) = \pm (w', b')$, using $\tanh(-z) = -\tanh(z)$. Since the map is nonconstant, at least one neuron survives in every hidden layer. The resulting network is *regular* in the sense of Vlačić & Bölcskei (2021) (every surviving neuron depends structurally on the input and reaches a nonzero readout column, with no signed clones). By Vlačić & Bölcskei (2021, Lemma 1 and Proposition 2), any two regular $\tanh$-networks realizing the same function have the same widths and are related by layerwise signed permutations.

To verify identifiability across the entire fiber of the reduced architecture, let $\tilde{\theta}'$ be any competitor with the same reduced widths. Its regular reduction must be isomorphic to the first regular network and therefore has the same total neuron count, so no deletion can occur—$\tilde{\theta}'$ is itself regular and isomorphic to $\tilde{\theta}$ by some signed permutation sequence $(k_1, \dots, k_{L-1})$. Uniqueness of $(k_1, \dots, k_{L-1})$ follows by induction on $i$ from the fixed input: no two surviving neurons share the same incoming affine form up to sign, and $\tanh$ is injective and odd, so matching layer maps determines each $k_i$ uniquely. Coordinate padding embeds $\tilde{\theta}$ as a weakly identifiable representative in the ambient architecture. Rescalings $x \mapsto \alpha \tanh(\beta x)$ ($\alpha\beta \neq 0$) absorb into adjacent affine layers. $\square$

*Plain logistic counterexample.* Plain logistic $s(x) = (1 + e^{-x})^{-1}$ has $s(0) = \tfrac{1}{2} \neq 0$ (outside the paper's origin-passing convention). Under linear latent groups fixing $0$, the identity $s(x) = 1 - s(-x)$ produces two width-$1$ realizations with distinct readout biases $0$ and $1$ (padded to any positive width), while width $0$ cannot realize $s$; thus $s(x)$ has no weakly identifiable representative under linear latent actions.

**Theorem 5 (Dense sigmoidal family under $d_i \ge 0$).** *For every origin-passing piecewise $C^1$ function $\rho$ with $\rho' \in BV(\mathbb{R})$ and every $\varepsilon > 0$, there exists an origin-passing activation $\sigma$ with $\|\sigma - \rho\|_\infty < \varepsilon$ such that Conjecture 1 holds at all finite depths and widths with $K_i = S_{d_i}$ (when zero hidden widths are allowed).*

*Proof.*
1. **Scalar sigmoidal activation:** Apply Vlačić & Bölcskei (2022, Theorem 1) to $\rho$ with tolerance $\varepsilon/2$ to obtain an activation $s$ such that any two scalar-input, activated-scalar-output networks with no zero weight rows/columns and no exact clone pairs realizing the same function have equal widths and are isomorphic by layerwise permutations. Set $\sigma(z) := s(z) - s(0)$; then $\sigma(0) = 0$ and $\|\sigma - \rho\|_\infty \le \|s - \rho\|_\infty + |s(0) - \rho(0)| < \varepsilon$.
2. **Reduction and shift back to $s$:** For a nonconstant $\sigma$-MLP realizing $F : \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$, reduce by deleting unused/constant neurons and merging exact clones (identical incoming affine forms) until irreducible. Given two reduced realizations of $F$, convert hidden activations from $\sigma$ to $s$ by subtracting $s(0) W_i \mathbf{1}$ from each bias after layer $1$ (including the readout bias). This bijection preserves $F$, nonzero rows/columns, absence of exact clones, and commutes with permutations.
3. **Generic 1D slicing:** For $(u, v) \in \mathbb{R}^{d_L} \times \mathbb{R}^{d_0}$, restrict to $t \mapsto s(u^\top F(tv))$. Outside a finite union of proper hyperplanes, all nonzero rows of $W_1$ remain nonzero on $v$, distinct first-layer affine forms remain distinct on $v$, and all nonzero columns of $W_L$ have nonzero inner product with $u$. All interior layers already satisfy the Vlačić–Bölcskei hypotheses, and the scalar output adds a single unit with no clone. Hence Vlačić & Bölcskei (2022, Theorem 1) gives equal widths and matching parameters up to a layerwise permutation sequence $\pi(u, v)$.
4. **Global permutation sequence:** For each of the finitely many permutation sequences $\pi$, the parameter-matching equations are affine-linear in $(u, v)$ (first weights $W_1 v$, readout $(u^\top W_L, u^\top c)$, interior layers constant). Because an open dense subset of $\mathbb{R}^{d_L} \times \mathbb{R}^{d_0}$ cannot lie in a finite union of proper affine subspaces, one fixed $\pi$ works for all $(u, v)$, matching the original matrices and biases identically. Any same-width competitor must itself be reduced (by neuron-count equality after reduction), proving uniqueness up to permutations across the full reduced fiber.
5. **Intertwiner group and uniqueness of $\pi$:** Applying the reduced-network theorem to the one-hidden-layer vector-output pairs $(A, I)$ and $(I, B)$ with zero biases shows that $\sigma(Ax) = B\sigma(x)$ for invertible $A, B$ forces $A = B \in S_d$, so $K_i = S_{d_i}$. Applying it at width $1$ with readout $1$ shows that distinct nonzero incoming affine forms define distinct ridge functions (since $\sigma$ is nonconstant). Inducting from the input shows that the stabilizing permutation in each layer is unique. Coordinate padding completes the nonconstant case; the all-zero-width submodel covers constants. $\square$

---

## 5. Numerical Verification and Attribution

Run the verification script from the project root:

```sh
.venv/bin/python -B solutions/equivariant_mlp_ident/explore.py
```

The script evaluates analytic sample Jacobians on 257 deterministic points in $[-5.013, 5.017]$ (avoiding $z = 0$ for ReLU, SVD cutoff $10^{-10}\sigma_{\max}$) and exact integer interpolation for polynomials:

| Function and representation | Verified sample Jacobian rank / parameter count |
|---|---|
| Identity, minimal ReLU (36 tuples) | 4 / 7 |
| Identity, minimal GELU (18 tuples) | 5 / 7 |
| $\tanh$, centered logistic, plain logistic (both reflected width-1 tuples) | 4 / 4 |
| $\tanh$ with unused, zero-padded, or cloned second unit | 5 / 7, 4 / 7, 4 / 7 |
| Constant $\tanh$ (three positive-width tuples) | 2 / 4, 2 / 4, 1 / 4 |
| Constant, zero hidden width | 1 / 1 |

All assertions pass, including center-shift tangents lying in the Jacobian kernel for ReLU and GELU, and exact integer verification of the forward-difference identity for monomials of degrees $1$ through $12$ at three shifts each.

**References.**
- V. Shahverdi, G. L. Marchetti, G. Bökman, and K. Kohn, *Identifiable Equivariant Networks are Layerwise Equivariant*, [arXiv:2601.21645](https://arxiv.org/abs/2601.21645) (ICML 2026 camera-ready).
- V. Vlačić and H. Bölcskei, *Affine Symmetries and Neural Network Identifiability*, Advances in Mathematics **376** (2021), 107485.
- V. Vlačić and H. Bölcskei, *Neural Network Identifiability for a Family of Sigmoidal Nonlinearities*, Constructive Approximation **55** (2022), 173–224 ([arXiv:1906.06994](https://arxiv.org/abs/1906.06994)).
