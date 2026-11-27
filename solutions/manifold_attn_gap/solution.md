# Refutation of the Matched-Budget Attention Rate Gap (`conj:rate-gap`) and Depth-$L$ Polynomial Analysis

**Target paper:** Siddharth Roheda, Rohit Chowdhury, Amit Satish Unde, and Aniruddha Bala, *Polynomial Denoisers on the Image Manifold: Volterra Networks for Self-Contained Pixel Diffusion*, [arXiv:2611.10857](https://arxiv.org/abs/2611.10857), Section 3 and Appendix (Proposition 2 / `prop:separation` and Conjecture 1 / `conj:rate-gap`).  
**Companion files:** [`solutions/manifold_attn_gap/explore.py`](explore.py), [`solutions/manifold_attn_gap/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 1 (`conj:rate-gap` — Quantitative rate gap at matched budget): FALSE.** On the uniform 2-sphere $\mathcal{M} = R S^2 \subset \mathbb{R}^D$ ($D \ge 5$) with $t = \coth(1) - 1$ and $R = (1-t)/t$, the optimal Gaussian denoiser satisfies $f_t(z) = z$ identically on $t\mathcal{M}$ (while ambient $f_t$ is non-affine). Identity residual softmax attention ($W_O = 0$ in every layer) therefore achieves **zero worst-case error** on $t\mathcal{M}$ at every depth $L \ge 1$ and every common parameter budget $\mathsf{P} \ge \mathsf{P}_0(L, d')$, contradicting any strictly positive lower bound $C_2 K_L^{-2s}(1 + \Delta(s, L))$.
- **Depth-$L$ exact polynomial content (open extension of Proposition 2(ii)): PROVED.** Any finite-depth stack of residual softmax attention blocks with affine values and affine input/output maps that agrees with a polynomial on a nonempty ambient open set is globally affine.
- **Small-logit uniform approximation (matched budget order):** With distinguishable fixed positional encodings, every degree-$2^L$ patchwise polynomial lies in the compact-uniform closure of depth-$L$ residual softmax attention at the same per-layer budget order $O(H^2)$ as the Volterra construction. **Important distinction:** this small-logit Taylor construction requires output weights growing as $O(\epsilon^{-2})$ as $\epsilon \to 0$ (which parameter-count budgets permit, though uniform weight-norm bounds would not), whereas the sphere counterexample uses **zero** output weights ($W_O = 0$) and requires no limiting argument.

---

## 1. Problem Setup and Conjecture Statement

Let $\mathcal{M} \subset \mathbb{R}^D$ be a compact $C^\infty$ submanifold of dimension $d_{\mathcal{M}}$ carrying a probability measure $\rho_{\mathrm{data}}$ that is not a single point mass. For $t \in (0, 1)$ and $\sigma := 1 - t$, observe $z = tX + \sigma \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I_D)$ independent of $X \sim \rho_{\mathrm{data}}$, and let $f_t(z) := \mathbb{E}[X \mid tX + \sigma\varepsilon = z]$ be the MMSE denoiser. Let $\pi : \mathbb{R}^D \to \mathbb{R}^{d'}$ ($d' = 2d_{\mathcal{M}} + 1$) be a linear bottleneck embedding $t\mathcal{M}$, and let $K_L := 2^L$ and $H := \binom{d' + K_L}{d'}$. Proposition 2(i) of arXiv:2611.10857 realizes all degree-$K_L$ polynomials in $\pi(z)$ via a depth-$L$ normalization-free Volterra stack with per-layer budget $\mathsf{P}_0(L, d') = c_0 H^2$, attaining $\sup_{z \in t\mathcal{M}} \|f_t(z) - g(z)\|^2 \le C_1 K_L^{-2s}$.

**Conjecture 1 (`conj:rate-gap`, arXiv:2611.10857, Appendix).** *In the setting of Proposition 2, let $\mathcal{A}_{L, \mathsf{P}}$ denote the class of depth-$L$ stacks of softmax self-attention blocks with residual connections and at most $\mathsf{P}$ parameters per layer. For matched depth $L$ and every $\mathsf{P} \ge \mathsf{P}_0(L, d')$,*
$$
\inf_{h \in \mathcal{A}_{L, \mathsf{P}}} \sup_{z \in t\mathcal{M}} \|f_t(z) - h(z)\|^2 \ge C_2 \cdot K_L^{-2s} \cdot \bigl(1 + \Delta(s, L)\bigr)
$$
*for a constant $C_2 > 0$ and a gap $\Delta(s, L) > 0$ nondecreasing in the smoothness order $s$.*

*(Note: Proposition 2(ii) uses multi-patch token matrices while `conj:rate-gap` uses single-patch notation; the sphere counterexample below works in both regimes.)*

---

## 2. Exact Sphere Counterexample ($W_O = 0$)

**Theorem 1 (Zero error on $t\mathcal{M}$ at every depth).** *For every $D \ge 5$, there exist a connected, compact, $C^\infty$ 2-dimensional submanifold $\mathcal{M} \subset \mathbb{R}^D$ with uniform surface measure $\rho_{\mathrm{data}}$ (not a point mass) and a time $t \in (0, 1)$ such that $f_t(z) = z$ for all $z \in t\mathcal{M}$. Consequently, for every depth $L \ge 1$ and every common budget cap $\mathsf{P}$ covering $\mathsf{P}_0(L, 5)$ and the finite identity residual stack (with $W_O = 0$ in every layer),*
$$
\inf_{h \in \mathcal{A}_{L, \mathsf{P}}} \sup_{z \in t\mathcal{M}} \|f_t(z) - h(z)\|^2 = 0,
$$
*refuting Conjecture 1 for every $C_2 > 0$, $s > 0$, and $\Delta(s, L) \ge 0$.*

*Proof.* Let $E := \operatorname{span}(e_1, e_2, e_3) \subset \mathbb{R}^D$ ($D \ge 5$), let $U$ be uniform on the unit sphere $S^2 \subset E$, and set $X = R U$ on $\mathcal{M} = R S^2$ ($d_{\mathcal{M}} = 2$, $d' = 5$). Projection onto the first five coordinates is an isometric linear embedding of $\mathcal{M}$.

For $z = t R u \in t\mathcal{M}$ ($\|u\| = 1$, $u \in E$), expanding $\|z - tRv\|^2$ and using $\|v\| = 1$ shows that the posterior density of $U = v$ with respect to normalized surface area is proportional to $\exp(\kappa u^\top v)$, where $\kappa := t^2 R^2 / \sigma^2$. By rotational invariance about $u$, the posterior mean lies along $u$. Under uniform surface area on $S^2$, the height $a := u^\top v$ is uniform on $[-1, 1]$, so its tilted expectation is the Langevin function
$$
A(\kappa) := \frac{d}{d\kappa}\log\biggl(\frac{1}{2}\int_{-1}^1 e^{\kappa a}\,da\biggr)
= \coth\kappa - \frac{1}{\kappa},
\qquad
f_t(tRu) = R A(\kappa)\, u.
$$
Thus $f_t|_{t\mathcal{M}}$ is a scalar linear map for *every* radius $R > 0$ and time $t \in (0, 1)$. Setting
$$
t := A(1) = \coth(1) - 1 \approx 0.313035285499,
\qquad
R := \frac{1 - t}{t} \approx 2.194528049465,
$$
we have $0 < t < 1$ (as $a \in [-1, 1]$ is nondegenerate and positively tilted), $\kappa = (tR/\sigma)^2 = 1$, and $R A(1) = tR$, so **$f_t(z) = z$ for every $z \in t\mathcal{M}$**. (Globally on $\mathbb{R}^D$, $f_t$ is non-affine because $\|f_t(z)\| \le R$ everywhere while $f_t$ is nonconstant on $t\mathcal{M}$, consistent with Lemma `lem:not-affine`.)

Now take any depth-$L$ residual softmax attention stack and set $W_O = 0$ in every attention block. Each residual layer acts as the identity, and the affine input/output maps copy the three signal coordinates (with zero in the remaining output coordinates). If fixed additive positional encodings are present, reserve a signal subspace orthogonal to the span of the position vectors and let the readout project onto that subspace (finite width for any fixed token count $N$). On an image of $N$ independent sphere patches with independent Gaussian noise, the posterior factorizes across patches, so the exact same $W_O = 0$ stack attains zero error simultaneously on $(t\mathcal{M})^N$. Since the left-hand side of Conjecture 1 is $0$ while the right-hand side is strictly positive for all $C_2 > 0$ and $\Delta(s, L) \ge 0$ (even allowing instance-dependent constants), Conjecture 1 fails. $\square$

---

## 3. Depth-$L$ Exact Polynomial Content

The paper left open whether Proposition 2(ii) extends from a single softmax attention layer to arbitrary finite depth $L$. The following linear-growth argument settles it completely.

**Theorem 2 (Depth-$L$ affine polynomial content).** *Let $F : \mathbb{R}^{N \times d_{\mathrm{in}}} \to \mathbb{R}^{N \times d_{\mathrm{out}}}$ be a finite-depth composition of an affine input map, $L$ residual multi-head softmax attention blocks with affine value projections, and an affine readout. If $F$ agrees with a polynomial map on a nonempty open subset of $\mathbb{R}^{N \times d_{\mathrm{in}}}$, then $F$ is affine on all of $\mathbb{R}^{N \times d_{\mathrm{in}}}$.*

*Proof.* For a token matrix $X = (x_1, \dots, x_N)$, define $M(X) := \max_j \|x_j\|$. Because each softmax head outputs a convex combination of affine transforms of the tokens $x_j$, every residual block $h$ satisfies a linear growth bound $M(h(X)) \le a M(X) + b$ with finite parameter-dependent constants $a, b \ge 0$. Composing $L$ such blocks with affine input and output maps gives $M(F(X)) \le \tilde{a} M(X) + \tilde{b}$ globally. Moreover, every softmax denominator is strictly positive, so $F$ is real-analytic on the connected space $\mathbb{R}^{N \times d_{\mathrm{in}}}$. Agreement with a polynomial $P$ on a nonempty open set therefore forces $F \equiv P$ everywhere by real-analytic uniqueness. If $\deg(P) \ge 2$, choose a direction $V$ where the top-degree homogeneous part of $P$ is nonzero; evaluating $P(\lambda V)$ as $\lambda \to \infty$ contradicts linear growth. Hence $\deg(P) \le 1$, so $F$ is affine. $\square$

**Token conventions:** On a single token ($N = 1$), every softmax weight equals $1$, so all pure-attention stacks are affine and form a closed finite-dimensional space. On $N \ge 2$ tokens with shared affine embedding and *no* positional encodings, identical input tokens $(x, \dots, x)$ remain identical through all layers, forcing affine dependence along the diagonal and obstructing uniform approximation of nonlinear patchwise polynomials on sets containing the diagonal. (Theorem 1 works in both conventions.)

---

## 4. Small-Logit Taylor Squaring and Matched-Order Uniform Closure

Although finite-depth pure softmax attention contains no exact nonlinear polynomials (Theorem 2), its compact-uniform closure *does* contain all degree-$2^L$ patchwise polynomials at the same budget order $O(H^2)$ once fixed distinguishable positional encodings are present—at the cost of diverging output weight norms.

**Theorem 3 (Small-logit squaring gadget and uniform closure).** *Suppose $N = 2$ patches carry fixed distinguishable positional encodings (or any fixed $N \ge 2$ patches with a nonconstant linear position tag isolated from the signal channels). Then every degree-$K_L$ ($K_L = 2^L$) vector polynomial in $\pi(z)$ on a compact domain lies in the uniform closure of depth-$L$ residual softmax attention with stream width $2H + 1$ and $4(2H + 1)^2 = O(H^2)$ parameters per layer, where $H = \binom{d' + K_L}{d'}$. The approximating family uses output projection entries scaling as $O(\epsilon^{-2})$ as the uniform error $O(\epsilon^2) \to 0$.*

*Proof.* For $N = 2$, translate the midpoint of the two distinct additive position vectors using the embedding bias and allocate stream coordinates for $H$ data channels $w_1, \dots, w_H$, a position tag $p_j \in \{+1, -1\}$, and a constant channel $1$ (orthogonal to the position difference). A scalar head with query $q_i$ and keys and values equal to $p_j$ outputs $T(q_i) = \tanh(q_i)$ at patch $i$. Fix $b := \operatorname{arctanh}(1/2)$, where $T'(b) = 3/4$ and $T''(b) = -3/4$. Using three heads with queries $b + \epsilon w_i$, $b - \epsilon w_i$, and $b$ (the constant-query head $b$ is shared across all $H$ channels, giving $2H + 1$ scalar heads total), set the output projection to combine them with the residual into
$$
S_\epsilon(w) := w
+ \frac{T(b + \epsilon w) + T(b - \epsilon w) - 2T(b)}{\epsilon^2 T''(b)}
- \frac{T(b + \epsilon w) - T(b - \epsilon w)}{2\epsilon T'(b)}.
$$
By Taylor's theorem, $S_\epsilon(w) = w^2 + O(\epsilon^2)$ uniformly on every compact interval (the odd difference cancels the residual $w$ to $O(\epsilon^2)$). Output projections write only to the data channels, leaving the tag and constant channels intact. Iterating $L$ blocks yields uniform convergence to $w^{2^L}$ on any compact set by induction using the local Lipschitz constant of squaring on an enlarged compact interval. Initializing the $H$ channels to a basis of affine forms $\ell_r(\pi(z))$ whose $K_L$-th powers span $\Pi_{K_L}(\mathbb{R}^{d'})$ (Lemma `lem:waring` of arXiv:2611.10857) and applying the same linear readout as the Volterra stack approximates any degree-$K_L$ polynomial uniformly on compact sets with $4(2H+1)^2$ dense projection parameters per layer.

For general fixed $N \ge 2$ with any nonconstant scalar position tag $p_j$, the head response is $T(q) = \sum_{j=1}^N p_j e^{q p_j} / \sum_{j=1}^N e^{q p_j}$. Since $T'(q)$ is the strictly positive tilted variance of $p_j$ and $T$ is bounded and nonconstant, there exists $b \in \mathbb{R}$ with $T''(b) \neq 0$, and the exact same formula for $S_\epsilon$ applies at $b$. $\square$

*(Weight-norm contrast: In Theorem 3, the output weights grow as $O(\epsilon^{-2})$—permissible under the parameter-count budget of Conjecture 1, though precluded by a fixed operator-norm cap. By contrast, Theorem 1 sets $W_O = 0$ identically and avoids any weight growth or limiting procedure.)*

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/manifold_attn_gap/explore.py
```

1. **Sphere quadrature check:** Evaluates the exact Gaussian posterior integral on $R S^2$ ($t = 0.313035285499$, $R = 2.194528049465$) using 32 Gauss–Legendre heights $\times$ 64 azimuths across 97 seeded directions; maximum squared discrepancy from $f_t(z) = z$ is **$1.855 \times 10^{-29}$**.
2. **Small-logit polynomial check:** Evaluates explicit $(W_Q, W_K, W_V, W_O)$ matrices (width $7$, $4 \times 7^2 = 196$ parameters/layer) and row softmax on 1,681 distinct image pairs for $K = 2^L$ ($L = 1, 2, 3$), targeting $(x^K - y^K/2,\, ((x-y)/2)^K + y^K/4)$:

| Degree $2^L$ | Depth $L$ | Max sq.\ err.\ ($\epsilon = 0.2$) | ($\epsilon = 0.025$) | ($\epsilon = 0.00625$) |
|---:|---:|---:|---:|---:|
| 2 | 1 | $7.163 \times 10^{-4}$ | $1.763 \times 10^{-7}$ | $6.888 \times 10^{-10}$ |
| 4 | 2 | $4.766 \times 10^{-3}$ | $1.249 \times 10^{-6}$ | $4.885 \times 10^{-9}$ |
| 8 | 3 | $2.206 \times 10^{-2}$ | $6.312 \times 10^{-6}$ | $2.472 \times 10^{-8}$ |

Across the last three halvings of $\epsilon$, every maximum squared error drops by a factor between $15$ and $17$, confirming the $O(\epsilon^2)$ function error ($O(\epsilon^4)$ squared error).
