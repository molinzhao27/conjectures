# Refutation of the Four Architectural Conjectures in Power Law Graph Attention

**Target paper:** Burc Gokden, *Power law graph attention: exact generalization of scaled dot-product attention, empirical collapse at inference*, [arXiv:2608.10288](https://arxiv.org/abs/2608.10288), Section 11 (`sec:conjectures`: `conj:invariance`, `conj:spectral`, `conj:functorial`, `conj:modeltransfer`).  
**Companion files:** [`solutions/powerlaw_graph_attn/explore.py`](explore.py), [`solutions/powerlaw_graph_attn/proof.pdf`](proof.pdf).

---

## Summary of Verdicts and Epistemic Scope

All **four** formal conjectures in arXiv:2608.10288 are **DISPROVED** as mathematical implications of the stated PLGA architecture, definitions, and refit rules (in exact real arithmetic). At the same time, we separate what is refuted mathematically for the stated architecture from what remains an open empirical question for trained checkpoints under specific optimizers, modified symmetrizations, or expanded refit sets:

| Conjecture | Mathematical verdict (stated architecture & definitions) | Open empirical / training-dependent scope |
|---|---|---|
| `conj:invariance` (Rigidity & power-law exponents) | **DISPROVED (Theorem 1).** Uniform pairwise relative fluctuation $m_T \to 0$ across all four tensor types $(A, B, U, G)$ still permits a positive limiting absolute RMS radius ($\approx 0.829364$). Exact $m = 0$ still permits alternating constant limits; every positive vanishing rate $h(d, N)$ (including $1/\log(2+dN)$) is realizable, even at population cross-entropy optima. | A specific optimizer, initialization, and schedule might select bounded, coherently centered operators and an empirical scaling rate. |
| `conj:spectral` (Spectral order parameter & monotonicity) | **DISPROVED (Theorem 2).** Under the ideal causal mask, the unique stationary law is $\delta_1$ and $L^2_0(\delta_1) = \{0\}$ for *every* checkpoint. Under a finite additive mask, $m = 0$ allows either a uniform spectral gap or an atomic edge limit $\delta_1$, and even any prescribed probability law on $[0, 1]$ (including any $\vartheta > 0$). Conversely, $m_G \to 2$ is compatible with an arbitrarily small gap via invisible key coordinates. Readout bias breaks universal monotonicity in $\vartheta$. | A different symmetrization, observable rule, or performance statistic restricted to specific trained checkpoints would define a separate empirical hypothesis. |
| `conj:functorial` (Congruence alignment) | **DISPROVED (Theorem 3).** Exact key sign-flip invariance produces predictively identical optima on the same data law with exact $m = 0$ whose invariant forms $G_1 = J$ and $G_2 = -J$ (or full-rank $\pm(I+J)$) have entry-RMS distance $\ge 1$ under *every* invertible real congruence $S$, even with unbounded condition number. | Whether a specific training trajectory preferentially selects aligned query/key frames or restricted congruence classes requires additional evidence. |
| `conj:modeltransfer` (Whole-model transfer) | **DISPROVED (Theorem 4).** Even literal equality of frozen $(G^*, P)$ on two domains with identical long-range two-time laws at all lags $k \ge 2$ gives an irreducible 25-percentage-point accuracy deficit ($0.50$ vs. $0.75$) when LayerNorm stays frozen under the stated refit list, at matched parameter/token/FLOP budgets. | Parity under expanded refit lists (e.g., unfreezing LayerNorm), specific pretrained checkpoints, domains, or wider tolerances remains empirical. |

---

## 1. Exact Pipeline, Diagnostic Order Parameter, and Dependencies

In one PLGA head of width $d = d_k$, write $Q_r, K_r$ for the projected, rotary-rotated queries and keys (RoPE applies *before* both the Gram and score computations). With $f(z) = z^2 / (1 + e^{-z})$, the exact pipeline is
$$
D = Q_r^\top Q_r, \quad
A = \Phi_{\mathrm{res}}(\mathrm{LN}(D)), \quad
B = A_{\mathrm{lm}} = f(W A + b_W) + 10^{-9}, \quad
U = A_p = B^{\odot P}, \quad
G = G_{\mathrm{lm}} = a U + b_a,
$$
where nonlinearities and powers act entrywise, left multiplications are ordinary matrix products, and biases are full matrices. The residual network $\Phi_{\mathrm{res}}$ acts row-wise with shared parameters: each of its $8$ reference units adds a two-SwiGLU branch and applies feature-axis LayerNorm with learned affine parameters and variance offset $\epsilon_{\mathrm{LN}} = 10^{-6}$. The position-space attention kernel is $E_{\mathrm{lm}} = \operatorname{softmax}_{\mathrm{mask}}(Q_r G K_r^\top / \sqrt{d})$, where the ideal causal mask admits precisely $j \le i$ in row $i$, whereas a finite additive penalty $-C$ ($0 < C < \infty$) for $j > i$ has different real-arithmetic semantics.

For a tensor $T$ with $\nu$ entries, define $r(T) := \|T\|_F / \sqrt{\nu}$. Across the diagnostic family $\mathcal{D} = (A, B, U, G)$, the pairwise relative fluctuation for independent inputs $x, y$ is
$$
m_T(x, y) := \frac{2\, r(T(x) - T(y))}{r(T(x)) + r(T(y))}
$$
when $r(T(x) - T(y)) > 0$, and $0$ otherwise (including $T(x) = T(y) = 0$). Absolute invariance requires a deterministic center $T^*$ with $\operatorname{ess\,sup}_x r(T(x) - T^*) \le \varepsilon$; below, the **radius** means the smallest such $\varepsilon$.

- **Exact conditional implication:** At fixed width, $m_T = 0$ almost surely for independent draws iff $T$ is almost surely constant (by Fubini). If additionally $r(T) \le M$ almost surely and $m_T \le \eta$ a.s., the same reference-input argument gives radius $\le M\eta$ (and in $L^2$, $\mathbb{E}\,r(T - \mathbb{E}T)^2 = \tfrac{1}{2}\mathbb{E}\,r(T(x) - T(y))^2 \le \tfrac{1}{2} M^2 \mathbb{E} m_T^2$). However, $M$ and the center depend on width; without a uniform RMS bound and an identification of changing tensor spaces, $m_T \to 0$ implies neither absolute convergence nor a rate.
- **Logical note on `thm:collapse` and `prop:gap`:** Exact constancy of $G$ permits replacing the generator at inference (absorbing $G$ into query projections without RoPE, or with RoPE iff $G$ commutes with all rotations on reachable states); at training time, the full gradient includes the adjoint-Jacobian path through the generator. Proposition `prop:gap` assumes a reversible Markov family with a limiting edge measure near $\lambda = 1$ of density $\sim s^{\vartheta - 1}$ and a uniform lower-edge gap, yielding limiting correlation $O(t^{-\vartheta})$ and propagated norm $O(t^{-\vartheta/2})$ (with constants $c\Gamma(\vartheta)$ and $2^{-\vartheta/2}\sqrt{c\Gamma(\vartheta)}$ for density $(c + o(1)) s^{\vartheta - 1}$).

---

## 2. Conjecture 1 (`conj:invariance`): Rigidity and Finite-Size Exponents

**Theorem 1 (Failure of rigidity and universal power-law exponents).** *In the PLGA architecture with even width $d$ and any diverging token budget $N$:*
1. *There is a family of models whose pairwise relative fluctuation satisfies $m_T(x, y) = O(1/d) \to 0$ uniformly across all input pairs and all four tensor types $(A, B, U, G)$, yet the optimal absolute RMS radius of $G$ converges to $c f'(1) / 2 \approx 0.829364 > 0$ (and no sequence of deterministic centers approximates $G$ in RMS even in probability).*
2. *For every prescribed function $h(d, N) > 0$ (e.g., $h(d, N) = 1 / \log(2 + dN)$), an admissible choice of $(a, b_a)$ yields bounded $G'$ with uniform $m_{G'} \to 0$ and exact RMS radius $h(d, N)$, refuting any universal monomial scaling $d^{-\alpha} N^{-\beta}$.*
3. *Exact $m_T \equiv 0$ across all four types still allows a bounded alternating limit $G = (-1)^{d/2} J$ with no cross-width limit.*
4. *All constructions above can be embedded at exact population (and rational balanced-batch) cross-entropy optima on a nondegenerate binary Markov source.*

*Proof.*
1. Use one head, one layer, even width $d$, and two equiprobable one-token inputs at position $0$ (where RoPE is the identity). Let $J = \mathbf{1}\mathbf{1}^\top$, let $v \in \{\pm 1\}^d$ alternate $+1, -1$, and realize queries $0$ and $v$ via embeddings $0, v$, unit-gain zero-shift normalization, and compensating scalar query projection. Set all $8$ residual SwiGLU branches to zero and all LayerNorms to unit gain and zero shift, except the final residual LayerNorm which has gain $t > 0$ and shift $1$.
   The two Grams are $0$ and $v v^\top$. Centered rows of $v v^\top$ have variance $1$, and each unaffined LayerNorm maps amplitude $z$ to $z / \sqrt{z^2 + 10^{-6}}$. After the initial and $8$ residual LayerNorms, the amplitude is a width-independent constant $c > 0$. Setting $W = I$, $b_W = 0$, $P = J$, $a = I/t$, $b_a = 0$ yields
   $$
   A_0 = J, \quad A_1 = J + t c\, v v^\top, \qquad
   B_i = f(A_i) + 10^{-9}, \quad U_i = B_i, \quad G_i = B_i / t.
   $$
   As $t \downarrow 0$, $r(A_1 - A_0) = O(t)$ while $r(A_0) = 1$, and $r(B_1 - B_0) = O(t)$ while $r(B_0) = f(1) + 10^{-9} > 0$. Because $m_T$ is scale-invariant, $m_T = O(t)$ uniformly for all four types $(A, B, U, G)$. Yet by Taylor expansion of $f$ at $1$, $r(G_1 - G_0) = t^{-1} r(f(J + t c\, v v^\top) - f(J)) \to c f'(1) > 0$. By the triangle inequality and midpoint center $(G_0 + G_1)/2$, the optimal radius is exactly $r(G_1 - G_0)/2 \to c f'(1)/2 \approx 0.829364$. Taking $t = 1/d$ gives uniform $m_T = O(1/d) \to 0$ with persistent radius $> 0.829$ (and since each input has probability $1/2$, no center sequence approximates both in probability).
2. Let $R := r(G_1 - G_0)/2 > 0$. Replacing $(a, b_a)$ so that $G'_i = J + (h/R)(G_i - G_0)$ gives exact radius $h(d, N)$ and bounded entry RMS with $m_{G'} \to 0$ whenever $h \to 0$. Choosing $h(d, N) = 1 / \log(2 + dN)$ contradicts any power-law equivalent $d^{-\alpha} N^{-\beta}$.
3. Setting zero final generator gain, $a = 0$, and $b_a = (-1)^{d/2} J$ gives $m_T \equiv 0$ identically for all four types while $G$ alternates between $+J$ and $-J$.
4. For the stationary two-state Markov source repeating its bit with probability $3/4$ under shifted two-token training blocks (one input token), setting the attention output projection and FFN branch to zero carries the two distinct embeddings through the residual stream, and an affine readout fits the two Bayes logits exactly, attaining conditional entropy for every generator choice above. $\square$

---

## 3. Conjecture 2 (`conj:spectral`): Spectral Order Parameter and Monotonicity

**Theorem 2 (Spectral obstructions and arbitrariness of edge laws).**
1. *(Ideal causal mask — universal obstruction for every checkpoint.)* For any attention weights under the ideal causal mask ($j \le i$), row $1$ is absorbing and $E_{i1} > 0$ for all $i$. Stationarity $\pi_1 = \pi_1 + \sum_{i > 1} \pi_i E_{i1}$ forces $\pi = \delta_1$ uniquely. Thus $L^2_0(\delta_1) = \{0\}$: no normalized stationary-mean-zero observable exists for Proposition `prop:gap` at any width or context length, for *every* trained checkpoint. (For instance, zero scores have triangular eigenvalues $1, 1/2, \dots, 1/S$, all transient except $1$.)
2. *(Finite additive mask — uniform gap at $m = 0$.)* With zero scores, constant generator ($m = 0$), and fixed finite penalty $-C$ ($0 < C < \infty$) for $j > i$, every entry satisfies $e^{-C}/S \le E_{ij} \le e^C/S$, so the unique stationary law satisfies $\pi_j \le e^C/S$ and $E_{ij} \ge e^{-2C} \pi_j$. Writing $E = \eta \mathbf{1}\pi^\top + (1 - \eta)K$ with $\eta = e^{-2C}$ makes $K$ Markov with stationary law $\pi$ (hence an $L^2(\pi)$-contraction by Jensen). On $L^2_0(\pi)$, $\|E\| \le 1 - \eta$, so the additive reversibilization $(E + E^*)/2$ has spectral radius $\le 1 - \eta$ and $E^*E$ has spectral radius $\le (1 - \eta)^2$, uniformly in $S$ and $d$.
3. *(Arbitrary edge measure at $m = 0$.)* Fix context length $S$, even $d \ge S$, finite mask penalty $C$, and any strictly positive row-stochastic matrix $H \in \mathbb{R}^{S \times S}$. In a RoPE decoder, set $W_Q = W_K = 0$ with constant biases $(1, 0)$ in every rotary plane, giving common rotated matrix $Z \in \mathbb{R}^{S \times d}$ with columns $\cos(n\theta_j), \sin(n\theta_j)$ ($n = 0, \dots, S-1$). Under the complex basis change to $e^{\pm i n \theta_j}$, any $S$ columns form a nonsingular Vandermonde minor, so $Z$ has full row rank $S$ and admits a right inverse $V$. Setting $L_{ij} := \log H_{ij} + C \mathbf{1}_{j > i}$, $a = 0$, and $b_a = \sqrt{d}\, V L V^\top$ yields pre-mask logits $L$ and attention kernel $E_{\mathrm{lm}} = H$ identically on all inputs of length $S$ with $m = 0$.
   - Taking $H = r I + (1 - r) J/S$ ($0 \le r < 1$) is reversible with uniform stationary law and spectral measure $\delta_r$ (additive) or $\delta_{r^2}$ (multiplicative): fixed $r$ gives a uniform gap, while $r = 1 - 1/S$ closes the gap to an atomic limit $\delta_1$ (limiting correlation $1$ at every fixed time).
   - More generally, **every probability measure $\mu$ on $[0, 1]$** is a weak limit at $m = 0$: at $S = 2k$, pick $k$ equal-weight quantiles $r_1, \dots, r_k$ approximating $\mu$, form $2 \times 2$ diagonal blocks $\tfrac{1}{2}\begin{psmallmatrix} 1+r_i & 1-r_i \\ 1-r_i & 1+r_i \end{psmallmatrix}$, and mix with $J/S$ of weight $1/S$ for strict positivity. For the alternating observable $f = (+1, -1, \dots, +1, -1)$, each block contributes mass $1/k$ at eigenvalue $(1 - 1/S) r_i$. This realizes both any power-law density $s^{\vartheta - 1}$ for arbitrary $\vartheta > 0$ (e.g., $\vartheta = 2$, density $2(1 - \lambda)$, fourth moment $1/15$) and singular/atomic laws, all at $m = 0$.
4. *(Order-one $m_G$ without a spectral gap.)* Reserve one extra rotary plane ($d - 2 \ge S$) with zero key coordinates, using the active $d - 2$ coordinates for the lift of $H$ above. Let two equiprobable contexts differ only at position $0$ with extra-plane query $(0, 0)$ vs. $(1, 0)$. With zero residual branches and unit LayerNorms, the extra-plane Gram row differs between the two inputs and remains distinct after LayerNorm and $f$. Setting $W = I$, $b_W = 0$, $P = 1$ in the two extra columns and $0$ elsewhere, shifting $b_a$ to cancel the first input's extra columns and add the lifted $H$-form, and scaling $a = u I$ produces a difference $\Delta G$ supported solely on zero-key columns. As $u \to \infty$, $m_G \to 2$ while both inputs retain the exact attention kernel $H$ (even with $r \uparrow 1$, violating any uniform subcritical gap under a diagnostic-only criterion).
5. *(Failure of universal benchmark monotonicity.)* Changing only the readout bias alters argmax predictions arbitrarily without changing any attention matrix, stationary measure, or edge exponent $\vartheta$. $\square$

*(Epistemic note: modifying the problem via non-stationary symmetrizations, restart kernels, or undirected graphs, or restricting to an empirical population correlation over specific trained checkpoints, would pose a different question not fixed by `conj:spectral`.)*

---

## 4. Conjecture 3 (`conj:functorial`): Congruence Alignment

Under a common query/key frame change $q' = S q$, $k' = S k$, preserving $q^\top G k$ requires congruence $G' = S^{-\top} G S^{-1}$. However, the PLGA architecture has an exact **independent key sign-flip symmetry**: negating $(a, b_a, W_K, b_K)$ negates both $G$ and $K_r$, leaves $Q_r$ and the entire generator pipeline $(D, A, B, U)$ unchanged, and preserves all bilinear scores $Q_r G K_r^\top$ at every position (including under RoPE), hence the exact decoder function and training loss.

**Theorem 3 (Failure of congruence alignment).** *On the same nondegenerate data distribution (either the two-token Markov source of Section 2 or the long-memory source of Section 5), there exist two PLGA population optima with exact invariance $m_T \equiv 0$ for all four tensor types and identical predictive functions, whose invariant forms $G_1 = J$ and $G_2 = -J$ (or full-rank $G_1 = I + J$ and $G_2 = -(I + J)$) satisfy*
$$
r(G_2 - S^{-\top} G_1 S^{-1}) \ge 1
$$
*for every invertible real matrix $S \in \mathrm{GL}(d)$, uniformly in width $d$ and without any condition-number restriction on $S$.*

*Proof.* Construct the first optimum with $a = 0$, constant final generator gain, and $b_a = J$ (resp. $I + J$), then apply the key sign-flip symmetry to obtain the second optimum with $G_2 = -G_1$ and identical predictions. For any $S \in \mathrm{GL}(d)$, $B := S^{-\top} G_1 S^{-1}$ is symmetric positive semidefinite. Taking the unit vector $w = \mathbf{1} / \sqrt{d}$ gives $w^\top G_2 w \le -d$ and $w^\top B w \ge 0$, so
$$
\|G_2 - B\|_F \ge \|G_2 - B\|_{\mathrm{op}} \ge |w^\top (G_2 - B) w| = |w^\top G_2 w| + w^\top B w \ge d.
$$
Dividing by $\sqrt{d^2} = d$ gives $r(G_2 - B) \ge 1$ (and operator-norm error $\ge d$). $\square$

---

## 5. Conjecture 4 (`conj:modeltransfer`): Whole-Model Transfer with Frozen Normalization

We construct a matched-architecture synthetic benchmark adhering literally to the refit list of `conj:modeltransfer` (freezing the extracted pair $(G^*, P)$ and core including the final post-FFN LayerNorm, while refitting embeddings, linear attention projections, and readout) on two domains with identical long-range statistics.

**Two domains in the same long-range class.** Let tokens be three sign bits $(H_t, B_t, C_t) \in \{\pm 1\}^3$ under shifted two-token training blocks (one input token), evaluated on next-token accuracy for bit $C_{t+1}$. In both domains, generate an independent stationary long-memory sign process $H$: draw $R$ once per trajectory from $\tfrac{1}{2}\operatorname{Unif}[0, 1] + \tfrac{1}{2}\delta_{-1/2}$, start a fair sign $H_0$, and preserve the sign at each step with probability $(1 + R)/2$. Then for all $k \ge 1$,
$$
\mathbb{E}[H_t H_{t+k}] = \mathbb{E}[R^k] = \frac{1}{2(k + 1)} + \frac{1}{2}\Bigl(-\frac{1}{2}\Bigr)^k \sim \frac{1}{2k},
$$
with neighboring signs independent ($\mathbb{E}[H_t H_{t+1}] = 0$).
- **Source domain $\mathcal{D}_1$:** $B, C$ are independent fair i.i.d. sign sequences, independent of $H$.
- **Target domain $\mathcal{D}_2$:** $B$ is fair i.i.d., independent of $H$, and $C_t = B_{t-1} Z_t$ where $Z_t \in \{\pm 1\}$ are i.i.d. with $\mathbb{P}(Z_t = +1) = 3/4$.

At every lag $k \ge 2$, the **entire two-time joint distribution** of $(H_t, B_t, C_t)$ and $(H_{t+k}, B_{t+k}, C_{t+k})$ is identical across $\mathcal{D}_1$ and $\mathcal{D}_2$, so both domains share the exact long-range exponent and all large-lag two-point statistics.

**Theorem 4 (Irreducible 25-point transfer deficit under frozen LayerNorm).** *At any even width $d$ and one layer, with literal equality of the frozen pair $(G^*, P) = (J, J)$:*
1. *($\mathcal{D}_1$ optimum):* Setting zero gain and shift in the final post-FFN LayerNorm (sending every final hidden row to $0$), zero readout weights/biases, and constant generator $(G^*, P) = (J, J)$ attains the exact population cross-entropy minimum on $\mathcal{D}_1$ (since $x_{t+1}$ is conditionally uniform given $x_t$).
2. *(Frozen transport ceiling):* Keeping the core and final LayerNorm frozen while refitting embeddings, linear projections, and readout arbitrarily still sends every final hidden row to $0$. The readout can emit only a constant distribution; because $C_{t+1}$ is marginally fair on $\mathcal{D}_2$, accuracy is identically $1/2$ at every width and token/FLOP budget.
3. *(Matched target control):* Keeping the exact same $(G^*, P) = (J, J)$ and generator but setting unit gain on the final LayerNorm, embedding $(H_t, B_t, C_t) \mapsto B_t v$ (balanced $v \in \{\pm 1\}^d$), and zeroing attention output projection and FFN passes a fixed positive multiple of $B_t v$ to the readout. Linear readout weights giving logits $\tfrac{1}{2}(\log 3) B_t c'$ attain the exact one-step Bayes law on $\mathcal{D}_2$ with accuracy $3/4$. Moreover, starting from zero readout weights, a **single** gradient step on a batch with the rational population pair frequencies already achieves accuracy $3/4$ (since $\mathbb{E}[B_t C_{t+1}] = 1/2$ and readout biases have zero gradient).

*Thus frozen-LayerNorm transport suffers an irreducible $0.25$ accuracy deficit against a matched-budget control.* $\square$

*(Epistemic caveat: Theorem 4 refutes a universal parity guarantee for the stated frozen-LayerNorm refit list and any tolerance $< 0.25$. It does not rule out empirical parity under expanded refit lists such as unfreezing LayerNorm, specific pretrained checkpoints, or real-world corpora.)*

---

## 6. Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/powerlaw_graph_attn/explore.py
```

All assertions pass in `float64` (NumPy):
- **Rigidity (`conj:invariance`):** Across widths $d \in \{8, 32, 128, 256\}$ with $t = 1/d$, $\max_{T \in \{A, B, U, G\}} m_T$ drops from `0.276294557` to `0.008862811` while the absolute $G$-radius remains `0.832874631` to `0.829367569`, and exact inverse-logarithmic scaling $h = 1/\log(d+1)$ is verified.
- **Spectral checks (`conj:spectral`):** At $S \in \{4, 16, 64, 128\}$, ideal causal stationary law $\delta_1$ and triangular eigenvalues $1/k$, finite-mask ($C = 2$) additive/multiplicative reversibilization gaps (`0.89642652` and `0.90701275` at $S = 128$), RoPE Vandermonde lifts for $r \in \{0, 0.5, 0.99\}$ with invisible-key variation $m_G = 1.980100$, and the $128$-pair $\vartheta = 2$ density construction (fourth moment `0.06548019` vs. limit $1/15 \approx 0.06666667$).
- **Alignment & transfer (`conj:functorial`, `conj:modeltransfer`):** Key sign-flip score invariance, 16 sample congruences verifying $\mathrm{RMS}(-J - S^{-\top} J S^{-1}) \ge 1$, and the full one-token residual-stream transfer benchmark confirming frozen accuracy `0.50` vs. one-step control optimum `0.75`.
