# Refutation of Cross-Task Linearity Transitivity

**Target paper:** Zhanpeng Zhou, Zijun Chen, Yilan Chen, Bo Zhang, and Junchi Yan, *On the Emergence of Cross-Task Linearity in Pretraining-Finetuning Paradigm*, [arXiv:2402.03660](https://arxiv.org/abs/2402.03660) (ICML 2024), Section 4.1, Conjecture 4.1 (`con:transitivity`; restated in the Appendix as `def:suppl_transitivity` and used in `lem:multi_interpolation`, `thm:linearity_multi`, and `thm:ctl_connects_wd`).  
**Companion files:** [`solutions/cross_task_linearity/explore.py`](explore.py), [`solutions/cross_task_linearity/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 4.1 (`con:transitivity`): FALSE.**
- **Collinear one-layer refutation (Theorem 1):** Three collinear checkpoints $A, B, C$ of a one-layer ReLU network satisfy **exact** cross-task linearity (CTL) on $(A, B)$ and $(B, C)$ for every positive input $x > 0$ and every interpolation weight $t \in [0, 1]$, while on $(A, C)$ at $t = 1/2$ the Euclidean interpolation gap grows without bound ($Mx/\sqrt{2}$), the relative error tends to $1$, and both empirical diagnostics used in Zhou et al.—$\mathrm{cosine}$ and $\mathrm{coef}$—tend to $0$ (from $1$ on both premise pairs).
- **Shared-checkpoint fine-tuning and strictly active depth-two refutations (Propositions 2 and 3):** The one-layer failure is realized by three exact global minimizers reached in one full-batch gradient descent step from a single pretrained global minimizer (in a stylized setup with a frozen baseline unit and step size $2$). A second, two-layer counterexample exhibits unbounded failure inside a single open activation cone where every ReLU is strictly active throughout the triangle $\operatorname{conv}(A, B, C)$ (and identically in a deep linear network).
- **Exact characterizations (Theorem 4):** We establish exact necessary and sufficient conditions for transitivity of exact CTL in one-layer ReLU networks (failing iff some neuron and input satisfy $b = 0$ and $ac < 0$), in arbitrary-depth feedforward affine/ReLU networks (via an exact layer-order quadratic endpoint criterion), and on general convex parameter domains (via pairwise disjointness of maximal convex affine regions).
- **Scope note:** Zhou et al. state `con:transitivity` informally using $\approx$ without fixing a norm or tolerance, motivated empirically by large pretrained models (such as ViT and ResNet). Our counterexamples refute exact CTL, any uniform $\varepsilon$-CTL bound depending only on the two premise errors, and arbitrary near-$1$ acceptance thresholds on both $\mathrm{cosine}$ and $\mathrm{coef}$; they do not address how frequently approximate CTL holds empirically in practical large-scale fine-tuning.

---

## 1. Problem Setup and Quantitative Formulations

Given models $\theta_p, \theta_q$ fine-tuned on datasets $\mathcal{D}_p, \mathcal{D}_q$ and interpolation parameter $t \in [0, 1]$, write $\theta_t := (1 - t)\theta_p + t\theta_q$. Zhou et al. say that $(\theta_p, \theta_q)$ satisfies **Cross-Task Linearity (CTL)** on $\mathcal{D}_p \cup \mathcal{D}_q$ if for every layer $\ell \in \{1, \dots, L\}$, every $t \in [0, 1]$, and every input $x \in \mathcal{D}_p \cup \mathcal{D}_q$,
$$
f^{(\ell)}(\theta_t; x) \approx (1 - t) f^{(\ell)}(\theta_p; x) + t f^{(\ell)}(\theta_q; x),
$$
where $f^{(\ell)}(\theta; x)$ denotes the post-activation feature vector at layer $\ell$ (with layer $L$ being the network output). Because $\approx$ has no formal norm or tolerance in the paper, the authors evaluate CTL via two per-example diagnostics comparing $h := f^{(\ell)}(\theta_t; x)$ and $g := (1 - t)f^{(\ell)}(\theta_p; x) + t f^{(\ell)}(\theta_q; x)$:
$$
\mathrm{cosine}_t^{(\ell)}(x) := \frac{\langle h, g \rangle}{\|h\|_2 \|g\|_2}, \qquad
\mathrm{coef}_t^{(\ell)}(x) := \frac{\|h\|_2\,\mathrm{cosine}_t^{(\ell)}(x)}{\|g\|_2} = \frac{\langle h, g \rangle}{\|g\|_2^2}.
$$
**Conjecture 4.1 (`con:transitivity`)** asserts that for any three models $A, B, C$, if $(A, B)$ and $(B, C)$ satisfy CTL, then $(A, C)$ satisfies CTL.

To cover every formal reading of $\approx$, we consider:
1. **Exact CTL:** equality $h = g$ for all $\ell, t, x$.
2. **Uniform $\varepsilon$-CTL:** $E_{\mathcal{D}}(p, q) := \sup_{\ell, x \in \mathcal{D}, t \in [0, 1]} \|f^{(\ell)}(\theta_t; x) - ((1 - t)f^{(\ell)}(\theta_p; x) + t f^{(\ell)}(\theta_q; x))\|_2 \le \varepsilon$.
3. **Diagnostic thresholds:** $\mathrm{cosine}_t^{(\ell)}(x) \ge 1 - \tau$ and $|\mathrm{coef}_t^{(\ell)}(x) - 1| \le \tau$.

In all counterexamples below, the premise identities hold identically for **every** positive input $x > 0$ simultaneously, so any pair-specific choice of positive datasets $\mathcal{D}_A, \mathcal{D}_B, \mathcal{D}_C \subset (0, \infty)$ is handled uniformly.

---

## 2. One-Layer Collinear Counterexample and Shared-Checkpoint GD

Let $\sigma(s) := \max(s, 0)$ denote ReLU. On scalar input $x > 0$, define the one-layer network with two trainable weights $(u, v) \in \mathbb{R}^2$ and a fixed bias $d > 0$ on a third constant ReLU unit:
$$
F((u, v); x) := \bigl(\sigma(ux),\; \sigma(vx),\; d\bigr) \in \mathbb{R}^3.
$$

**Theorem 1 (Collinear one-layer refutation).** *Fix $M > 0$ and $d > 0$, and take the three collinear checkpoints*
$$
A = (M, -M), \qquad B = (0, 0), \qquad C = (-M, M).
$$
1. *For every $x > 0$ and every $t \in [0, 1]$, exact CTL holds on both $(A, B)$ and $(B, C)$, with $\mathrm{cosine}_t^{(1)}(x) = \mathrm{coef}_t^{(1)}(x) = 1$.*
2. *On $(A, C)$ at the midpoint $t = 1/2$ (where $\frac{1}{2}A + \frac{1}{2}C = B$), the actual and target features are $h = (0, 0, d)$ and $g = (Mx/2, Mx/2, d)$, giving*
   $$
   \|h - g\|_2 = \frac{Mx}{\sqrt{2}}, \qquad
   \frac{\|h - g\|_2}{\|g\|_2} = \frac{Mx/\sqrt{2}}{\sqrt{d^2 + (Mx)^2/2}},
   $$
   $$
   \mathrm{cosine}_{1/2}^{(1)}(x) = \frac{d}{\sqrt{d^2 + (Mx)^2/2}}, \qquad
   \mathrm{coef}_{1/2}^{(1)}(x) = \frac{d^2}{d^2 + (Mx)^2/2}.
   $$

*Proof.* Along $(A, B)$, the interpolated weights are $((1 - t)M, -(1 - t)M)$, so for every $x > 0$ and $t \in [0, 1]$,
$$
F((1 - t)A + tB; x) = \bigl((1 - t)Mx,\; 0,\; d\bigr) = (1 - t)F(A; x) + tF(B; x).
$$
Symmetrically, along $(B, C)$, the weights are $(-tM, tM)$ and $F((1 - t)B + tC; x) = (0, tMx, d) = (1 - t)F(B; x) + tF(C; x)$. Since $d > 0$, every feature vector is nonzero, so $\mathrm{cosine} = \mathrm{coef} = 1$ everywhere on both segments. On $(A, C)$ at $t = 1/2$, $\frac{1}{2}A + \frac{1}{2}C = (0, 0) = B$, yielding $h = (0, 0, d)$ and $g = (Mx/2, Mx/2, d)$. Thus $\|h - g\|_2^2 = (Mx/2)^2 + (Mx/2)^2 = (Mx)^2/2$, $\langle h, g \rangle = d^2$, $\|h\|_2 = d$, and $\|g\|_2^2 = d^2 + (Mx)^2/2$, which gives all four formulas immediately. $\square$

Taking $x = 1$ (a singleton evaluation set, so supremum and empirical mean coincide) and letting $M/d \to \infty$ drives $\|h - g\|_2 \to \infty$, $\|h - g\|_2 / \|g\|_2 \to 1$, $\mathrm{cosine}_{1/2}^{(1)}(1) \to 0$, and $\mathrm{coef}_{1/2}^{(1)}(1) \to 0$, while both premise errors are identically $0$. Hence exact CTL, every uniform $\varepsilon$-CTL implication depending only on premise errors, and every nontrivial near-$1$ threshold on $\mathrm{cosine}$ or $\mathrm{coef}$ fail—even for three collinear checkpoints.

**Proposition 2 (Shared-checkpoint gradient-descent construction).** *Set $M = 1$ and $d = 1/10$ in $F((u, v); x)$, keep the constant unit $d$ frozen, and train $(u, v) \in \mathbb{R}^2$ under the average squared one-hot loss $\mathcal{L}_{\mathcal{D}}(u, v) := \frac{1}{|\mathcal{D}|} \sum_{(x, y) \in \mathcal{D}} \frac{1}{2}\|F((u, v); x) - e_y\|_2^2$ for $y \in \{1, 2, 3\}$.*
1. *On the pretraining set $\mathcal{D}_0 = \{(1/2, 1), (1/2, 2)\}$, the point $P = (1, 1)$ is the unique global minimizer of $\mathcal{L}_{\mathcal{D}_0}$ over all $(u, v) \in \mathbb{R}^2$.*
2. *Starting from $P = (1, 1)$, one full-batch gradient descent step of step size $\eta = 2$ on the three downstream tasks*
   $$
   \mathcal{D}_A = \{(1, 1)\}, \qquad
   \mathcal{D}_B = \bigl\{(3/5, 3),\, (4/5, 3)\bigr\}, \qquad
   \mathcal{D}_C = \{(1, 2)\}
   $$
   *produces $A = (1, -1)$, $B = (0, 0)$, and $C = (-1, 1)$, respectively. Each model attains the exact global minimum of its task loss over $\mathbb{R}^2$ and achieves $100\%$ $\arg\max$ classification accuracy on its training task and on any held-out positive input $x \ge 1$ labeled by its task.*

*Proof.* Set $a := \sigma(u/2) \ge 0$ and $b := \sigma(v/2) \ge 0$. On $\mathcal{D}_0$,
$$
\mathcal{L}_{\mathcal{D}_0}(u, v)
= \frac{1}{4}\bigl((a - 1)^2 + a^2 + b^2 + (b - 1)^2 + 2d^2\bigr)
= \frac{1}{2}\Bigl(a - \frac{1}{2}\Bigr)^2 + \frac{1}{2}\Bigl(b - \frac{1}{2}\Bigr)^2 + \frac{1}{4} + \frac{d^2}{2}.
$$
The unique minimum occurs at $a = b = 1/2$ (any nonpositive weight gives $a = 0$ or $b = 0$, raising that coordinate's contribution from $1/8$ to $1/4$ in the averaged coordinate sum), forcing $u = v = 1$.

At $P = (1, 1)$, every preactivation $ux, vx$ on $\mathcal{D}_A, \mathcal{D}_B, \mathcal{D}_C$ is strictly positive, so $\mathcal{L}$ is smooth near $P$ with gradients
$$
\nabla \mathcal{L}_{\mathcal{D}_A}(P) = (0, 1), \qquad
\nabla \mathcal{L}_{\mathcal{D}_B}(P) = \Bigl(\frac{(3/5)^2 + (4/5)^2}{2},\, \frac{(3/5)^2 + (4/5)^2}{2}\Bigr) = \Bigl(\frac{1}{2}, \frac{1}{2}\Bigr), \qquad
\nabla \mathcal{L}_{\mathcal{D}_C}(P) = (1, 0).
$$
Applying $P - 2\nabla \mathcal{L}(P)$ yields $A = (1, -1)$, $B = (0, 0)$, and $C = (-1, 1)$. For each task, the resulting model matches both trainable target coordinates identically (hence is a global minimizer over $\mathbb{R}^2$ with $d$ frozen), and for every $x \ge 1$ the score vectors $(x, 0, 1/10)$, $(0, 0, 1/10)$, and $(0, x, 1/10)$ attain their unique maximum at classes $1, 3, 2$, respectively. Evaluating on held-out inputs $x = 2$ (or arbitrarily large $x > 0$) realizes Theorem 1 on test unions with fixed fine-tuned weights. $\square$

*Remark (Stylized GD caveat).* Proposition 2 is an intentionally stylized training construction: the constant baseline unit $d$ is frozen during fine-tuning (otherwise class-3 examples would produce a nonzero gradient on $d$), and the step size $\eta = 2$ is large enough to step across $0$ in a single update.

---

## 3. Failure Inside a Single Strictly Active Cone

The failure of transitivity is not solely an artifact of ReLU kinks crossing zero.

**Proposition 3 (Strictly active two-layer counterexample).** *Consider the two-layer scalar network $h((u, v); x) := \sigma(ux)$ and $y((u, v); x) := \sigma(v\,h((u, v); x))$ for $x > 0$ and $(u, v) \in (0, \infty)^2$ (where $h = ux$ and $y = uvx$, identical to a two-layer deep linear network). For any $r, s > 0$, set*
$$
A = (1 + r, 1), \qquad B = (1, 1), \qquad C = (1, 1 + s).
$$
*Then every neuron is strictly active throughout $\operatorname{conv}(A, B, C)$, exact layerwise CTL holds on $(A, B)$ and $(B, C)$ for all $x > 0$ and $t \in [0, 1]$, and along $(A, C)$ layer $1$ interpolates exactly while layer $2$ exhibits the exact positive gap*
$$
y\bigl((1 - t)A + tC; x\bigr) - \bigl((1 - t)y(A; x) + t\,y(C; x)\bigr) = t(1 - t)rsx,
$$
*attaining its maximum $rsx/4$ at $t = 1/2$ (e.g., for $r = s = x = 1$, endpoints $2$ vs. midpoint $9/4$).*

*Proof.* On $(A, B)$, $v \equiv 1$ is constant; on $(B, C)$, $u \equiv 1$ is constant; hence both $h = ux$ and $y = uvx$ are affine along each premise segment. On $(A, C)$, $u_t = 1 + (1 - t)r$ and $v_t = 1 + ts$, so $h_t = u_t x$ is affine while
$$
y_t - \bigl((1 - t)y(A; x) + t\,y(C; x)\bigr)
= \bigl((1 + (1 - t)r)(1 + ts) - (1 + (1 - t)r + ts)\bigr)x
= t(1 - t)rsx.
$$
More generally, on $(0, \infty)^2$, exact CTL holds on a pair $(p, q)$ iff $(u_q - u_p)(v_q - v_p) = 0$; two premise pairs can fix distinct coordinates while the third varies both. $\square$

*Remark (Diagnostics on the two-layer example).* Because the second-layer output $y > 0$ is one-dimensional and positive, $\mathrm{cosine}_t^{(2)}(x) \equiv 1$ trivially, whereas at $t = 1/2$,
$$
\mathrm{coef}_{1/2}^{(2)}(x) = \frac{y(p_{1/2}; x)}{\frac{1}{2}(y(A; x) + y(C; x))} = 1 + \frac{rs/4}{1 + (r + s)/2} \xrightarrow[r, s \to \infty]{} \infty.
$$
Appending a constant second output coordinate $d > 0$ (forming $\tilde{y} = (y, d) \in \mathbb{R}^2$) breaks the cosine diagnostic as well: for $r = s = M$ and $d = M^{3/2}x$, we have $\tilde{g}/d \to (0, 1)$ while $\tilde{h}/h \to (1, 0)$, driving $\mathrm{cosine}_{1/2}^{(2)}(x) \to 0$ as $M \to \infty$ while keeping both premise pairs exact.

---

## 4. Exact Characterizations of Transitivity

**Theorem 4 (Exact conditions for transitivity of exact CTL).**
1. *(One ReLU layer.)* For a scalar preactivation taking values $a, c \in \mathbb{R}$ at endpoints $p, q$, the ReLU chord identity $\sigma((1 - t)a + tc) = (1 - t)\sigma(a) + t\sigma(c)$ holds for all $t \in [0, 1]$ if and only if $ac \ge 0$. When $ac < 0$, the maximum gap over $t \in [0, 1]$ is exactly $|ac|/(|a| + |c|) > 0$. Consequently, on a common evaluation set, if $(A, B)$ and $(B, C)$ satisfy exact CTL for a one-layer ReLU network (with weights and biases allowed to vary), then $(A, C)$ fails exact CTL **if and only if** some neuron and input satisfy $b = 0$ and $ac < 0$ (where $a, b, c$ are the preactivations at $A, B, C$).
2. *(Arbitrary-depth feedforward affine/ReLU networks.)* For any two checkpoints $p, q$ and input $x$, suppose all layers $1, \dots, \ell - 1$ satisfy exact CTL on $(p, q)$ at $x$, with endpoint layer-$\ell$ inputs $h_p, h_q$ ($h_p = h_q = x$ for $\ell = 1$), endpoint preactivations $z_p, z_q$, and mixed increment $k := (W_q^{(\ell)} - W_p^{(\ell)})(h_q - h_p)$. Then the interpolated preactivation at layer $\ell$ is exactly
   $$
   z(t) = (1 - t)z_p + t z_q - t(1 - t)k.
   $$
   Layer $\ell$ satisfies exact CTL on $(p, q)$ at $x$ if and only if, for each coordinate $i$:
   - if layer $\ell$ is ReLU and $\max(z_{p,i}, z_{q,i}) > 0$: $z_{p,i} \ge 0$, $z_{q,i} \ge 0$, and $k_i = 0$;
   - if layer $\ell$ is ReLU and $z_{p,i}, z_{q,i} \le 0$: $\max_{t \in [0, 1]} z_i(t) \le 0$;
   - if layer $\ell$ has identity activation: $k_i = 0$.
3. *(Global characterization on a convex domain.)* Let $F : \Omega \to Y$ be any map on a convex subset $\Omega$ of a real vector space (e.g., all stacked layer features over a common dataset). Call $p \sim q$ if $F$ is affine on the line segment $[p, q]$. Then $\sim$ is transitive on $\Omega$ **if and only if** the maximal convex subsets of $\Omega$ on which $F$ is affine are pairwise disjoint.

*Proof.* **(1)** If $ac \ge 0$, $(1 - t)a + tc$ never changes sign on $[0, 1]$, so $\sigma$ acts affinely. If $ac < 0$, the piecewise-affine gap $(1 - t)\sigma(a) + t\sigma(c) - \sigma((1 - t)a + tc)$ vanishes at $t = 0, 1$ and has a unique breakpoint at $t^* = a/(a - c) \in (0, 1)$, where its value is $|ac|/(|a| + |c|) > 0$. With exact premises $ab \ge 0$ and $bc \ge 0$, we have $ac < 0$ iff $b = 0$ and $ac < 0$. (For pair-specific datasets, the same criterion applies on each pair's dataset union.)

**(2)** Expanding $z(t) = ((1 - t)W_p^{(\ell)} + tW_q^{(\ell)})((1 - t)h_p + th_q) + (1 - t)b_p^{(\ell)} + tb_q^{(\ell)}$ gives $(1 - t)z_p + t z_q - t(1 - t)(W_q^{(\ell)} - W_p^{(\ell)})(h_q - h_p)$ because $(1 - t)^2 = (1 - t) - t(1 - t)$ and $t^2 = t - t(1 - t)$. For an identity layer, $z(t) \equiv (1 - t)z_p + t z_q$ iff $k = 0$. For a ReLU coordinate $i$: if $\max(z_{p,i}, z_{q,i}) > 0$, the target chord $(1 - t)\sigma(z_{p,i}) + t\sigma(z_{q,i})$ is strictly positive on $(0, 1)$, so $\sigma(z_i(t))$ equals this chord on $(0, 1)$ iff $z_i(t) = (1 - t)\sigma(z_{p,i}) + t\sigma(z_{q,i})$ on $(0, 1)$; comparing quadratic terms forces $k_i = 0$, and continuity at $t \in \{0, 1\}$ forces $z_{p,i} = \sigma(z_{p,i}) \ge 0$ and $z_{q,i} = \sigma(z_{q,i}) \ge 0$ (which is visibly sufficient). If $z_{p,i}, z_{q,i} \le 0$, the target chord is identically $0$, so $\sigma(z_i(t)) \equiv 0$ iff $\max_{t \in [0, 1]} z_i(t) \le 0$ (checked at endpoints and any interior vertex of the quadratic). Induction over $\ell = 1, \dots, L$ finishes the proof.

**(3)** The relation $\sim$ is always reflexive and symmetric. Suppose $\sim$ is transitive, hence an equivalence relation. For $p, q$ in one equivalence class $E$ and any $r \in [p, q]$, affinity of $F$ on $[p, q]$ implies $F$ is affine on $[p, r]$, so $r \sim p$ and $r \in E$; thus $E$ is convex and $F|_E$ is affine. Any convex affine set $K$ meeting $E$ at $p$ satisfies $q \sim p$ for all $q \in K$, hence $K \subseteq E$. Therefore the equivalence classes are precisely the maximal convex affine subsets, and they are pairwise disjoint. Conversely, by Zorn's lemma (the union of any chain of convex subsets on which $F$ is affine is again convex and affine), every exact segment $[p, q]$ lies in some maximal convex affine subset. If maximal convex affine subsets are pairwise disjoint and $A \sim B$, $B \sim C$, then the maximal sets containing $[A, B]$ and $[B, C]$ share $B$ and must coincide, so $[A, C]$ lies in that common convex affine set and $A \sim C$. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/cross_task_linearity/explore.py
```

The script [`explore.py`](explore.py) performs exact rational arithmetic (`fractions.Fraction`) verifying:
- The pretraining fixed point $P = (1, 1)$ and the three single-step GD updates to $A = (1, -1)$, $B = (0, 0)$, and $C = (-1, 1)$;
- Exact premise CTL on $65$ rational weights $t = n/64$ across $12$ combinations of $(M, x, d) \in \{1, 8, 1000\} \times \{1, 2\} \times \{1/10, 1/1000\}$, together with the exact closed-form $\mathrm{cosine}^2$ and $\mathrm{coef}$ identities at $t = 1/2$;
- The one-gate sign criterion and exact gap formula $|ab|/(|a| + |b|)$ across all $121$ pairs $(a, b) \in \{-5, \dots, 5\}^2$;
- The strictly active two-layer identity $y(p_t) - ((1 - t)y(A) + t\,y(C)) = t(1 - t)rsx$ across $18$ parameter settings $(r, s, x) \in \{1, 3, 10\} \times \{1, 4, 20\} \times \{1, 3\}$ and $65$ interpolation weights each.

All assertions pass with exit code `0`.

---

## References

- Z. Zhou, Z. Chen, Y. Chen, B. Zhang, and J. Yan, *On the Emergence of Cross-Task Linearity in Pretraining-Finetuning Paradigm*, Proceedings of the 41st International Conference on Machine Learning (ICML 2024), [arXiv:2402.03660](https://arxiv.org/abs/2402.03660), 2024.
