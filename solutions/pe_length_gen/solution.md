# Length-Generalization Impossibility and Exact Feasibility for a Positional Attention Head

**Target paper:** Jiasong Han, Yunkai Chai, and Jintao Wang, *A Positional Realizability Theory of Length Generalization in Single-Head Attention*, [arXiv:2610.41218](https://arxiv.org/abs/2610.41218) (TMLR, October 2026), Conjecture `conj:general`.  
**Companion files:** [`solutions/pe_length_gen/explore.py`](explore.py), [`solutions/pe_length_gen/proof.pdf`](proof.pdf).

---

## Verdict

- **Printed implication of Conjecture `conj:general` under S1–S3: PROVED** by positive scaling of $W$. Any unconstrained head strictly solving both $L_0$ and $L$ can be positively rescaled to meet any prescribed training margin $\gamma > 0$, contradicting the quantified separation hypothesis directly. The projection premise $R_L^{\mathrm{new}} \notin \Pi_{\mathcal{K}_L}$ is superfluous and can be deleted.
- **Substantive finite feasibility criterion:** Joint strict-argmax realizability of any finite set of comparisons is characterized exactly by **Gordan's theorem of the alternative (1873)** for unconstrained $W$ (and by **Farkas' lemma (1902)** for sign-constrained families such as ALiBi).
- **Nonzero projection residual alone: DISPROVED** by an explicit degree-4 monomial `copy` witness at $L_0 = 5, L = 10$ with full-column-rank $\Phi_5$. A single admissible parameter $W$ achieves strict positive margins ($1$ at $L_0 = 5$ and $2$ at $L = 10$), yet a fifth-forward-difference obstruction proves $R_{10}^{\mathrm{new}}$ is non-realizable under **every** masking and padding convention.
- **Exact selection failure (for the min-norm interpolant $W_*$):** On the same degree-4 `copy` instance, the unique minimum-Frobenius-norm exact old-block interpolant $W_* = \Phi_5^{-1} R_5 \Phi_5^{-\top}$ attains new-row margins $(-15, 15, -35, -70, -126)$ ($\alpha_{10} = 1/5$). Note that this selection failure is established for $W_*$ and does not evaluate the paper's regularized cross-entropy selector.

---

## 1. Setup and Notation Caveats

Under hypotheses S1–S3 of arXiv:2610.41218, a deterministic positional feature map $\varphi : \mathbb{N} \to \mathbb{R}^m$ defines positional logits $s_W(i, j) = \varphi(i)^\top W \varphi(j)$ for an unconstrained parameter $W \in \mathbb{R}^{m \times m}$, with feature matrix $\Phi_n \in \mathbb{R}^{n \times m}$ and realizable score space $\mathcal{K}_n := \{\Phi_n W \Phi_n^\top : W \in \mathbb{R}^{m \times m}\} \subseteq \mathbb{R}^{n \times n}$. At length $n$, a position-driven task specifies scored query rows $Q_n \subseteq \{0, \dots, n - 1\}$ and causal targets $t_n(i) \in \{0, \dots, i\}$ for $i \in Q_n$, with one-hot indicator entries $R_n[i, j] = \mathbf{1}[j = t_n(i)]$. Under S2, $W$ *solves* length $n$ if
$$
s_W(i, t_n(i)) > s_W(i, j) \qquad \text{for all } i \in Q_n \text{ and all } 0 \le j \le i,\; j \neq t_n(i).
$$
(A row with $i = 0$ has no competitors and imposes zero comparisons.) The scored new rows at length $L > L_0$ are $\mathcal{S}_L := Q_L \cap [L_0, L)$.

**Notation and convention caveats in `conj:general`.**
1. *Ill-typed projection membership:* Conjecture `conj:general` writes $R_L^{\mathrm{new}} \notin \Pi_{\mathcal{K}_L}$, where $\Pi_{\mathcal{K}_L}$ is a projection operator onto $\mathcal{K}_L$, without specifying how $R_L^{\mathrm{new}}$ is embedded into $\mathbb{R}^{L \times L}$, what inner product defines $\Pi_{\mathcal{K}_L}$, or whether masked cells are excluded. The natural reading is nonmembership in the range of $\Pi_{\mathcal{K}_L}$, i.e., a nonzero projection residual $(I - \Pi_{\mathcal{K}_L}) R_L^{\mathrm{new}} \neq 0$. Zero-padding to $\mathbb{R}^{L \times L}$ versus restricting to scored causal cells can yield different residuals. Our proof of the printed implication holds regardless of convention (since the projection premise is unused), and our counterexample in Section 4 has a nonzero residual under **every** masking and padding convention.
2. *Old-row target shifts in `copy` (`old+new` vs. `old+long`):* For `copy`, $Q_n = \{\lceil n/2 \rceil, \dots, n - 1\}$ and $t_n(i) = i - \lceil n/2 \rceil$. Whenever $L < 2L_0 - 1$, we have $\lceil L/2 \rceil \le L_0 - 1$, so length $L$ places scored rows inside $[0, L_0)$ whose targets differ from $t_{L_0}(i)$. Consequently, checking only $Q_{L_0}$ together with $\mathcal{S}_L = Q_L \cap [L_0, L)$ (`old+new`) can strictly underestimate the full two-length constraint set $Q_{L_0} \cup Q_L$ (`old+long`); when $L \ge 2L_0 - 1$, $Q_L \cap [0, L_0) = \varnothing$ and the two systems coincide.

---

## 2. Proof of the Printed Implication of Conjecture `conj:general`

**Theorem 1 (Printed implication via positive scaling).** *Under S1–S3 with unconstrained scale $W \in \mathbb{R}^{m \times m}$ (or any parameter cone closed under positive scalar multiplication) and any fixed $\gamma > 0$, suppose that every $W$ realizing $R_{L_0}$ with old-row margin at least $\gamma$ satisfies $\max_{0 \le j \le i,\, j \neq t_L(i)} s_W(i, j) \ge s_W(i, t_L(i))$ on some scored new row $i \in \mathcal{S}_L$. Then no single head $W$ solves both lengths $L_0$ and $L$.*

*Proof.* Suppose for contradiction that a parameter $W$ solves both $L_0$ and $L$ under S2. Then every required comparison at both lengths has a strictly positive gap. If the set of old comparisons $\{(i, j) : i \in Q_{L_0},\, 0 \le j \le i,\, j \neq t_{L_0}(i)\}$ is nonempty, let $m_0 > 0$ be its minimum gap, and set $c := \max(1, \gamma / m_0) > 0$; if the old comparison set is empty, set $c := 1$ (old margin $\ge \gamma$ holds vacuously). Because $s_{cW} = c\, s_W$ and $c > 0$, the rescaled parameter $cW$ realizes $R_{L_0}$ with old-row margin at least $\gamma$, while preserving strict positivity of every comparison on $Q_L$ (and in particular on $\mathcal{S}_L$). Applying the quantified separation hypothesis to $cW$ yields a scored new row $i \in \mathcal{S}_L$ with a nonpositive gap, contradicting that $cW$ strictly solves $L$. $\square$

*Scope remarks.* Positive-scaling closure also covers sign-constrained cones such as canonical ALiBi (nonpositive recency slope). Imposing a norm cap or frozen logit scale breaks the scaling step (for instance, choosing $\gamma$ above every attainable old margin makes the premise vacuous under a norm cap); weight decay in a selection rule does not impose a hard norm cap on the existence question. Under S2, softmax at any finite positive temperature preserves all strict comparisons and ties; values, output projections, and content logits do not enter S2.

---

## 3. Exact Finite Feasibility Criterion (Gordan and Farkas Alternatives)

To decide the substantive question of whether *any* head solves a finite collection $\mathcal{H}$ of required comparisons $h = (n, i, j)$ ($i \in Q_n$, $0 \le j \le i$, $j \neq t_n(i)$), associate to each $h \in \mathcal{H}$ the comparison matrix
$$
D_h := \varphi(i)\bigl(\varphi(t_n(i)) - \varphi(j)\bigr)^\top \in \mathbb{R}^{m \times m},
$$
so that $\langle D_h, W \rangle_F = s_W(i, t_n(i)) - s_W(i, j)$. Stack the flattened vectors $\operatorname{vec}(D_h)^\top$ as rows of a matrix $A \in \mathbb{R}^{|\mathcal{H}| \times m^2}$ and set $w := \operatorname{vec}(W)$.

**Theorem 2 (Exact finite feasibility criterion).** *Assume $\mathcal{H} \neq \varnothing$.*
1. *(Unconstrained $W$, Gordan 1873).* *There exists $W \in \mathbb{R}^{m \times m}$ strictly satisfying all comparisons in $\mathcal{H}$ if and only if the dual system*
   $$
   y \ge 0, \qquad \mathbf{1}^\top y = 1, \qquad A^\top y = 0
   $$
   *has no solution.*
2. *(Sign-constrained basis $w \ge 0$, Farkas 1902).* *For a linear relative-score family $s_w(i, j) = \psi(i, j)^\top w$ with $w \ge 0$ (such as canonical ALiBi with feature $j - i$ and nonnegative recency magnitude), strict feasibility $Aw > 0, w \ge 0$ holds if and only if the dual system*
   $$
   y \ge 0, \qquad \mathbf{1}^\top y = 1, \qquad A^\top y \le 0
   $$
   *has no solution.*

*Proof.* For (1), strict feasibility $Aw > 0$ is equivalent by positive scaling to $Aw \ge \mathbf{1}$. If a dual vector $y$ exists, then $y^\top A w = (A^\top y)^\top w = 0$ for every $w$, ruling out $Aw > 0$. Conversely, if no dual solution exists, the compact convex hull $C := \operatorname{conv}\{\operatorname{vec}(D_h) : h \in \mathcal{H}\}$ excludes $0$. Its unique minimum-norm point $p \in C$ satisfies $\|p\|_2 > 0$; differentiating $\|(1 - \theta)p + \theta d\|_2^2$ at $\theta = 0^+$ for any $d \in C$ yields $d^\top p \ge \|p\|_2^2 > 0$, so $w := p / \|p\|_2^2$ satisfies $Aw \ge \mathbf{1}$ (Gordan, 1873). Part (2) is the standard homogeneous Farkas alternative (Farkas, 1902): $Aw \ge \mathbf{1}, w \ge 0$ fails if and only if $y \ge 0, \mathbf{1}^\top y = 1, A^\top y \le 0$ is solvable. $\square$

**Consequences for the two comparison sets:**
- Taking $\mathcal{H} = \mathcal{H}_{\text{old}} \cup \mathcal{H}_{\text{new}}$ characterizes the displayed separation premise of `conj:general` after rescaling: if $\mathcal{H}_{\text{old}}$ is strictly feasible and $y = (y_{\text{old}}, y_{\text{new}})$ is a dual certificate with $a := \mathbf{1}^\top y_{\text{old}}$ and $b := \mathbf{1}^\top y_{\text{new}} = 1 - a$, then $b > 0$, and every $W$ with old margin $\ge \gamma$ has a new comparison gap at most $-(a/b)\gamma \le 0$.
- Taking $\mathcal{H} = \mathcal{H}_{\text{old}} \cup \mathcal{H}_{\text{long}}$ characterizes whether a single head solves both lengths $L_0$ and $L$. When a task modifies targets on old rows (e.g., `copy` with $L < 2L_0 - 1$), $\mathcal{H}_{\text{old}} \cup \mathcal{H}_{\text{new}}$ can be feasible while $\mathcal{H}_{\text{old}} \cup \mathcal{H}_{\text{long}}$ is infeasible.

---

## 4. Growing-Offset Counterexample and Exact Selection Failure

The stronger claim—that a nonzero projection residual $(I - \Pi_{\mathcal{K}_L}) R_L^{\mathrm{new}} \neq 0$ (even with full-column-rank $\Phi_{L_0}$) forces joint impossibility—is **false**.

**Theorem 3 (Degree-4 monomial `copy` counterexample at $L_0 = 5, L = 10$).** *Take $\varphi(i) = (1, i, i^2, i^3, i^4)^\top \in \mathbb{R}^5$ and the `copy` task ($Q_n = \{\lceil n/2 \rceil, \dots, n - 1\}$, $t_n(i) = i - \lceil n/2 \rceil$) at $L_0 = 5$ (offset $3$) and $L = 10$ (offset $5$). Note $\Phi_5$ is invertible ($5 \times 5$ Vandermonde) and $L = 10 \ge 2L_0 - 1$, so `old+new` and `old+long` coincide. Then:*
1. *(Joint strict realizability.)* *The single admissible bilinear kernel $s_W(i, j) = g(i - j)$ with*
   $$
   g(d) := d - 2(d - 3)^2(d - 5)^2
   $$
   *has degree at most $4$ in each of $i$ and $j$ (hence $s_W \in \mathcal{K}_n$ for all $n$ via one $5 \times 5$ matrix $W$), strictly solves $L_0 = 5$ with minimum margin $1$, and strictly solves $L = 10$ with minimum margin $2$.*
2. *(Convention-independent non-realizability of $R_{10}^{\mathrm{new}}$.)* *On the last scored row $i = 9$ (whose causal keys $j \in \{0, \dots, 9\}$ include no masked entries), the fifth forward difference $\Delta_5(f) := \sum_{k=0}^5 (-1)^{5 - k}\binom{5}{k} f(k)$ vanishes on every realizable score row $j \mapsto s_W(9, j)$, yet evaluates to $-5 \neq 0$ on the one-hot target $j \mapsto \mathbf{1}[j = 4]$. Thus $R_{10}^{\mathrm{new}}$ has nonzero projection residual under **every** masking and padding convention.*
3. *(Exact selection failure of the min-norm interpolant $W_*$.)* *Extending $R_5$ by zeros on unscored rows (so its ones lie at $(3, 0)$ and $(4, 1)$), the unique exact old-block interpolant $W_* := \Phi_5^{-1} R_5 \Phi_5^{-\top}$ solves $L_0 = 5$ exactly, yet achieves new-row margins $(-15, 15, -35, -70, -126)$ on $i = 5, 6, 7, 8, 9$ (hence $\alpha_{10} = 1/5$).*

*Proof.*
For (1), expanding $g(i - j)$ produces monomials $i^a j^b$ with $a + b \le 4$, so $a, b \le 4$ and $g(i - j) = \varphi(i)^\top W \varphi(j)$ for a single $W \in \mathbb{R}^{5 \times 5}$. At $L_0 = 5$, scored rows $i \in \{3, 4\}$ admit offsets $d \in \{0, \dots, i\} \subseteq \{0, 1, 2, 3, 4\}$: we have $g(3) = 3$, $g(4) = 2$, $g(2) = -16$, $g(1) = -127$, $g(0) = -450$, so offset $3$ wins with minimum margin $3 - 2 = 1$. At $L = 10$, scored rows $i \in \{5, 6, 7, 8, 9\}$ admit offset $5$ with $g(5) = 5$; the best competitor is offset $3$ ($g(3) = 3$), since $g(4) = 2$ and for $6 \le d \le 9$ we have $2(d - 3)^2(d - 5)^2 \ge 18 > d$ (hence $g(d) \le -12$), giving minimum margin $5 - 3 = 2$.

For (2), every score row $j \mapsto \varphi(9)^\top W \varphi(j)$ is a polynomial in $j$ of degree at most $4$, so the fifth forward difference operator $\Delta_5$ on the causal keys $\{0, 1, 2, 3, 4, 5\} \subseteq \{0, \dots, 9\}$ annihilates it identically. On the one-hot target $\mathbf{1}[j = 4]$ of row $i = 9$, $\Delta_5$ evaluates to $(-1)^{5 - 4}\binom{5}{4} = -5 \neq 0$. Because only scored causal cells on row $9$ are used, no masking or padding convention can place $R_{10}^{\mathrm{new}}$ in the realizable score subspace.

For (3), $\Phi_5$ is an invertible $5 \times 5$ Vandermonde matrix, so $W_* = \Phi_5^{-1} R_5 \Phi_5^{-\top}$ is the unique exact interpolant of $R_5$. Direct exact rational evaluation of $\Phi_{10} W_* \Phi_{10}^\top$ on rows $i = 5, 6, 7, 8, 9$ yields target-minus-best-competitor gaps $(-15, 15, -35, -70, -126)$ (only row $6$ has a positive gap, giving $\alpha_{10} = 1/5$). $\square$

*Scope remark on selection failure.* Theorem 3(3) exhibits the growing-offset non-realizable nonempty-joint-cone phenomenon requested at the end of Section 4 of arXiv:2610.41218 (under the exact one-hot score-realizability reading) for the **minimum-Frobenius-norm exact old-block interpolant $W_*$**. It does not evaluate the paper's regularized cross-entropy selector (400 Adam steps, weight decay $10^{-4}$).

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/pe_length_gen/explore.py
```

The script [`explore.py`](explore.py) runs 12 deterministic cases across `RoPE`, `sinusoidal`, `ALiBi`, and `monomial` families, testing four comparison sets per case (`old`, `new`, `old+new`, `old+long`) with independent primal substitution and dual certificate verification (tolerance $10^{-7}$), plus exact SymPy verification of Theorem 3. All 48 verdicts and exact symbolic assertions passed (NumPy 2.5.5, SciPy 1.18.1, SymPy 1.14.0):

| Family | Task | Lengths ($L_0 \to L$) | `old` / `new` / `old+new` / `old+long` |
|---|---|---:|---|
| `RoPE` | `prev` | $4 \to 10$ | `YYYY` |
| `RoPE` | `copy` | $4 \to 6$ | `YYNN` |
| `sinusoidal` | `prev` | $4 \to 10$ | `YYYY` |
| `sinusoidal` | `copy` | $4 \to 6$ | `YYYN` |
| `ALiBi` | `self` | $4 \to 10$ | `YYYY` |
| `ALiBi` | `first` | $4 \to 10$ | `NNNN` |
| `ALiBi` | `prev` | $1 \to 4$ | `YNNN` |
| `monomial1` | `first` | $4 \to 10$ | `YYYY` |
| `monomial1` | `prev` | $2 \to 5$ | `YNNN` |
| `monomial2` | `prev` | $4 \to 10$ | `YYYY` |
| `monomial4` | `copy` | $5 \to 10$ | `YYYY` |
| `monomial1` | `changed prefix` | $3 \to 5$ | `YYYN` |

Notice how `sinusoidal` `copy` ($4 \to 6$, where $L = 6 < 2L_0 - 1 = 7$) yields `YYYN`: `old+new` is feasible, while `old+long` is infeasible because row $3$ demands key $1$ at length $4$ and key $0$ at length $6$. By contrast, for `RoPE` `copy` ($4 \to 6$), single-plane relative scores require both $\kappa(2) > \kappa(3)$ on old row $3$ and $\kappa(3) > \kappa(2)$ on new row $4$, so the two comparison vectors cancel directly (`YYNN`).

---

## References

- J. Farkas, *Theorie der einfachen Ungleichungen*, Journal für die reine und angewandte Mathematik 124:1–27, 1902.
- P. Gordan, *Ueber die Auflösung linearer Gleichungen mit reellen Coefficienten*, Mathematische Annalen 6:23–28, 1873.
- J. Han, Y. Chai, and J. Wang, *A Positional Realizability Theory of Length Generalization in Single-Head Attention*, [arXiv:2610.41218](https://arxiv.org/abs/2610.41218) (TMLR), October 2026.
