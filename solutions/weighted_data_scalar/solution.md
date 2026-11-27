# Exact Risk Ratios for Weighted Scalar Regression on the Upper Half of Budgets

**Target paper:** Guangjian Zhang, *Exact Risk Ratios for Weighted Data Selection in Linear Regression*, [arXiv:2608.28007](https://arxiv.org/abs/2608.28007) (October 2026), Conjecture `conj:gamma`.  
**Companion files:** [`solutions/weighted_data_scalar/explore.py`](explore.py), [`solutions/weighted_data_scalar/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:gamma`: PROVED ON THE ENTIRE UPPER HALF OF BUDGETS** ($\lfloor d/2 \rfloor \le k \le d - 1$ for every dimension $d \ge 2$):
  $$
  \mathrm{Fw}(d, d + k) = 1 + \Gamma_{d, k} = 2 - \frac{k}{d}.
  $$
  In particular, the first two previously open cells in dimension $5$ are $\mathrm{Fw}(5, 7) = 8/5$ and $\mathrm{Fw}(5, 8) = 7/5$.
- **Unrestricted generality:** Unlike the paper's block-model minimax theorem, no support hypothesis is imposed; the theorem holds for all finite datasets under the original minimum-Euclidean-norm tie rule.
- **Scope caveat & first open cell:** Conjecture `conj:gamma` remains **open** on the lower half $1 \le k < \lfloor d/2 \rfloor$, starting at $(d, n) = (5, 6)$. For this first unresolved cell we prove the bracket
  $$
  \frac{11}{5} \le \mathrm{Fw}(5, 6) \le \frac{12}{5},
  $$
  where the upper bound combines one closest-hyperplane reduction with Zhang's Theorem `thm:fw45` (strictly convex / full-rank selection for $(4, 5)$).

---

## 1. Setting and Exact Value of $\Gamma_{d, k}$

A finite multiset of rows $\{(x_i, y_i)\}_{i=1}^N \subset \mathbb{R}^d \times \mathbb{R}$ has full squared loss $L(w) = N^{-1}\sum_{i=1}^N (x_i^\top w - y_i)^2$ and minimum $L^* = \min_{w \in \mathbb{R}^d} L(w)$. A weighted selector chooses nonnegative weights supported on at most $n = d + k$ distinct row indices, and the learner outputs the minimum-Euclidean-norm minimizer of the weighted loss. The instance risk ratio is the infimum of $L(w)/L^*$ over all such selections (with $0/0 = 1$ and positive$/0 = \infty$), and $\mathrm{Fw}(d, n)$ is the supremum of this ratio over all finite datasets.

Writing $d = q s + a$ ($0 \le a < s$), Zhang (2026) defines
$$
\Gamma_{d, k} := \max_{k < s \le d} \frac{s - k}{(s - a)/q + a/(q + 1)}
$$
and conjectures $\mathrm{Fw}(d, d + k) = 1 + \Gamma_{d, k}$ for all $1 \le k \le d - 1$ (Conjecture `conj:gamma`). Prior unrestricted results (Hanneke, Moran, Shlimovich, and Yehudayoff, 2025; Dewasurendra, 2026; Zhang, 2026) settled only $k = d - 1$ and the isolated cells $(3, 4)$, $(4, 5)$, and $(4, 6)$.

**Proposition 1 (Evaluation of $\Gamma_{d,k}$ and matching lower bound).** *For every $d \ge 2$ and $\lceil d/3 \rceil \le k \le d - 1$ (in particular, all $\lfloor d/2 \rfloor \le k \le d - 1$),*
$$
\Gamma_{d, k} = \frac{d - k}{d} \qquad\text{and}\qquad \mathrm{Fw}(d, d + k) \ge 2 - \frac{k}{d}.
$$

*Proof.* For every positive integer $r \ge 1$, $(r - 1)(r - 2) \ge 0$ gives $1/r \ge (3 - r)/2$. Summing over any partition of $d$ into $s > k$ positive integer parts yields denominator $\ge (3s - d)/2 > 0$. Since $3k \ge d$ and $s \le d$, $(3k - d)(d - s) \ge 0$ implies $2(s - k)/(3s - d) \le (d - k)/d$, with equality at $s = d$ (all parts equal to $1$). Thus $\Gamma_{d, k} = (d - k)/d$.

For the lower bound, take $2d$ rows with features $e_j$ and labels $M \pm 1$ ($j = 1, \dots, d$), where $M := d + 1$. Then $w^* = M\mathbf{1}$, $L^* = 1$, and the excess loss at $w$ is $d^{-1}\|w - M\mathbf{1}\|_2^2$. Under the minimum-norm rule, omitting coordinate $j$ forces $w_j = 0$ and incurs cost $M^2/d > 1$, selecting one row on coordinate $j$ costs $1/d$, and selecting both rows permits zero cost with equal weights. With at most $d + k$ rows and every coordinate covered, at least $d - k$ coordinates receive a single row, giving excess $\ge (d - k)/d$ and ratio $\ge 2 - k/d$. $\square$

---

## 2. Whitened Reductions and Positive Circuits

Work inside $\operatorname{span}\{x_i\}$, of dimension $m \le d$ (where the minimum-norm minimizer lies). Zero feature rank gives ratio $1$; $L^* = 0$ is solved exactly by $m$ linearly independent rows. Otherwise center at $w^*$ and whiten by $H := \sum x_i x_i^\top \succ 0$: set $z_i := H^{-1/2} x_i$, $t_i := y_i - x_i^\top w^*$, and $u := H^{1/2}(w - w^*)$. Then
$$
\sum z_i z_i^\top = I_m, \qquad \sum t_i z_i = 0, \qquad \sum (z_i^\top u - t_i)^2 = E + \|u\|_2^2 \qquad \Bigl(E := \sum t_i^2\Bigr).
$$
An *excess coefficient* $c$ means an infimum of $\|u\|_2^2 \le c E$ over **full-rank** selections (which have unique minimizers, so whitening introduces no tie-rule discrepancy). Rows with $z_i = 0$ may be discarded for an upper bound, and we normalize $E = 1$ whenever $E > 0$. Write $g_i := t_i z_i$ for the signed gradients.

**Lemma 2 (Flat completion, single rows, and orthogonal lines).**
1. *(Positive circuits and flat completion.)* A *positive circuit* is a minimal support of a positive linear dependence among nonzero gradients $g_i$ (its size equals its rank plus $1$; Davis 1954; Steinitz 1916). A rank-$r$ circuit positively spans its $r$-plane $P$, and weighting its $r + 1$ rows by the dependence coefficients produces a quadratic minimized precisely on $P^\perp$. The same holds for any direct-sum union of circuits spanning $P$. Projecting all remaining features onto $P^\perp$ (discarding zero projections) while keeping their labels yields a whitened problem on $P^\perp$ with zero gradient sum and residual energy equal to the sum off $P$. Any full-rank selection in the quotient lifts to a full-rank selection in $\mathbb{R}^m$ with the same infimal norm bound (by scaling the completed flat objective by a penalty $\lambda \to \infty$ after choosing finite inner weights).
2. *(Anchors and closest hyperplane.)* A nonzero row with $t_i = 0$ (an *anchor*) fixes one coordinate to $0$ using $1$ row, reducing $(m, k)$ to $(m - 1, k)$. In general, since $\sum \|z_i\|_2^2 = m$, the closest affine interpolation hyperplane $\{u : z_{i_0}^\top u = t_{i_0}\}$ has squared distance $h = t_{i_0}^2/\|z_{i_0}\|_2^2 \le E/m$, achieved at $u_0 = (t_{i_0}/\|z_{i_0}\|_2^2)z_{i_0}$. Centering on $z_{i_0}^\perp$ at $u_0$ yields projected gradient sum $\Pi_{z_{i_0}^\perp}\sum (t_i - z_i^\top u_0)z_i = \Pi_{z_{i_0}^\perp}(-u_0) = 0$ and residual energy $\le E + h$. If the $(m - 1)$-dimensional quotient has excess coefficient $D$, lifting gives excess $\le h + D(E + h)$, hence ratio factor $\le (1 + 1/m)(1 + D)$. Iterating $m$ times gives ratio $\le m + 1$ with $m$ rows.
3. *(All rank-$1$ circuits $\implies$ orthogonal lines.)* If there are no anchors, the nonzero gradients positively span $\mathbb{R}^m$ and every $g_i$ lies in a positive circuit. If every positive circuit has rank $1$, then every $g_i$ has an opposite partner and the distinct lines are linearly independent (any minimal dependence among distinct lines could be oriented using opposite partners to form a positive circuit of rank $\ge 2$). From $I_m = \sum_{j=1}^m Q_j v_j v_j^\top$ (unit $v_j$), the square matrix with columns $\sqrt{Q_j}\,v_j$ is orthogonal, so the $m$ lines are mutually orthogonal with $Q_j = 1$. On line $j$ (residual energy $e_j$), averaging $t_i^2/(z_i^\top v_j)^2$ with weights $(z_i^\top v_j)^2$ gives a single interpolation of cost $\le e_j$, while two opposite gradients recover that coordinate with zero cost. Pairing the $k$ heaviest lines and interpolating the rest achieves excess $\le (m - k)E/m$.
4. *(Exact recovery for $k \ge m$.)* Repeatedly fixing an anchor or completing a positive circuit in the quotient uses at most $2$ rows per removed dimension, achieving zero excess in the infimum with at most $2m \le m + k$ rows. $\square$

---

## 3. Exact Ratio for $2k \ge d$

**Theorem 3 (Upper half $2k \ge d$).** *For every $d \ge 2$ and $\lceil d/2 \rceil \le k \le d - 1$, $\mathrm{Fw}(d, d + k) = 2 - k/d$.*

*Proof.* We prove the full-rank excess coefficient $1 - k/d$ by induction on $d \ge 1$ (base $d = 1$ and boundary $k \ge d$ are immediate from Lemma 2(4)). If an anchor exists, Lemma 2(2) reduces to $(d - 1, k)$, improving $1 - k/(d - 1) < 1 - k/d$ (or giving exact recovery if $k \ge d - 1$). If all positive circuits have rank $1$, Lemma 2(3) gives excess $\le (1 - k/d)E$. Otherwise, complete a positive circuit of rank $r \ge 2$ using $r + 1$ rows (Lemma 2(1)), leaving dimension $m = d - r$ and extra budget $k' = k - 1$. Since $2k \ge d$ and $r \ge 2$, we have $2k' = 2k - 2 \ge d - r = m$. Either $k' \ge m$ (exact recovery) or induction applies with coefficient
$$
1 - \frac{k - 1}{d - r} \le 1 - \frac{k}{d} \iff kr \ge d,
$$
which holds since $r \ge 2$ and $2k \ge d$. Finally, if an input dataset has feature rank $m < d$, its extra budget in $\operatorname{span}\{x_i\}$ is $k' = d + k - m > k$, satisfying $k'/m > k/d$. Combining with Proposition 1 proves Theorem 3. $\square$

---

## 4. Structure of Rank-$\le 2$ Systems and the Odd Boundary $d = 2k + 1$

To handle the remaining odd midpoint $d = 2k + 1$ ($k = (d - 1)/2$), assume there are no anchors and every positive circuit has rank $\le 2$ (call a rank-$2$ positive circuit a *triangle*). Choose a maximal family of triangles whose planes $P_1, \dots, P_a$ form a direct sum $U := P_1 \oplus \cdots \oplus P_a$, and set $b := d - 2a$.

**Lemma 4 (Quotient decomposition and $3a$-plane cover).**
1. *(Projection lemma.)* Modulo $U$, the nonzero projected gradients lie on $b$ linearly independent lines with both signs.
2. *(Group cover.)* Partition the rows outside $U$ into $b$ groups according to their quotient line. Each group is either a *single line in $\mathbb{R}^d$* (which is an orthogonal direct summand of the whitened configuration) or is covered by at most $3a$ planes, each carrying an original positive triangle. Moreover, for $a = 1$, any group that is not a single line yields a $3$-plane cover of the entire $3$-space $P_1 \oplus \text{group}$.

*Proof.* (1) Suppose a positive circuit of rank $s \ge 2$ existed in the quotient $\mathbb{R}^d / U$. Lifting its positive coefficients gives a vector $v \in U$. If $v = 0$, the lifted circuit has rank $s$ and meets $U$ trivially, contradicting either the rank-$\le 2$ assumption ($s \ge 3$) or maximality of $a$ ($s = 2$). If $v \neq 0$, represent $-v$ positively by an independent subset of the chosen triangles; appending these vectors creates a positive circuit of rank $\ge s + 1 \ge 3$, contradiction. Thus all quotient circuits have rank $1$, and Lemma 2(3) gives $b$ independent two-sided lines in $\mathbb{R}^d / U$.

(2) Fix a group and a representative $v \notin U$ of its quotient line. Positive rescaling puts all gradients of the group in the form $v + p$ ($p \in A \subset U$) or $-(v + q)$ ($q \in B \subset U$) for nonempty finite sets $A, B \subset U$. For any $p \in A$ and $q \in B$, the sum of the two gradients is $p - q \in U$; expressing $q - p$ positively by a minimal subset of the chosen triangles must use at most $1$ vector (otherwise we obtain a positive circuit of rank $\ge 3$). Hence every difference $q - p$ is either $0$ or lies on one of the $3a$ positive rays of the chosen triangles, and every nonzero difference yields a triangle plane containing both rows and that ray in $U$.

- If $A = B$ is a singleton, the group is a single original line $L$. All other rows lie in a complementary $(d - 1)$-space $Q$, so the covariance splits as $M_L + M_Q = I_d$ with $\operatorname{rank}(M_L) = 1$ and $\operatorname{rank}(M_Q) = d - 1$; hence $L \perp Q$, and costs and residual energies of $L$ and $Q$ add.
- If one of $A, B$ is a singleton and some difference is nonzero, at most $3a$ triangle planes through that fixed row cover the group.
- For $a \ge 2$ with $|A|, |B| \ge 2$: if $|A| \le 2$ and $|B| \le 2$, at most $4 \le 3a$ unequal pairs cover the group. Otherwise, swap $A$ and $B$ if needed so $|A| \ge 3$; note that negating differences $q - p \mapsto p - q$ preserves the property that every difference lies in a single direct-sum summand $P_r$, i.e., Hamming distance $d_H(p, q) \le 1$ in $P_1 \times \cdots \times P_a$. Two points of $B$ at Hamming distance $2$ have only $2$ common neighbors at distance $\le 1$ (and distance $\ge 3$ has none), contradicting $|A| \ge 3$. Thus any two distinct points of $B$ differ in exactly one coordinate $P_r$, forcing all of $A$ (and symmetrically all of $B$) to agree outside $P_r$. All nonzero differences therefore use only the $3$ rays of $P_r$. Fixing one unequal center pair $(p_0, q_0) \in A \times B$, the triangle planes containing $p_0$ or $q_0$ number at most $3 + 3 - 1 = 5 \le 3a$ (the shared plane of $(p_0, q_0)$ is counted once) and cover the entire group.
- For $a = 1$ (in the $3$-space $W := P_1 + \mathbb{R}v$): a nonzero difference produces a second triangle sharing a ray with the first. Rescale the two triangles to $c + x_1 + x_2 = c + y_1 + y_2 = 0$; these $5$ vectors positively span $W$. Every gradient $g \in W$ outside these two planes has $-g$ positively spanned by at most $2$ independent vectors from $\{c, x_1, x_2, y_1, y_2\}$ (otherwise a rank-$3$ positive circuit arises), so $g = -\alpha x_i - \beta y_j$ with $\alpha, \beta > 0$. We claim only one cross type $(i, j)$ can occur. Suppose two gradients $\tilde{g}_1, \tilde{g}_2$ have distinct $j_1 \neq j_2$ (if only $i_1 \neq i_2$, swap $x$ and $y$). Scaling $\tilde{g}_1, \tilde{g}_2$ so their $y_{j_1}, y_{j_2}$ coefficients match and using $y_1 + y_2 = -c = x_1 + x_2$ yields $\tilde{g}_1 + \tilde{g}_2 + \gamma_1 x_1 + \gamma_2 x_2 = 0$ with $\gamma_1, \gamma_2 > 0$. Because $x_1, x_2$ span the $2$-plane $P_1$ and $\tilde{g}_1 = -\alpha_1 x_{i_1} - \beta_1 y_{j_1}$ ($\beta_1 > 0$) lies outside $P_1$, we have $\dim\operatorname{span}\{\tilde{g}_1, x_1, x_2\} = 3$. Hence the $4$-vector strictly positive dependence has a $1$-dimensional kernel and forms a genuine rank-$3$ positive circuit, contradiction. Thus at most one additional triangle plane exists, giving a $3$-plane cover of all data in $W$. $\square$

**Theorem 5 (Odd boundary $d = 2k + 1$).** *For every odd $d = 2k + 1 \ge 3$, $\mathrm{Fw}(d, d + k) = 2 - k/d = 1 + c_d$, where $c_d := (d + 1)/(2d) = 1 - k/d$.*

*Proof.* We induct on odd $d \ge 3$, normalizing $E = 1$. An anchor reduces to the even midpoint $d - 1 = 2k$ (Theorem 3, excess $\le 1/2 < c_d$). A positive circuit of rank $r \ge 3$ leaves dimension $\le 2(k - 1)$ with $k - 1$ extras, giving excess $\le 1/2 < c_d$ by Theorem 3 (or exact recovery). If all circuits have rank $1$, Lemma 2(3) gives $c_d$. Thus Lemma 4 applies with $a \ge 1$, $b = d - 2a \ge 1$ odd, and $k = a + (b - 1)/2$.

**Base $d = 3$ ($a = 1, b = 1, c_3 = 2/3$):** If the sole quotient group is a single original line of energy $e$, completing the triangle costs $\le e$, while pairing the line and interpolating $2$ rows in its orthogonal plane costs $\le 2(1 - e)$ by Lemma 2(2); $\min(e, 2(1 - e)) \le 2/3$. Otherwise Lemma 4(2) gives a $3$-plane cover of all data; some triangle plane holds energy $\ge 1/3$, and completing it plus interpolating one quotient point costs $\le 2/3$.

**Inductive step $d \ge 5$:** Set $h := 1/(kd)$.
1. *Heavy triangle plane:* If some triangle plane has energy $\ge h$, complete it; the $(d - 2, k - 1)$ quotient gives cost $\le (1 - h)c_{d-2} = c_d$. Henceforth every triangle plane has energy $< h$, so every triangle-covered group has energy $\le 3ah$.
2. *Light single original line:* If any single-original-line group has energy $e \le 1/d$, interpolate it and apply the even midpoint (Theorem 3) on its orthogonal complement for cost $\le e + (1 - e)/2 \le c_d$.
3. *Two single original lines:* If at least two single-original-line groups remain, their combined energy is $s > 2/d$. Pair the heavier and interpolate the lighter, applying odd induction on the remaining $(d - 2, k - 1)$ complement: cost $\le s/2 + (1 - s)c_{d-2} \le c_d$ (equality at $s = 2/d$, decreasing in $s$ since $c_{d-2} > 1/2$).
4. *Complete all $a$ triangles:* At most one single-original-line group remains. Complete all $a$ chosen triangles using $3a$ rows; the remaining budget is $b + (b - 1)/2$ on $b$ orthogonal quotient lines. If $b \ge 3$ (use one of the $(b - 1)/2$ pairs on the exceptional single line if present), or if $b = 1$ with no single-original-line group, all $(b + 1)/2$ interpolated groups are triangle-covered and have energy $\le 3ah$. Their total cost is at most
   $$
   \frac{b + 1}{2}\cdot 3ah \le c_d \iff 4a^2 - 2ab - 6a + b^2 - 1 \ge 0.
   $$
   At $b = 1$, this is $4a(a - 2) \ge 0$ (since $d = 2a + 1 \ge 5 \implies a \ge 2$); at $b = 3$, it is $4(a - 1)(a - 2) \ge 0$ for integer $a$; for $b \ge 5$, the quadratic in $a$ has leading coefficient $4 > 0$ and negative discriminant $4(-3b^2 + 6b + 13) < 0$.
5. *Remaining case $b = 1$ with a single original line of energy $e$:* If $e \le c_d$, completing all $a$ triangles and interpolating that line costs $\le e \le c_d$. If $e > c_d$, pair the line; on its $(d - 1)$-dimensional orthogonal complement apply one closest-hyperplane reduction (Lemma 2(2)) followed by odd induction on $(d - 2, k - 1)$. The complement excess coefficient is at most
   $$
   D := \left(1 + \frac{1}{d - 1}\right)(1 + c_{d-2}) - 1 = \frac{d^2 + d - 4}{2(d - 2)(d - 1)} \le \frac{d + 1}{d - 1}
   $$
   for $d \ge 5$ (cross-multiplied difference $d(d - 3) > 0$). Thus the cost is at most $D(1 - e) \le D(1 - c_d) \le c_d$.

Rank-deficient inputs ($m < d$) have $k' = d + k - m \ge \lfloor m/2 \rfloor$ and satisfy $1 - k'/m \le 1 - k/d$. Combining with Proposition 1 completes the proof. $\square$

---

## 5. Bounds for the First Unresolved Cell $(5, 6)$

**Proposition 6.** *$11/5 \le \mathrm{Fw}(5, 6) \le 12/5$.*

*Proof.* For the upper bound, full-rank datasets use one closest-hyperplane reduction (Lemma 2(2), factor $1 + 1/5$) followed by Zhang's Theorem `thm:fw45` (strictly convex / full-rank selection for $(4, 5)$ with ratio $\le 2$; Zhang 2026, Appendix F), yielding $\mathrm{Fw}(5, 6) \le (6/5) \cdot 2 = 12/5$; rank-deficient datasets ($m \le 4$) have budget $6 \ge m + 2$, which satisfies a strictly smaller ratio. For the lower bound, take two orthogonal blocks of ranks $r = 2$ and $r = 3$, each consisting of $r + 1$ rows $e_1, \dots, e_r, -\sum_{j=1}^r e_j$ with residual $a_r > 0$ chosen so $r(r + 1)a_r^2 = C$, shifted by a common linear predictor $M v$ with $v$ outside all finitely many proper row spans (making any rank-deficient selection diverge as $M \to \infty$). Selecting $r$ rows in a rank-$r$ block incurs excess $C$; with $6$ rows at most one block can be completed, giving excess $C$ against optimal full loss $C(1/2 + 1/3) = \frac{5}{6}C$, hence ratio $1 + 6/5 = 11/5$. $\square$

---

## 6. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/weighted_data_scalar/explore.py
```

The script [`explore.py`](explore.py) verifies:
- $\Gamma_{d, k} = (d - k)/d$ across all $3{,}333$ cells with $2 \le d \le 100$ and $\lceil d/3 \rceil \le k \le d - 1$, and all $5{,}049$ odd packing branch inequalities ($5 \le d \le 201$);
- Exhaustive shifted-axis lower-bound checks (including omitted axes) for $2 \le d \le 9$;
- $33$ seeded coupled graph-data searches in dimensions $3$–$6$ (max witness excess $0.30307844$ for $(5, 6)$ and $0.10169485$ for $(5, 7)$) and $96$ coupled $2 + 3$ direct-sum compositions for $(5, 6)$ (max excess $0.91619392$, versus $1.2$ for two orthogonal simplices).

All assertions passed.

---

## References

- C. Davis, *Theory of Positive Linear Dependence*, American Journal of Mathematics 76(4):733–746, 1954.
- M. Dewasurendra, *Risk Ratios for Weighted Linear Regression*, Zenodo, 2026 ([doi:10.5281/zenodo.21881535](https://doi.org/10.5281/zenodo.21881535)).
- S. Hanneke, S. Moran, A. Shlimovich, and A. Yehudayoff, *Weighted Data Selection for Linear Regression*, COLT 2025, PMLR 291, 2025.
- E. Steinitz, *Bedingt konvergente Reihen und konvexe Systeme*, Journal für die reine und angewandte Mathematik 146:1–52, 1916.
- G. Zhang, *Exact Risk Ratios for Weighted Data Selection in Linear Regression*, [arXiv:2608.28007](https://arxiv.org/abs/2608.28007), October 2026.
