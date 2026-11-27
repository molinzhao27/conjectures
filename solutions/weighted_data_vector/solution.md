# Bounds, Exact Optima, and Dual Formulations for the Seven-Atom Circle Moment Problem

**Target paper:** Guangjian Zhang, *Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression*, [arXiv:2608.30254](https://arxiv.org/abs/2608.30254) (October 2026), Section `sec:intermediate` and Conjecture `conj:CE`.  
**Companion files:** [`solutions/weighted_data_vector/explore.py`](explore.py), [`solutions/weighted_data_vector/search.c`](search.c), [`solutions/weighted_data_vector/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:CE` ($\Gamma_3 = 5/8$ and $\Gamma_4 = 1/4$, equivalently $F_{\mathrm{weighted}}(2,2,3) = 13/8$ and $F_{\mathrm{weighted}}(2,2,4) = 5/4$): OPEN in general**, with the following strict improvements and structural theorems:
  $$
  \frac{5}{8} \le \Gamma_3 \le \frac{5}{6}, \qquad \frac{1}{4} \le \Gamma_4 < \frac{1}{2}.
  $$
  1. **Improved unrestricted 3-point bound (Theorem 2):** For every moment system of energy $E = \operatorname{tr}(\Sigma) > 0$ and normalized output covariance anisotropy $d := |\lambda_1(\Sigma) - \lambda_2(\Sigma)| / \operatorname{tr}(\Sigma) \in [0, 1]$,
     $$
     \frac{C_3}{E} \le \frac{5}{6} - \frac{7}{24} d^2 \le \frac{5}{6}, \qquad \frac{C_3}{E} \le \frac{7}{8} - \frac{3}{8} d^2
     $$
     (strictly improving Zhang's $\Gamma_3 \le 7/8$ and proving $C_3 \le 5E/8$ whenever $d^2 \ge 2/3$).
  2. **Strict uniform 4-point improvement (Theorem 3):** $\Gamma_4 < 1/2$ strictly (improving Zhang's non-strict $\Gamma_4 \le 1/2$; proved via an explicit rank-one perturbation and 7-atom compactness, yielding no explicit numerical gap).
  3. **Two real output lines (Theorem 4):** Both conjectured bounds hold whenever all real output residuals $y_i$ lie on the union of at most two vector lines through the origin: $C_4 = 0$ and $C_3 / E \le \frac{1}{2\sqrt{2} - 1} < \frac{5}{8}$.
  4. **Isotropic equal-energy 6-atom systems and 5-atom classification (Theorem 5 & Proposition 6):** For equal-mass, equal-norm, isotropic-output 6-atom systems, four equally weighted atoms suffice for $C_4 / E \le 1/4$. Under the same hypotheses for 5 atoms, the system is uniquely classified up to orthogonal symmetries as the fifth-roots second-harmonic system with exact optimum $C_3 / E = (3 - \sqrt{5})/4$.
  5. **Minimax dualities and exact rational 7-atom certificates (Theorem 7 & Proposition 8):** Two bilinear minimax characterizations (via Sion, 1958), illustrated by exact rational dual certificates valid for a continuum of compatible 7-atom test measures ($560659/7066875 < 1/10$ at $k=3$ and $381447541/7312728225 < 1/16$ at $k=4$).
  6. **Exact cyclotomic 3-point optima (Proposition 9):** Exact algebraic KKT certificates across all dihedral triangle types determine the exact three-point optima for $(n,h) = (5,2), (6,3), (7,3)$ as $(3 - \sqrt{5})/4$, $1/2$, and $1/4$—sharpening Zhang's seventh-roots witness $2/7$ to the exact optimum $1/4$ on $\{0, 1, 3\}$.
  7. **Seven-atom original-weight obstruction at both budgets (Proposition 10):** An explicit 7-atom system with distinct atoms whose original-weight optima exceed both targets ($700205/1107014 > 5/8$ at $k=3$ and $295205/1107014 > 1/4$ at $k=4$), while reweighting one cross pair to equal weights beats both targets—strictly strengthening Proposition `prop:averaging` of `arXiv:2608.30254`.

---

## 1. Moment System, Real Dictionary, and Planar Covariance Mean Lemma

A *moment system* consists of weights $p_i > 0$ ($\sum p_i = 1$), unit complex numbers $t_i \in S^1$, and residuals $z_i \in \mathbb{C}$ satisfying
$$
\sum p_i t_i = \sum p_i z_i = \sum p_i \bar{t}_i z_i = 0, \qquad E := \sum p_i |z_i|^2 > 0.
$$
For probability weights $q$ with $|\sum q_i t_i| < 1$, let $a + b t$ be the $q$-weighted least-squares fit to $z$, and set $C(q) := |a|^2 + |b|^2$. Write $C_k := \inf \{C(q) : |\operatorname{supp} q| \le k,\ |\sum q_i t_i| < 1\}$ and $\Gamma_k := \sup C_k / E$. Conjecture `conj:CE` asks whether $\Gamma_3 = 5/8$ and $\Gamma_4 = 1/4$ (equivalently $F_{\mathrm{weighted}}(2,2,3) = 13/8$ and $F_{\mathrm{weighted}}(2,2,4) = 5/4$). By minimizing linear energy over the probability polytope with the three complex moment constraints (6 real equations + mass 1), a minimizing vertex has at most $7$ atoms and every witness on it lifts to the full system (a zero-energy vertex gives a zero fit on two distinct directions); hence $7$ atoms suffice (Proposition `prop:seven` of `arXiv:2608.30254`).

**Real dictionary.** Choose unit real feature vectors $u_i \in \mathbb{R}^2$ (identified with $S^1 \subset \mathbb{C}$) such that $\bar{u}_i / u_i = t_i$, and put $y_i := z_i u_i \in \mathbb{R}^2$ (flipping the sign of $u_i$ flips $y_i$ simultaneously). A real $2 \times 2$ matrix $B$ acts on $\mathbb{C}$ as $u \mapsto a u + b \bar{u}$ with $\|B\|_F^2 = 2(|a|^2 + |b|^2)$. In real coordinates, the moment equations, energy, output covariance, and selection cost become
$$
\sum p_i u_i u_i^\top = \tfrac{1}{2} I_2, \qquad \sum p_i y_i u_i^\top = 0, \qquad \Sigma := \sum p_i y_i y_i^\top, \qquad E = \operatorname{tr}(\Sigma) = \sum p_i \|y_i\|^2, \qquad C(q) = \tfrac{1}{2}\|B(q)\|_F^2.
$$
By scaling $y_i$, we normalize $E = 1$ in the proofs. (The known lower-bound example places a uniform centered unit equilateral output triangle on each of two perpendicular feature lines of mass $1/2$: per-class costs $1, 1/4, 0$ at supports $1, 2, 3$ yield $C_3/E = 5/8$ and $C_4/E = 1/4$.)

### Lemma 1 (Planar covariance mean lemma)
*Let $v_i \in \mathbb{R}^2$ carry weights $p_i > 0$ with $\sum p_i = 1$, $\sum p_i v_i = 0$, and covariance $S = \sum p_i v_i v_i^\top \succeq 0$. Define $H(S) := \det(S) / \operatorname{tr}(S)$ for $S \succ 0$ and $H(S) := 0$ for singular $S$. Then some convex combination of at most two support points has squared norm at most $H(S)$. Moreover, $H(S) = \inf_{\operatorname{tr} L = 1} \operatorname{tr}(L^\top S L)$, so $H$ is concave and monotone in the positive semidefinite order.*

*Remark on relation to prior lemmas:* Since $\det(S) / \operatorname{tr}(S) \le \frac{1}{4}\operatorname{tr}(S)$, Lemma 1 strictly sharpens Zhang's isotropic $m$-point mean lemma (`lem:mmean` in `arXiv:2608.30254`) at $m = 2$ whenever $S$ is anisotropic (note that Theorem 2 of `SharpPair`, as discussed in Section 1 of `arXiv:2608.30254`, covers $k = 2$ for dimensions $M \ge 3$ rather than $M = 2$).

*Proof.* Decompose $p$ into a convex combination of extreme points of the mass-one, mean-zero polytope. Each extreme point uses at most $3$ points. A collinear centered subsystem contains $0$ on a segment (cost $0 = H(S)$). For a nondegenerate centered triangle with vertices $v_1, v_2, v_3$, weights $r_1, r_2, r_3 > 0$, and $S = \sum r_i v_i v_i^\top \succ 0$, the $3 \times 3$ matrix $M$ with columns $\sqrt{r_i}(1, S^{-1/2} v_i)^\top$ satisfies $M M^\top = I_3$, hence $M^\top M = I_3$. Its off-diagonal entries give $1 + v_i^\top S^{-1} v_j = 0$ for $j \neq i$, so the edge opposite $v_i$ lies on the line $v_i^\top S^{-1} v = -1$. Because $0 \in \operatorname{int}\operatorname{conv}\{v_1, v_2, v_3\}$, the largest centered disk inside the triangle touches the interior of some edge, so the squared distance from $0$ to the boundary equals $\min_i \|S^{-1} v_i\|^{-2}$. By averaging with weights $r_i$,
$$
\min_i \|S^{-1} v_i\|^{-2} \le \frac{1}{\sum_i r_i \|S^{-1} v_i\|^2} = \frac{1}{\operatorname{tr}(S^{-1} S S^{-1})} = \frac{1}{\operatorname{tr}(S^{-1})} = \frac{\det S}{\operatorname{tr} S} = H(S).
$$
For the variational formula, if $S \succ 0$ then Cauchy–Schwarz for the Frobenius inner product gives
$$
1 = (\operatorname{tr} L)^2 = \langle S^{-1/2}, S^{1/2} L \rangle_F^2 \le \operatorname{tr}(S^{-1})\operatorname{tr}(L^\top S L),
$$
with equality at $L = S^{-1} / \operatorname{tr}(S^{-1})$; if $S$ is singular, a rank-one orthogonal projector onto $\ker(S)$ attains $0$. As an infimum of linear functionals in $S$ with $L L^\top \succeq 0$, $H$ is concave and PSD-monotone. Averaging the vertex bounds by concavity finishes the proof. $\square$

---

## 2. Three-Point Bound $\Gamma_3 \le 5/6$ and Strict Four-Point Bound $\Gamma_4 < 1/2$

### Theorem 2 (Anisotropic 3-point bound and $\Gamma_3 \le 5/6$)
*For every moment system with energy $E = \operatorname{tr}(\Sigma) > 0$ and normalized output covariance anisotropy $d := |\lambda_1(\Sigma) - \lambda_2(\Sigma)| / \operatorname{tr}(\Sigma) \in [0, 1]$,*
$$
\frac{C_3}{E} \le \frac{5}{6} - \frac{7}{24} d^2 \le \frac{5}{6}, \qquad \frac{C_3}{E} \le \frac{7}{8} - \frac{3}{8} d^2.
$$
*In particular, $\Gamma_3 \le 5/6$, and Conjecture `conj:CE`'s 3-point bound $C_3 \le 5E/8$ holds whenever $d^2 \ge 2/3$.*

*Proof.* Normalize $E = 1$. Fix an anchor atom $i$. In an orthonormal input basis whose first vector is $u_i$, write $u_j = (c_j, s_j)$. Exact two-point interpolation on $\{i, j\}$ ($s_j \neq 0$) produces the matrix with columns $(y_i, w_j)$, where $w_j := (y_j - c_j y_i) / s_j$. Under weights $2p_j s_j^2$ (summing to $1$ by $\sum p_j u_j u_j^\top = \frac{1}{2} I_2$), the vectors $w_j$ have mean zero by $\sum p_j y_j u_j^\top = 0$ and covariance
$$
S_i := \sum_{s_j \neq 0} 2p_j (y_j - c_j y_i)(y_j - c_j y_i)^\top \preceq \sum_{\text{all } j} 2p_j (y_j - c_j y_i)(y_j - c_j y_i)^\top = 2\Sigma + y_i y_i^\top.
$$
Any convex combination of two vectors $w_j$ is achievable in the limit by weighting anchor $i$ by $1$ and the two other atoms by $\varepsilon$ times their desired coefficients divided by $s_j^2$ ($\varepsilon \to 0^+$): the first fitted column tends to $y_i$, and the second normal equation yields the convex combination. By planar Carathéodory, three other atoms reach $(y_i, 0)$. Applying Lemma 1 and PSD monotonicity of $H$ gives, for every atom $i$,
$$
2 C_3 \le \|y_i\|^2 + H(2\Sigma + y_i y_i^\top), \qquad 2 C_4 \le \|y_i\|^2. \tag{1}
$$

Choose output axes so $\Sigma = \operatorname{diag}((1 + d)/2, (1 - d)/2)$ with $d \in [0, 1]$. Under the distribution $p$, set $s := \|y\|^2$ and $u := \exp(2\mathrm{i}\arg y)$ (arbitrary unit phase when $s = 0$), so $\mathbb{E}[s] = 1$ and $\mathbb{E}[s u] = d$. For the real symmetric matrix $2\Sigma + y y^\top$, the trace is $2 + s$ and the difference of diagonal entries plus $2\mathrm{i}$ times the off-diagonal entry is $2d + s u$. Using $4\det M = (\operatorname{tr} M)^2 - |(M_{11} - M_{22}) + 2\mathrm{i} M_{12}|^2$,
$$
4\,\mathbb{E} H(2\Sigma + y y^\top) = \mathbb{E}\left[(2 + s) - \frac{|2d + s u|^2}{2 + s}\right] = 3 - \mathbb{E}\frac{|2d + s u|^2}{2 + s}.
$$
Put $A := \mathbb{E}[(2 + s)^{-1}]$, $U := \mathbb{E}[(s - 1)^2 / (2 + s)]$, and $V := \mathbb{E}[s^2 / (2 + s)]$. Using $\mathbb{E}[s] = 1$, the polynomial identities $\frac{(s-1)^2}{2+s} = s - 4 + \frac{9}{2+s}$ and $\frac{s^2}{2+s} = s - 2 + \frac{4}{2+s}$ give $U = 9A - 3$, $V = 4A - 1$, and $V - \frac{1}{3} = \frac{4}{9}U$. Using $\frac{1}{2+s} = \frac{1}{3} - \frac{s-1}{3(2+s)}$ in the cross term $4d\,\mathbb{E}[s\operatorname{Re}(u)/(2+s)]$ together with $\mathbb{E}[s\operatorname{Re}(u)] = d$, we compute
$$
\frac{1}{4}\left(\mathbb{E}\frac{|2d + su|^2}{2+s} - \frac{1}{3}\right)
= \frac{U}{9} + d^2\Bigl(A + \frac{1}{3}\Bigr) - \frac{d}{3}\,\mathbb{E}\frac{s\operatorname{Re}(u)(s - 1)}{2 + s}.
$$
By Cauchy–Schwarz, $|\mathbb{E}[s\operatorname{Re}(u)(s - 1)/(2 + s)]| \le \sqrt{U V}$, so completing the square in $\sqrt{U}$ gives
$$
\frac{U}{9} - \frac{d}{3}\sqrt{UV} + d^2\Bigl(A + \frac{1}{3}\Bigr) \ge d^2\Bigl(A + \frac{1}{3} - \frac{V}{4}\Bigr) = d^2\Bigl(A + \frac{1}{3} - \frac{4A - 1}{4}\Bigr) = \frac{7}{12} d^2.
$$
Thus $\mathbb{E} H(2\Sigma + y y^\top) \le \frac{2}{3} - \frac{7}{12} d^2$. Averaging (1) yields $C_3 \le \frac{1}{2}(1 + \frac{2}{3} - \frac{7}{12}d^2) = \frac{5}{6} - \frac{7}{24}d^2$.
Finally, applying concavity of $H$ directly to (1) gives $\mathbb{E} H(2\Sigma + y y^\top) \le H(3\Sigma) = \frac{3}{4}(1 - d^2)$, whence $C_3 \le \frac{7}{8} - \frac{3}{8} d^2$, which is $\le 5/8$ whenever $d^2 \ge 2/3$. $\square$

### Theorem 3 (Strict uniform four-point improvement)
*We have $\Gamma_4 < 1/2$ strictly.*

*Proof.* First we show $C_4 < 1/2$ for every normalized ($E = 1$) moment system. By (1), if $C_4 \ge 1/2$ then $\|y_i\| = 1$ for every positive-weight atom. Whitening $\sum p_i u_i u_i^\top = \frac{1}{2}I_2$ implies every feature line has total mass $\le 1/2$. If some feature line has mass $1/2$, all other features are perpendicular to it, so each of the two orthogonal classes has mean-zero outputs after consistent orientation; applying Lemma 1 and $H(S) \le \operatorname{tr}(S)/4$ to two points per class gives $C_4 \le 1/4 < 1/2$.

Otherwise, choose a feature line of total mass $P < 1/2$, orient all $u_j$ on that line consistently as $u_i$, and pick an atom $i$ on it satisfying $y_i^\top \sum_{j \text{ on line}} p_j y_j \ge 0$ (which exists because $\sum_{i \text{ on line}} p_i y_i^\top \sum_{j \text{ on line}} p_j y_j = \|\sum_{j \text{ on line}} p_j y_j\|^2 \ge 0$). In the orthonormal coordinates $u_j = (c_j, s_j)$ of Theorem 2, the moment equations give $\sum_{j \text{ off line}} p_j s_j (y_j - c_j y_i) = 0$. Decompose the vector $(p_j)_{j \text{ off line}}$ into a nonnegative sum of vectors $r$ supported on at most $3$ off-line indices each satisfying $\sum_j r_j s_j (y_j - c_j y_i) = 0$ (by normalizing and taking vertices of the two-constraint probability polytope). Define $g(r) := \sum_j r_j c_j (y_j - c_j y_i)$. Summing over all atoms gives $\sum_{\text{all } j} p_j c_j (y_j - c_j y_i) = -\frac{1}{2} y_i$; subtracting the on-line contribution (where $c_j = 1$) yields
$$
\sum_r y_i^\top g(r) = -\frac{1}{2}\|y_i\|^2 - y_i^\top \sum_{j \text{ on line}} p_j (y_j - y_i) = P - \frac{1}{2} - y_i^\top \sum_{j \text{ on line}} p_j y_j < 0.
$$
Pick a component $r$ (with $|\operatorname{supp} r| \le 3$) having $y_i^\top g(r) < 0$. Assign anchor $i$ weight $1$ and off-line atoms weights $\tau r_j$ ($\tau > 0$). Put $D := \sum r_j s_j^2 > 0$, $h := (\sum r_j c_j s_j) / D$, and $L := \sum r_j c_j^2 - D h^2 \ge 0$ (by Cauchy–Schwarz). The weighted Gram matrix has determinant $\tau D(1 + \tau L) > 0$, so the support of size $\le 4$ spans $\mathbb{R}^2$. Direct substitution verifies that the exact normal equations $B G = R$ are solved by
$$
B(\tau) = (y_i + v,\, -h v), \qquad v := \frac{\tau g(r)}{1 + \tau L}.
$$
Hence $\|B(\tau)\|_F^2 = \|y_i\|^2 + 2\tau y_i^\top g(r) + O(\tau^2) < 1$ for sufficiently small $\tau > 0$, proving $C_4 < 1/2$.

**Uniformity via 7-atom compactness.** By the 7-atom reduction, pad every 7-atom system to $7 \times 2$ matrices $X, Y$ with rows $X_i = \sqrt{p_i} u_i^\top$ and $Y_i = \sqrt{p_i} y_i^\top$. The set $\mathcal{K} := \{(X, Y) : X^\top X = \frac{1}{2} I_2,\ X^\top Y = 0,\ \|Y\|_F = 1\}$ is compact. Suppose for contradiction that $(X^{(m)}, Y^{(m)}) \in \mathcal{K}$ has $C_4 \to 1/2$. Passing to a convergent subsequence $(X^{(m)}, Y^{(m)}) \to (X^*, Y^*) \in \mathcal{K}$, drop any rows with $X_i^* = 0$. The remaining indices $I_+$ satisfy $\sum_{i \in I_+} (X_i^*)^\top X_i^* = \frac{1}{2} I_2$, $\sum_{i \in I_+} (Y_i^*)^\top X_i^* = 0$, and residual energy $E^* = \sum_{i \in I_+} \|Y_i^*\|^2 \le 1$. If $E^* > 0$, the strict point-by-point result produces a spanning selection $S \subseteq I_+$ ($|S| \le 4$) with weights $q_i^* > 0$ and cost $< E^*/2 \le 1/2$; if $E^* = 0$, any spanning pair $S \subseteq I_+$ has cost $0$. Fixing positive row multipliers $w_i := q_i^* / \|X_i^*\|^2$ on $S$, the fitted matrix $B_S(X, Y)$ is continuous in a neighborhood of $(X^*, Y^*)$ (as the selected Gram matrix stays nonsingular), so for large $m$ we obtain $\frac{1}{2}\|B_S(X^{(m)}, Y^{(m)})\|_F^2 < 1/2 - \delta$, contradicting $C_4 \to 1/2$. $\square$

---

## 3. Closed Structural Classes

### Theorem 4 (Two real output lines)
*If all real output residuals $y_i$ lie on the union of at most two vector lines through the origin in $\mathbb{R}^2$, then $C_4 = 0$ and $C_3 / E \le \frac{1}{2\sqrt{2} - 1} < \frac{5}{8}$.*

*Proof.* Normalize $E = 1$. If any $y_i = 0$, (1) immediately yields $C_4 = 0$ and $C_3 \le \frac{1}{2} H(2\Sigma) = \frac{1 - d^2}{4} \le \frac{1}{4}$. Assume all $y_i \neq 0$ and partition atoms by their output line. Within each group, write $y_i = r_i v$ for a fixed unit vector $v$. Because distinct output lines are linearly independent, $\sum p_i y_i u_i^\top = 0$ splits into $\sum_{i \in \text{group}} p_i r_i u_i = 0$ for each group (and trivially for one group).

Decompose each planar relation into positive minimal relations (vertices of the fixed-mass polytope, each of size $\le 3$). If some minimal relation spans $\mathbb{R}^2$, its $\le 3$ atoms give an exact zero fit ($C_3 = C_4 = 0$). Otherwise every atom lies in a 2-atom balancing pair on a single feature line (consistent orientation gives opposite signs for $r_i$). Picking one balancing pair on each of two independent feature lines gives an exact zero fit at budget $4$, so $C_4 = 0$.

For $k = 3$: if a group spans multiple feature lines, let $b := \min \|y_i\|^2$ in that group. Take a minimizing atom $i$ and a balancing pair from the same group on another feature line; its output segment contains every scalar of magnitude $\le \sqrt{b}$ and in particular the prediction of the rank-one map $B = y_i u_i^\top$, giving cost $b/2$. Thus $C_3 \le 1/2$ if the global minimum-norm atom lies in a multiline group. If every group uses one feature line, whitening forces two perpendicular feature lines, and pairing a smallest atom on one with a balancing pair on the other gives cost $\le 1/2$.

In the only remaining case, a single-line group of mass $P \in (0, 1)$ (unit feature $u$) has minimum squared radius $a$, while the other group is multiline with minimum squared radius $b$. Whitening yields $\sum_{j \text{ in other}} p_j \sin^2 \angle(u, u_j) = 1/2$, so some feature line in the other group has $\sin^2 \angle(u, u_j) \ge \frac{1}{2(1 - P)}$. Using its zero-mean balancing pair together with the minimizing atom of the single-line group gives an exact fit of squared Frobenius norm $a / \sin^2 \angle(u, u_j) \le 2(1 - P)a$. Combining with $C_3 \le b/2$ and $1 = E \ge P a + (1 - P)b$ (put $t := 1 - P \in (0, 1)$) gives
$$
C_3 \le \min\Bigl(\frac{b}{2},\, t a\Bigr) \le \frac{t}{1 - t + 2t^2} \le \frac{1}{2\sqrt{2} - 1} \approx 0.5469 < \frac{5}{8},
$$
with the maximum attained at $t = 1/\sqrt{2}$. $\square$

### Theorem 5 (Equal-energy isotropic 6-atom systems at budget 4)
*Suppose $N = 6$, $p_i = 1/6$, $\|y_i\| = 1$, and $\Sigma = \frac{1}{2} I_2$. Then some subset of $4$ equally weighted atoms spans $\mathbb{R}^2$ and achieves $C_4 \le 1/4$.*

*Proof.* Let $X, U \in \mathbb{R}^{6 \times 2}$ have rows $u_i^\top / \sqrt{3}$ and $y_i^\top / \sqrt{3}$. The four columns of $[X\ U]$ are orthonormal; complete them to a $6 \times 6$ orthogonal matrix $[X\ U\ Z]$. Each row of $Z$ has squared norm $1/3$, so write it as $w_i^\top / \sqrt{3}$ with $\|w_i\| = 1$. Row orthogonality gives $a_{ij} + b_{ij} + c_{ij} = 0$ for every $i \neq j$, where $a_{ij} = u_i^\top u_j$, $b_{ij} = y_i^\top y_j$, and $c_{ij} = w_i^\top w_j$.

There exists a pair $J = \{i, j\}$ ($i \neq j$) with $a_{ij} c_{ij} \ge 0$: otherwise the $6$ tensors $u_k \otimes w_k \in \mathbb{R}^4$ would have pairwise strictly negative inner products, which is impossible for $> d + 1 = 5$ vectors in $\mathbb{R}^d$ (Rankin, 1955; directly, an affine dependence among $6$ vectors in $\mathbb{R}^4$ equates two convex combinations on disjoint supports, whose inner product would be simultaneously $\ge 0$ and $< 0$).

Delete that pair $J$ and give the remaining $4$ atoms equal weight. Write $G := X_J X_J^\top$, $a := a_{ij}$, $b := b_{ij}$, $c := c_{ij}$. Using $X^\top X = I_2$ and $X^\top U = 0$, the remaining four-atom Gram matrix is $I_2 - X_J^\top X_J$ with eigenvalues $1 - (1 \pm a)/3 \ge 1/3 > 0$ (so it spans), and the fit is $B^\top = -(I_2 - X_J^\top X_J)^{-1} X_J^\top U_J = -X_J^\top (I_2 - G)^{-1} U_J$. Simultaneously diagonalizing the $2 \times 2$ matrices $G = \frac{1}{3}\bigl(\begin{smallmatrix} 1 & a \\ a & 1 \end{smallmatrix}\bigr)$ and $U_J U_J^\top = \frac{1}{3}\bigl(\begin{smallmatrix} 1 & b \\ b & 1 \end{smallmatrix}\bigr)$ in the basis $(1, \pm 1)/\sqrt{2}$ yields
$$
\frac{1}{2}\|B\|_F^2 = \frac{1}{6}\Bigl(f_+ + f_- + b(f_+ - f_-)\Bigr), \qquad f_\pm := f\Bigl(\frac{1 \pm a}{3}\Bigr), \qquad f(t) := \frac{t}{(1 - t)^2}.
$$
Since $f$ is strictly increasing on $[0, 2/3]$, $f_+ - f_-$ has the same sign as $a$. Since $b = -a - c$ and $ac \ge 0$, replacing $b$ by $-a$ increases (or preserves) the cost, and exact algebraic simplification gives
$$
\frac{1}{6}\Bigl(f_+ + f_- - a(f_+ - f_-)\Bigr) - \frac{1}{4} = -\frac{a^2(5a^2 + 4)}{4(4 - a^2)^2} \le 0. \qquad \square
$$

### Proposition 6 (Classification of equal-energy isotropic 5-atom systems)
*If $N = 5$, $p_i = 1/5$, $\|y_i\| = 1$, and $\Sigma = \frac{1}{2} I_2$, then up to sign flips of $(u_i, y_i)$ and orthogonal transformations of input and output, $t_j = \zeta^j$ and $z_j = \zeta^{2j}$ ($\zeta = e^{2\pi\mathrm{i}/5}$), with exact three-point optimum $C_3 = (3 - \sqrt{5})/4$.*

*Proof.* Scale the two input and two output columns to an orthonormal $5 \times 4$ matrix; its orthogonal complement is spanned by a unit vector whose entries have equal magnitude $1/\sqrt{5}$ (since all rows have norm $\sqrt{4/5}$). Simultaneously flipping signs of $(u_j, y_j)$ makes this $5$th column constant. Viewing $u_j$ as unit complex numbers $v_j$, orthogonality to $\mathbf{1}$ gives $\sum v_j = 0$ and input isotropy gives $\sum v_j^2 = 0$. Hence the first two elementary symmetric polynomials $e_1, e_2$ vanish; since $|v_j| = 1$, $e_3 = \bar{e}_2 \prod v_j = 0$ and $e_4 = \bar{e}_1 \prod v_j = 0$. Thus $v_j^5 = \prod v_k$, so after rotation $v_j = \zeta^j$. The output columns span the real second-harmonic space (orthogonal complement of $\mathbf{1}$ and the first harmonic), so after an orthogonal output map $y_j = v_j^2$. Therefore $t_j = v_j^{-2} = v_j^3$ and $z_j = y_j / v_j = v_j = t_j^2$. Proposition 9 below gives $C_3 = (3 - \sqrt{5})/4$. $\square$

---

## 4. Two Minimax Formulations and Rational Dual Certificates

### Theorem 7 (Fixed-atom and fixed-subspace minimax dualities)
1. ***Fixed atoms, varying compatible test masses.*** *Fix full-rank input rows $x_i^\top \in \mathbb{R}^2$ and nonzero residual rows $y_i^\top \in \mathbb{R}^2$ admitting strictly positive compatible masses $p > 0$ with $\sum p_i y_i x_i^\top = 0$. Let $\mathcal{B}_k$ be the closure of all spanning $k$-sparse fits (using row multipliers). Then $\mathcal{B}_k$ is compact, and the supremum of the excess ratio $\sum p_i \|B x_i\|^2 / \sum p_i \|y_i\|^2$ over compatible $p > 0$ equals*
   $$
   \inf_{\pi,\, D \in \mathbb{R}^{2 \times 2}} \max_i \frac{\mathbb{E}_{B \sim \pi}\|B x_i\|^2 - \langle D, y_i x_i^\top \rangle_F}{\|y_i\|^2}, \tag{2}
   $$
   *where $\pi$ ranges over probability distributions on $\mathcal{B}_k$ (four matrices suffice by Carathéodory). If an interior positive compatible $p^*$ attains the maximum $c > 0$ with dual certificate $(\pi, D)$ and $H := \mathbb{E}_\pi[B^\top B]$, complementary slackness forces the exact quadric identity*
   $$
   c\left\|y_i + \frac{D x_i}{2c}\right\|^2 = x_i^\top \left(H + \frac{D^\top D}{4c}\right) x_i \qquad (\forall i).
   $$
2. ***Fixed comparator dual.*** *For arbitrary fixed observations $(x_i, y_i)$ (excluding trivial $(0,0)$ observations, which contribute zero to both losses and can be dropped without affecting the ratio) and any comparator $W$ with $d_i := \|y_i - W x_i\|^2 > 0$ for all $i$, writing $L_p(B) := \sum p_i \|y_i - B x_i\|^2$,*
   $$
   \sup_{p > 0} \frac{\inf_{B \in \mathcal{B}_k} L_p(B)}{L_p(W)} = \min_\pi \max_i \frac{\mathbb{E}_{B \sim \pi}\|y_i - B x_i\|^2}{d_i}.
   $$
   *Comparators with strictly positive $d_i$ approximate any comparator having some zero losses by an arbitrarily small generic matrix perturbation (since $(x_i, y_i) \neq (0,0)$).*
3. ***Fixed orthogonal input/output subspaces.*** *Fix $X, U \in \mathbb{R}^{N \times 2}$ with $X^\top X = U^\top U = I_2$ and $X^\top U = 0$, and vary $Y = U R$ over $\operatorname{tr}(R R^\top) = 1$. For diagonal row multipliers $D_q$ on $\le k$ spanning rows, set $T_q := (X^\top D_q X)^{-1} X^\top D_q U$. The worst normalized loss over $R$ equals*
   $$
   \min_{H \in \operatorname{conv} \overline{\{T_q^\top T_q\}}} \lambda_{\max}(H).
   $$

*Proof.* By $2 \times 2$ Cramer's rule, every spanning fit $B$ is a convex combination of the finitely many pair interpolants with weights proportional to $q_i q_j \det(x_i, x_j)^2$, so $\mathcal{B}_k$ is compact.
(1) Normalize $\sum p_i \|y_i\|^2 = 1$; the nonnegative closure of compatible $p$ is compact and convex, with strictly positive $p$ dense (mix with one positive feasible point). The payoff is bilinear in $p$ and $\mathbb{E}_\pi[B^\top B]$ (which runs over a compact convex subset of $\mathbb{S}^2$), so Sion's minimax theorem (Sion, 1958) interchanges $\sup_p$ and $\inf_\pi$. Strong LP duality in $p$ yields (2); completing the square in the active inequalities gives the quadric identity.
(2) Reparameterize via $\tilde{p}_i \propto p_i d_i$ (ranging densely over the simplex) and apply Sion's minimax theorem to the bilinear payoff in $\tilde{p}$ and $\pi$.
(3) The normalized loss is $\operatorname{tr}(R R^\top T_q^\top T_q)$, bilinear in $M = R R^\top \succeq 0$ ($\operatorname{tr} M = 1$) and $H \in \operatorname{conv}\overline{\{T_q^\top T_q\}}$; Sion's minimax theorem and $\max_{\operatorname{tr} M = 1} \operatorname{tr}(MH) = \lambda_{\max}(H)$ finish the proof. $\square$

### Proposition 8 (Exact rational dual certificates for a 7-atom continuum)
*Let*
$$
X = \begin{pmatrix} 1&0 \\ 0&1 \\ 1&1 \\ 1&-1 \\ 2&1 \\ 1&2 \\ 2&-1 \end{pmatrix}, \qquad
A = \begin{pmatrix} 1&-2 \\ -1&3 \\ 2&1 \\ 0&3 \\ -1&-1 \\ 2&-2 \\ -3&1 \end{pmatrix}, \qquad
Y = \bigl(I - X(X^\top X)^{-1}X^\top\bigr) A.
$$
*Uniform positive masses are compatible, and the compatible mass cone has at least two free parameters after normalization. Using equal row multipliers on each support (0-indexed):*
- *At $k = 3$, mixing supports $(1,5,6)$ and $(3,5,6)$ in proportions $93:7$ with $100D = \bigl(\begin{smallmatrix}-22 & 24 \\ 1 & -20\end{smallmatrix}\bigr)$ certifies the exact maximum $560659/7066875 < 1/10$ in (2) across the entire continuum of compatible test masses.*
- *At $k = 4$, mixing supports $(1,3,4,5)$ and $(1,3,5,6)$ in proportions $33:67$ with $100D = \bigl(\begin{smallmatrix}-14 & 15 \\ 2 & -12\end{smallmatrix}\bigr)$ certifies the exact maximum $381447541/7312728225 < 1/16$ in (2).*

Verified in exact rational arithmetic in [`explore.py`](explore.py).

---

## 5. Exact Cyclotomic Optima and Dual-Budget Original-Weight Obstruction

### Proposition 9 (Exact 3-point optima for regular polygons)
*For the uniform harmonic moment systems $t_j = e^{2\pi\mathrm{i} j/n}$, $z_j = t_j^h$ with $(n, h) \in \{(5, 2), (6, 3), (7, 3)\}$, let $x := 2\cos(2\pi/n)$. The exact squared-distance minima on all dihedral triangle types are:*

| $n$ | Support representative | Exact squared distance | Minimal polynomial of $x$ |
|---:|---|---|---|
| 5 | $\{0,1,2\}$ | $1 + x/2$ | $x^2 + x - 1$ |
| 5 | $\{0,1,3\}$ | $(1 - x)/2$ | $x^2 + x - 1$ |
| 6 | $\{0,1,2\}, \{0,1,3\}$ | $1/2$ | $x - 1$ |
| 6 | $\{0,2,4\}$ | $1$ | $x - 1$ |
| 7 | $\{0,1,2\}$ | $1 + x/2$ | $x^3 + x^2 - 2x - 1$ |
| 7 | $\{0,1,3\}$ | $1/4$ | $x^3 + x^2 - 2x - 1$ |
| 7 | $\{0,1,4\}$ | $2 - (x^2 + x)/2$ | $x^3 + x^2 - 2x - 1$ |
| 7 | $\{0,2,4\}$ | $x^2/2$ | $x^3 + x^2 - 2x - 1$ |

*The overall three-point optima are $(3 - \sqrt{5})/4$ ($n = 5$), $1/2$ ($n = 6$), and $1/4$ ($n = 7$, attained on $\{0,1,3\}$, sharpening Zhang's witness $2/7$).*

*Proof.* For each pair $i \neq j$, the exact pair interpolant is $b_{ij} = \sum_{l=0}^{h-1} t_i^l t_j^{h-1-l}$, $a_{ij} = -t_i t_j \sum_{l=0}^{h-2} t_i^l t_j^{h-2-l}$. On any triangle $\{i,j,l\}$, Cramer's rule expresses fits as convex combinations of the three edge fits with weights proportional to $q_i q_j |t_i - t_j|^2$; any strictly positive barycentric weights $(r_{ij}, r_{il}, r_{jl})$ are realized by choosing $q_i \propto |t_j - t_l|^2 / r_{jl}$. Thus $C_3$ equals the minimum squared distance from $0$ to the convex hulls of edge triples. Form the real $3 \times 3$ Gram matrix $G_{ef} = \operatorname{Re}(a_e \bar{a}_f + b_e \bar{b}_f)$, whose entries are polynomials in $x = 2\cos(2\pi/n)$ via Chebyshev polynomials $T_m(x/2)$. For each active face $J \subseteq \{1, 2, 3\}$, solving $G_{JJ} r_J = c \mathbf{1}$, $\mathbf{1}^\top r_J = 1$ and verifying $r \ge 0$, $G r \ge c \mathbf{1}$, $r^\top G r = c$ certifies global KKT optimality on that triangle. The script [`explore.py`](explore.py) certifies every row exactly in $\mathbb{Q}[x]/(\text{minpoly})$ with rational interval arithmetic on $[0.61803, 0.61804]$, $\{1\}$, and $[1.24697, 1.24698]$, and verifies dihedral coverage of all $\binom{5}{3} = 10$, $\binom{6}{3} = 20$, and $\binom{7}{3} = 35$ triangles. (As Zhang observed, restricting to $\mu = 0$ changes the heptagon optimum: Parseval forces $C \ge (7/k - 1)/2$ at budget $k$.) $\square$

### Proposition 10 (Simultaneous 7-atom original-weight obstruction at both budgets)
*There exists a 7-atom moment system with all atoms distinct such that retaining the original relative weights on **every** spanning support of size $\le k$ yields minimum cost strictly above $5/8$ (at $k = 3$) and strictly above $1/4$ (at $k = 4$), whereas reweighting one cross pair to equal weights beats both targets.*

*Proof.* Take two perpendicular feature lines of mass $1/2$ each. On the first line place a uniform centered equilateral triangle of squared radius $e := 20/21$. On the second line place the four real output vectors
$$
(-1,\, \pm s/100), \qquad (5/9,\, \pm s), \qquad s := \frac{2\sqrt{14}}{9},
$$
with conditional weights $5/28, 5/28, 9/28, 9/28$, respectively. Both classes have zero conditional mean, so all moment conditions hold; the second class has conditional energy $e' = 1 + 1/40500$. For any spanning selection, $C/E$ equals the sum of the two squared conditional means divided by $e + e'$.

Under original relative weights, the first-class costs for $1, 2, 3$ atoms are $e, e/4, 0$. In the second class, a single atom costs at least $1$; the two same-$x$-sign pairs cost $1$ and $25/81$; and an opposite-vertical-sign cross pair has zero $x$-mean and minimum squared mean
$$
d := \left(\frac{9 - 5/100}{14} s\right)^2 = \frac{32041}{113400} < \frac{25}{81}.
$$
Comparing all spanning allocations (since $e/4 + 1 > e + d$, and $(1,3)$ and $(3,1)$ cost at least $e$ and $1$ before division) yields the exact original-weight optima
$$
\frac{e + d}{e + e'} = \frac{700205}{1107014} > \frac{5}{8}, \qquad \frac{e/4 + d}{e + e'} = \frac{295205}{1107014} > \frac{1}{4}.
$$
In contrast, giving the opposite-vertical-sign cross pair **equal selected weights** lowers its squared mean to $d' = ((4/9)^2 + (99s/100)^2)/4 = 88607/405000$, yielding ratios
$$
\frac{e + d'}{e + e'} = \frac{3320249}{5535070} < \frac{5}{8}, \qquad \frac{e/4 + d'}{e + e'} = \frac{1295249}{5535070} < \frac{1}{4}. \qquad \square
$$

---

## 6. Reproducible Verification and Numerical Exploration

Run from the repository root (requires a C compiler and the project Python environment; the C helper [`search.c`](search.c) is compiled into a temporary directory):

```sh
.venv/bin/python -B solutions/weighted_data_vector/explore.py
.venv/bin/python -B solutions/weighted_data_vector/explore.py --suite
```

The first command verifies all exact certificates (cyclotomic KKT certificates of Proposition 9, rational dual certificates of Proposition 8, original-weight enumeration of Proposition 10, polynomial identities of Theorems 2 and 5, and reference sanity checks) and exits with code `0`.

The optional `--suite` flag runs 84 heuristic global searches (552,066 differential-evolution evaluations followed by SLSQP epigraph refinement across all 22 partitions of $5, 6, 7$ atoms among $\ge 3$ direction groups). The largest heuristic witness costs observed were:

| Atom count | Budget $k = 3$ | Budget $k = 4$ |
|---:|---:|---:|
| 5 | 0.527457 | 0.200000 |
| 6 | 0.570584 | 0.250000 |
| 7 | 0.533421 | 0.249988 |

These are floating-point search records only (not certified maxima), completely decoupled from all proofs and exact certificates above.

---

## References

- R. A. Rankin, *The closest packing of spherical caps in $n$ dimensions*, Proceedings of the Glasgow Mathematical Association 2(3):139–144, 1955.
- M. Sion, *On general minimax theorems*, Pacific Journal of Mathematics 8(1):171–176, 1958.
- G. Zhang, *Exact Recovery Thresholds for Weighted Data Selection in Vector-Valued Linear Regression*, [arXiv:2608.30254](https://arxiv.org/abs/2608.30254), October 2026.
