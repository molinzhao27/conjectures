# Continuous Piecewise Linear Triangulations of the Product Surface

**Target paper:** Robert Burlacu, Lukas Hager, and Robert Hildebrand, *Optimal Triangulations for Piecewise Linear Approximations of Non-Convex Variable Products*, [arXiv:2604.04026](https://arxiv.org/abs/2604.04026) (2026), Conjecture `conj:continuous-optimality` and Remark `lem:no-optimal-tiling`.  
**Companion files:** [`solutions/pwl_product_triang/explore.py`](explore.py), [`solutions/pwl_product_triang/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:continuous-optimality`: OPEN in general**, with the following rigorous universal and structural progress:
  $$
  \frac{0.16240658646\ldots}{\varepsilon} \le \inf_{\mathcal{T}, f} \delta(\mathcal{T}) \le \frac{\sqrt{3}}{8\varepsilon} = \frac{0.21650635094\ldots}{\varepsilon}.
  $$
- **Universal density improvement (Theorem 3):** For *every* locally finite triangulation $\mathcal{T}$ of $\mathbb{R}^2$ admitting a continuous PWL $\varepsilon$-approximation of $F(x,y) = xy$ (with **no** bounded-diameter assumption), the triangle density strictly exceeds the unrestricted (possibly discontinuous) optimum $3\sqrt{3}/(32\varepsilon) = 0.16237976320\ldots/\varepsilon$:
  $$
  \delta(\mathcal{T}) \ge \frac{1}{M\varepsilon}\left(1 + \frac{1 - (123/20)/M}{1 + 6(11/10)^2}\right) = \frac{0.16240658646\ldots}{\varepsilon}, \qquad M := \frac{32\sqrt{3}}{9}.
  $$
- **Resolution of Remark `lem:no-optimal-tiling`:** As an immediate corollary of our quantitative vertex obstruction (Lemma 3), we completely prove the open sketch in Remark `lem:no-optimal-tiling` of `arXiv:2604.04026`: no continuous triangulation can consist exclusively of optimal general triangles of area $M\varepsilon$ (or even exclusively of triangles of area $> 6.15\varepsilon$).
- **Sharp bound for one signed edge-product multiset (Theorem 2):** The conjectured bound $\delta(\mathcal{T}) \ge \sqrt{3}/(8\varepsilon)$ **holds sharply** for every continuous triangulation in which all triangles share the same unordered multiset of signed edge products, allowing arbitrary globally consistent vertex deviations (strictly generalizing Theorem `thm:optimal-triangulation-cont` of `arXiv:2604.04026` for parallelogram tilings with constant deviations).
- **Explicit continuous periodic mesh with an oversized face (Proposition 1):** We construct an explicit 10-face continuous periodic triangulation on $[0, 21/10] \times [0, 51/10]$ with exact global error $169/180 < 1$ (at $\varepsilon = 1$) containing an individual triangle of area $5\varepsilon > 8\varepsilon/\sqrt{3}$. This rules out any facewise area cap $A(T) \le 8\varepsilon/\sqrt{3}$ by exhibiting an explicit continuous periodic mesh (concrete realization of the phenomenon noted without a periodic construction in Remark `rem:conjecture-evidence` of `arXiv:2604.04026`), while its density $1000/(1071\varepsilon)$ stays above $\sqrt{3}/(8\varepsilon)$.

---

## 1. Setting and Local Error Characterization

Following Burlacu, Hager, and Hildebrand (`arXiv:2604.04026`, Definitions 2.1–2.6) and Atariah, Rote, and Wintraecken (2018), let $\mathcal{T}$ be a locally finite face-to-face triangulation of $\mathbb{R}^2$ by closed nondegenerate triangles (every compact square meets finitely many triangles). A continuous piecewise linear (PWL) approximation $f$ of $F(x,y) = xy$ on $\mathcal{T}$ is uniquely determined by a single signed deviation $d_v := f(v) - F(v)$ at each vertex $v$, subject to $|f(x,y) - xy| \le \varepsilon$ everywhere. Fixing concentric squares $Q_\ell := [-\ell/2, \ell/2]^2$ of side length $\ell$, the triangle density is
$$
\delta(\mathcal{T}) := \limsup_{\ell \to \infty} \frac{\#\{T \in \mathcal{T} : T \cap Q_\ell \neq \varnothing\}}{\ell^2}.
$$
Scaling $(x,y) \mapsto (\sqrt{\varepsilon}\,x, \sqrt{\varepsilon}\,y)$ and $f, F \mapsto \varepsilon f, \varepsilon F$ reduces the problem to $\varepsilon = 1$, which we assume throughout the proofs.

Write $q(u) := u_x u_y$ for $u = (u_x, u_y) \in \mathbb{R}^2$. Along the edge $[v_i, v_j]$ parameterized by $(1 - t)v_i + t v_j$ ($t \in [0, 1]$), the signed error is
$$
E(t) = (1 - t)d_i + t d_j + t(1 - t)\,q(v_j - v_i).
$$
Thus $|E(t)| \le 1$ for all $t \in [0, 1]$ if and only if $d_i, d_j \in [-1, 1]$ and
$$
-\bigl(\sqrt{1 + d_i} + \sqrt{1 + d_j}\bigr)^2 \le q(v_j - v_i) \le \bigl(\sqrt{1 - d_i} + \sqrt{1 - d_j}\bigr)^2.
$$
Indeed, for $t \in (0, 1)$, $E(t) \le 1$ is equivalent to $q(v_j - v_i) \le \frac{1 - d_i}{t} + \frac{1 - d_j}{1 - t}$, whose minimum over $t \in (0, 1)$ is $(\sqrt{1 - d_i} + \sqrt{1 - d_j})^2$; sign reversal yields the lower bound. Inside any triangle, the error $f - F$ has constant Hessian $\bigl(\begin{smallmatrix} 0 & -1 \\ -1 & 0 \end{smallmatrix}\bigr)$ of determinant $-1$, so any interior critical point is a saddle. Hence the edge tests are necessary and sufficient for $|f - F| \le 1$ on the entire face. For a triangle with two edge vectors $u, v$ and signed edge products $p = q(u)$, $r = q(v)$, $s = q(u - v)$, direct expansion gives the area identity
$$
4A^2 = \det(u, v)^2 = (p + r - s)^2 - 4pr,
$$
where $A = |\det(u, v)|/2$ is the triangle area.

---

## 2. Continuous Periodic Mesh with an Oversized Face

### Proposition 1 (Explicit continuous periodic mesh with a face of area $5\varepsilon$)
*At $\varepsilon = 1$, there exists a continuous periodic triangulation of $\mathbb{R}^2$ with period box $[0, 21/10] \times [0, 51/10]$ (area $1071/100$), $10$ triangles per period, density $1000/1071$, exact global maximum error $169/180 < 1$, and an individual triangle of area $5 > 8/\sqrt{3}$.*

*Proof.* In the subrectangle $[0, 2] \times [0, 5]$, take $A = (0, 0)$, $B = (0, 5)$, $D = (2, 0)$, $C = (2, 5/2)$, and $E = (2, 5)$, with deviations $d_A = -2/3$, $d_B = +2/3$, and $d_C = d_D = d_E = 0$. Triangulate $[0, 2] \times [0, 5]$ into $ABC$, $ADC$, and $BCE$. The only nonzero edge products are $q(C - A) = +5$ on $AC$ and $q(C - B) = -5$ on $BC$. Both pass the edge bounds strictly because $5 < (1 + \sqrt{5/3})^2$, and triangle $ABC$ has exact area $5$.

Complete the period rectangle $[0, 21/10] \times [0, 51/10]$ as follows:
- On the right strip $[2, 21/10] \times [0, 5]$, place the translates $A' = (21/10, 0)$ and $B' = (21/10, 5)$ with deviations $-2/3$ and $+2/3$, and form the three triangles $A' B' C$, $A' D C$, and $B' E C$.
- On the top strip $[0, 21/10] \times [5, 51/10]$, split at $x = 2$ into two rectangles and triangulate each with one diagonal (joining $(0, 5)$ to $(2, 51/10)$ and $(2, 5)$ to $(21/10, 51/10)$), assigning to the top boundary $(x, 51/10)$ the same deviations as on the bottom boundary $(x, 0)$.

Extend the mesh and deviations periodically by $(21/10, 0)$ and $(0, 51/10)$ and interpolate $f(v) = F(v) + d_v$ affinely on each face. Every edge on the torus has incidence $2$, so matching deviations along opposite boundary edges ensures continuity of $f$ everywhere. Every added edge has $|q| \le 1/2$, while all vertex deviations satisfy $|d_v| \le 2/3$, so both allowable edge-product caps are at least $(1 + \sqrt{1/3})^2 > 4/3 > 1/2$. Exact rational evaluation over all edges gives the global maximum error $169/180 < 1$ (attained on $AC$ and $BC$).

Scaling back to arbitrary $\varepsilon > 0$ gives a face of area $5\varepsilon > 8\varepsilon/\sqrt{3}$, ruling out any facewise area cap $A(T) \le 8\varepsilon/\sqrt{3}$ even for continuous periodic meshes (realizing the non-constant-deviation effect noted in Remark `rem:conjecture-evidence` of `arXiv:2604.04026`). Its density $1000/(1071\varepsilon) \approx 0.9337/\varepsilon$ lies well above $\sqrt{3}/(8\varepsilon)$. $\square$

---

## 3. Sharp Bound for One Signed Edge-Product Multiset

### Theorem 2 (Sharp density bound for one signed edge-product multiset)
*Let $\mathcal{T}$ be a locally finite triangulation of $\mathbb{R}^2$ admitting a continuous PWL $\varepsilon$-approximation of $F(x,y) = xy$ (with arbitrary globally consistent vertex deviations). If every triangle in $\mathcal{T}$ has the same unordered multiset of three signed edge products, then every triangle has area at most $8\varepsilon/\sqrt{3}$ and $\delta(\mathcal{T}) \ge \sqrt{3}/(8\varepsilon)$. Equality is attained by the constant-deviation parallelogram tiling with $\Delta = \varepsilon/3$.*

*Proof.* Scale to $\varepsilon = 1$.

**Case 1 (All products weakly of one sign).** Reflecting one coordinate and negating all deviations if needed, assume all three edge products are $\ge 0$. Order a triangle's vertices $v_1, v_2, v_3$ by increasing $x$-coordinate (breaking ties by increasing $y$-coordinate). Since $q(v_j - v_i) \ge 0$, their $y$-coordinates are also nondecreasing. Hence $v_1$ and $v_3$ are opposite corners of an axis-aligned bounding box containing the triangle, so $A \le \frac{1}{2} q(v_3 - v_1) \le \frac{1}{2} \cdot 8 = 4 < 8/\sqrt{3}$ (this also covers triangles with two null edges).

**Case 2 (Mixed signs).** Reflecting one coordinate and negating deviations if needed, write the common multiset as $\{a, b, -w\}$ with $0 \le a \le b$ and $b, w > 0$. We claim that **every vertex of $\mathcal{T}$ is incident to at least one edge of product $b$ and at least one edge of product $-w$**.
- *Subcase $a > 0$, no incident $-w$-edge:* Every corner at the vertex would be bounded by two positive-product spokes with a negative-product opposite edge. Because two spokes in opposite positive quadrants have a difference with positive product, all spokes must lie in a single open positive quadrant, which cannot cover a full $360^\circ$ neighborhood of the vertex. (If $a = b$, every corner has a positive spoke, completing the claim.)
- *Subcase $0 < a < b$, no incident $b$-edge:* Every incident corner is bounded by an $a$-spoke and a $-w$-spoke. Since opposite-sign spokes bound a sector containing exactly one of the four open coordinate-axis rays, the vertex has degree $4$ with alternating spokes $p_1, n_1, p_2, n_2$ (shifted to the origin) satisfying $q(p_i) = a$, $q(n_j) = -w$, and $q(p_i - n_j) = b$ for all $i, j \in \{1, 2\}$. Subtracting the equations for $i = 1, 2$ at fixed $j$ yields $B(p_1 - p_2, n_j) = 0$ for the nondegenerate symmetric bilinear form $B(u, v) = \frac{1}{2}(u_x v_y + u_y v_x)$ polarizing $q$. Since $p_1 \neq p_2$, $n_1$ and $n_2$ are collinear; $q(n_1) = q(n_2) = -w$ and distinctness force $n_2 = -n_1$. Averaging $q(p_1 - n_1) = b$ and $q(p_1 + n_1) = b$ yields $b = q(p_1) + q(n_1) = a - w < a$, a contradiction.
- *Subcase $a = 0$:* Any vertex missing either a $b$-edge or a $-w$-edge must be incident to a null edge ($q = 0$). Place that vertex at the origin and assume the null edge is horizontal, $(s, 0)$ with $s \neq 0$ (vertical is identical after swapping coordinates). If both flanking triangles presented the same nonnull product ($b$ or $-w$) at the origin, their third vertices $(x, y)$ would both satisfy $sy = q(x, y) - q(x - s, y) = \pm(b + w)$, placing them on the same side of the horizontal edge—impossible for a face-to-face triangulation. Thus the origin is incident to both $b$ and $-w$.

Now start at any vertex $v_0$ and build a walk $v_0, v_1, v_2, \dots$ alternating a $b$-edge $(v_{2k}, v_{2k+1})$ with a $-w$-edge $(v_{2k+1}, v_{2k+2})$. The midpoint error test ($t = 1/2$) on these two edges gives
$$
b \le 4 - 2(d_{v_{2k}} + d_{v_{2k+1}}), \qquad w \le 4 + 2(d_{v_{2k+1}} + d_{v_{2k+2}}).
$$
Adding these inequalities gives $d_{v_{2k+2}} - d_{v_{2k}} \ge (b + w - 8)/2$. Because $d_v \in [-1, 1]$ is bounded, we must have $b + w \le 8$.

With $s = -w$, the area formula reads $4A^2 = (b - a)^2 + 2w(a + b) + w^2$, which is convex in $a \in [0, b]$. Checking the endpoints:
- At $a = 0$: $4A^2 = (b + w)^2 \le 64$.
- At $a = b$: $4A^2 = w(4b + w) \le w(32 - 3w) = \frac{256}{3} - 3\bigl(w - \frac{16}{3}\bigr)^2 \le \frac{256}{3}$.

Therefore $A \le 8/\sqrt{3}$ for every face. Since the triangles meeting $Q_\ell$ cover $Q_\ell$, we have $\#\{T : T \cap Q_\ell \neq \varnothing\} \cdot (8/\sqrt{3}) \ge \ell^2$, giving $\delta(\mathcal{T}) \ge \sqrt{3}/8$.

For sharpness, take $u = (s, t)$ and $v = (t, s)$ with $st = 8/3$ and $s - t = 4/\sqrt{3}$ ($s > t > 0$), split every translate of the parallelogram spanned by $u, v$ along $v - u$, and assign constant deviation $d_v = 1/3$. Every face has edge-product multiset $\{8/3, 8/3, -16/3\}$ and area $8/\sqrt{3}$; boundary cells of $Q_\ell$ contribute $O(\ell)$, so $\delta(\mathcal{T}) = \sqrt{3}/8$. $\square$

---

## 4. Universal Density Improvement for Arbitrary Continuous Meshes

Set $M := 32\sqrt{3}/9 \approx 6.1584028714$ and $B := 123/20 = 6.15$. Call a triangle $T$ **large** if $A(T) > B$.

### Lemma 3 (Unrestricted area bound, quantitative stability, and vertex obstruction)
*At $\varepsilon = 1$, every valid triangle satisfies $A(T) \le M$. Moreover:*
1. ***Quantitative stability.*** *Every large triangle ($A(T) > B$) has mixed edge-product signs. After possibly reflecting one coordinate and negating all deviations so that two positive-product edges $p, r > 0$ meet at vertex $0$ opposite a negative-product edge $-w < 0$, its parameters satisfy*
   $$
   d_0 < -0.96, \qquad 0.5 < d_1, d_2 < 0.92, \qquad w > \max(p, r).
   \tag{1}
   $$
2. ***Vertex obstruction.*** *Mark the two endpoints of the negative-product base in (1) (using the reflected convention on sign-reversed faces). Every vertex receives at most $4$ marks, and every marked vertex is incident to at least one triangle of area at most $B$. In particular, no locally finite continuous triangulation can consist exclusively of large triangles (completing Remark `lem:no-optimal-tiling` of `arXiv:2604.04026`).*

*Proof.*
**(1)** One-sign triangles have $A \le 4 < B$. For mixed signs, put $a_i := \sqrt{(1 - d_i)/2} \in [0, 1]$ and $b_i := \sqrt{1 - a_i^2} = \sqrt{(1 + d_i)/2}$. The edge caps are $p \le 2(a_0 + a_1)^2$, $r \le 2(a_0 + a_2)^2$, and $w \le 2(b_1 + b_2)^2$. Since $4A^2 = (p - r)^2 + 2w(p + r) + w^2$ increases in $w$ and is separately convex in $p, r$, it is bounded by evaluating at $w = 2(b_1 + b_2)^2$ and the four endpoint pairs for $(p, r)$. If $p = 0$ (or $r = 0$), using the vertex $i$ shared by the remaining edges gives $2A \le 2[(1 + a_i)^2 + (1 + b_i)^2] \le 6 + 4\sqrt{2} < 2B$.

At the remaining corner $(p, r, w) = (2(a_0 + a_1)^2, 2(a_0 + a_2)^2, 2Q)$ with $Q := (b_1 + b_2)^2$, write $m := (a_1 + a_2)/2$, $H := (a_1 - a_2)/2$, $h := H^2$, $S := 1 - m^2$, and $C := (a_0 + m)^2$. Since $Q \le 2(b_1^2 + b_2^2) = 4(S - h)$, direct polynomial expansion yields
$$
A_{\mathrm{corner}}^2 = 16Ch + Q(Q + 4C + 4h) \le 16S(S + C) - 16Sh = P(m) - 16S\bigl\{h + (1 - a_0)(1 + a_0 + 2m)\bigr\},
$$
where $P(m) := 32(1 - m)(1 + m)^2$. Differentiating $P'(m) = -32(m + 1)(3m - 1)$ shows $P(m) \le P(1/3) = 1024/27 = M^2$ on $[0, 1]$, proving $A \le M$.

If $A > B$, then $P(m) > B^2$. Checking the rational endpoints $P(29/100) < B^2$ and $P(39/100) < B^2$ forces $0.29 < m < 0.39$, so $S = 1 - m^2 > 0.84$. Since $M^2 - B^2 < 0.104$, we obtain
$$
h < \frac{0.104}{16 \cdot 0.84} < 0.09^2, \qquad 1 - a_0 < \frac{0.104}{16 \cdot 0.84} < 0.01,
$$
whence $a_0 > 0.99$ and $0.20 < a_1, a_2 < 0.48$. Converting via $d_i = 1 - 2a_i^2$ yields $d_0 < -0.96$ and $0.5 < d_1, d_2 < 0.92$. In particular $p, r < 2(1 + 0.48)^2 < 4.5$. If $w \le \max(p, r)$, maximizing $(p - r)^2 + 2w(p + r) + w^2$ over the cube $[0, 4.5]^3$ by separate convexity gives $4A^2 \le 5(4.5)^2 < 4B^2$, a contradiction. Thus $w > \max(p, r)$.

**(2)** Each marked corner is bounded by one positive-product spoke and one negative-product spoke, so it contains exactly one of the four open coordinate-axis rays from its vertex; hence each vertex receives at most $4$ marks.

Suppose for contradiction that all triangles incident to a marked vertex $v$ are large (forming a finite complete fan by local finiteness). Because the four deviation intervals $(-1, -0.96)$, $(-0.92, -0.5)$, $(0.5, 0.92)$, and $(0.96, 1)$ in (1) and its sign-reversal are pairwise disjoint, every incident triangle must share the same sign convention and mark $v$. Translating $v$ to the origin and fixing the convention of (1), the fan has degree $4$ with spokes alternating in sign, one in each open quadrant:
$$
p_1 = (u, v), \quad p_2 = (-s, -t) \quad (q > 0), \qquad
n_1 = (h, -g), \quad n_2 = (-h', g') \quad (q < 0),
$$
with all eight scalars $u, v, s, t, h, g, h', g' > 0$. Every positive spoke $p \in \{p_1, p_2\}$ is adjacent to both negative spokes $n \in \{n_1, n_2\}$, the opposite edge satisfies $q(p - n) > 0$, and (1) gives $q(n) < -q(p)$. Therefore
$$
p_x n_y + p_y n_x = q(p) + q(n) - q(p - n) < 0
$$
for all four pairs $(p, n)$. Expanding the four inequalities yields $g'/h' < v/u < g/h$ from $p_1$ and $g/h < t/s < g'/h'$ from $p_2$, a contradiction. $\square$

### Theorem 3 (Universal density improvement without diameter bounds)
*For every locally finite triangulation $\mathcal{T}$ of $\mathbb{R}^2$ admitting a continuous PWL $\varepsilon$-approximation of $F(x,y) = xy$,*
$$
\delta(\mathcal{T}) \ge \frac{1}{M\varepsilon}\left(1 + \frac{1 - (123/20)/M}{1 + 6(11/10)^2}\right) = \frac{0.16240658646\ldots}{\varepsilon} > \frac{3\sqrt{3}}{32\varepsilon}.
$$

*Proof.* Scale to $\varepsilon = 1$. Put $\eta := 1 - B/M$ (note $\eta < 1/484$ since $B^2 > M^2(1 - 1/484)^2$), $\alpha := 11/10$, $N_\ell := \#\{T \in \mathcal{T} : T \cap Q_\ell \neq \varnothing\}$, and $D_\ell := M N_\ell / \ell^2 - 1 \ge 0$. Since $\sum_{T \cap Q_\ell \neq \varnothing} |T \cap Q_\ell| = \ell^2$, the deficit sum satisfies
$$
\sum_{T \cap Q_\ell \neq \varnothing} \bigl(M - |T \cap Q_\ell|\bigr) = M N_\ell - \ell^2 = \ell^2 D_\ell.
$$
Let $G_\ell$ be the number of triangles with $|T \cap Q_\ell| > B$. Each triangle with $|T \cap Q_\ell| \le B$ contributes at least $M - B = M\eta$ to the deficit sum, so $N_\ell - G_\ell \le \ell^2 D_\ell / (M\eta)$. Since $N_\ell \ge \ell^2 / M$, we get
$$
G_\ell \ge \frac{\ell^2}{M}\Bigl(1 - \frac{D_\ell}{\eta}\Bigr).
$$

**Overhang control (no bounded-diameter assumption).** We claim every one of these $G_\ell$ triangles lies wholly inside the concentric enlarged square $Q_{\alpha\ell}$. Indeed, $|T \cap Q_\ell| > B = (1 - \eta)M \ge (1 - \eta)|T|$, so $|T \setminus Q_\ell| < \eta |T|$. Along either coordinate axis, let $W$ be the projection width of $T$. If an extremal vertex of $T$ extends distance $z > 0$ past a side of $Q_\ell$, the slice of $T$ beyond that side line is a homothetic subtriangle based at that vertex of area fraction $(z/W)^2 < \eta$, giving $z < \sqrt{\eta}\,W$. Summing both possible overhangs yields $W < \ell + 2\sqrt{\eta}\,W$, so $W < \ell / (1 - 2\sqrt{\eta})$. Consequently every point of $T$ has both coordinates bounded in magnitude by $\frac{\ell}{2} + \sqrt{\eta}\,W < \frac{\ell}{2(1 - 2\sqrt{\eta})}$, i.e. $T \subseteq Q_{\ell / (1 - 2\sqrt{\eta})} \subset Q_{\alpha\ell}$ because $\eta < 1/484 \implies (1 - 2\sqrt{\eta})^{-1} < 11/10 = \alpha$.

**Double-counting marks.** Each of the $G_\ell$ large triangles supplies $2$ marks on vertices inside $Q_{\alpha\ell}$, giving $2G_\ell$ marks total. By Lemma 3(2), each marked vertex receives at most $4$ marks and touches a triangle of area $\le B$ (which therefore meets $Q_{\alpha\ell}$). Because each triangle of area $\le B$ has at most $3$ vertices, $Q_{\alpha\ell}$ meets at least $\frac{2G_\ell}{4 \cdot 3} = \frac{G_\ell}{6}$ triangles of area $\le B$. Each such triangle contributes at least $M\eta$ to the deficit sum on $Q_{\alpha\ell}$, so
$$
(\alpha\ell)^2 D_{\alpha\ell} \ge \frac{G_\ell}{6} \cdot M\eta \ge \frac{\eta \ell^2}{6}\Bigl(1 - \frac{D_\ell}{\eta}\Bigr)
\quad\Longrightarrow\quad
D_{\alpha\ell} \ge \frac{\eta}{6\alpha^2}\Bigl(1 - \frac{D_\ell}{\eta}\Bigr).
$$
Let $D := \limsup_{\ell \to \infty} D_\ell$. If $D = \infty$ the theorem is immediate; otherwise, for every $\delta > 0$ eventually $D_\ell, D_{\alpha\ell} \le D + \delta$, yielding $D + \delta \ge \frac{\eta}{6\alpha^2}(1 - (D + \delta)/\eta)$. Letting $\delta \to 0^+$ gives $D \ge \frac{\eta}{1 + 6\alpha^2}$. Restoring $\delta(\mathcal{T}) = (1 + D)/(M\varepsilon)$ completes the proof. $\square$

The remaining open gap is to control the average area of continuous triangulations whose faces lie in the intermediate range $(8\varepsilon/\sqrt{3}, 6.15\varepsilon]$, where the disjoint-band obstruction of Lemma 3 no longer applies.

---

## 5. Reproducible Verification and Periodic Sanity Check

Run from the repository root:

```sh
.venv/bin/python -B solutions/pwl_product_triang/explore.py
```

The script [`explore.py`](explore.py) performs:
1. **Exact rational verification of Proposition 1:** Checks all $10$ triangles on $[0, 21/10] \times [0, 51/10]$, torus edge incidence $2$, total area $1071/100$, maximum face area $5$, and exact global error $169/180 < 1$.
2. **Exact symbolic and rational verification of Theorem 2, Lemma 3, and Theorem 3:** Verifies the corner-area polynomial identities in SymPy, all rational cutoffs ($M^2 = 1024/27$, $B = 123/20$, $P(0.29), P(0.39) < B^2$, $w > \max(p, r)$, $\eta < 1/484$), and the resulting lower bound $0.162406586464$.
3. **Finite periodic SLSQP sanity check:** Optimizes four vertex orbits and two period vectors on a $2 \times 2$ periodic grid across all $16$ diagonal patterns and $4$ starting orientations ($64$ runs total). All $64/64$ runs converged and passed feasibility checks, achieving best mean area $4.6188021535$ (matching $8/\sqrt{3}$ to $10^{-10}$). This finite search serves only as a sanity check on the scale of the conjectured bound and does not establish optimality over arbitrary meshes.

---

## References

- D. Atariah, G. Rote, and M. Wintraecken, *Optimal triangulation of saddle surfaces*, Beiträge zur Algebra und Geometrie 59(1):113–126, 2018.
- R. Burlacu, L. Hager, and R. Hildebrand, *Optimal Triangulations for Piecewise Linear Approximations of Non-Convex Variable Products*, [arXiv:2604.04026](https://arxiv.org/abs/2604.04026), 2026.
- H. Pottmann, R. Krasauskas, B. Hamann, K. Joy, and W. Seibold, *On piecewise linear approximation of quadratic functions*, Journal for Geometry and Graphics 4(1):31–53, 2000.
