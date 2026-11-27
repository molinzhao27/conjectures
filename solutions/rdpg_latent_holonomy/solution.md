# Complete Resolution of Restricted Holonomy for the RDPG Bundle and Laplacian Dynamics

**Target paper:** Giulio Valentino Dalla Riva, *Random Dot Product Graphs as Dynamical Systems: Limitations and Opportunities*, [arXiv:2603.05703](https://arxiv.org/abs/2603.05703), Conjecture `conj:full-holonomy` (Full Holonomy in Higher Dimensions).  
**Companion files:** [`solutions/rdpg_latent_holonomy/explore.py`](explore.py), [`solutions/rdpg_latent_holonomy/proof.pdf`](proof.pdf).

---

## Verdict

Conjecture `conj:full-holonomy` is **completely settled across all three natural readings** of its statement for $n > d \ge 2$:

1. **Ambient RDPG principal $O(d)$-bundle (Theorem 1 — TRUE everywhere, including regular graphs):** On the full-rank interior $\mathcal{X} = \{X \in \mathbb{R}^{n \times d} : \operatorname{rank}(X) = d,\ 0 < (XX^\top)_{ij} < 1\}$ with $n > d$, the restricted holonomy group satisfies $\mathrm{Hol}_p^0 = \mathrm{SO}(d)$ at **every** point $p = XX^\top$ (and within every open neighborhood of $p$). Curvature values at the base point alone already span $\mathfrak{so}(d)$, so Ambrose–Singer immediately yields $\mathrm{SO}(d)$.
2. **Single autonomous Laplacian orbit (Theorem 2 — FALSE everywhere):** If holonomy is restricted to loops lying inside a single trajectory of the autonomous ODE $\dot{X} = -L(P)X$, the restricted holonomy is **trivial** at every interior initial condition (generic or exceptional): $\frac{d}{dt}\operatorname{tr}(P) = -2\sum_{i<j} P_{ij}\|x_i - x_j\|^2 < 0$ strictly, so each orbit is an embedded interval carrying a global horizontal section with zero pullback curvature.
3. **Finite-time projected-commutator rank criterion (Theorems 3–4 — FALSE for arbitrary non-regular graphs, TRUE generically):**
   - *Non-regular counterexample (Theorem 3):* For every $d \ge 2$, the explicit $S_d$-symmetric interior frame on $n = d + 1$ nodes has full rank, strictly interior probabilities, and non-regular degrees, yet all $L(t)$ commute for all $t$, forcing every frozen projected commutator $X^\top[L(t_1), L(t_2)]X$ to vanish identically (and giving a flat 3-dimensional horizontal submanifold). Nevertheless, an explicit contractible ambient disk loop of radius $r = 1/(40d)$ based at this exact graph produces the exact nonidentity rotation $\exp(-\frac{2\pi}{201}(uv^\top - vu^\top)) \in \mathrm{SO}(d)$.
   - *Generic full-rank theorem (Theorem 4):* For every $n > d \ge 2$, outside a proper real algebraic subset of initial frames (hence on a Zariski-open, dense, full-measure subset of $\mathcal{X}$), the jets $J_k(X) = X^\top[L(0), L^{(k)}(0)]X$ ($1 \le k \le d(d+1)/2$) span $\mathfrak{so}(d)$, proved via base-4 Vandermonde rates $\mu_{ij} = 4^i + 4^j$ around an orthogonal-row equilibrium. Consequently, on every short forward interval $[0, \delta)$ there exist $d(d-1)/2$ times at which $B(s) = X^\top[L(0), L(s)]X$ are linearly independent.

---

## 1. The RDPG Principal Bundle and Ambient Holonomy

Let $n > d \ge 2$. Work on the open total space of full-column-rank interior frames
$$
\mathcal{X} := \bigl\{X \in \mathbb{R}^{n \times d} : \operatorname{rank}(X) = d,\ 0 < (XX^\top)_{ij} < 1\ \forall\, i,j \in [n]\bigr\},
$$
with right free $O(d)$-action $X \mapsto XQ$ and quotient base $\mathcal{B} = \mathcal{X}/O(d) = \{P = XX^\top : X \in \mathcal{X}\}$. For $X \in \mathcal{X}$, write $G := X^\top X \succ 0$ and define the Lyapunov operator $S_G(A) := GA + AG$ on skew-symmetric matrices $\mathfrak{so}(d)$. Because $G \succ 0$, $S_G : \mathfrak{so}(d) \to \mathfrak{so}(d)$ is a linear isomorphism (even when $G$ has repeated eigenvalues: in an eigenbasis of $G$ with eigenvalues $\lambda_i > 0$, $(S_G(A))_{ij} = (\lambda_i + \lambda_j)A_{ij}$ with $\lambda_i + \lambda_j > 0$).

The vertical tangent space at $X$ is $\{XA : A \in \mathfrak{so}(d)\}$. Frobenius orthogonality $\operatorname{tr}(H^\top XA) = 0$ for all $A \in \mathfrak{so}(d)$ is equivalent to symmetry of $X^\top H$, i.e., $X^\top H = H^\top X$. The unique orthogonal principal connection one-form $\omega$ and horizontal projection are therefore
$$
\omega_X(V) = S_G^{-1}(X^\top V - V^\top X), \qquad V^h = V - X\omega_X(V).
$$

**Conjecture `conj:full-holonomy` ([arXiv:2603.05703](https://arxiv.org/abs/2603.05703)).** *For $d \ge 3$, $n > d$, and initial conditions with $D(P(0))$ not scalar (non-regular graph), the restricted holonomy group of Laplacian dynamics $\dot{X} = -L(P)X$ is $\mathrm{SO}(d)$.*

**Theorem 1 (Full ambient restricted holonomy everywhere).** *For every $n > d \ge 2$ and every $X \in \mathcal{X}$ (including regular graphs), the curvature two-form with sign convention $R(H, K) = -\omega([H, K])$ is*
$$
R_X(V, W) = 2 S_G^{-1}\bigl((V^h)^\top W^h - (W^h)^\top V^h\bigr),
$$
*and its values at $X$ already span $\mathfrak{so}(d)$. Consequently, the restricted holonomy group of the RDPG bundle at $p = XX^\top$ (and even restricted to loops inside any open neighborhood of $p$ in $\mathcal{B}$) is $\mathrm{Hol}_p^0 = \mathrm{SO}(d)$.*

*Proof.* On horizontal tangent vectors $H, K$ extended as constant vector fields on $\mathcal{X}$, Cartan's structure equation gives $R_X(H, K) = d\omega_X(H, K)$. Differentiating $\omega_X = S_{X^\top X}^{-1}(X^\top dX - dX^\top X)$, terms differentiating $S_{X^\top X}^{-1}$ multiply $(X^\top H - H^\top X) = 0$ or $(X^\top K - K^\top X) = 0$ and vanish, while $d(X^\top dX - dX^\top X)(H, K) = 2(H^\top K - K^\top H)$. Since $R$ is horizontal, $R_X(V, W) = R_X(V^h, W^h)$ gives the displayed formula.

Because $n > d$, $\ker(X^\top)$ is nonzero; pick a unit vector $z \in \ker(X^\top)$. For each $i \in \{1, \dots, d\}$, set $H_i := z e_i^\top \in \mathbb{R}^{n \times d}$. Then $X^\top H_i = 0$, so every $H_i$ is horizontal, and for all $1 \le i < j \le d$,
$$
R_X(H_i, H_j) = 2 S_G^{-1}(e_i e_j^\top - e_j e_i^\top).
$$
Since $\{e_i e_j^\top - e_j e_i^\top : i < j\}$ is a basis of $\mathfrak{so}(d)$ and $S_G^{-1}$ is an isomorphism on $\mathfrak{so}(d)$, the curvature values at $X$ alone span $\mathfrak{so}(d)$. By the Ambrose–Singer theorem, the Lie algebra of $\mathrm{Hol}_p^0$ equals $\mathfrak{so}(d)$. Because $\mathrm{Hol}_p^0$ is a connected Lie subgroup of $O(d)$ with Lie algebra $\mathfrak{so}(d)$, $\mathrm{Hol}_p^0 = \mathrm{SO}(d)$ (and the same holds in any open neighborhood of $p$). $\square$

---

## 2. Trivial Holonomy on Autonomous Laplacian Orbits

Set $L(P) := \operatorname{diag}(P\mathbf{1}) - P$. The Laplacian vector field $F(X) := -L(XX^\top)X$ is $O(d)$-equivariant and horizontal (since $X^\top L(P) X$ is symmetric), projecting to the autonomous base ODE $\dot{P} = -L(P)P - P L(P)$.

**Theorem 2 (Strict trace decay and trivial orbit holonomy).** *Along every interior solution $X(t) \in \mathcal{X}$ of $\dot{X} = -L(P)X$,*
$$
\frac{d}{dt}\operatorname{tr} P(t) = -2\operatorname{tr}\bigl(X^\top L(P) X\bigr) = -2 \sum_{1 \le i < j \le n} P_{ij} \|x_i - x_j\|_2^2 < 0.
$$
*Hence there are no nonconstant closed trajectories; each orbit in $\mathcal{B}$ is a smooth 1-dimensional interval carrying the global horizontal section $P(t) \mapsto X(t)$, and parallel transport around any loop supported on the orbit (even allowing stops and reversals) is the identity.*

*Proof.* The identity $\operatorname{tr}(X^\top L(P) X) = \sum_{i<j} P_{ij}\|x_i - x_j\|_2^2$ is standard for symmetric weights $P_{ij}$. Because $X \in \mathcal{X}$ has $P_{ij} > 0$ for all $i, j$ and $\operatorname{rank}(X) = d \ge 2$, the rows $x_i$ cannot all be equal, so the sum is strictly positive. Thus $\operatorname{tr} P$ is a strictly decreasing smooth coordinate along the orbit, which is diffeomorphic to an interval. The solution curve $t \mapsto X(t)$ is a horizontal section over the entire orbit, so the pullback of $\omega$ vanishes identically and every orbit-supported loop has identity holonomy. $\square$

*(Connection to frozen commutators: freezing two Laplacians $L(t_1), L(t_2)$ and applying both at the **same** frame $X$ yields legitimate ambient horizontal vectors $H_k = -L(t_k)X$ with ambient curvature $R_X(H_1, H_2) = 2 S_G^{-1}(X^\top[L(t_1), L(t_2)]X)$. Independently moving along $H_1, H_2$ leaves the 1-dimensional autonomous orbit and probes ambient 2-planes in $\mathcal{B}$.)*

---

## 3. Non-Regular $S_d$-Symmetric Counterexample and Exact Disk Loop

**Theorem 3 (Non-regular commuting counterexample and explicit ambient loop).** *For every $d \ge 2$, set $n := d + 1$ and define $X \in \mathbb{R}^{(d+1) \times d}$ by*
$$
x_i := \frac{\mathbf{1}_d + e_i}{4d} \quad (1 \le i \le d), \qquad
x_{d+1} := \frac{\mathbf{1}_d}{4d}.
$$
*Then:*
1. *$X \in \mathcal{X}$ is full-rank, strictly interior ($0 < P_{ij} < 1$), and non-regular (the first $d$ node degrees exceed the last by $(d+2)/(4d)^2 > 0$).*
2. *Along the Laplacian flow from $X$, all matrices $L(t)$ commute for all $t \ge 0$, so every frozen projected commutator $X^\top[L(t_1), L(t_2)]X$ vanishes identically. Moreover, the $S_d$-symmetric frames form a 3-dimensional flat horizontal submanifold through $X$.*
3. *Choose unit $z \in \ker(X^\top)$ and orthonormal latent contrasts $u, v \perp \mathbf{1}_d$, and consider the contractible ambient loop based at $p = XX^\top$ given by going radially out to radius $r := 1/(40d)$ along $u$, traversing the circle $Y(\theta) = X + r z(\cos\theta\, u + \sin\theta\, v)^\top$ counterclockwise, and retracing the radius. The entire disk lies in $\mathcal{X}$, and the exact holonomy matrix is*
   $$
   Q = \exp\!\left(-\frac{2\pi r^2}{2g + r^2} K\right) = \exp\!\left(-\frac{2\pi}{201} K\right) \in \mathrm{SO}(d),
   \qquad g := (4d)^{-2},\quad K := uv^\top - vu^\top.
   $$

*Proof.* For (1), all entries of $X$ are positive, each row has squared norm at most $(d+3)/(4d)^2 < 1$, and the top $d \times d$ block $(I_d + \mathbf{1}\mathbf{1}^\top)/(4d)$ is invertible. Since $\sum_{i=1}^{d+1} x_i = \frac{d+2}{4d}\mathbf{1}_d$, the degrees satisfy $(P\mathbf{1})_i - (P\mathbf{1})_{d+1} = (x_i - x_{d+1})^\top \sum_j x_j = (d+2)/(4d)^2 > 0$ for $1 \le i \le d$.

For (2), simultaneous permutation of the first $d$ nodes and the $d$ latent coordinates fixes $X$. By $O(d)$-equivariance and ODE uniqueness, this $S_d$ symmetry is preserved for all $t$. Thus at every $t$, off-diagonal weights among $\{1,\dots,d\}$ share a common value $a(t)$ and weights from $\{1,\dots,d\}$ to $d+1$ share a common value $b(t)$. Hence $L(t)$ acts as a scalar on three fixed orthogonal subspaces: $0$ on $\operatorname{span}(\mathbf{1}_{d+1})$, $(d+1)b(t)$ on $\operatorname{span}((1,\dots,1,-d)^\top)$, and $d a(t) + b(t)$ on $\{w \in \mathbb{R}^{d+1} : w_{d+1} = 0,\, \sum_{i=1}^d w_i = 0\}$. Therefore $[L(t_1), L(t_2)] = 0$ for all $t_1, t_2$. Furthermore, every $S_d$-fixed frame has top block $\alpha I_d + \beta \mathbf{1}\mathbf{1}^\top$ and last row $\gamma \mathbf{1}^\top$; for any tangent velocity $V$ in this 3-dimensional family, $X^\top V$ commutes with all coordinate permutations and is therefore symmetric, making the entire family horizontal (and locally a flat section over its 3-dimensional image in $\mathcal{B}$).

For (3), on $\operatorname{span}(u, v) \subset \mathbf{1}_d^\perp$, the Gram matrix $G = X^\top X = \frac{1}{(4d)^2}(I_d + (d+3)\mathbf{1}_d\mathbf{1}_d^\top)$ acts as the scalar $g = (4d)^{-2}$. On the disk $Y = X + z h^\top$ with $h = au + bv$ and $a^2 + b^2 \le r^2$, we have $Y^\top Y = G + hh^\top \succ 0$, every entry of $Y$ is at least $\frac{1}{4d} - \frac{1}{40d} > 0$, and every row norm is at most $\frac{\sqrt{d+3}}{4d} + r < 1$, so the disk maps into $\mathcal{X}$. Moreover, $Y^\top dY - dY^\top Y = h\,dh^\top - dh\,h^\top = (a\,db - b\,da)K$, and $S_{Y^\top Y}(K) = (G + hh^\top)K + K(G + hh^\top) = (2g + a^2 + b^2)K$ (using $h^\top K = (au+bv)^\top(uv^\top - vu^\top) = av^\top - bu^\top$ so $hh^\top K + K hh^\top = (a^2+b^2)K$). Along the radial segments $a\,db - b\,da = 0$, so $\omega = 0$; along the circle $a^2 + b^2 = r^2$, $\omega_Y(\dot{Y}) = \frac{r^2}{2g + r^2} \dot{\theta}\, K$ is constant in $\theta$. Integrating $\dot{Q} = -\omega_Y(\dot{Y})Q$ over $\theta \in [0, 2\pi]$ gives $Q = \exp(-\frac{2\pi r^2}{2g + r^2} K) = \exp(-\frac{2\pi}{201} K)$ since $2g/r^2 = 2 \cdot 10^2 = 200$. $\square$

---

## 4. Generic Full-Rank Theorem for the Finite-Time Criterion

**Theorem 4 (Generic full rank of projected commutators).** *Fix $n > d \ge 2$, $m := d(d-1)/2$, and $N := d(d+1)/2$. Outside a proper real algebraic subset of initial frames $X \in \mathbb{R}^{n \times d}$ (hence on a Zariski-open, dense, full-Lebesgue-measure subset of $\mathcal{X}$), the jet matrices*
$$
J_k(X) := X^\top \bigl[L(0), L^{(k)}(0)\bigr] X \in \mathfrak{so}(d), \qquad 1 \le k \le N,
$$
*span $\mathfrak{so}(d)$. Consequently, in every sufficiently short forward time interval $[0, \delta)$ there exist $m$ sample times $s_1, \dots, s_m$ for which the matrices $B(s) := X^\top[L(0), L(s)]X$ are linearly independent.*

*Proof.* **Step 1 (Polynomial reduction).** Because $\dot{P} = -LP - PL$ is polynomial in $P = XX^\top$ and $L$ is linear in $P$, every time derivative $L^{(k)}(0)$ and hence every $J_k(X)$ is a polynomial in the entries of $X$. Rank deficiency ($\operatorname{rank}\{J_1(X), \dots, J_N(X)\} < m$) is cut out by the vanishing of all $m \times m$ minors of the $m \times N$ coordinate matrix of $J_1, \dots, J_N$ in $\mathfrak{so}(d)$. To prove that at least one minor is a nonzero polynomial—and thus nonvanishing outside a proper real algebraic subset—it suffices to exhibit a single real perturbation witness $X_\epsilon$ (not required to lie in $\mathcal{X}$) on which the leading $\epsilon$-term of $\{J_k(X_\epsilon)\}$ spans $\mathfrak{so}(d)$.

**Step 2 (Base-4 Vandermonde equilibrium and first variation for $n = d + 1$).** Index rows by $0, 1, \dots, d$. Set $X_* \in \mathbb{R}^{(d+1) \times d}$ to have top $d \times d$ block $\operatorname{diag}(2^0, 2^1, \dots, 2^{d-1})$ and last row $0^\top$. Then $P_* = X_* X_*^\top = \operatorname{diag}(s_0, \dots, s_d)$ with $s_i = 4^i$ for $0 \le i < d$ and $s_d = 0$, so $L_* = L(P_*) = 0$ and $X_*$ is an equilibrium of $\dot{P} = -LP - PL$.

For $0 \le i < j \le d$, set $E_{ij} := (e_i - e_j)(e_i - e_j)^\top$ and prescribe positive first-order off-diagonal weights $w_{ij} := 1 + \mathbf{1}_{j = i+1}$. Define $V \in \mathbb{R}^{(d+1) \times d}$ by $V_{ji} := w_{ij} / 2^i$ for $0 \le i < j \le d$ and $0$ elsewhere, and set $X_\epsilon := X_* + \epsilon V$. Then the first-order perturbation $\delta P(0) = X_* V^\top + V X_*^\top$ has off-diagonal entries $\delta P_{ij}(0) = 2^i V_{ji} = w_{ij}$ for $i < j$ (and $\delta P_{ii}(0) = 0$).

Linearizing $\dot{P} = -LP - PL$ around $P_* = \operatorname{diag}(s_0, \dots, s_d)$ ($L_* = 0$) gives $\delta\dot{P} = -(\delta L) P_* - P_* (\delta L)$. For $i \neq j$, $\delta L_{ij} = -\delta P_{ij}$, so
$$
\delta\dot{P}_{ij} = (s_i + s_j)\,\delta P_{ij} \qquad (i \neq j).
$$
Writing $\mu_{ij} := s_i + s_j$ ($0 \le i < j \le d$), the solution is $\delta P_{ij}(t) = w_{ij} e^{\mu_{ij} t}$, whence
$$
\delta L(t) = \sum_{0 \le i < j \le d} w_{ij} e^{\mu_{ij} t} E_{ij}, \qquad
A_k := \delta L^{(k)}(0) = \sum_{0 \le i < j \le d} w_{ij} \mu_{ij}^k E_{ij} \quad (k \ge 0).
$$
Every rate $\mu_{ij}$ is either $4^i + 4^j$ ($0 \le i < j < d$) or $4^i$ ($j = d$). By uniqueness of base-4 representations, all $N = \binom{d+1}{2}$ rates $\mu_{ij}$ are strictly positive and pairwise distinct. Hence the $N \times N$ matrix $(\mu_{ij}^k)_{(i<j),\, 1 \le k \le N}$ is an invertible Vandermonde matrix (with nonzero row scalings $\mu_{ij}$), and multiplying by the positive diagonal weights $w_{ij}$ preserves invertibility. Because $\{E_{ij} : 0 \le i < j \le d\}$ is a basis of the $N$-dimensional space $\mathcal{S}_0(W)$ of symmetric $(d+1) \times (d+1)$ matrices annihilating $\mathbf{1}_{d+1}$ (each has independent off-diagonal entry $-1$ and zero row sums), the matrices $A_1, \dots, A_N$ span all of $\mathcal{S}_0(W)$.

**Step 3 (Leading commutators and surjectivity onto $\mathfrak{so}(d)$).** Restrict to $W := \mathbf{1}_{d+1}^\perp \subset \mathbb{R}^{d+1}$. Since $w_{ij} = 1 + \mathbf{1}_{j=i+1}$, $A_0|_W = (d+1)I_W + L_{\mathrm{path}}|_W$, where $L_{\mathrm{path}}$ is the Laplacian of the path graph on $\{0, \dots, d\}$. As a real symmetric tridiagonal matrix with strictly nonzero subdiagonal entries $-1$, $L_{\mathrm{path}}$ has simple spectrum (any eigenvector is uniquely determined up to scale by its first coordinate via the three-term recurrence), so $A_0|_W$ has $d$ distinct eigenvalues. In an orthonormal eigenbasis of $A_0|_W$, the commutator map $B \mapsto [A_0|_W, B]$ from symmetric operators on $W$ to skew-symmetric operators $\mathfrak{so}(W)$ multiplies the $(a,b)$-entry by $(\lambda_a - \lambda_b) \neq 0$ for all $a \neq b$, and is therefore **surjective** onto $\mathfrak{so}(W)$.

Let $T := \operatorname{proj}_W X_* : \mathbb{R}^d \to W$. Since the last entry of $X_* v$ is $0$, $X_* v \in \operatorname{span}(\mathbf{1}_{d+1})$ forces $X_* v = 0$ and thus $v = 0$; hence $T$ is a linear isomorphism, and congruence $K \mapsto T^\top K T$ maps $\mathfrak{so}(W)$ isomorphically onto $\mathfrak{so}(d)$. Because $L(0) = \epsilon A_0 + O(\epsilon^2)$ and $L^{(k)}(0) = \epsilon A_k + O(\epsilon^2)$ (by commuting initial-state and time derivatives of the polynomial ODE),
$$
J_k(X_\epsilon) = \epsilon^2 X_*^\top [A_0, A_k] X_* + O(\epsilon^3) = \epsilon^2 T^\top \bigl[A_0|_W, A_k|_W\bigr] T + O(\epsilon^3).
$$
As $A_1, \dots, A_N$ span $\mathcal{S}_0(W)$, the leading coefficients $T^\top[A_0|_W, A_k|_W]T$ span $\mathfrak{so}(d)$. Selecting $m$ indices with independent leading terms produces an $m \times m$ minor with nonzero $\epsilon^{2m}$ coefficient, proving polynomial nonvanishing for $n = d + 1$.

**Step 4 (Extension to $n > d + 1$ and short forward intervals).** For any $n > d + 1$, append $n - (d+1)$ zero rows to $X_\epsilon$; zero rows remain identically zero under $\dot{P} = -LP - PL$, preserving all $J_k$ identically and proving polynomial nonvanishing on $\mathbb{R}^{n \times d}$. Finally, at any frame $X \in \mathcal{X}$ outside the resulting proper algebraic set, if $\{B(s) : s \in [0, \delta)\}$ lay in a proper subspace of $\mathfrak{so}(d)$, some nonzero linear functional $\phi \in \mathfrak{so}(d)^*$ would satisfy $\phi(B(s)) \equiv 0$ on $[0, \delta)$, forcing $0 = \frac{d^k}{ds^k}\big|_{s=0} \phi(B(s)) = \phi(J_k(X))$ for all $k \ge 1$, contradicting $\operatorname{span}\{J_1(X), \dots, J_N(X)\} = \mathfrak{so}(d)$. $\square$

---

## 5. Verification and Reproduction

Run from the repository root:

```sh
.venv/bin/python -B solutions/rdpg_latent_holonomy/explore.py
```

The script uses exact rational arithmetic (SymPy) for the connection, curvature, iterated Lie brackets, and time derivatives $L^{(k)}(0)$ via the product rule on $\dot{P} = -LP - PL$, and integrates the full matrix horizontal-lift ODE (`DOP853`, `rtol=1e-12, atol=1e-14`) around the explicit contractible disk loop at the $S_d$-symmetric counterexample. All assertions passed:

| $d$ | Ambient curvature rank | Adjacent-pair Lie dimensions | Frozen jet rank (symmetric) | Frozen jet rank (perturbed) | Disk loop angle $2\pi/201$ | Transport error |
|---:|---:|:---|---:|---:|---:|---:|
| 3 | 3 | 2 $\to$ 3 | 0 | 3 | `0.031259628394` | `2.95e-16` |
| 4 | 6 | 3 $\to$ 6 | 0 | 6 | `0.031259628394` | `5.32e-16` |
| 5 | 10 | 4 $\to$ 10 | 0 | 10 | `0.031259628394` | `7.31e-16` |

All integrated endpoint determinants rounded to `1.000000000000`, matching $\exp(-\frac{2\pi}{201}K) \in \mathrm{SO}(d)$ to within $8 \times 10^{-16}$ in Frobenius norm at the exact non-regular frames where all frozen Laplacian commutators vanish.
