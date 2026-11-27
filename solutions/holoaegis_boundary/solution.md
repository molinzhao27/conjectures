# Refutation of Reference-Noise Boundary Stability in HoloAegis

**Target paper:** Tak Ho Alex Li, Kaijie Liu, Lik-Hang Lee, Kin Chung Ho, Ping Shum, and Michael K. Ng, *HoloAegis: Frozen Representation, Topological Inference — Minimally Parametric Safety Manifolds and Their Capability Boundaries for LLM Guardrails*, [arXiv:2608.08485](https://arxiv.org/abs/2608.08485) (v2, September 2026), Section 3.4, Conjecture `conj:main` (Reference-Noise Boundary Stability; v2 restatement).  
**Companion files:** [`solutions/holoaegis_boundary/explore.py`](explore.py), [`solutions/holoaegis_boundary/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `conj:main`: FALSE** at the paper's exact default hyperparameters ($N = 30{,}000$, $K \in \{30, 100, 300\}$, $k = 10$, $\tau = 0.05$, unit embeddings in $\mathbb{R}^{384}$, unit queries $5.7^\circ$ apart) across *all* standard realizations of the unspecified noise law and centroid scoring convention (both raw and unit-normalized centroids, always with cosine neighbor selection and globally optimal $K$-means partitions).
- **Exact range $[0, \infty)$ under independent spherical noise (Theorem 1):** At fixed $N = 30{,}000$, $K \in \{30, 100, 300\}$, $1 \le k \le K$, and $\tau > 0$, every finite nonnegative real $r \in [0, \infty)$ is realized as $\mathbb{E} S(A_K) / \mathbb{E} S(X_N)$ with $\mathbb{E} S(X_N) > 0$, under both raw and unit-normalized centroid scoring, with nonconstant independent spherical perturbations and a unique global $K$-means optimum on every noise realization. The same family also realizes $\mathbb{E} S(A_K) > 0$ with $\mathbb{E} S(X_N) = 0$.
- **Arbitrarily large ratio under ordinary i.i.d. bootstrap (Theorem 2):** Under uniform i.i.d. sampling with replacement from one balanced 300-site unit corpus ($N = 30{,}000$, $K = 300$, $k = 10$, $\tau = 0.05$), the exact expectation ratio exceeds $1.36 \times 10^{16}$.
- **No uniform approach to $1$ as $K / N \to 1$ (Theorem 3):** For every $N \ge k + 1$ with $K = N - 1$ (a single merger) in fixed dimension $d = 4$, the shift ratio diverges to $+\infty$ as a single parameter $t \uparrow 0$, keeping the ratio above any prescribed finite bound along $K / N \to 1$.

---

## 1. Setup and Conventions

Following Li et al. ([arXiv:2608.08485](https://arxiv.org/abs/2608.08485), Section 3.2 and 3.4), for a query $q \in \mathbb{S}^{d-1}$ and an indexed reference bank $R = \{r_1, \dots, r_m\}$ (where repeated observations count separately), the top-$k$ Gibbs–Boltzmann potential of one tower is
$$
U(q; R) := 1 - \tau \log\!\left(\frac{1}{k} \sum_{i \in I_k(q; R)} \exp\!\left(\frac{q \cdot r_i}{\tau}\right)\right)
= -\tau \log\!\left(\frac{1}{k} \sum_{i \in I_k(q; R)} \exp\!\left(\frac{q \cdot r_i - 1}{\tau}\right)\right),
\tag{1}
$$
where $I_k(q; R)$ selects the $k$ nearest references to $q$ by cosine similarity. For a query perturbation $\delta$ (with $q, q + \delta \in \mathbb{S}^{d-1}$), the one-tower boundary shift is
$$
S(R) := \mathrm{Shift}(q, \delta; R) := \bigl|U(q + \delta; R) - U(q; R)\bigr|.
$$
Conjecture `conj:main` compares $\mathbb{E}_\eta[S(A_K)]$ with $\mathbb{E}_\eta[S(X_N)]$, where $X_N$ is a realization of $N$ noisy samples and $A_K$ is the set of $K$-means centroids of that same realization, asserting $\mathbb{E}_\eta[S(A_K)] < \mathbb{E}_\eta[S(X_N)]$ with ratio $\approx 0.44 + 0.23\sqrt{k/K}$ on $K/N \in [10^{-3}, 10^{-2}]$ ($k = 10$, $\tau = 0.05$, $N = 30{,}000$) and approaching $1$ as $K \to N$.

Because the conjecture leaves the noise law $\eta$, sampling protocol, and centroid normalization unspecified (the exponent writes $q \cdot r_i$ while $I_k$ ranks by cosine similarity), we prove all counterexamples for *both* raw arithmetic centroids (cosine-ranked) and unit-normalized centroids, and certify global optimality of every $K$-means partition under squared Euclidean distortion with $K$ nonempty clusters.

We use one elementary monotonicity lemma throughout:

**Lemma 1 (Log-sum-exp shift inequality).** *For $\tau > 0$ and vectors $x, y \in \mathbb{R}^k$ with $u \le y_i - x_i \le v$ for all $i = 1, \dots, k$,*
$$
u \le \tau\log\!\left(\frac{1}{k}\sum_{i=1}^k e^{y_i/\tau}\right) - \tau\log\!\left(\frac{1}{k}\sum_{i=1}^k e^{x_i/\tau}\right) \le v.
$$
*In particular, for any bank $R \subset \mathbb{S}^{d-1}$ of unit vectors and any query shift $\delta$, $S(R) \le \|\delta\|_2$.*

*Proof.* Factoring $e^{u/\tau} e^{x_i/\tau} \le e^{y_i/\tau} \le e^{v/\tau} e^{x_i/\tau}$ and summing over $i$ yields the first claim. For unit banks, $|(q + \delta)\cdot r_i - q\cdot r_i| \le \|\delta\|_2$ for all $i$, so the sorted order statistics change by at most $\|\delta\|_2$ coordinatewise (by Weyl/order-statistic $L^\infty$-contraction), and the first claim gives $|U(q+\delta; R) - U(q; R)| \le \|\delta\|_2$. $\square$

---

## 2. Exact Ratio Range $[0, \infty)$ Under Independent Spherical Noise

**Theorem 1 (Surjectivity onto $[0, \infty)$ at fixed defaults).** *Fix $N = 30{,}000$, $K \in \{30, 100, 300\}$, $1 \le k \le K$, $\tau > 0$, $d = 384$, and unit queries $5.7^\circ$ apart. Under either raw or unit-normalized centroid scoring, every finite $r \ge 0$ is realized as $\mathbb{E} S(A_K) / \mathbb{E} S(X_N)$ with $\mathbb{E} S(X_N) > 0$, using nonconstant independent spherical perturbations, an almost surely distinct sample bank, equal cluster sizes, and a unique global $K$-means partition on every noise realization. The same family also realizes $\mathbb{E} S(A_K) > 0$ with $\mathbb{E} S(X_N) = 0$.*

*Proof.* Set $h := 2.85^\circ$ (so the angle between $q_\pm := \cos h\, e_1 \pm \sin h\, e_2$ is $5.7^\circ$), $q := q_-$, $\delta := q_+ - q_-$. Put $a := 0.600$, $b := 0.599$, $s := 0.001$, and vary $t \in [-s, 0]$. Let $n := N / K \in \{1000, 300, 100\}$ (always even). Writing $(x, y) := x e_1 + y e_2$, for each cluster $j = 1, \dots, K$ take $n/2$ copies of each of two unit sites
$$
v_j := (a, t) + r_v\, e_{j+2}, \qquad w_j := (b, s) + r_w\, e_{j+2},
$$
where $r_v := \sqrt{1 - a^2 - t^2}$ and $r_w := \sqrt{1 - b^2 - s^2}$. To introduce independent spherical noise, replace $e_{j+2}$ in each individual observation by
$$
\cos\phi\, e_{j+2} + \sin\phi\, e_{K+3}, \qquad \phi \sim \mathrm{Unif}[-0.001, 0.001] \text{ radians},
$$
independently across all $N$ observations. This embeds in $\mathbb{R}^{K+3} \subseteq \mathbb{R}^{384}$ (since $K + 3 \le 303 \le 384$), keeps every sample on $\mathbb{S}^{d-1}$, and makes all $N$ observations distinct almost surely. (The identical proof holds for $\phi \equiv 0$.)

**Global $K$-means uniqueness certificate.** Since $r_v, r_w \ge \sqrt{1 - 0.6^2 - 0.001^2} > 0.7999$,
$$
|r_w - r_v| = \frac{|(a^2 - b^2) + (t^2 - s^2)|}{r_w + r_v} \le \frac{0.001199 + 0.000001}{1.5998} < 0.000751 < 0.001,
$$
hence $\|w_j - v_j\|_2 < \sqrt{0.001^2 + 0.002^2 + 0.001^2} < 0.00245 < 0.003$. Angular perturbation moves a vector by at most $r_w |\phi| < 0.001$, so every observation in cluster $j$ lies inside $B(v_j, 0.004)$. For $j \neq \ell$, $\|v_j - v_\ell\|_2 = \sqrt{2}\, r_v > 1.131$, so the balls $B(v_j, 0.25)$ are pairwise disjoint. Using $\{v_j\}_{j=1}^K$ as candidate centers gives total distortion at most $N(0.004)^2$. Conversely, any $K$-center set omitting a center in $B(v_j, 0.25)$ incurs distortion at least $n(0.25 - 0.004)^2 = 0.060516\, n > 300(0.004)^2 n = 0.0048\, n \ge N(0.004)^2$ in cluster $j$ alone (since $K \le 300$). Therefore every global minimizer places exactly one center in each $B(v_j, 0.25)$. Every sample in cluster $j$ then lies within $0.254$ of its local center and at distance $> 1.131 - 0.25 - 0.004 = 0.877$ from every other center. Thus optimal assignments are forced to the $K$ clusters, and the unique global $K$-means bank (up to relabeling) is the set of exact arithmetic cluster means for *every* noise realization.

**Full-bank shift.** Because $(a - b)\cos h = 0.001\cos 2.85^\circ > 0.000998 > 0.002\sin 2.85^\circ \ge |s - t|\sin h$, every type-$v$ sample has strictly larger dot product with both $q_-$ and $q_+$ than every type-$w$ sample. Since $N/2 \ge K \ge k$, all $k$ selected samples are of type $v$ with identical dot products $a\cos h \pm t\sin h$, giving identically
$$
S(X_N) = 2|t|\sin h.
\tag{2}
$$

**Anchor shift.** Every cluster mean $c_j$ has identical first two coordinates $C := (a + b)/2 = 0.5995$ and $D := (t + s)/2 \ge 0$, and norm $L_j := \|c_j\|_2 \in [0.996, 1]$ (as $\|c_j - v_j\|_2 \le 0.004$ and $\|v_j\|_2 = 1$).
- *Raw centroids:* All $K$ anchors have identical dot products $C\cos h \pm D\sin h$ at $q_\pm$, so $S(A_K) = 2D\sin h$ identically, independent of noise and top-$k$ selection. (At $t = -s/10$, the ratio is exactly $D/|t| = 4.5$.)
- *Unit-normalized centroids:* At $q_\pm$, the scores are $(C\cos h \pm D\sin h)/L_j$. Both numerators are strictly positive (since $C\cos h > 0.598 > D\sin h$), so both queries select the same $k$ smallest norms $L_{(1)} \le \cdots \le L_{(k)}$ (even at ties, the multiset of selected norms is identical). Each selected score rises by $2D\sin h / L_{(i)} \in [2D\sin h,\, 2D\sin h / 0.996]$, so Lemma 1 yields
  $$
  2D\sin h \le S(A_K) \le \frac{2D\sin h}{0.996}.
  \tag{3}
  $$

**Attainment of every ratio $r \in [0, \infty)$.** Couple the family across $t \in [-s, 0]$ using the same angles $\phi$. For each realization, the norms $L_j(t)$ and hence their order statistics $L_{(1)}(t), \dots, L_{(k)}(t)$ vary continuously in $t$, so $S(A_K)(t)$ is continuous in $t$ and uniformly bounded by $2s\sin h / 0.996$. By dominated convergence, $t \mapsto \mathbb{E} S(A_K)(t)$ is continuous on $[-s, 0]$. Under either convention, the ratio $R(t) := \mathbb{E} S(A_K)(t) / (2|t|\sin h)$ is continuous on $[-s, 0)$, satisfies $R(-s) = 0$ (since $D = 0$), and diverges as $t \uparrow 0$ (since $D \to s/2 > 0$ while $|t| \to 0$). By the intermediate value theorem, $R([-s, 0)) = [0, \infty)$. At $t = 0$, $\mathbb{E} S(A_K) \ge s\sin h > 0$ while $S(X_N) \equiv 0$. $\square$

---

## 3. Ordinary Paired i.i.d. Bootstrap From One Corpus

**Theorem 2 (Extreme ratio under ordinary i.i.d. bootstrap).** *At the paper's defaults ($N = 30{,}000$, $K = 300$, $k = 10$, $\tau = 0.05$, $d = 384$, queries $q_\pm$ at angle $5.7^\circ$), let the corpus consist of $100$ copies of each of $300$ distinct unit sites: site $0$ is $e_1$, and sites $j = 1, \dots, 299$ are $(0.99, 0.04) + \sqrt{1 - 0.99^2 - 0.04^2}\, e_{j+2} \in \mathbb{R}^{302} \subseteq \mathbb{R}^{384}$. Drawing $X_N$ as $N$ i.i.d. samples uniformly with replacement from this corpus yields*
$$
\frac{\mathbb{E} S(A_K)}{\mathbb{E} S(X_N)} > 1.36 \times 10^{16} \qquad (\mathbb{E} S(X_N) > 0),
\tag{4}
$$
*for both raw and unit-normalized centroids (which coincide on the high-probability event).*

*Proof.* First note $1 - 0.99^2 - 0.04^2 = 0.0183 > 0$, so all $300$ sites are valid unit vectors. Let $E_{\mathrm{all}}$ be the event that all $300$ sites appear at least once in $X_N$. On $E_{\mathrm{all}}$, the unique zero-cost $K$-means bank $A_K$ consists of the $300$ distinct unit sites themselves. Because $0.01\cos h > 0.00998 > 0.04\sin h \approx 0.00199$, site $0$ ranks first at both $q_+$ and $q_-$, while the next $9$ selected sites all have coordinates $(0.99, 0.04)$ and identical dot products $0.99\cos h \pm 0.04\sin h$. Factoring out $\exp(\cos h / \tau)$ gives the exact shift on $E_{\mathrm{all}}$:
$$
S(A_K) = L_0 := \tau \log\!\left(\frac{1 + 9\exp\bigl((-0.01\cos h + 0.04\sin h)/\tau\bigr)}{1 + 9\exp\bigl((-0.01\cos h - 0.04\sin h)/\tau\bigr)}\right) = 0.003502414473\ldots
$$
By a union bound, $\Pr(E_{\mathrm{all}}^c) \le 300(1 - 1/300)^{30000} \le 300 e^{-100}$, so $\mathbb{E} S(A_K) \ge (1 - 300 e^{-100}) L_0$.

Now let $M \sim \mathrm{Binomial}(30000, 1/300)$ (mean $\mu = 100$) be the multiplicity of site $0$ in $X_N$. Whenever $M \ge 10$, all $k = 10$ nearest samples in $X_N$ at both $q_+$ and $q_-$ are copies of $e_1$, whose dot product $\cos h$ is identical at $q_\pm$, so $S(X_N) = 0$. By the multiplicative Chernoff lower-tail bound, $\Pr(M < 10) \le e^{-100(1 - 0.1)^2/2} = e^{-40.5}$ (and directly via Markov's inequality on $e^{-u M}$ at $u = \log 10$, $\Pr(M \le 10) \le e^{10u}(1 + (e^{-u}-1)/300)^{30000} \le \exp(10\log 10 - 90) < e^{-66.97} < e^{-40.5}$). Since $S(X_N) \le \|\delta\|_2 = 2\sin h$ pointwise by Lemma 1, we obtain
$$
0 < \mathbb{E} S(X_N) \le 2\sin h\, e^{-40.5},
$$
where strict positivity follows from $\Pr(M = 0) > 0$ (on $\{M = 0\}$, every sample shifts by $0.08\sin h > 0$). Dividing the two bounds yields
$$
\frac{\mathbb{E} S(A_K)}{\mathbb{E} S(X_N)} \ge \frac{(1 - 300 e^{-100}) L_0}{2\sin h\, e^{-40.5}} > 1.367 \times 10^{16}. \qquad \square
$$

---

## 4. Non-Uniform Behavior as $K / N \to 1$

At $K = N$, nonempty $K$-means clusters are singletons, so $A_N = X_N$ and the shift ratio equals $1$ whenever defined. However, there is no uniform approach to $1$ as $K/N \to 1$:

**Theorem 3 (Single-merger divergence at $K = N - 1$).** *Fix $k = 10$, $\tau = 0.05$, queries $q_\pm$ at angle $5.7^\circ$, and ambient dimension $d = 4$. For every $N \ge k + 1$ and $K = N - 1$, there exists a deterministic unit-sphere family $X_N(t) \subset \mathbb{S}^3$ ($t \in [-s, 0)$) with unique zero-cost $K$-means bank $A_{N-1}(t)$ such that*
$$
\lim_{t \uparrow 0} \frac{S(A_{N-1}(t))}{S(X_N(t))} = +\infty
$$
*uniformly in $N$. In particular, fixing a sufficiently small $t < 0$ keeps the ratio above any prescribed finite constant along the entire sequence $K/N = (N-1)/N \to 1$.*

*Proof.* With $a = 0.600$, $b = 0.599$, $s = 0.001$, and $t \in [-s, 0]$, choose $k - 1$ distinct unit sites with first two coordinates $(a, t)$ and distinct angles on the transverse circle in $\operatorname{span}(e_3, e_4)$, duplicate one of them once (giving $k$ observations of type $(a, t)$), add one unit site with coordinates $(b, s)$, and add $N - k - 1$ distinct unit fillers with coordinates $(-a, 0)$ on their transverse circle. There are exactly $N - 1 = K$ distinct support sites, so the unique zero-cost $K$-means bank $A_{N-1}$ consists of those $N - 1$ distinct unit sites.

In $X_N$, all $k$ nearest observations at both $q_\pm$ are the $k$ observations of type $(a, t)$, so $S(X_N) = 2|t|\sin h$. In $A_{N-1}$, only $k - 1$ anchors have type $(a, t)$, so the $k$-th selected anchor at both $q_\pm$ is the unique $(b, s)$ site (as $b\cos h - s\sin h > 0 > -a\cos h$). At $t = 0$, the $k - 1$ sites of type $(a, 0)$ have constant dot product $a\cos h$ while the $(b, s)$ site's dot product strictly increases by $2s\sin h$, so $S(A_{N-1}(0)) > 0$ (independent of $N$). By continuity in $t$, $S(A_{N-1}(t)) / (2|t|\sin h) \to +\infty$ as $t \uparrow 0$, identically for all $N \ge k + 1$. $\square$

---

## 5. Reproducible Numerical Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/holoaegis_boundary/explore.py
```

The script [`explore.py`](explore.py) evaluates the exact top-$k$ potential (1) under both raw (cosine-ranked) and unit-normalized centroid scoring:
- **Paired family (72 configurations):** Tests $K \in \{30, 100, 300\}$, $k \in \{1, 10, K\}$, $t \in \{-s, -s/10, -s/10000, 0\}$, both noiseless and independently perturbed ($\phi \sim \mathrm{Unif}[-0.001, 0.001]$), certifying the ball radius $< 0.004$, the $K$-means uniqueness inequality, exact $S(X_N) = 2|t|\sin h$, exact raw $S(A_K) = (t + s)\sin h$ (ratio $4.5$ at $t = -s/10$), and the unit-normalized bracket (3).
- **Ordinary i.i.d. bootstrap:** Verifies $L_0 = 0.003502414473$, checks 128 seeded multinomial draws (every draw has all 300 sites present, $M \ge 10$, and $S(X_N) = 0$), and verifies the proved expectation ratio $> 1.367 \times 10^{16}$.
- **Single-merger family ($K = N - 1$):** Verifies $N \in \{100, 1000, 30{,}000\}$ at $t = 0$ (positive anchor shift, zero full-bank shift) and $t = -10^{-7}$ (ratio $> 900$).

All assertions pass with exit status `0`.

---

## References

- T. H. A. Li, K. Liu, L.-H. Lee, K. C. Ho, P. Shum, and M. K. Ng, *HoloAegis: Frozen Representation, Topological Inference — Minimally Parametric Safety Manifolds and Their Capability Boundaries for LLM Guardrails*, [arXiv:2608.08485](https://arxiv.org/abs/2608.08485) (v2), September 2026.
