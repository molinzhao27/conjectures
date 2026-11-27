# Resolution of the Golden-Ratio Barrier Conjecture for Timely Language Generation

**Target paper:** Atul Ganju, Travis McVoy, Shaddin Dughmi, and Shang-Hua Teng, *A Theory of Time-Sensitive Language Generation: Sparse Hallucination Beats Mode Collapse*, [arXiv:2605.11302](https://arxiv.org/abs/2605.11302) (v2), Section 5 (Discussion and Open Problems), Conjecture 1 (*The Golden-Ratio Barrier for Timely Language Generation*).  
**Companion files:** [`solutions/timely_lm_hallucination/explore.py`](explore.py), [`solutions/timely_lm_hallucination/proof.pdf`](proof.pdf).

---

## Verdict

The conjecture mentions the time of first speculation from the target language while motivating a guarantee for sustained meaningful generation, without fixing a formal definition of language-level timeliness, probability quantifier, or density threshold. Resolving both natural formalizations settles the conjecture completely:

- **Reading 1 (First actual speculation from target $K = L^s$): DISPROVED** (beats $\varphi$ down to joint polynomial infimum $1$; Theorem 2). A single online generator using $(s, k)$ product-order probe scheduling achieves element-wise timely lower density $\ge 1/2$ almost surely, pointwise vanishing hallucination (both in expectation and on every run), string deadline $D(i) = \lceil i \log^3(2i) \rceil$ (and simultaneously every $D(i) = \lceil i^{1+\alpha} \rceil$, $\alpha > 0$), and first actual probe of hypothesis $s$ by $O(s \log^2(2s))$ rounds on every run. Pure powers $3/2$ for both bounds already beat $\varphi$; the joint polynomial infimum is $1$, and the pure-power endpoint $(O(s), O(i))$ is impossible.
- **Reading 2 (Uniform sustained positive density in prefix ranks): PROVED — Golden-Ratio Theorem** (Theorem 3). If $D(i) = \lceil i^p \rceil$ and a generator guarantees uniform positive expected timely density $\mathbb{E}[\mu_i] \ge \rho > 0$ for all prefix ranks $i \ge C s^q$, then $p \ge 1 + 1/q$ is necessary (even with unrestricted hallucination). For the balanced objective $p = q$, the infimum is **exactly** $\varphi = (1 + \sqrt{5})/2$:
  - *Achievability with polylog slack:* At the critical string deadline $D(i) = \lceil i^\varphi \rceil$, our same online generator achieves $\mathbb{E}[\mu_i] \ge 1/2 - o(1)$ and exponential tail $\Pr(\mu_i < 1/2 - \epsilon) \le \exp(-c_\epsilon i)$ uniformly over all sufficiently large $i \ge s^\varphi \log^4(2s)$ (so every pure balanced power $d > \varphi$ is achievable).
  - *Refutation of the pure-power endpoint:* Under pointwise vanishing hallucination, even a small constant positive expected density $\rho > 0$ is impossible uniformly at the exact pure-power boundary $p = 1 + 1/q$ (in particular $D(i) = O(i^\varphi)$, $i \ge C s^\varphi$).
- **Quantifier catch (`introduction.tex` vs. `possibility.tex`, Proposition 1):** The introduction defines vanishing hallucination pointwise (supremum over instances *outside* $\lim_{t \to \infty}$), whereas a display in the positive theorem places the supremum *inside* a decay bound at time $t$. Uniform decay across instances is impossible in the total-enumeration game (two infinite targets sharing one symbol force worst-instance expected hallucination fraction $\ge 1/2$ at every prescribed round $t$). Thus pointwise vanishing hallucination is essential.
- **Scope notes:** The full unbalanced $(p, q)$ achievability region (pairs with $p > 1 + 1/q$ and $p \neq q$) remains open here. The companion simulation [`explore.py`](explore.py) tests the unbounded-interval idealization (directly taking the earliest fresh string of the chosen viable intersection rather than truncating at $M = (t+4)^2$).

---

## 1. Game, Clocks, and the Quantifier Catch

A countable family of infinite languages $L^1, L^2, \dots$ over a countably infinite ordered universe $\mathcal{U}$ is known in advance. Before the generator's random bits are drawn, an adversary fixes a target index $s \ge 1$ ($K := L^s$) and a total enumeration $E = (e_1, e_2, \dots)$ of $K$ (repetitions allowed). At each round $t \ge 1$, the generator observes $e_t$ and must output a fresh string $o_t \in \mathcal{U}$ outside $\{e_1, \dots, e_t\} \cup \{o_1, \dots, o_{t-1}\}$.

Write $K_i := \{k_1, \dots, k_i\}$ for the first $i$ strings of $K$ in universe preference order. Given a nondecreasing string deadline $D : \mathbb{N}_+ \to \mathbb{N}_+$, a string $k_j \in K$ earns timely credit only if the generator outputs $k_j$ by round $D(j)$ (strings supplied only by the adversary earn zero credit). Let $\mu_i$ be the fraction of strings in $K_i$ earning timely credit. An output $o_t \notin K$ is a **hallucination**. Following the introduction of [arXiv:2605.11302](https://arxiv.org/abs/2605.11302), *vanishing hallucination* is defined **pointwise**: for every fixed instance $(s, E)$, the expected hallucination fraction through round $t$ tends to $0$ as $t \to \infty$ (whereas *eventual consistency* allows only finitely many hallucinations on $(s, E)$).

**Proposition 1 (Impossibility of uniform-over-instances hallucination decay).** *In the total-enumeration game, no generator can satisfy $\sup_{(s, E)} \mathbb{E}[\text{hallucination fraction through } t] \to 0$ as $t \to \infty$, even on a two-language family.*

*Proof.* Let $L^1, L^2$ be two infinite languages sharing exactly one common string $c$, and fix any horizon $t \ge 1$. For each $s \in \{1, 2\}$, choose a total enumeration $E^{(s)}$ of $L^s$ beginning with $t$ repetitions of $c$. Both instances induce the identical transcript through round $t$. After round $1$, $c$ has been observed, so every legal output $o_1, \dots, o_t$ differs from $c$ and therefore lies outside at least one of $L^1, L^2$. Summing hallucination counts over $s = 1, 2$ gives at least $t$ hallucinations across the two instances, so $\max_{s \in \{1, 2\}} \mathbb{E}[\text{hallucination fraction through } t] \ge 1/2$ for every $t \ge 1$. $\square$

**Clarifying clocks and probes.** A **probe of $s$** designates hypothesis index $s$ and outputs the earliest unused string of $L^s$ (as in the speculative step of `possibility.tex`). Its first occurrence is a round number. In the paper's `possibility.tex` construction, $m$ indexes epochs ending at round $D(m)$, with constraint $C(m) = o(D(m) - D(m-1)) = o(m^\alpha)$ for $D(m) = \lceil m^{1+\alpha} \rceil$. Consequently, at first eligibility, $m / s^{1/\alpha} \to \infty$ as $s \to \infty$, and converting epoch index $m$ to round number $D(m)$ raises the near-extremal exponent from $1/\alpha$ to $(1+\alpha)/\alpha$. Moreover, eligibility in an epoch does not equate to an immediate probe or to sustained timely density.

---

## 2. A Single Online Generator

We construct one randomized online generator with two alternating roles: **probe rounds** (chosen randomly inside growing deterministic blocks) and **ordinary rounds**.

### 2.1 Product-order probe allocation
Order all pairs $(s, k) \in \mathbb{N}_+^2$ by increasing product $sk$, breaking ties by increasing $s$. Assign the $n$-th pair $(s_n, k_n)$ a contiguous block of length $\ell_n := 1 + \lfloor \log_2 n \rfloor$ rounds. Independently for each block $n$, choose one uniformly random slot within that block to **probe $s_n$** (outputting the earliest unused string of $L^{s_n}$); this executes the $k_n$-th probe of $s_n$.

Define
$$
A(m) := \sum_{d=1}^m \left\lfloor \frac{m}{d} \right\rfloor,
\qquad
B(n) := \sum_{b=1}^n \ell_b,
\qquad
T_s(i) := B\bigl(A(si)\bigr).
$$
Since $A(m) \le m(1 + \log m)$ and $B(n) \le n(1 + \log_2 n)$, we have
$$
T_s(i) = O\bigl(si \log^2(2si)\bigr).
$$
Every pair $(s, k)$ with $1 \le k \le i$ has product $sk \le si$ and therefore appears among the first $A(si)$ blocks. Hence by the deterministic round $T_s(i)$, at least $i$ probes of $s$ have executed on **every** realization of the random slots. Because each probe of $s$ outputs the earliest unused string of $K = L^s$, while any string of $K_i$ remains unused, a probe of $s$ consumes one; thus **every string of $K_i$ has been observed or output by round $T_s(i)$**. Taking $i = 1$ gives the deterministic first-probe bound $T_s(1) = O(s \log^2(2s))$. Because $\ell_n \to \infty$, the total number of probe rounds through time $t$ is $o(t)$ deterministically on every run.

### 2.2 Ordinary rounds and pointwise vanishing hallucination
Let $r(t) := \lfloor t^{1/\varphi} \rfloor$ be a nondecreasing integer cutoff. Call a language $L^j$ *viable* at round $t$ if it contains all observations $\{e_1, \dots, e_t\}$ seen so far. For $0 \le j \le r(t)$, let $V_j$ be the intersection of the viable languages among $L^1, \dots, L^j$ (with $V_0 := \mathcal{U}$); these sets are nested downward in $j$. Set $M := (t + 4)^2$, and let $I_t$ be the finite universe prefix ending at the largest of the $M$-th strings of $\mathcal{U}$ and of $L^1, \dots, L^{r(t)}$. On an ordinary round $t$, pick the largest $j \le r(t)$ for which $V_j \cap I_t$ contains a fresh (unused) string, and output its earliest such string. Since $|I_t| \ge M > 2t$ while at most $2t - 1$ strings are used before the generator moves, $j = 0$ is always feasible, and each round uses finite computation given membership queries.

Fix any instance $(s, E)$ with $K = L^s$. Every language among $L^1, \dots, L^s$ not containing $K$ is permanently eliminated once a witnessing string of $K$ is observed in $E$. After finite instance-dependent time, $r(t) \ge s$ and $V_s = K$; since $I_t$ contains at least $M > 2t - 1$ strings of $K$, the chosen index satisfies $j \ge s$, so every subsequent ordinary output belongs to $K$. Thus total hallucinations through round $t$ are bounded by a finite instance-dependent constant plus the number of probe blocks started through $t$, which is $o(t)$. This proves **pointwise vanishing hallucination** both in expectation and on every run.

---

## 3. Turning Joint Coverage into Generator Credit

**Lemma 1 (Pathwise charging and exponential tail).** *Fix any family, any instance $(s, E)$, and any rank $i \ge 1$, and set $T := T_s(i)$. Let $G$ be the number of strings in $K_i$ output by the generator by round $T$, and let $Z_i$ count strings in $K_i$ whose first occurrence in $E$ falls in a probe slot. Then, pathwise,*
$$
2G \ge i - O\bigl(s^\varphi + \sqrt{i} + 1\bigr) - 2r(T) - Z_i,
$$
*and for any deadline function $D$, letting $b_D(T) := |\{j \le i : D(j) < T\}|$,*
$$
\mu_i \ge \frac{1}{2} - O\!\left(\frac{s^\varphi + \sqrt{i} + 1 + r(T)}{i}\right) - \frac{Z_i}{2i} - \frac{b_D(T)}{i}.
$$
*Moreover, $\mathbb{E}[Z_i] = o(i)$ uniformly over all instances, and for every $\eta > 0$ there is $i_0(\eta)$ (independent of the family, $s$, and $E$) such that $\Pr(Z_i > \eta i) \le e^{-\eta i / 2}$ for all $i \ge i_0(\eta)$. In particular, $Z_i / i \to 0$ almost surely on every fixed instance.*

*Proof.* Assign ownership of each string in $K_i$ to whichever player (adversary observation or generator output) produces it first (within a round, the observation comes first). By Section 2.1, all $i$ strings of $K_i$ have owners by round $T$.

Discard the initial $t_0 := O(s^\varphi + \sqrt{i} + 1)$ rounds so that for all $t > t_0$ we have $r(t) \ge s$ and $M = (t + 4)^2 \ge i$, ensuring $K_i \subset I_t$. Partition $(t_0, T]$ into maximal stretches on which $r(t)$ and the set of viable languages among $L^1, \dots, L^{r(t)}$ (evaluated *after* observing $e_t$) remain constant. Since each index $j \le r(T)$ enters once and can be eliminated at most once, there are at most $2r(T) + 1$ stretches.

Throughout one stretch, let $V := V_{r(t)}$; then $V \subset K$ (since $L^s$ is viable and $s \le r(t)$) and $e_t \in V$ on every round of the stretch. Whenever the adversary first claims a string of $K_i$ on an *ordinary* round in this stretch, either a fresh string in $V \cap K_i \subset I_t$ remains—in which case $j = r(t)$ is feasible and the generator outputs a fresh string from $V \cap K_i$ on that very round—or no fresh string remains in $V \cap K_i$, making this the *last* new adversary claim in $K_i$ during the stretch. Thus at most one ordinary adversary claim per stretch goes unmatched by a distinct generator output in $K_i$.

The remaining uncharged adversary claims in $K_i$ either occurred during the first $t_0$ rounds (at most $t_0$) or on a probe slot (at most $Z_i$). Hence the number of observation-owned strings in $K_i$ is at most $G + t_0 + (2r(T) + 1) + Z_i$. Adding $G$ gives $i \le 2G + O(s^\varphi + \sqrt{i} + 1) + 2r(T) + Z_i$, establishing the pathwise bound. At most $b_D(T)$ of the $G$ generator outputs in $K_i$ by round $T$ can miss their own deadlines $D(j)$, yielding the displayed bound on $\mu_i$.

For the tail of $Z_i$: the first-occurrence times of $K_i$ in $E$ are at most $i$ distinct deterministic round numbers fixed independently of the random probe slots. At most $\sqrt{i}$ fall before round $\sqrt{i}$; every later block $b$ has length $\ell_b \ge L(i) \to \infty$ (independent of the instance). Each block $b$ contributes an independent Bernoulli variable $X_b \in \{0, 1\}$ to $Z_i$ with mean $p_b = m_b / \ell_b$, where $m_b$ is the number of first-occurrence times in block $b$. Since $\sum_b m_b \le i$,
$$
\mathbb{E}[Z_i] \le \sqrt{i} + \frac{i}{L(i)} = o(i)
\qquad \text{uniformly over all instances.}
$$
By independence and $1 + (e - 1)p_b \le e^{(e-1)p_b}$, we have $\mathbb{E}[e^{Z_i}] \le \exp\bigl((e - 1)\mathbb{E}[Z_i]\bigr)$. For any $\eta > 0$, Markov's inequality yields
$$
\Pr(Z_i > \eta i) \le \exp\bigl((e - 1)\mathbb{E}[Z_i] - \eta i\bigr) \le e^{-\eta i / 2}
$$
for all $i \ge i_0(\eta)$ (since $\mathbb{E}[Z_i]/i \to 0$ uniformly). Summing over $i$ gives $Z_i / i \to 0$ almost surely by Borel–Cantelli. $\square$

---

## 4. Reading 1: First Actual Speculation Beats $\varphi$ (Infimum $1$)

**Theorem 2 (Refutation under first-speculation timeliness).**
1. *Upper bound:* The generator of Section 2 achieves pointwise vanishing hallucination (in expectation and pathwise), first actual probe of hypothesis $s$ by $O(s \log^2(2s))$ rounds on every run, and element-wise timely lower density $\liminf_{i \to \infty} \mu_i \ge 1/2$ almost surely under the string deadline $D(i) = \lceil i \log^3(2i) \rceil$ (and simultaneously under every $D(i) = \lceil i^{1+\alpha} \rceil$, $\alpha > 0$). In particular, pure powers $3/2$ for both bounds already beat $\varphi$.
2. *Sharpness of infimum $1$ and endpoint impossibility:* No generator can guarantee a first probe for every $s \le S$ with constant probability $c > 0$ in $o(S)$ rounds, nor achieve positive limiting timely density under sublinear (or, with vanishing hallucination, linear) string deadlines. Under pointwise vanishing hallucination, even a uniform $O(s)$ first-probe bound (with constant probability) for earliest-fresh probes is impossible.

*Proof.* For (1), fix $(s, E)$. Then $T = T_s(i) = O_s(i \log^2(2i))$, so $r(T) = O(T^{1/\varphi}) = o(i)$. Under $D(i) = \lceil i \log^3(2i) \rceil$ (or any $D(i) = \lceil i^{1+\alpha} \rceil$, $\alpha > 0$), for every fixed $\delta > 0$ we have $T < D(\lceil \delta i \rceil)$ for all sufficiently large $i$, so $b_D(T) \le \delta i$, hence $b_D(T) = o(i)$. Lemma 1 then gives $\liminf_{i \to \infty} \mu_i \ge 1/2$ almost surely.

For (2), construct the *one-common-symbol petal family*: every language $L^s$ ($s \ge 1$) contains one shared symbol $c$ together with its own infinite disjoint set of private strings (its *petal*). Repeating $c$ through any horizon $t$ is extendible to a total enumeration of every $L^s$, producing one shared transcript per random seed. Since at most $t$ distinct indices can be probed in $t$ rounds, summing probabilities over $s \le S$ gives $cS \le t$, so $t = \Omega(S)$ is required. Sublinear string deadlines give zero limiting density because at most one string is output per round, and [arXiv:2605.11302](https://arxiv.org/abs/2605.11302) rules out linear string deadlines under vanishing hallucination.

To rule out an $O(s)$ constant-probability first-probe bound for earliest-fresh probes under pointwise vanishing hallucination, use the *powers-of-two core family* from Theorem 3(3) below: $L^1 = \{2^m : m \ge 0\}$ is the shared core, and for $s \ge 1$, $L^{s+1}$ is the union of $L^1$ and its own infinite private petal, where pairs $(s, k) \in \mathbb{N}_+^2$ are placed in diagonal order of $s + k$ (ties by $s$) onto $\mathbb{N}_+ \setminus L^1$. The first private point of petal $s \le S$ has universe position $O(S^2)$. On the fixed total enumeration $1, 2, 4, 8, \dots$ of $L^1$, after $O(\log(2S))$ rounds every core point preceding the first private points of petals $1, \dots, S$ has been observed. Any later earliest-fresh probe of petal $s \le S$ either outputs a private point of petal $s$ or finds one already output; since no private point is ever observed on $L^1$, at most $O(\log(2S))$ of these $S$ labels can be probed earlier without triggering a private output. Thus a constant-probability first-probe guarantee by round $t = O(S)$ for each $s \le S$ forces $\Omega(S)$ expected hallucinations by round $t = O(S)$ on the fixed run of $L^1$, contradicting pointwise vanishing hallucination. $\square$

---

## 5. Reading 2: The Golden-Ratio Theorem for Sustained Prefix-Rank Density

Now measure language-level timeliness by the prefix rank $i_0(s)$ at which uniform positive timely density begins.

**Theorem 3 (Golden-Ratio Barrier for sustained prefix-rank density).**
1. *Packing lower bound (unrestricted hallucination):* Suppose $D(i) = O(i^p)$ and a generator guarantees $\mathbb{E}[\mu_i] \ge \rho > 0$ uniformly over all families, instances $(s, E)$, and prefix ranks $i \ge C s^q$ (for constants $\rho, C > 0$). Then necessarily
   $$
   p \ge 1 + \frac{1}{q}.
   $$
   In particular, for the balanced objective $p = q$, necessarily $p \ge \varphi = \frac{1 + \sqrt{5}}{2}$.
2. *Matching upper bound with polylog slack at $p = q = \varphi$:* At the critical string deadline $D(i) = \lceil i^\varphi \rceil$, the generator of Section 2 achieves pointwise vanishing hallucination and, for every $\epsilon \in (0, 1/2)$, constants $c_\epsilon > 0$ and $i_1(\epsilon)$ independent of the family, $s$, and $E$ such that
   $$
   \Pr\!\left(\mu_i < \frac{1}{2} - \epsilon\right) \le \exp(-c_\epsilon i)
   \qquad \text{for all } i \ge \max\!\bigl\{i_1(\epsilon),\; s^\varphi \log^4(2s)\bigr\}
   $$
   (and hence $\mathbb{E}[\mu_i] \ge 1/2 - o(1)$ uniformly over $i \ge s^\varphi \log^4(2s)$, and simultaneous high-probability density across all qualifying prefixes above any sufficiently large rank). Consequently, every balanced pure power $d > \varphi$ (deadline $\lceil i^d \rceil$, onset $i \ge C s^d$) is achievable.
3. *Impossibility of the pure-power endpoint:* Under pointwise vanishing hallucination, no generator can guarantee $\mathbb{E}[\mu_i] \ge \rho > 0$ uniformly for $D(i) = O(i^p)$ and $i \ge C s^q$ on the boundary $p = 1 + 1/q$—in particular at $D(i) = O(i^\varphi)$, $i \ge C s^\varphi$.

*Proof.*
**Part (1).** Use the one-common-symbol petal family of Theorem 2(2). For large $S$, choose $i = \lceil C S^q \rceil$ and present $c$ repeatedly through round $t := D(i)$. This prefix is extendible to a total enumeration of every $L^s$ ($1 \le s \le S$). On the shared transcript, $c$ is already observed and ineligible for generator credit, while the remaining strings of each $K_i^{(s)}$ lie in disjoint private petals. Hence, for every random seed, the sum of timely credit counts across $s = 1, \dots, S$ is at most $t = D(i)$. Taking expectations gives $\rho S i \le D(i) = O(i^p)$. Since $S = \Theta(i^{1/q})$, we obtain $i^{1 + 1/q} = O(i^p)$, hence $p \ge 1 + 1/q$. Setting $p = q$ gives $p^2 - p - 1 \ge 0$, so $p \ge \varphi$.

**Part (2).** Set $p = \varphi$, $D(i) = \lceil i^\varphi \rceil$, and assume $i \ge s^\varphi \log^4(2s)$. We verify uniformly as $i \to \infty$ that $s = O\bigl(i^{1/\varphi} / \log^{4/\varphi}(2i)\bigr)$: if $s \le i^{1/(2\varphi)}$ this is immediate; if $s > i^{1/(2\varphi)}$ then $\log(2s) = \Theta(\log(2i))$, so $s^\varphi \le i / \log^4(2s) = O(i / \log^4(2i))$, giving the claim. Therefore $s^\varphi = o(i)$ and, using $1 + 1/\varphi = \varphi$ and $4/\varphi > 2$,
$$
T = T_s(i) = O\bigl(s i \log^2(2si)\bigr)
= O\!\left(\frac{i^{1 + 1/\varphi} \log^2(2i)}{\log^{4/\varphi}(2i)}\right)
= O\!\left(i^\varphi \log^{2 - 4/\varphi}(2i)\right)
= o(i^\varphi).
$$
Hence $r(T) = O(T^{1/\varphi}) = o(i)$ and $b_D(T) \le T^{1/\varphi} = o(i)$ uniformly in the range $i \ge s^\varphi \log^4(2s)$. Substituting into Lemma 1 yields $\Pr(\mu_i < 1/2 - \epsilon) \le \exp(-c_\epsilon i)$ for all sufficiently large $i \ge s^\varphi \log^4(2s)$, with constants independent of the instance.

**Part (3).** Use the powers-of-two core family: $L^1 = \{2^m : m \ge 0\}$, and for $s \ge 1$, $L^{s+1}$ consists of $L^1$ together with an infinite private petal placed onto $\mathbb{N}_+ \setminus L^1$ by ordering pairs $(s, k) \in \mathbb{N}_+^2$ along diagonals of increasing $s + k$ (ties by $s$). For any $1 \le s \le S$ and $k \le i$, the pair $(s, k)$ lies on diagonal $s + k \le S + i$, so at most $(S + i)^2$ private points precede it; accounting for intervening powers of two, the $i$-th private point of petal $s$ has universe position $O((S + i)^2)$. Hence among the first $i$ strings of $L^{s+1}$, at most $O(\log(2 + S + i))$ belong to the core $L^1$, and all others are private to $L^{s+1}$.

Run the generator on the fixed total enumeration $E^* = (1, 2, 4, 8, \dots)$ of $L^1$. For each $S \ge 1$, set $i := \lceil C(S + 1)^q \rceil$ and $t := D(i)$. Every petal language $L^{s+1}$ ($1 \le s \le S$) admits a total enumeration agreeing with $E^*$ through round $t$. If every $L^{s+1}$ had expected timely density at least $\rho$ at rank $i$, then each $L^{s+1}$ would receive at least $\rho i - O(\log(2 + S + i))$ expected timely outputs in its disjoint private petal by round $t$. Summing over $s = 1, \dots, S$, the fixed run on $(L^1, E^*)$ would produce at least
$$
S\bigl(\rho i - O(\log(2 + S + i))\bigr) = \Omega(S i)
$$
expected hallucinations (outputs outside $L^1$) by round $t$. On the boundary $p = 1 + 1/q$, we have $t = D(i) = O(i^{1 + 1/q}) = O(S i)$, so the expected hallucination fraction on the single fixed instance $(L^1, E^*)$ remains bounded below by a positive constant along the unbounded sequence of horizons $t = D(\lceil C(S+1)^q \rceil)$, contradicting pointwise vanishing hallucination. $\square$

---

## 6. Scope and Reproducibility

- **Unbalanced $(p, q)$ region:** Theorem 3 establishes the necessary condition $p \ge 1 + 1/q$ everywhere, refutes the pure-power boundary $p = 1 + 1/q$ under pointwise vanishing hallucination, and resolves the balanced diagonal $p = q$ (exact infimum $\varphi$). Characterizing the full unbalanced achievability region $\{(p, q) : p > 1 + 1/q,\; p \neq q\}$ remains open. Note also that a prefix-rank onset $i_0(s)$ corresponds to round $D(i_0(s))$ under the string deadline, and neither onset provides an identification guarantee.
- **Computational verification:** Run from the repository root with Python 3.10+ (standard library, exact integer arithmetic):

```sh
.venv/bin/python -B solutions/timely_lm_hallucination/explore.py
```

The script tests:
1. **Exact probe schedule:** All $15{,}518$ allocation pairs $(s, k)$ with $sk \le 2{,}000$ against the divisor-sum count $A(m)$ and exact block endpoints $B(n)$.
2. **Online game simulations (unbounded-interval idealization):** $48$ games spanning $2$ core types (singleton finite core and infinite core $7\mathbb{Z}_{\ge 0}$), $4$ target indices ($s \in \{1, 2, 5, 8\}$), $3$ enumeration policies (`ordered`, `delayed`, `mixed`), and $2$ fixed seeds (`17`, `2605`). Across $1{,}828{,}104$ rounds ($1{,}276{,}373$ repeat observations and $8{,}152$ finite-core fallback rounds), all $384$ tested prefixes ($i \in \{1, 2, 4, 8, 16, 32, 64, 128\}$) pass the deterministic coverage, pathwise charging, and element-wise deadline accounting checks (with $D(j) = \lceil j^{3/2} \rceil$). *(Honest caveat: the simulation implements the unbounded-interval idealization, drawing the earliest fresh string of the chosen viable language directly rather than truncating to the finite prefix $I_t$ of size parameter $M = (t + 4)^2$; both agree once $I_t$ contains the earliest fresh string, as ensured after warmup in the proof.)*
3. **Sparse-core packing checks:** $20$ checks over $(S, i) \in \{2, 5, 13, 32, 80\} \times \{16, 64, 256, 1024\}$ verifying the $O((S+i)^2)$ position bound and disjoint private witnesses in the powers-of-two core construction.
