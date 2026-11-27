# Resolution of the Natural Language Autoencoder Monitorability Conjectures

**Target paper:** Kevin Yandoka Denamganaï, *On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination*, [arXiv:2610.18727](https://arxiv.org/abs/2610.18727), Section `sec:open`, Conjectures `conj:idnodisc` (parts (a)–(d)) and `conj:paraphrase` (parts (a)–(c)).  
**Companion files:** [`solutions/nla_monitorability/explore.py`](explore.py), [`solutions/nla_monitorability/proof.pdf`](proof.pdf).

---

## Summary of Verdicts

All seven precise sub-claims across Conjectures `conj:idnodisc` and `conj:paraphrase` are settled:

1. **Conjecture `conj:idnodisc` (Identification without discrimination):**
   - **(a) Existence — PROVED (with a tie-vs.-$J=0$ structural caveat).** On an $n$-state orthogonal activation codebook ($n \ge 2$), a generatively optimal pair $(\mathrm{AV}, \mathrm{AR})$ attains the global minimum $J = 0$ and provenance $\mathrm{AUC} = 1$, while achieving strict truth $\mathrm{AUC} = 0$ (midrank $1/2$, zero score bits), strict truth $\mathrm{AUC} = 1/2$ (mixed rivals), or strict truth $\mathrm{AUC} = 1$ at the exact same loss $J = 0$.
     *Tie-vs.-$J=0$ tension:* In the unrestricted Setting 1 game, any attained global minimum is $J = 0$, which forces $S(c_0, H) = 1$ almost surely when $c_0 = Z$, so strict losses are impossible and every unrestricted $J = 0$ witness with strict truth $\mathrm{AUC} < 1$ **must** rely on exact score ties. Imposing `method.tex`'s informal remark ("Since scores are real-valued cosines, ties do not occur") as a mathematical hypothesis would make (a) impossible for $\varepsilon < 1/2$ in the unrestricted game. Under a 2-symbol codebook restriction on $H \sim \mathcal{N}(0, I_2)$, however, a **tie-free** joint optimum ($J = 1 - 2/\pi$) attains provenance $\mathrm{AUC} = 1$ and strict truth $\mathrm{AUC} = \frac{1}{2} + \frac{\alpha}{2\pi}$ for any $0 < \alpha < \pi$.
   - **(b) Surface-Lipschitz bound — MARGIN BOUND PROVED; LITERAL NOISE MODEL INCONSISTENT; COHERENT NOISY-READOUT VARIANT PROVED.** Angular $L$-Lipschitzness gives the sharp chord bound $|D| \le 2\sin(\frac{1}{2}\min\{Lw,\pi\}) = B_w \le Lw$, which alone cannot bound ranks (any tiny positive margin gives $\mathrm{AUC} = 1$). Additive independent nondegenerate Gaussian noise inside the bounded cosine difference $\Delta_k \in [-2,2]$ cannot hold literally. In the coherent external-noise readout model $D + \sigma G$ ($G \sim \mathcal{N}(0,1)$ independent of $D$), $|\mathrm{AUC} - 1/2| \le \Phi(B_w/\sigma) - 1/2 \le Lw/(\sigma\sqrt{2\pi})$, proving the displayed inequality with $C = 1$ and without requiring independence across rivals.
   - **(c) Information form — REFUTED AS STATED (depends on unspecified candidate coupling).** Under the literal $\mathrm{AV}$ message $Z$, a deterministic speaker makes both $I(P; Z \mid H)$ and $I(T; Z \mid H, P=1)$ vanish identically even with perfect rankings; under $Z$-dependent candidate sampling, selection effects allow either quantity to be $0$ or $1$ bit at $J = 0$. Replacing $Z$ by the offered candidate $C$ forces $I(T; C \mid H, P=1) = \mathsf{H}(T \mid H, P=1)$ identically (1 bit at $J = 0$), while $I(P; C \mid H)$ can still be $0$ or $1$ bit at $J = 0$. The intrinsic zero-distortion identity is simply $U = v(Z)$ a.s. ($\mathsf{H}(U \mid Z) = 0$).
   - **(d) Genericity — REFUTED under a natural measure on minimizers.** For every $0 < \varepsilon < 1/2$, fixing a bijective grounded speaker and its unique on-support decoder and parameterizing rival extensions by one unit vector per distinct rival in dimension $n \ge 2$ under the product uniform sphere measure, the extensions achieving strict truth $\mathrm{AUC} = 1$ form an **open set of full measure** (1), not measure zero.

2. **Conjecture `conj:paraphrase` (Invariance licenses the truth reading):**
   - **(a) Additive decomposition — PROVED AND SHARP.** When $S(c,h) = g(m(c),h) + e(c,h)$ with $|e| \le \varepsilon$ on **both** candidates in every lineup and $M = g(m(c_0),h) - g(m(c_k),h)$,
     $$
     \Pr(M > 2\varepsilon) \le \mathrm{AUC}_{\mathrm{strict}} \le \Pr(M > -2\varepsilon).
     $$
     Both endpoints are sharp, the bound extends to a failure event of probability $q$, and uniform within-class oscillation $d$ supplies $|e| \le d/2$. However, even arbitrarily small uniform $\varepsilon \in (0, 1/2)$ can flip strict $\mathrm{AUC}$ from $1$ to $0$ when margins are small.
   - **(b) Authorship — REFUTED.** An explicit 50-token counterexample ("blue" vs. native "red" and external "scarlet", keeping 49 framing tokens fixed) has **zero** matched authorship control damage on both true-account controls, retains 98% $\mathrm{AV}$-authored tokens on the rival, yet shifts the rival score by $0.6$ and reverses strict $\mathrm{AUC}$ from $1$ to $0$ (and any value in $[0,1]$ by mixing). Experiment 2's reported median normalized angular movement ($0.021$) and matched damage difference ($0.073$) therefore do not bound unseen rival forms or settle the causal contribution of foreign wording.
   - **(c) Steganographic gap — REFUTED IN BOTH DIRECTIONS.** Under the decision-theoretic steganographic gap of Anwar et al. ([arXiv:2602.23163](https://arxiv.org/abs/2602.23163)), a lexical-marker example ("indeed"/"certainly") gives raw gap $1/2$ (relative gap $1$) with **zero** paraphrase angular damage, refuting the upper bound; a public word-order example ("Alice leads Bob" vs. "Bob leads Alice") gives gap **zero** with normalized shuffle damage $1$, refuting the lower bound.

---

## 1. Setup: The Two Games and Scoring Conventions

In **Setting 1** (generative training), let $U := H/\|H\| \in S^{n-1}$ be the unit activation direction, $Z \sim \mathrm{AV}(\cdot \mid H)$ the emitted explanation, and $v(c) := \mathrm{AR}(c)/\|\mathrm{AR}(c)\| \in S^{n-1}$ the normalized nonzero $\mathrm{AR}$ output. The generative cosine loss is $J := 1 - \mathbb{E}[U \cdot v(Z)]$. Conditioning on each emitted string $z$ and maximizing $v(z) \cdot \mathbb{E}[U \mid Z = z]$ over $v(z) \in S^{n-1}$ gives the exact Bayes risk for any fixed speaker:
$$
\inf_v J = 1 - \mathbb{E}\bigl\|\mathbb{E}[U \mid Z]\bigr\|,
$$
attained whenever $v(z) = \mathbb{E}[U \mid Z = z] / \|\mathbb{E}[U \mid Z = z]\|$ on the support of $Z$ (with nonzero conditional mean). Crucially, strings with emission probability zero impose no constraint on $v$, and neither the truth relation $R \subseteq \mathcal{H} \times \mathcal{A}$ nor any distractor distribution enters $J$.

In **Setting 2** (oversight evaluation), the experimental surrogate listener scores a candidate account $c$ against $h$ by $S(c,h) := u \cdot v(c)$, where $u = h/\|h\|$. Each lineup pairs $c_0 = z \in A(h)$ with false rivals $c_1, \dots, c_K$ formed by editing $c_0$. Provenance $P \in \{0,1\}$ indicates whether a candidate originated from activation $h$ (including edited rivals derived from $z$), and $T := \mathbf{1}_R(h, c)$ indicates truth. Sampling a conversation first and a rival uniformly from its lineup second, the paper defines lineup AUC using a **strict** win:
$$
\mathrm{AUC}_{\mathrm{strict}} := \Pr\bigl(S(c_0, H) > S(c_k, H)\bigr).
$$
An uninformative tie therefore contributes $0$ to $\mathrm{AUC}_{\mathrm{strict}}$ (and $1/2$ under the conventional midrank rule).

---

## 2. Conjecture `conj:idnodisc`: Identification without Discrimination

### 2.1 Part (a): Orthogonal codebook witnesses, tie-vs.-$J=0$ tension, and tie-free Gaussian witness

**Theorem 1 (Unrestricted $J=0$ codebook witnesses).** *For every $n \ge 2$, there exists a finite activation distribution $(H, R)$ and a pair $(\mathrm{AV}, \mathrm{AR})$ attaining the global minimum $J = 0$ and $I(H; Z) = \log_2 n$ with provenance $\mathrm{AUC}_{\mathrm{strict}} = 1$, such that changing only the zero-probability out-of-support decoder outputs produces strict truth $\mathrm{AUC}_{\mathrm{strict}}$ equal to $0$ (midrank $1/2$, zero score bits), $1/2$, or $1$.*

*Proof.* Let $H$ be uniform over the $n$ standard basis vectors $e_0, \dots, e_{n-1} \in \mathbb{R}^n$. Construct $n$ distinct templates indexed by $j \in \{0,\dots,n-1\}$ expressing the color claim "blue" ($b_j$), with one-word false edits "red" ($r_j$) and "green" ($g_j$). Declare all $b_j$ true of every $e_i$ and all $r_j, g_j$ false of every $e_i$ (padding the template makes the token edit fraction arbitrarily small; if false other-activation provenance controls are desired, include a true transcript-specific identifier in each template and retain it in its rivals).

For any permutation $p$ of $\{0,\dots,n-1\}$, let $\mathrm{AV}(e_i) = b_{p(i)}$ deterministically and set $v(b_{p(i)}) = e_i$. Then $J = 0$ (the global lower bound), $I(H; Z) = \log_2 n$, and on provenance lineups ($c_0 = b_{p(i)}$ vs. $c_k = b_{p(i')}$, $i' \neq i$), $S(c_0, e_i) = 1 > 0 = S(c_k, e_i)$, so provenance $\mathrm{AUC}_{\mathrm{strict}} = 1$. Because $r_j, g_j$ have emission probability $0$, their decoder outputs do not affect $J$:
- Setting $v(r_{p(i)}) = e_i$ gives $S(r_{p(i)}, e_i) = 1 = S(b_{p(i)}, e_i)$ on every red rival. With balanced labels, $S$ carries zero bits about $T$ given $(H, P=1)$, strict truth $\mathrm{AUC}_{\mathrm{strict}} = 0$, and midrank truth $\mathrm{AUC} = 1/2$. This proves `conj:idnodisc`(a) for every $\varepsilon > 0$.
- Setting $v(g_{p(i)}) = e_{(i+1) \bmod n}$ gives $S(g_{p(i)}, e_i) = 0 < 1 = S(b_{p(i)}, e_i)$. Mixing red and green rivals equally yields strict truth $\mathrm{AUC}_{\mathrm{strict}} = 1/2$ (half strict wins, half ties), separating a numerical value of $1/2$ from zero score information.
- Using only green rivals—or keeping the red-only lineup fixed and redefining $v(r_{p(i)}) = e_{(i+1) \bmod n}$—gives strict truth $\mathrm{AUC}_{\mathrm{strict}} = 1$ at the exact same training loss $J = 0$. $\square$

**Remark 1 (Tension between $J=0$ and `method.tex`'s "ties do not occur" remark).** In `method.tex`, the prose states: *"Since scores are real-valued cosines, ties do not occur."* Real-valued cosines can in fact tie exactly, and this remark is mathematically incompatible with `conj:idnodisc`(a) in the unrestricted Setting 1 game. Indeed, for any activation distribution on $\mathbb{R}^n$, assigning distinct strings to a finite angular net of mesh $a$ achieves loss at most $1 - \cos a$, so the unrestricted infimum is $0$ and any attained global minimum satisfies $J = 0$. Whenever $J = 0$ and $c_0 = Z$, we have $S(c_0, H) = U \cdot v(Z) = 1$ almost surely, making strict losses $S(c_0, H) < S(c_k, H)$ impossible. Consequently, any $J = 0$ witness with $c_0 = Z$ and $\mathrm{AUC}_{\mathrm{strict}} < 1$ **must** produce ties; imposing "no ties" as a mathematical hypothesis would force $\mathrm{AUC}_{\mathrm{strict}} = 1$ and render `conj:idnodisc`(a) false for every $\varepsilon < 1/2$ in the unrestricted game. The displayed definition of lineup AUC itself imposes no no-ties hypothesis.

**Remark 2 (Deployed monitors vs. surrogate scores: an information-theoretic witness).** When $c_0 = Z$ and $Z$ recovers $H$ bijectively, a deployed listener with access to $(Z, c)$ and $R$ can decide $T = \mathbf{1}_R(H, c)$ perfectly (or simply check string equality $c = Z$). Genuine information loss persists even for arbitrary monitors when truths depend on activation magnitude, which cosine loss ignores: let $H = D(1 + B)$ with independent uniform $D \in \{-1, 1\}$ and $B \in \{0, 1\}$. The grounded sign message $Z = D$ achieves $J = 0$ and provenance $\mathrm{AUC} = 1$ against opposite-direction distractors, while any independently presented magnitude query satisfies $I(T; Z, C) = 0$, forcing every decision rule to score $1/2$.

**Proposition 2 (Tie-free 2-symbol Gaussian optimum).** *Let $H \sim \mathcal{N}(0, I_2)$ and restrict $\mathrm{AV}$ to two emitted messages $\{b_+, b_-\}$ (with other strings available as edited candidates). Then:*
1. *The joint global minimum of $J$ is $1 - 2/\pi$, attained uniquely (up to rotation) by antipodal decoder directions $\pm e_1$ and $\mathrm{AV}(h) = b_{\operatorname{sgn}(h_1)}$.*
2. *Against opposite-half-plane provenance distractors, provenance $\mathrm{AUC}_{\mathrm{strict}} = 1$.*
3. *Decoding one-word false rivals $r_\pm$ to $\pm(\cos\alpha, \sin\alpha)$ for $0 < \alpha < \pi$ gives a tie-free strict truth $\mathrm{AUC}_{\mathrm{strict}} = \frac{1}{2} + \frac{\alpha}{2\pi}$, arbitrarily close to $1/2$.*

*Proof.* For any two unit decoder vectors $a, b \in S^1$, the optimal encoder chooses $\arg\max_{w \in \{a,b\}} U \cdot w$. Since $U$ is uniform on $S^1$, $U \cdot a + U \cdot b$ has mean $0$, so
$$
\mathbb{E}\max(U \cdot a, U \cdot b)
= \frac{1}{2}\mathbb{E}|U \cdot (a - b)|
= \frac{\|a - b\|}{\pi} \le \frac{2}{\pi},
$$
with equality if and only if $b = -a$. Thus $J^* = 1 - 2/\pi$, attained by $v(b_\pm) = \pm e_1$ and $\mathrm{AV}(h) = b_{\operatorname{sgn}(h_1)}$, yielding $S(b_{\operatorname{sgn}(h_1)}, h) = |u_1| > -|u_1|$ almost surely against the opposite-half-plane message (provenance $\mathrm{AUC}_{\mathrm{strict}} = 1$). Conditional on $\mathrm{AV}(h) = b_\pm$, the angular offset $t$ of $U$ from $v(b_\pm)$ is uniform on $(-\pi/2, \pi/2)$. Since $S(b_\pm, h) = \cos t$ and $S(r_\pm, h) = \cos(t - \alpha)$, ties occur with probability $0$ and the true account wins iff $\cos t > \cos(t - \alpha) \iff t < \alpha/2$ (up to endpoints), giving
$$
\mathrm{AUC}_{\mathrm{strict}} = \frac{\alpha/2 - (-\pi/2)}{\pi} = \frac{1}{2} + \frac{\alpha}{2\pi}. \quad \square
$$
*(The 2-symbol restriction is essential: for unrestricted finite strings on $\mathcal{N}(0, I_2)$, finite angular quantizers drive $\inf J = 0$, while no countable codebook attains $J = 0$ because a continuous uniform direction hits any countable subset with probability $0$.)*

---

### 2.2 Part (b): Surface smoothness, chord bound, and noisy readout

**Proposition 3 (Chord bound and external Gaussian readout).** *Suppose $\angle(v(c), v(c')) \le L\,\delta_s(c, c')$ (in radians). For any pair $(c_0, c_k)$ with $\delta_s(c_0, c_k) \le w$, the latent cosine margin $D := u \cdot (v(c_0) - v(c_k))$ satisfies the sharp bound*
$$
|D| \le 2\sin\!\Bigl(\frac{1}{2}\min\{Lw, \pi\}\Bigr) =: B_w \le Lw.
$$
*Moreover:*
1. *Without noise, the Lipschitz bound alone does not bound $\mathrm{AUC}_{\mathrm{strict}}$: any strictly positive $D > 0$ on every pair yields $\mathrm{AUC}_{\mathrm{strict}} = 1$ even for arbitrarily small $Lw$.*
2. *The printed assumption that the cosine difference $\Delta_k = S(c_0, H) - S(c_k, H)$ itself equals a mean term plus independent nondegenerate Gaussian noise cannot hold literally, because $\Delta_k \in [-2, 2]$ has bounded support whereas any finite random variable plus an independent nondegenerate Gaussian has unbounded support.*
3. *Under the coherent external-noise readout model $\widetilde{\Delta}_k := D + \sigma G$ with $G \sim \mathcal{N}(0, 1)$ independent of $D$ and $\sigma > 0$,*
   $$
   \Bigl|\Pr(\widetilde{\Delta}_k > 0) - \frac{1}{2}\Bigr|
   = \Bigl|\mathbb{E}\Phi(D/\sigma) - \frac{1}{2}\Bigr|
   \le \Phi(B_w / \sigma) - \frac{1}{2}
   \le \frac{Lw}{\sigma\sqrt{2\pi}},
   $$
   *establishing `conj:idnodisc`(b) with $C = 1$ and without requiring independence across rivals.*

*Proof.* Since $v(c_0), v(c_k)$ are unit vectors separated by angle $\theta \le \min\{Lw, \pi\}$, $\|v(c_0) - v(c_k)\| = 2\sin(\theta/2) \le B_w \le Lw$; Cauchy–Schwarz gives $|D| \le B_w$, with equality when $u$ is aligned with the chord. For (1), decode $c_0$ to $u$ and $c_k$ to a vector rotated by any $\theta > 0$ (Lipschitz ratio $\theta / w$). Part (2) is immediate from support comparison (and also rules out a fixed-mean nondegenerate Gaussian across conversations). For (3), conditioning on $D$ gives $\Pr(D + \sigma G > 0 \mid D) = \Phi(D/\sigma)$, and since $|\Phi(x) - 1/2| \le |x|/\sqrt{2\pi}$ by the maximum standard normal density $1/\sqrt{2\pi}$, averaging over $(H, c_0, c_k)$ yields $|\mathrm{AUC} - 1/2| \le \Phi(B_w/\sigma) - 1/2 \le B_w/(\sigma\sqrt{2\pi}) \le Lw/(\sigma\sqrt{2\pi})$. Note that empirical spread of actual cosine differences supplies no independent noise mechanism. $\square$

---

### 2.3 Part (c): Why the proposed mutual-information bounds fail

Setting 1 specifies only the law of $(H, Z)$. Evaluating $I(P; \cdot \mid H)$ and $I(T; \cdot \mid H, P=1)$ requires a candidate sampler and its joint coupling with $Z$. Every natural interpretation fails as stated:

1. **Literal $\mathrm{AV}$ message $Z$ with a deterministic speaker (or independent label draw):** When $Z$ is a deterministic function of $H$ (as in Theorem 1), $\mathsf{H}(Z \mid H) = 0$, so
   $$
   I(P; Z \mid H) = 0 \qquad \text{and} \qquad I(T; Z \mid H, P=1) = 0
   $$
   identically—even when both provenance and truth rankings are perfect. More generally, sampling $(P, T)$ independently of speaker randomness conditional on $H$ makes both conditional mutual informations vanish, rendering "maximal $I(P; Z \mid H)$" vacuous.
2. **Offered candidate $C$ in place of $Z$:** Since $T = \mathbf{1}_R(H, C)$ is a deterministic function of $(H, C)$, we have $I(T; C \mid H, P=1) = \mathsf{H}(T \mid H, P=1)$ identically. On balanced blue/red lineups in Theorem 1, $I(T; C \mid H, P=1) = 1$ bit even though truth carries zero weight in $J = 0$. Conversely, $I(P; C \mid H)$ need not be maximal at $J = 0$: in the sign/magnitude model $H = D(1 + B)$ of Remark 2, offering with equal probability the original message or a message from the *same sign and opposite magnitude* yields $I(P; C \mid H) = 0$ for the sign-only encoder $Z = D$, versus $1$ bit for an encoder that adds a grounded magnitude tag—both attaining $J = 0$.
3. **Literal $Z$ with $Z$-dependent candidate sampling (selection effects):** At fixed $H$, let $\mathrm{AV}$ emit two grounded synonymous blue wordings $b^{(1)}, b^{(2)}$ with equal probability, both decoded to $u$ ($J = 0$). If the sampler offers the original ($T = 1$) after $b^{(1)}$ and its false red edit ($T = 0$) after $b^{(2)}$ (always $P = 1$), then $T$ is a fair bit determined by $Z$ given $H$, so $I(T; Z \mid H, P=1) = 1$ bit at $J = 0$ (whereas an independent coin flip gives $0$ bits for the same $(\mathrm{AV}, \mathrm{AR})$). Likewise, triggering $P = 1$ on $b^{(1)}$, $P = 0$ on $b^{(2)}$, and an independent fair $P$-coin on a third synonym $b^{(3)}$ produces $I(P; Z \mid H) = 1$ bit for the uniform $(b^{(1)}, b^{(2)})$ speaker and $0$ bits for the deterministic $b^{(3)}$ speaker, both at $J = 0$ with balanced provenance at every $h$.

The intrinsic information-theoretic consequence of $J = 0$ is simply $U = v(Z)$ almost surely (so $\mathsf{H}(U \mid Z) = 0$ for discrete directions); additional string features remain free to carry information ignored by the decoder, and the paper defines no componentwise truth distortion that would yield the claimed upper bound.

---

### 2.4 Part (d): Refutation of genericity

**Proposition 4 (Open full-measure set of perfect-ranking minimizers).** *Let $n \ge 2$ and fix the activation/truth setup of Theorem 1 with any bijective grounded speaker $e_i \mapsto b_{p(i)}$ and its unique on-support minimizer $v(b_{p(i)}) = e_i$. Parameterize the space of $J = 0$ extensions over the $n$ distinct red rivals by $(v(r_{p(0)}), \dots, v(r_{p(n-1)})) \in (S^{n-1})^n$, equipped with the product of uniform surface measures on $S^{n-1}$. Then for every $0 < \varepsilon < 1/2$, the set of minimizers achieving strict truth $\mathrm{AUC}_{\mathrm{strict}} = 1 > 1/2 + \varepsilon$ is an open subset of $(S^{n-1})^n$ of measure $1$.*

*Proof.* Every point in $(S^{n-1})^n$ achieves $J = 0$ and provenance $\mathrm{AUC}_{\mathrm{strict}} = 1$. For each $i$, strict victory on rival $r_{p(i)}$ holds iff $e_i \cdot v(r_{p(i)}) < 1$, i.e., $v(r_{p(i)}) \neq e_i$. Therefore
$$
\mathcal{G} := \prod_{i=0}^{n-1} \bigl(S^{n-1} \setminus \{e_i\}\bigr)
$$
is precisely the set of extensions with strict truth $\mathrm{AUC}_{\mathrm{strict}} = 1$. Since $n \ge 2$, each singleton $\{e_i\}$ is closed and has surface measure $0$ in $S^{n-1}$, so $\mathcal{G}$ is open and has product measure $1$. Mixing uniformly over the $n!$ bijective codebooks preserves measure $1$. $\square$

*(Note: an ambient parameter measure assigning all exact minimizers measure $0$ would not represent a measure on minimizers; on the minimizer manifold itself, natural uniform measure yields the opposite of `conj:idnodisc`(d).)*

---

## 3. Conjecture `conj:paraphrase`: What Paraphrase Invariance Certifies

### 3.1 Part (a): Sharp $2\varepsilon$ margin sandwich

**Theorem 5 (Additive form-error sandwich).** *Let $m(c)$ denote the mutual-entailment (paraphrase) class of $c$, and suppose that on a good event $E$ of probability at least $1 - q$, both candidates in a lineup satisfy $S(c, h) = g(m(c), h) + e(c, h)$ with $|e(c, h)| \le \varepsilon$. Setting $M := g(m(c_0), H) - g(m(c_k), H)$, we have*
$$
\max\bigl\{0,\, \Pr(M > 2\varepsilon) - q\bigr\}
\le \mathrm{AUC}_{\mathrm{strict}}
\le \min\bigl\{1,\, \Pr(M > -2\varepsilon) + q\bigr\}.
$$
*In particular, when $q = 0$,*
$$
\Pr(M > 2\varepsilon) \le \mathrm{AUC}_{\mathrm{strict}} \le \Pr(M > -2\varepsilon),
$$
*and $|\mathrm{AUC}_{\mathrm{strict}} - \Pr(M > 0)| \le \Pr(-2\varepsilon < M \le 2\varepsilon)$.*

*Proof.* On $E$, $S(c_0, H) - S(c_k, H) = M + e(c_0, H) - e(c_k, H) \in [M - 2\varepsilon,\, M + 2\varepsilon]$. Hence on $E$:
$$
\{M > 2\varepsilon\} \subseteq \{S(c_0, H) > S(c_k, H)\} \subseteq \{M > -2\varepsilon\}.
$$
Taking probabilities yields the $q = 0$ inequalities immediately; intersecting with $E$ and using $\Pr(E^c) \le q$ yields the general bounds. $\square$

**Sharpness, oscillation equivalence, and small-$\varepsilon$ reversal:**
- *Boundary sharpness:* At $M = -2\varepsilon$, even the best form error $(+\varepsilon, -\varepsilon)$ produces a tie (contributing $0$ to $\mathrm{AUC}_{\mathrm{strict}}$); at $M = 2\varepsilon$, the worst error $(-\varepsilon, +\varepsilon)$ also produces a tie. Throughout $(-2\varepsilon, 2\varepsilon)$, choosing opposite extreme errors realizes either a strict win or a strict loss.
- *Oscillation equivalence:* If $S(\cdot, h)$ has uniform within-class oscillation $\sup_{c, c' \in m} |S(c,h) - S(c',h)| \le d$ on every class $m$, defining $g(m, h) := \frac{1}{2}(\sup_{c \in m} S(c,h) + \inf_{c \in m} S(c,h))$ yields $|e(c, h)| \le d/2 =: \varepsilon$ (minimal by the triangle inequality). Exact invariance ($\varepsilon = 0$) makes $S$ a function of meaning alone, yet still allows arbitrary preferences over meanings—including insensitivity to truth.
- *Arbitrarily small $\varepsilon$ can flip $\mathrm{AUC}_{\mathrm{strict}}$ from $1$ to $0$:* Fix $h = e_1 \in \mathbb{R}^2$, where any scalar $s \in [-1, 1]$ is realized as a unit cosine by $v = (s, \sqrt{1-s^2})$. For any $0 < \varepsilon < 1/2$, set $g(\text{true}, h) = \varepsilon$, $g(\text{false}, h) = 0$, and assign form errors $-\varepsilon$ to the observed true form and $+\varepsilon$ to the observed false form (with zero-error paraphrases available). Then $M = \varepsilon > 0$ (meaning-only strict AUC $= 1$), while $S(c_0, h) = 0 < \varepsilon = S(c_k, h)$ (observed strict AUC $= 0$).

---

### 3.2 Part (b): Refutation of the authorship bound (50-token red/scarlet counterexample)

**Proposition 6 (Zero matched control damage with reversed rival ranking).** *There exists a listener $S(c, h) = u \cdot v(c)$ on 50-token accounts for which:*
1. *Every true-account control—both an externally authored paraphrase and an $\mathrm{AV}$ substitution, each changing $1$ of $50$ tokens—reconstructs to the exact same vector as the original, giving matched authorship angular damage difference equal to $0$ (for every positive normalizer);*
2. *Both false rivals change only the single color token (retaining $98\%$ $\mathrm{AV}$-authored tokens), yet switching the rival's wording from native "red" to external "scarlet" shifts the cosine score by $3/5 = 0.6$ and reverses strict $\mathrm{AUC}_{\mathrm{strict}}$ from $1$ to $0$.*

*Proof.* Fix $h = e_1 \in \mathbb{R}^2$ (with another activation $h' = e_2$ supplying a positive angular normalizer). Consider 50-token accounts with token $1$ naming a color and tokens $2,\dots,50$ fixed framing words. Define $v(c) = (s(c), \sqrt{1 - s(c)^2})$ solely as a function of token $1$:

| Account | Token 1 | Cosine score at $h = e_1$ |
|---|---:|---:|
| Original true claim ("blue") | `blue` | $3/5 = 0.6$ |
| External true paraphrase (1 framing word changed) | `blue` | $3/5 = 0.6$ |
| $\mathrm{AV}$ substitution control (1 framing word changed) | `blue` | $3/5 = 0.6$ |
| Native false rival ("red") | `red` | $1/5 = 0.2$ |
| External false rival ("scarlet") | `scarlet` | $4/5 = 0.8$ |

On both true-account controls, $v(c)$ is identical to the original (indeed this can be extended to *every* paraphrase of the true meaning), so the matched angular damage difference is identically $0$. Yet on the unseen false meaning class, "red" scores $0.2 < 0.6$ (strict win) while "scarlet" scores $0.8 > 0.6$ (strict loss), a shift of $0.6$. Mixing the two false wordings in proportion $\lambda \in [0, 1]$ realizes any strict win rate in $[0, 1]$ with identical zero control damage. $\square$

**Scope remarks on linear costs and Experiment 2:** Even for a genuinely linear form cost $K\delta_s$, measuring an excess $\le \varepsilon$ at displacement $d > 0$ would only bound $K \le \varepsilon/d$ (not $K \le \varepsilon$ without dividing by $d$), and fails completely for nonlinear costs. A sufficient extra hypothesis is a bounded per-token replacement effect $\le K/N$ at fixed meaning *in every intermediate context* along a path of $\le fN$ edits, which telescopes to $Kf$ in score units (still requiring margins to bound ranks). Consequently, Experiment 2's reported median normalized angular movement ($0.021$) and matched damage difference ($0.073$)—measured on edits of true accounts, using a median rather than uniform bound and normalized by per-item denominators—leave the causal contribution of foreign wording on rivals undetermined.

---

### 3.3 Part (c): Refutation of both steganographic-gap inequalities

Recall the decision-theoretic steganographic gap of Anwar et al. ([arXiv:2602.23163](https://arxiv.org/abs/2602.23163)): given an audit target $Y$, utility function, public context, and decoder families for Receiver and Sentinel, the raw gap is
$$
\Delta_{\mathrm{steg}} := \bigl(U_{\mathrm{Rec}} - U_{\mathrm{Rec}}^0\bigr) - \bigl(U_{\mathrm{Sent}} - U_{\mathrm{Sent}}^0\bigr),
$$
comparing each player's achievable utility gain from observing the message over its message-free baseline (with relative normalization dividing by the positive Receiver gain). A string-rewriting operation supplies no audit rule on its own; fixing constant public context, a fair binary target $Y \in \{0, 1\}$, accuracy utility, and constant actions in both families refutes both inequalities:

1. **Word-identity channel (refutes upper bound by paraphrase damage).** Encode $Y = 0$ by `("text", "indeed", "today")` and $Y = 1$ by `("text", "certainly", "today")`. Let the rewriter paraphrase the trailing word `"today"` $\mapsto$ `"now"` while preserving the marker. Let $\mathrm{AR}$ map the presence of `"indeed"` vs. `"certainly"` to orthogonal directions $e_1, e_2$, which the Receiver reads to achieve accuracy $1$ (gain $1/2$ over baseline $1/2$). Let the Sentinel's available audit rules ignore the lexical marker (scoring $1/2$, gain $0$). Then $\Delta_{\mathrm{steg}} = 1/2$ (relative gap $1$), while the angular paraphrase damage is $0$ on every input.
2. **Word-order channel (refutes lower bound by shuffle damage).** Encode $Y = 0$ by `("Alice", "leads", "Bob")` and $Y = 1$ by `("Bob", "leads", "Alice")`. Both Receiver and Sentinel can read the leading name (public compositional meaning), so both achieve accuracy $1$ (gain $1/2$) and $\Delta_{\mathrm{steg}} = 0$ (relative gap $0$). Reversing token order swaps the two messages; with orthogonal $\mathrm{AR}$ outputs $e_1, e_2$ determined by the first token, the normalized shuffle angular damage is $\frac{\pi/2}{\pi/2} = 1$ on every input (and strictly positive in expectation under a uniform random permutation).

Thus angular rewrite and shuffle damage neither upper-bound nor lower-bound the decision-theoretic steganographic gap.

---

## 4. Verification and Reproduction

Run `.venv/bin/python -B solutions/nla_monitorability/explore.py` from the repository root (Python 3 standard library only; no downloads or external dependencies). The script verifies:
- All $5! = 120$ five-message codebooks (2,400/2,400 provenance wins, 600 tied red rivals, 600 strictly beaten green rivals) and the conditional mutual information values ($0$ or $1$ bit);
- The independent-magnitude query model ($J = 0$ and $I(T; Z, C) = 0$);
- 200,000 planar Gaussian trials (seed `261018727`, $\alpha = \pi/24$): 200,000/200,000 provenance wins, truth AUC $0.521925$ vs. exact $25/48 \approx 0.520833$, loss $0.364401$ vs. $1 - 2/\pi \approx 0.363380$, plus 33 deterministic angular quadratures and 48 independent numerical checks of the noisy-readout bound;
- 99 exact rational margin checks for the $2\varepsilon$ sandwich (including endpoint attainment and reversals), the 50-token red/scarlet counterexample, and both steganographic-gap counterexamples.
