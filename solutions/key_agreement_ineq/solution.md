# Logical Incompatibility of the Key-Agreement Conjectures and Exact Collapse of $\Psi_\Delta$

**Target paper:** Hamidreza Abin and Amin Gohari, *On the Source Model Key Agreement Problem*, [arXiv:2502.00294](https://arxiv.org/abs/2502.00294) (*IEEE Transactions on Information Theory* 72(8):5317–5334, August 2026), Conjecture 1 (`conjecture:1`) and Conjecture 2 (`conj:2`).  
**Companion files:** [`solutions/key_agreement_ineq/explore.py`](explore.py), [`solutions/key_agreement_ineq/proof.pdf`](proof.pdf).

---

## Verdict

- **Logical incompatibility (Theorem 1): Conjecture 1 $\implies \neg$Conjecture 2.** For every binary XOR source $(A, B, E)$ with $E = A \oplus B$, there is an explicit six-variable common-observation instance of Conjecture 1 whose left-hand side minus right-hand side equals
  $$
  I(A; B) - S(A; B \| A \oplus B).
  $$
  Consequently, Conjecture 1 and Conjecture 2 cannot both hold: any strict-gap witness for Conjecture 2 immediately yields a counterexample to Conjecture 1 with the exact same gap. The same construction shows that Conjecture 1 forces $S(A; B \| E) = I(A; B)$ for every source satisfying the hypothesis of Theorem `Th:gen:classic` of Abin and Gohari (2026).
- **Exact collapse of $\Psi_\Delta$ (Theorem 2): PROVED.** For every binary XOR source and every feasible parameter $0 \le \Delta \le H(E)$, the paper's proposed constrained upper bound evaluates identically to the trivial mutual-information ceiling:
  $$
  \Psi_\Delta(A; B \| A \oplus B) = I(A; B).
  $$
  The same proof yields $I(A; B \downarrow A \oplus B) = I(A; B)$ for the intrinsic information, refuting the paper's proposed bounding strategy for Conjecture 2.
- **Scope caveat:** Neither Conjecture 1 nor Conjecture 2 is individually proved or refuted here; their individual truth values remain open.
- **Operational structure (Propositions 3–5):** We also establish an exact one-way equality criterion ($S_{\mathrm{ow}}(A \to B \| E) = I(A; B) \iff A \perp E$ for dependent binary $A, B$), a strictly positive two-sample interactive rate for every dependent binary XOR source, and a triple-grouping protocol for the symmetric triangle source $(1/3, 1/3, 1/3, 0)$ (where both one-way capacities vanish) achieving $(1 + \log_2 3)/12 \approx 0.215414 \le S(A; B \| E) \le \log_2 3 - 4/3 \approx 0.251629$.

---

## 1. Model and Conjecture Statements

All random variables take values in finite alphabets and all logarithms are base $2$. Alice, Bob, and Eve observe i.i.d.\ copies of a discrete memoryless source $(X, Y, Z)$, receiving $X^n, Y^n, Z^n$ respectively. Using independent private randomness and an authenticated public interactive transcript $F$, Alice and Bob output keys $K_A, K_B$ in a common finite alphabet $\mathcal{K}$. Following Ahlswede and Csiszár (1993) and Abin and Gohari (2026), a rate $R \ge 0$ is achievable if for every $\delta > 0$ and sufficiently large $n$ there exists a protocol with $\Pr[K_A \neq K_B] \le \delta$, $R = H(K_A)/n$, and
$$
\frac{1}{n}\log_2 |\mathcal{K}| - R \le \delta, \qquad \frac{1}{n} I(K_A; Z^n, F) \le \delta.
$$
The secret-key capacity $S(X; Y \| Z)$ is the supremum of achievable rates. The one-way secret-key capacity from $X$ to $Y$ is given by the single-letter formula (Ahlswede and Csiszár, 1993; Abin and Gohari, 2026)
$$
S_{\mathrm{ow}}(X \to Y \| Z) = \max_{(U, V) - X - (Y, Z)} \bigl[I(U; Y \mid V) - I(U; Z \mid V)\bigr],
$$
where finite cardinality bounds on $U, V$ ensure the maximum is attained. We use the standard bounds $S_{\mathrm{ow}}(X \to Y \| Z) \le S(X; Y \| Z) \le \min\{I(X; Y),\, I(X; Y \mid Z)\}$; in particular, $S(T; T \| E) = H(T \mid E)$ (choosing $U = T$ and $V$ constant matches the upper bound $I(T; T \mid E) = H(T \mid E)$).

- **Conjecture 1 (`conjecture:1`, Abin and Gohari 2026):** *For every finite joint distribution $P_{X,Y,Z,X',Y',Z'}$,*
  $$
  S(X; Y \| Z) - S(X'; Y' \| Z') \le S_{\mathrm{ow}}(X, Y \to Z' \| Z) + I(Y', Z'; X \mid X') + I(X', Z'; Y \mid Y') + I(X; Y \mid X', Y', Z').
  \tag{1}
  $$
- **Conjecture 2 (`conj:2`, Abin and Gohari 2026):** *There exists a binary joint distribution $P_{X,Y}$ on $\{0, 1\}^2$ such that $S(X; Y \| X \oplus Y) < I(X; Y)$.*

---

## 2. Joint Incompatibility: Conjecture 1 Implies $\neg$Conjecture 2

**Lemma 1 (Common-observation entropy identity and XOR lift).**
1. *For any finite joint law $P_{T,A,B,E}$,*
   $$
   H(T \mid E) - H(T \mid A) - H(T \mid B) + H(T \mid A, B, E) = I(A; B) - I(A; B \mid T) - I(T; E) - I(T; E \mid A, B).
   \tag{2}
   $$
2. *For every binary distribution $P_{A,B}$ on $\{0, 1\}^2$ with $E = A \oplus B$, there exists a random variable $T$ taking at most $4$ values such that*
   $$
   I(A; B \mid T) = 0, \qquad I(T; E) = 0, \qquad I(T; E \mid A, B) = 0, \qquad H(A, B \mid T, E) = 0.
   \tag{3}
   $$

*Proof.* For (2), expand both sides into joint entropies:
$$
\begin{aligned}
\text{LHS} &= H(T, E) - H(E) - H(T, A) + H(A) - H(T, B) + H(B) + H(T, A, B, E) - H(A, B, E), \\
\text{RHS} &= \bigl[H(A) + H(B) - H(A, B)\bigr] - \bigl[H(T, A) + H(T, B) - H(T, A, B) - H(T)\bigr] \\
&\quad - \bigl[H(T) + H(E) - H(T, E)\bigr] - \bigl[H(T, A, B) + H(A, B, E) - H(T, A, B, E) - H(A, B)\bigr],
\end{aligned}
$$
and all terms cancel identically.

For (3), let the probability masses of $(A, B)$ at $(00, 01, 10, 11)$ be $(a, b, c, d)$, set $s := a + d$ and $t := b + c$, and put $u := a / s$ (or $0$ if $s = 0$) and $v := b / t$ (or $0$ if $t = 0$). Define $T \in \{1, 2, 3, 4\}$ with weights and conditional laws for $(A, B)$ given by:

| State $T$ | Weight $P_T$ | Conditional masses at $(00, 01, 10, 11)$ | Fixed coordinate |
|---:|---|---|---|
| $1$ | $u v$ | $(s,\, t,\, 0,\, 0)$ | $A = 0$ |
| $2$ | $u(1 - v)$ | $(s,\, 0,\, t,\, 0)$ | $B = 0$ |
| $3$ | $(1 - u) v$ | $(0,\, t,\, 0,\, s)$ | $B = 1$ |
| $4$ | $(1 - u)(1 - v)$ | $(0,\, 0,\, t,\, s)$ | $A = 1$ |

The weighted column sums equal $(su, tv, t(1 - v), s(1 - u)) = (a, b, c, d)$, including the boundary cases $s = 0$ or $t = 0$. In every state of $T$, one of $A, B$ is deterministic, so $I(A; B \mid T) = 0$; the parity $E = A \oplus B$ has conditional distribution $(s, t)$ independent of $T$, so $I(T; E) = 0$; $E$ is a function of $(A, B)$, so $I(T; E \mid A, B) = 0$; and knowing both $T$ (which fixes one bit) and $E = A \oplus B$ determines the other bit, giving $H(A, B \mid T, E) = 0$. $\square$

**Theorem 1 (Incompatibility of Conjectures 1 and 2).** *Assume Conjecture 1 holds. Then for every finite source $(A, B, E)$ admitting an auxiliary variable $T$ with $I(A; B \mid T) = I(T; E) = I(T; E \mid A, B) = 0$ (in particular, every source satisfying Theorem `Th:gen:classic` of Abin and Gohari 2026), we have*
$$
S(A; B \| E) = I(A; B).
$$
*In particular, Conjecture 1 implies $S(A; B \| A \oplus B) = I(A; B)$ for every binary distribution $P_{A, B}$, which is the negation of Conjecture 2.*

*Proof.* Instantiate Conjecture 1 with unprimed triple $(X, Y, Z) = (T, T, E)$ and primed triple $(X', Y', Z') = (A, B, E)$. Then $S(T; T \| E) = H(T \mid E)$, and the one-way term $S_{\mathrm{ow}}(T, T \to E \| E)$ vanishes identically because its objective is $I(U; E \mid V) - I(U; E \mid V) = 0$. Using $I(B, E; T \mid A) = H(T \mid A) - H(T \mid A, B, E)$, $I(A, E; T \mid B) = H(T \mid B) - H(T \mid A, B, E)$, and $I(T; T \mid A, B, E) = H(T \mid A, B, E)$, the remaining three penalty terms in (1) sum to
$$
H(T \mid A) + H(T \mid B) - H(T \mid A, B, E).
$$
Subtracting the right-hand side of (1) from its left-hand side and applying Lemma 1(1) gives
$$
\begin{aligned}
\text{LHS}(1) - \text{RHS}(1)
&= H(T \mid E) - S(A; B \| E) - H(T \mid A) - H(T \mid B) + H(T \mid A, B, E) \\
&= I(A; B) - S(A; B \| E) - I(A; B \mid T) - I(T; E) - I(T; E \mid A, B).
\end{aligned}
$$
When $I(A; B \mid T) = I(T; E) = I(T; E \mid A, B) = 0$, Conjecture 1 gives $I(A; B) - S(A; B \| E) \le 0$. Combining with the universal upper bound $S(A; B \| E) \le I(A; B)$ yields $S(A; B \| E) = I(A; B)$. Applying Lemma 1(2) settles every binary XOR source. $\square$

*Remark (Restricted equivalence).* Conjecture 1 holds on the subclass of common-observation XOR instances $((T, T, E), (A, B, E))$ if and only if Conjecture 2 is false. Disproving Conjecture 2 would validate Conjecture 1 on this subclass while still leaving general six-variable instances of Conjecture 1 open; conversely, any witness to Conjecture 2 immediately refutes Conjecture 1 with the exact same gap.

---

## 3. Exact Collapse of $\Psi_\Delta$ and Intrinsic Information

To attack Conjecture 2, Abin and Gohari (2026) introduced the constrained upper bound $\Psi_\Delta(A; B \| E)$ for protocols whose normalized residual XOR entropy is at least $\Delta \in [0, H(E)]$:
$$
\Psi_\Delta(A; B \| E) := \inf_{P_{J|A,B,E}} \max_{\substack{(U,V) - (A,B) - (J,E) \\ I(V;E) \le I(V;J) + H(E) - \Delta}} \bigl[I(A; B \mid J) + I(U; J \mid V) - I(U; E \mid V)\bigr],
$$
where the inner maximization may also include the XOR restriction $H(A, B \mid U, V, E) = 0$ and the binary-alphabet cardinality bound $|\mathcal{U}| \le 4$.

**Theorem 2 (Collapse of $\Psi_\Delta$ and intrinsic information).** *For every binary law $P_{A, B}$ with $E = A \oplus B$ and every $0 \le \Delta \le H(E)$,*
$$
\Psi_\Delta(A; B \| E) = I(A; B).
\tag{4}
$$
*Likewise, the Maurer–Wolf intrinsic information $I(A; B \downarrow E) := \inf_{(A, B) - E - J} I(A; B \mid J)$ equals $I(A; B)$.*

*Proof.* Taking $J$ constant in the infimum gives $\Psi_\Delta(A; B \| E) \le I(A; B)$ because $I(U; J \mid V) = 0$ and $I(U; E \mid V) \ge 0$. For the reverse inequality, fix an arbitrary finite channel $P_{J|A,B}$ (identical to $P_{J|A,B,E}$ since $E = A \oplus B$). Couple the 4-state variable $T$ from Lemma 1(2) to $(A, B, J)$ so that $T - (A, B) - J$ is a Markov chain, i.e., $I(T; J \mid A, B) = 0$.

Choose $U = T$ and $V$ constant. Then $(U, V) - (A, B) - (J, E)$ holds, $|\mathcal{U}| \le 4$, $|\mathcal{V}| = 1$, $H(A, B \mid U, V, E) = H(A, B \mid T, E) = 0$, and the constraint $I(V; E) = 0 \le I(V; J) + H(E) - \Delta$ holds for every $\Delta \in [0, H(E)]$. Since $I(A; B \mid T) = 0$,
$$
I(J; A \mid T) + I(J; B \mid T) = I(J; A, B \mid T) - I(A; B \mid J, T) \le I(J; A, B \mid T).
$$
Applying the chain rule together with $I(T; J \mid A, B) = 0$ yields
$$
\begin{aligned}
I(A; B) - I(A; B \mid J)
&= I(J; A) + I(J; B) - I(J; A, B) \\
&= \bigl[I(J; T) + I(J; A \mid T) - I(J; T \mid A)\bigr] + \bigl[I(J; T) + I(J; B \mid T) - I(J; T \mid B)\bigr] \\
&\quad - \bigl[I(J; T) + I(J; A, B \mid T) - I(J; T \mid A, B)\bigr] \\
&\le I(J; T) - I(J; T \mid A) - I(J; T \mid B) \le I(J; T).
\end{aligned}
$$
Because $I(T; E) = 0$, the inner objective at $(U, V) = (T, \text{const})$ satisfies
$$
I(A; B \mid J) + I(T; J) - I(T; E) = I(A; B \mid J) + I(J; T) \ge I(A; B)
$$
for every channel $P_{J|A,B}$. Taking the infimum over $J$ proves (4). Finally, for any degradation $(A, B) - E - J$, the Markov chain $T - E - J$ combines with $I(T; E) = 0$ to give $I(J; T) = 0$, so the same chain-rule bound yields $I(A; B \mid J) \ge I(A; B)$, matched by constant $J$. $\square$

This evaluates the upper bound $\Psi_\Delta$ identically to $I(A; B)$ across all $\Delta \in [0, H(E)]$, even if the operational subclass at a given $\Delta$ had strictly smaller capacity. Neither optimizing Eve's degradation nor supplying a stronger residual-entropy bound to $\Psi_\Delta$ can prove Conjecture 2.

---

## 4. Operational Rates for Binary XOR Sources

**Proposition 3 (One-way equality criterion).** *Let $(A, B)$ be dependent binary random variables ($I(A; B) > 0$) and $E = A \oplus B$. Then $S_{\mathrm{ow}}(A \to B \| E) = I(A; B)$ if and only if $A \perp E$. By symmetry, $S_{\mathrm{ow}}(B \to A \| E) = I(A; B)$ if and only if $B \perp E$.*

*Proof.* For any $(U, V)$ satisfying $(U, V) - A - (B, E)$, the chain rule gives the exact gap identity
$$
I(A; B) - \bigl[I(U; B \mid V) - I(U; E \mid V)\bigr] = I(V; B) + I(A; B \mid U, V) + I(U; E \mid V).
$$
Since $A, B$ are binary and dependent, the two conditional distributions $P_{B|A=0}$ and $P_{B|A=1}$ are distinct. Hence $I(V; B) = 0$ forces $P(A = 1 \mid V) = P(A = 1)$ (as the mixing weight is uniquely recovered from $P_{B|V}$), so $V \perp A$. Similarly, $I(A; B \mid U, V) = 0$ forces $A$ to be a deterministic function of $(U, V)$ (otherwise some $(u, v)$ assigns positive probability to both values of $A$, which produce distinct laws for $B$). Under both conditions, the Markov chain $(U, V) - A - E$ and $V \perp A$ give $I(U; E \mid V) = I(A; E \mid V) = I(A; E)$. Since the one-way maximum is attained, $S_{\mathrm{ow}}(A \to B \| E) = I(A; B)$ forces $I(A; E) = 0$. Conversely, if $A \perp E$, taking $U = A$ and $V$ constant attains $I(A; B)$. $\square$

**Proposition 4 (Strictly positive two-sample interactive rate).** *For every binary XOR source with masses $(a, b, c, d)$ at $(00, 01, 10, 11)$,*
$$
S(A; B \| A \oplus B) \ge (ad + bc)\left[1 - h_2\!\left(\frac{ad}{ad + bc}\right)\right]
$$
*(defined as $0$ when $ad + bc = 0$), which is strictly positive if and only if $A$ and $B$ are dependent ($ad \neq bc$).*

*Proof.* Pair $2m$ i.i.d.\ samples into $m$ pairs $((A_{2i-1}, B_{2i-1}), (A_{2i}, B_{2i}))$. Alice announces $A_{2i-1} \oplus A_{2i}$ and Bob announces $B_{2i-1} \oplus B_{2i}$ for each pair $i$, retaining only pairs where both parities equal $1$. Each pair is retained independently with probability $2(ad + bc)$. Conditional on acceptance, the first sample $(A_{2i-1}, B_{2i-1})$ has joint law proportional to $(ad, bc, bc, ad)$ on $(00, 01, 10, 11)$: both marginal bits are uniform, and Eve's two parity observations $(E_{2i-1}, E_{2i})$ are identical and independent of $A_{2i-1}$ (as well as of $B_{2i-1}$). Retaining a deterministic count $\lfloor m \cdot 2(ad + bc)(1 - \varepsilon)\rfloor$ of accepted pairs (aborting on shortage, which has exponentially small probability) and applying one-way secret-key extraction from $A$ to $B$ yields $1 - h_2(ad/(ad + bc))$ bits per accepted pair, hence $(ad + bc)[1 - h_2(ad/(ad + bc))]$ per original sample. This is strictly positive precisely when $ad \neq bc$. $\square$

**Proposition 5 (Triangle source).** *For the symmetric binary XOR source $(a, b, c, d) = (1/3, 1/3, 1/3, 0)$, both one-way capacities vanish, $S_{\mathrm{ow}}(A \to B \| E) = S_{\mathrm{ow}}(B \to A \| E) = 0$, whereas*
$$
\frac{1 + \log_2 3}{12} = 0.215413541727\ldots \;\le\; S(A; B \| E) \;\le\; I(A; B) = \log_2 3 - \frac{4}{3} = 0.251629167388\ldots
$$
*Moreover, the common-observation instance of Theorem 1 for this source uses a single fair bit $T$ (all six variables binary), with unprimed capacity $1$ and total penalty $7/3 - \log_2 3$.*

*Proof.* Given $A$, the conditional laws of $B$ and $1 - E$ coincide, so $I(U; B \mid V) = I(U; E \mid V)$ for every $(U, V) - A - (B, E)$, giving $S_{\mathrm{ow}}(A \to B \| E) = 0$; symmetry gives $S_{\mathrm{ow}}(B \to A \| E) = 0$. For the common-observation lift, setting $T = 0$ at $(0, 1)$, $T = 1$ at $(1, 0)$, and a fair coin at $(0, 0)$ satisfies $I(A; B \mid T) = I(T; E) = I(T; E \mid A, B) = 0$ with $T \sim \mathrm{Bern}(1/2)$ and $H(T \mid E) = 1$.

For the interactive lower bound, Alice partitions as many coordinates of $A^n$ as possible into triples containing one $1$ and two $0$s: she independently permutes her lists of $1$-positions and $0$-positions, forms $m = \min(N_1, \lfloor N_0 / 2 \rfloor)$ disjoint triples (so $m/n \to 1/3$ in probability), and publicly reveals her total count $N_1$ and the list of unordered triples. In each triple, call the index of Alice's unique $1$ the *pointer*. Because permuting the three coordinates within any triple preserves both the source distribution and the grouping probability, independent within-triple permutations act transitively on pointer tuples conditional on $N_1$, the unordered triples, and the ungrouped symbols; thus the pointers are i.i.d.\ uniform trits.

Inside each triple, Bob holds $0$ at the pointer and two independent fair bits at the other two positions. Bob announces the Hamming weight of his three bits in each triple:
1. **Weight $2$ (probability $1/4$):** Bob identifies the pointer uniquely, while Eve observes $E = (1, 1, 1)$ regardless of the pointer. Conditional on this branch, the pointer is a uniform secret trit independent of Eve's view and the public transcript.
2. **Weight $1$ (probability $1/2$):** Alice's pointer, Bob's unique $1$, and Eve's unique $0$ occupy the three distinct positions in a uniformly random permutation. Alice publishes an unordered pair consisting of her pointer and one uniformly chosen other position in the triple; Bob announces success if his $1$ lies in the pair (probability $1/2$, total branch probability $1/4$). On success, both parties know the pointer, while Eve's unique $0$ lies outside the published pair and both endpoints of the pair remain equiprobable given Eve's view and the transcript, yielding one uniform secret bit.
3. **Remaining outcomes (probability $1/2$):** Discarded.

Retaining deterministic counts $M_1 = \lfloor m(1/4 - \varepsilon) \rfloor$ of successful weight-$2$ triples and $M_2 = \lfloor m(1/4 - \varepsilon) \rfloor$ of successful weight-$1$ triples (aborting with exponentially small probability on shortage) yields a uniformly distributed key in $\mathcal{K} = \{0, 1, 2\}^{M_1} \times \{0, 1\}^{M_2}$ that is independent of $(E^n, F)$. Under Abin and Gohari's (2026) definition of key agreement (which permits arbitrary finite key alphabets $\mathcal{K}$ with $\frac{1}{n}\log_2 |\mathcal{K}| - R \le \delta$), outputting these $M_1$ trits and $M_2$ bits directly already achieves rate $\frac{1}{3}(\frac{1}{4}\log_2 3 + \frac{1}{4}) = (1 + \log_2 3)/12$; if binary key strings are desired, grouping $3^{M_1}$ equiprobable strings into $2^{\lfloor M_1 \log_2 3 \rfloor}$ nearly equal bins incurs only $o(1)$ rate loss. $\square$

---

## 5. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/key_agreement_ineq/explore.py
```

The script [`explore.py`](explore.py) performs four independent suites of checks:
- **Exhaustive binary simplex grid:** Enumerates all $455$ binary laws with denominator $12$, verifies the 4-state lift identities and the exact Conjecture 1 gap identity (Theorem 1) on all $455$ laws, checks $36{,}855$ binary degradation channels (confirming intrinsic information equals $I(A; B)$), and tests $2{,}275$ random $3$-output auxiliary channels (confirming both $\hat{\Psi}$ and the universal $(U, V) = (T, \text{const})$ bound for $\Psi_\Delta$).
- **Two-sample rate:** Verifies that all $390$ dependent grid laws achieve a strictly positive two-sample rate bounded above by $I(A; B)$.
- **Six-variable Conjecture 1 checks:** Tests $500$ seeded joint laws where both capacities equal conditional mutual information analytically.
- **Triangle source:** Verifies the exact joint distributions of the trit branch (mass $1/4$, Eve observes $(1, 1, 1)$) and bit branch (mass $1/4$, exact bit symmetry conditional on Eve's mask and public pair), confirming $(1 + \log_2 3)/12 = 0.215413541727\ldots \le S(A; B \| E) \le 0.251629167388\ldots$.

All assertions passed.

---

## References

- H. Abin and A. Gohari, *On the Source Model Key Agreement Problem*, *IEEE Transactions on Information Theory* 72(8):5317–5334, August 2026 ([arXiv:2502.00294](https://arxiv.org/abs/2502.00294)).
- R. Ahlswede and I. Csiszár, *Common Randomness in Information Theory and Cryptography—Part I: Secret Sharing*, *IEEE Transactions on Information Theory* 39(4):1121–1132, 1993.
- A. Gohari and V. Anantharam, *Information-Theoretic Key Agreement of Multiple Terminals—Part I*, *IEEE Transactions on Information Theory* 56(8):3973–3996, 2010.
- U. M. Maurer and S. Wolf, *Unconditionally Secure Key Agreement and the Intrinsic Conditional Information*, *IEEE Transactions on Information Theory* 45(2):499–514, 1999.
