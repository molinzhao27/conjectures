# Refutation of ENTP Conjecture 6.1 (`Count3`) and Exact Streaming Bounds

**Target paper:** Ethan Ewer, Daewon Chae, Thomas Zeng, Jinkyu Kim, and Kangwook Lee, *ENTP: Encoder-only Next Token Prediction*, [arXiv:2410.01600](https://arxiv.org/abs/2410.01600) (v3, Feb 2025; published at TMLR 2025), Section 6, Conjecture 6.1 (`con:ts2`).  
**Companion files:** [`lean_proofs/Conjectures/ENTPCount3.lean`](../../lean_proofs/Conjectures/ENTPCount3.lean), [`solutions/entp_count3/explore.py`](explore.py), [`solutions/entp_count3/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture 6.1 (`con:ts2`): FALSE** on the paper's own read-only random-access model, even with zero preprocessing before reading $x_n$ and with space measured in bits.
- **Simple counterexample (Theorem 2):** Partitioning residues modulo $n$ into blocks of width $B = \lceil\sqrt{n}\rceil$ computes $\operatorname{Count3}(x_1,\dots,x_n)$ deterministically in $O(n^{3/2})$ time and $O(\sqrt{n})$ words ($O(\sqrt{n}\log n)$ bits) of workspace, strictly violating both horns of Conjecture 6.1.
- **Word-RAM refinements (Theorem 3):** Using packed counters on $w = \lceil\log_2(n+1)\rceil$-bit words (with inputs fitting in $O(1)$ words) gives:
  1. $O(n^2 \log\log n / \log n)$ time and $O(1)$ words (strictly beating Algorithm 1), and
  2. $O(n \log n)$ time and $O(n \log\log n / \log n) = o(n)$ bits (strictly beating Algorithm 2 in space at a $\log n$ time factor).
  All three algorithms lie on the trade-off curve $T \cdot S = \tilde{O}(n^2)$ bits, matching Beame–Clifford–Machmouchi's $F_2$ trade-off; the dichotomy fails, while $T \cdot S = \tilde{\Omega}(n^2)$ remains open as a natural repaired conjecture.
- **One-pass streaming (Theorem 4):** Under a strict *one-pass* streaming model with known length $n$, $\Theta(n)$ persistent bits are both necessary (via an explicit INDEX reduction for every $n \ge 12$, giving $\ge \lfloor n/12\rfloor$ deterministic bits) and sufficient (a $(2n-2)$-bit stars-and-bars unary histogram with an $O(n)$ post-$x_n$ pass). Note that $\Theta(n)$ bits is still $\Theta(n/\log n) = o(n)$ words.
- **Impact on Lemma 6.2:** Lemma 6.2 assumes Conjecture 6.1 as its premise, so its proof loses its hypothesis; however, the fixed-width causal Transformer depth question ($L = \Omega(n)$) remains **open** (the one-pass streaming lower bound does not transfer to a decoder that can query its entire KV cache after seeing $x_n$). Lemma 6.3 and Theorems 4.1–4.2 are unaffected.

---

## 1. Exact Problem Statement and Histogram Identity

For positive integers $x_1, \dots, x_n$, the paper defines
$$
\operatorname{Count3}(x_1, \dots, x_n)
= \bigl|\{(i, j) \in [n]^2 : x_i + x_j + x_n \equiv 0 \pmod n\}\bigr| \bmod n.
$$
As confirmed by both Algorithm 1 and Algorithm 2 in arXiv:2410.01600, pairs $(i,j)$ are ordered, $i=j$ is allowed, and the last index $n$ is included.

Let $t := x_n \bmod n$, let $h(a) := |\{i \in [n] : x_i \equiv a \pmod n\}|$ for $a \in \{0, \dots, n-1\}$, and let $c(a) := (-t - a) \bmod n$ be the unique complementary residue in $\{0,\dots,n-1\}$ with $a + c(a) + t \equiv 0 \pmod n$. Note that $c(c(a)) = a$. Grouping matching pairs by $a = x_i \bmod n$ immediately yields:

**Proposition 1 (Histogram identity).**
$$
\operatorname{Count3}(x_1, \dots, x_n) = \biggl(\sum_{a=0}^{n-1} h(a)\,h(c(a))\biggr) \bmod n.
$$

**Conjecture 6.1 (`con:ts2`, arXiv:2410.01600 v3).** *Any algorithm $\mathcal{A}$ computing $\operatorname{Count3}(x_1, \dots, x_n)$ must satisfy at least one of:*
1. *$\mathcal{A}$ requires $\Omega(n^2)$ time with access to $x_n$;*
2. *$\mathcal{A}$ requires $\Omega(n)$ space storing values unique to $n$.*

In the paper's own model (established by calling the double-loop Algorithm 1 "$O(1)$ space" and the histogram Algorithm 2 "$O(n)$ space"), the input array $x_1,\dots,x_n$ is read-only and not counted toward workspace, while space is measured in $O(\log n)$-bit words (or bits).

---

## 2. Refutation via Residue Blocks

**Theorem 2 (Residue-block algorithm).** *For any integer block width $1 \le B \le n$, there is a deterministic read-only algorithm that does no work before reading $x_n$, uses $B$ counters plus $O(1)$ scalar variables ($O(B)$ words, i.e., $O(B \log n)$ bits), and computes $\operatorname{Count3}(x_1,\dots,x_n)$ in $O(n \lceil n/B \rceil)$ time.*

*Proof.* Read $t = x_n \bmod n$ and initialize $\mathrm{total} \leftarrow 0$. Partition $\{0,\dots,n-1\}$ into $M = \lceil n/B \rceil$ disjoint intervals $I_k = [kB, \min((k+1)B, n))$ for $k = 0, \dots, M-1$. For each block $I_k$:
1. Allocate an array $\mathrm{cnt}[0 \dots |I_k|-1]$ of zeros.
2. Scan $i = 1, \dots, n$: if $a = x_i \bmod n \in I_k$, increment $\mathrm{cnt}[a - kB]$. After this scan, $\mathrm{cnt}[a - kB] = h(a)$ for all $a \in I_k$.
3. Scan $j = 1, \dots, n$: let $b = (-t - x_j) \bmod n = c(x_j \bmod n)$. If $b \in I_k$, add $\mathrm{cnt}[b - kB]$ to $\mathrm{total}$.

Because $c$ is an involution, each $b \in I_k$ is hit by exactly $h(c(b))$ indices $j \in [n]$, so step 3 adds $\sum_{b \in I_k} h(b) h(c(b))$ to $\mathrm{total}$. Summing over the disjoint partition $\{I_k\}$ gives $\sum_{a=0}^{n-1} h(a) h(c(a))$, whose remainder modulo $n$ equals $\operatorname{Count3}(x_1,\dots,x_n)$ by Proposition 1. Each block performs two scans of length $n$ and clears at most $B$ counters, taking $O(n)$ time per block and $O(n \lceil n/B \rceil)$ time in total. $\square$

Choosing $B = \lceil \sqrt{n} \rceil$ yields:
- **Time (entirely after reading $x_n$):** $O(n^{3/2}) = o(n^2)$, violating (i).
- **Space:** $O(\sqrt{n})$ words $= O(\sqrt{n}\log n) = o(n)$ bits, violating (ii) under both word and bit accounting.

---

## 3. Word-RAM Refinements and the Time–Space Trade-Off

On a standard word RAM with word size $w = \lceil \log_2(n+1)\rceil$ bits (assuming inputs fit in $O(1)$ words and standard shift/mask/arithmetic operations take $O(1)$ time, with $2w$-bit intermediate products or modular reductions), we can push both endpoints of the trade-off strictly past Algorithms 1 and 2 of the paper:

**Theorem 3 (Packed-counter endpoints).**
1. **$O(1)$ words, subquadratic time:** Set $f = \lceil \log_2(w+1)\rceil$, $S = 2^f - 1 \ge w$, and $B = \lfloor w/f \rfloor = \Theta(\log n / \log\log n)$. Pack $B$ saturating $f$-bit counters into a single $w$-bit word (incrementing a field only while strictly below $S$, preventing carries). In each of the $\lceil n/B \rceil$ blocks, scan $x_i$ to fill the packed word, scan $x_j$ to add every *unsaturated* partner count, and for each *saturated* residue $a$ in the block, run one extra linear scan over $x$ to compute $h(a)$ and $h(c(a))$ directly and add $h(a)h(c(a))$ (handling $c(a) = a$ automatically). Since at most $\lfloor n/S \rfloor \le n/w$ residues globally have $h(a) \ge S$, the total rescan cost is $O(n^2 / \log n)$ and the block scans cost $O(n^2 \log\log n / \log n)$, using $O(1)$ words.
2. **$O(n \log n)$ time, $o(n)$ bits:** Set $B = \max(1, \lfloor n/w \rfloor)$, $f = \min(w, \lceil \log_2(w^2+1)\rceil) = \Theta(\log\log n)$, and $S = 2^f - 1 \ge w^2$. Pack $B$ saturating $f$-bit counters into $O(n \log\log n / \log n)$ bits. For each of the $O(\log n)$ blocks, build the packed counters, collect the sorted list `heavy` of saturated offsets in that block (at most $n/S \le n/w^2$ globally, taking $O(n/\log n)$ bits), compute exact counts for `heavy` via one scan using binary search on `heavy`, and scan $x_j$ looking up either the packed counter or (if saturated) `heavy` via binary search. Across all blocks, each $x_i$ falls in one block and each $c(x_j \bmod n)$ falls in one block, so total binary searches are at most $2n$, giving $O(n \log n)$ total time and $O(n \log\log n / \log n) = o(n)$ bits.

**Connection to $F_2$ and repaired conjecture.** Letting $A = \{x_i \bmod n\}$ and $B = \{(-t - x_j) \bmod n\}$ as multisets, $\sum_a h_A(a) h_B(a) = \frac{1}{2}(F_2(A \uplus B) - F_2(A) - F_2(B))$. Every algorithm above sits on $T \cdot S = \tilde{O}(n^2)$ bits, matching Beame–Clifford–Machmouchi's exact $F_k$ trade-off ([arXiv:1309.3690](https://arxiv.org/abs/1309.3690)). While the dichotomy of Conjecture 6.1 is false, a time–space product lower bound $T \cdot S = \tilde{\Omega}(n^2)$ remains a natural repaired conjecture.

---

## 4. Exact One-Pass Streaming Complexity ($\Theta(n)$ Bits)

The only reading under which an $\Omega(n)$-bit bound holds is **one-pass streaming** (where $x_1, \dots, x_{n-1}$ cannot be re-read once $x_n$ arrives) with space counted in **bits** (note that $\Theta(n)$ bits is still $\Theta(n/\log n) = o(n)$ words).

**Theorem 4 (Tight one-pass streaming bound for known length $n$).**
1. **Lower bound ($n \ge 12$):** Set $m = \lfloor n/12 \rfloor \ge 1$ and $F = 4m + 1$. Given any bitstring $b = (b_0, \dots, b_{m-1}) \in \{0,1\}^m$, define the prefix of length $n-1$:
   $$
   x_i = \begin{cases}
   2(m + 1 + i) & \text{if } 0 \le i < m \text{ and } b_i = 1, \\
   F & \text{otherwise.}
   \end{cases}
   $$
   For any query $q \in \{0, \dots, m-1\}$, set $k = m + 1 + q \in [m+1, 2m]$ and append $x_n = n - k$. Every entry lies in $\{1, \dots, n-1\}$. Moreover:
   - *Prefix–prefix pairs ($i, j < n$):* $x_i, x_j \in [2m+2, 4m+1]$, so $x_i + x_j \in [4m+4, 8m+2]$. Thus $x_i + x_j + (n - k) \in [n + 2m + 4, n + 7m + 1] \subset (n, 2n)$, which is never $0 \bmod n$.
   - *Last–last pair ($i = j = n$):* $3(n - k) \equiv -3k \pmod n$, and $0 < 3k \le 6m \le n/2 < n$, so $3(n-k) \not\equiv 0 \pmod n$.
   - *Prefix–last pairs:* $x_i + 2(n - k) \equiv 0 \pmod n \iff x_i = 2k$, which holds if and only if $i = q$ and $b_q = 1$ (since $F = 4m+1$ is odd and $2(m+1+i)$ is strictly increasing in $i$). Counting both $(q, n)$ and $(n, q)$ gives
     $$
     \operatorname{Count3}(x_1, \dots, x_{n-1}, n - k) = 2 b_q \bmod n = 2 b_q.
     $$
   Hence distinct bitstrings $b \neq b'$ produce distinct outputs on some query $q$, forcing at least $2^{\lfloor n/12 \rfloor}$ states ($\ge \lfloor n/12 \rfloor$ bits) for any deterministic one-pass streaming algorithm (and $\Omega(n)$ bits for bounded-error randomized algorithms by the communication complexity of `INDEX`).
2. **Upper bound ($n \ge 1$):** Maintain the residue histogram $(h_0, \dots, h_{n-1})$ of the first $n-1$ elements ($\sum_{a=0}^{n-1} h_a = n - 1$). Its stars-and-bars unary code (`1^{h_0} 0 1^{h_1} 0 ... 0 1^{h_{n-1}}`) uses exactly $(n-1) + (n-1) = 2n - 2$ bits (equivalently, the indicator of the $n-1$ separator positions in $\{0, \dots, 2n-3\}$). When $x_n$ arrives, decoding the histogram and evaluating Proposition 1 takes $O(n)$ time.

*(Scope note: a causal decoder must output $\operatorname{Count3}$ for every prefix $m \le n$ modulo $m$, which is stronger than known-length streaming; conversely, the streaming lower bound does not bound decoder depth because a decoder retains random access to its KV cache after seeing $x_n$.)*

---

## 5. Verification, Formalization, and Attribution

- **Lean 4 formalization** ([`lean_proofs/Conjectures/ENTPCount3.lean`](../../lean_proofs/Conjectures/ENTPCount3.lean), Lean `v4.24.0` + Mathlib, zero `sorry`, standard axioms only): machine-checks `count3_eq_hist_sum` (Proposition 1), `blockedCount3_eq_count3` and `sqrt_block_bounds` (Theorem 2), `indexInstance_bounds`, `index_reduction_correct`, and `streaming_state_lower_bound` (Theorem 4, lower bound), and `unaryCode_injective` (Theorem 4, upper bound).
- **Python verification** ([`solutions/entp_count3/explore.py`](explore.py)): verifies `blocked`, `packed`, `near_linear`, and `streaming_unary` against `definition` on all 50,069 sequences for $n = 1, \dots, 6$, 260 saturation stress tests up to $n = 513$, 100 explicit multi-field / multi-heavy packing tests with no-carry and $\le 2n$ binary-search assertions, and 4,290 `INDEX` queries.
- **Literature note:** Checked against arXiv v3 (4 Feb 2025); the paper was published in TMLR 2025 (OpenReview camera-ready was behind a bot check and not inspected directly). No prior refutation of Conjecture 6.1 was found in literature searches or citing papers as of November 2026. The blocking/packing technique itself is elementary and aligns with Beame–Clifford–Machmouchi ([arXiv:1309.3690](https://arxiv.org/abs/1309.3690)).
