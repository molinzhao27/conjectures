# Product Depth Bounds for Cache-Restricted Pointer Chasing

**Target paper:** Xiao Wang, *How Much Cache Does Reasoning Need? Depth–Cache Tradeoffs in KV-Compressed Transformers*, [arXiv:2604.17935](https://arxiv.org/abs/2604.17935) (v1, April 2026), Conjecture `thm:lower`, Lemma `lem:exit`, probabilistic step $(\star)$, and Open Problem 2.  
**Companion files:** [`solutions/kv_cache_depth/explore.py`](explore.py), [`solutions/kv_cache_depth/proof.pdf`](proof.pdf).

---

## Verdict

- **Conjecture `thm:lower` and Open Problem 2: PROVED** with explicit universal constants $c = 1$ and $n_0 = 4$, for **every** cache size $1 \le s \le n$ and every $k \ge 1$ with $n \ge 4k$: every exact locality-respecting $(L, H, m, p, s)$-Cache-Transformer solving $\mathrm{PC}_{n,k}$ satisfies both
  $$
  2^{mp} \ge n \qquad\text{and}\qquad Ls \ge k,
  $$
  and therefore
  $$
  L \ge \left\lceil \frac{k}{s} \right\rceil
  = \left\lceil \frac{k}{s} \right\rceil \cdot \left\lceil \frac{\log_2 n}{Hmp} \right\rceil.
  $$
- **Modeling inconsistency in the paper's narrow regime:** Because the paper explicitly decodes the output $\pi^k(1) \in [n]$ from the single final query-token state $X^{(L)}_0 \in \mathcal{Q}_p^m$ (with $|\mathcal{Q}_p| \le 2^p$), exactness requires $2^{mp} \ge n$ at every depth $L$. In the paper's narrow/low-precision regime $Hmp < \log_2 n$ (where the bandwidth factor $\lceil \log_2 n / (Hmp) \rceil > 1$ was intended to bind), we have $mp \le Hmp < \log_2 n$, so **no exact solver exists at any depth $L$**. Thus Conjecture `thm:lower` holds **vacuously** when $Hmp < \log_2 n$, and whenever $2^{mp} \ge n$ (hence $Hmp \ge mp \ge \log_2 n$ for $H \ge 1$) the bandwidth factor equals $1$ and the bound reduces to $L \ge \lceil k/s \rceil$.
- **Step $(\star)$ and Lemma `lem:exit` (Open Problem 1's proposed route): DISPROVED** for every $n \ge 16$. An explicit exact $2$-layer, $s = 1$, $H = 1$ locality-respecting circuit for $k = 2$ satisfies every hypothesis of Lemma `lem:exit` and Conjecture `thm:lower`, yet places $z_1(\pi) = \pi(1) \in T(\pi)$ on **every** permutation $\pi \in S_n$. This refutes **Lemma `lem:exit` itself** (no chain-exiting permutation exists) as well as step $(\star)$ ($1 \le (3 - 1/n)/(n - 1)$ fails for all $n \ge 16$, e.g. $1 \le 47/240$ at $n = 16$).

---

## 1. Model and Output-Capacity Constraint

In Definition `def:transformer` and `def:local` of arXiv:2604.17935, an $(L, H, m, p, s)$-Cache-Transformer operates on $n + 1$ positions $\{0, 1, \dots, n\}$: position $0$ holds a fixed query token, and position $i \in [n]$ holds the local input pair $(i, \pi(i))$ for a permutation $\pi \in S_n$. Every token representation lies in $\mathcal{Q}_p^m$ with $|\mathcal{Q}_p| \le 2^p$. At each layer $\ell \in \{1, \dots, L\}$, all $H$ heads attend only to a shared cache $S^{(\ell)} \subseteq \{0, 1, \dots, n\}$ of size $|S^{(\ell)}| \le s$. The controller $C_\ell$ is *locality-respecting*: $S^{(\ell)}(\pi)$ is a deterministic function of $\{X^{(\ell-1)}_j(\pi) : j \in T_\ell(\pi)\}$, where $T_\ell(\pi) := \{0\} \cup \bigcup_{\ell' < \ell} S^{(\ell')}(\pi)$ is the cache support up to layer $\ell - 1$ (with inactive tokens carrying forward their previous representations via the residual connection). The final answer $z_k(\pi) = \pi^k(1) \in [n]$ is decoded from the final query-token state $X^{(L)}_0 \in \mathcal{Q}_p^m$ (as specified in the proof of Proposition `prop:max-bound` and reiterated in the joint-state discussion).

**Proposition 1 (Output-capacity lower bound).** *Let $k \ge 1$ and $n \ge 4k$ (in particular $n \ge 4$). Any exact $(L, H, m, p, s)$-Cache-Transformer solving $\mathrm{PC}_{n,k}$ via $X^{(L)}_0 \in \mathcal{Q}_p^m$ satisfies*
$$
2^{mp} \ge n.
$$
*Consequently:*
1. *In the narrow regime $Hmp < \log_2 n$, no exact solver exists at any depth $L$ (so Conjecture `thm:lower` holds vacuously there).*
2. *On every feasible parameter tuple ($2^{mp} \ge n$, with $H \ge 1$), $\lceil \log_2 n / (Hmp) \rceil = 1$.*

*Proof.* Every target value $v \in [n]$ is realized as $\pi^k(1)$ for some $\pi \in S_n$: for $v = 1$, take the identity permutation $\pi = \mathrm{id}$ (since $\mathrm{id}^k(1) = 1$ for all $k \ge 1$); for $v \neq 1$, pick $k - 1$ distinct labels $a_1, \dots, a_{k-1} \in [n] \setminus \{1, v\}$ (possible because $n \ge 4k \ge k + 1$ for $k \ge 1$), form the $(k+1)$-cycle $(1, a_1, \dots, a_{k-1}, v)$, and fix all remaining labels. Because $X^{(L)}_0$ determines $\pi^k(1)$, distinct target values require distinct final query states, whence
$$
n \le |\mathcal{Q}_p^m| \le 2^{mp}.
$$
If $Hmp < \log_2 n$, then $mp \le Hmp < \log_2 n$, contradicting $2^{mp} \ge n$. Conversely, whenever $2^{mp} \ge n$ and $H \ge 1$, we have $0 < \log_2 n / (Hmp) \le 1$, so $\lceil \log_2 n / (Hmp) \rceil = 1$. $\square$

*Modeling note.* Encoding successive bits of a pointer across multiple layers cannot bypass Proposition 1 so long as the decoder reads only $X^{(L)}_0 \in \mathcal{Q}_p^m$. Making the narrow regime $Hmp < \log_2 n$ non-vacuous would require modifying the readout interface (for example, decoding from the full query history $(X^{(1)}_0, \dots, X^{(L)}_0)$ or from multiple output tokens), which defines a different model requiring separate analysis.

---

## 2. Simulation by Permutation Probes and Worst-Case Lower Bound

We now establish $Ls \ge k$ unconditionally for every $k \ge 1$, $n \ge 4k$, and $1 \le s \le n$ (without any restriction $s \le \sqrt{n}/4$).

**Lemma 2 (Locality-respecting simulation).** *Every $(L, H, m, p, s)$-Cache-Transformer with a locality-respecting controller can be simulated by a deterministic adaptive permutation-probe algorithm (with unlimited internal memory and computation, where probing index $i \in [n]$ returns $\pi(i)$) that reproduces $X^{(L)}_0(\pi)$ using at most $Ls$ probes.*

*Proof.* The simulator maintains, for each completed layer $\ell' < \ell$, the selected cache $S^{(\ell')}$ and the entering representations $\{X^{(\ell'-1)}_j : j \in S^{(\ell')}\}$ that supplied keys and values at layer $\ell'$. From this history, once the initial state $X^{(0)}_i$ of any token $i$ is known, its entire trajectory $X^{(1)}_i, \dots, X^{(\ell-1)}_i$ can be replayed deterministically: at each layer $\ell'$, token $i$'s update depends only on $X^{(\ell'-1)}_i$ and the saved cache states from $S^{(\ell')}$ (including residual carry-forward when $i$ is inactive).

At $\ell = 1$, $T_1 = \{0\}$ and $X^{(0)}_0$ is fixed, so the simulator determines $S^{(1)}$ with zero probes. Inductively, before layer $\ell$, the simulator knows $T_\ell$ and $\{X^{(\ell-1)}_j : j \in T_\ell\}$ (including ghost states of previously evicted tokens), so locality determines $S^{(\ell)}$ *before* any new probes at layer $\ell$. For each newly selected data index $i \in S^{(\ell)} \setminus T_\ell$, the simulator probes $\pi(i)$, forms $X^{(0)}_i$, and replays layers $1, \dots, \ell - 1$ to obtain $X^{(\ell-1)}_i$. All cache states for layer $\ell$ are now known, allowing the simulator to advance all tokens in $T_{\ell+1} = T_\ell \cup S^{(\ell)}$ to layer $\ell$. Since $|S^{(\ell)} \setminus T_\ell| \le |S^{(\ell)}| \le s$ at each of the $L$ layers, the total number of probes is at most $Ls$. $\square$

**Lemma 3 (Adversarial probe lower bound).** *Let $k \ge 1$ and $n \ge 4k$. Any deterministic adaptive probe algorithm that computes $\pi^k(1)$ for every $\pi \in S_n$ makes at least $k$ probes on some input.*

*Proof.* Suppose for contradiction that an exact deterministic algorithm always terminates after fewer than $k$ probes. Run it against an adversary that maintains a simple directed path $P = (v_0, v_1, \dots, v_r)$ starting at $v_0 = 1$ (initially $r = 0$, $P = (1)$) and a disjoint set $F$ of fixed points (initially $F = \varnothing$):
- If the algorithm repeats a previously probed index $i$, return the previously recorded answer.
- If the algorithm probes the current path tip $v_r$ for the first time, pick any unused label $u \in [n] \setminus (P \cup F)$, answer $\pi(v_r) = u$, and append $u$ as the new tip (incrementing $r$).
- If the algorithm probes any other new index $i \notin P \cup F$ (note every non-tip vertex of $P$ was already probed), answer $\pi(i) = i$ and add $i$ to $F$.

After $q < k$ distinct probes, the path has $r \le q$ edges and $r + 1$ vertices, and $|F| = q - r$, so $|P \cup F| = q + 1 \le k$. Because $k \ge 1$ and $n \ge 4k$, the number of unused labels is
$$
n - (q + 1) \ge 4k - k = 3k \ge k + 1 \ge k - r + 1.
$$
In particular, fresh labels always exist whenever the tip is probed.

When the algorithm terminates with $q \le k - 1$ probes, set $d := k - r \ge 1$. Since at least $d + 1$ unused labels remain, choose $d - 1$ common unused labels $w_1, \dots, w_{d-1}$ and two distinct additional unused labels $a, b$. Form two permutations $\pi_a, \pi_b \in S_n$ by extending $P$ through $w_1, \dots, w_{d-1}$, then appending $a$ (respectively $b$), closing the cycle back to $1$, and fixing every remaining label outside the cycle. Both $\pi_a$ and $\pi_b$ agree with every recorded probe answer, so the deterministic algorithm produces the same output on both. Yet $\pi_a^k(1) = a \neq b = \pi_b^k(1)$, a contradiction. $\square$

**Theorem 4 (Resolution of Conjecture `thm:lower` and Open Problem 2).** *For all $k \ge 1$, $n \ge 4k$ (hence $n \ge 4$), and all $1 \le s \le n$, every exact locality-respecting $(L, H, m, p, s)$-Cache-Transformer solving $\mathrm{PC}_{n,k}$ satisfies*
$$
L \ge \left\lceil \frac{k}{s} \right\rceil
= \left\lceil \frac{k}{s} \right\rceil \cdot \left\lceil \frac{\log_2 n}{Hmp} \right\rceil.
$$
*Hence Conjecture `thm:lower` holds with $c = 1$ and $n_0 = 4$, and extends without change to the large-cache regime $s > \sqrt{n}/4$ (resolving Open Problem 2).*

*Proof.* Combine Proposition 1 with Lemmas 2 and 3: $Ls \ge k$ gives $L \ge \lceil k/s \rceil$ by integrality of $L$, and Proposition 1 gives $\lceil \log_2 n / (Hmp) \rceil = 1$ on every feasible instance (while infeasible tuples with $2^{mp} < n$ hold vacuously). $\square$

---

## 3. Refutation of Lemma `lem:exit` and Step $(\star)$

In arXiv:2604.17935, the proposed proof route for Conjecture `thm:lower` partitions the chain into windows of size $s$ with boundaries $\tau_j = \min(js, k)$ and $\sigma_j = \tau_j - 1$. Lemma `lem:exit` asserts that for any locality-respecting algorithm with $L \le n/(8s)$, $n \ge 4k$, and $\tau_j \ge 2$, there exists a permutation $\pi_1$ and an index $t^\star \in (\tau_{j-1}, \tau_j]$ such that $z_{t^\star - 1}(\pi_1) \notin T(\pi_1)$. Its proof sketch reduces this to the probabilistic inequality
$$
\Pr\bigl[z_{\sigma_j}(\pi) \in T(\pi) \mid \mathcal{D}_{\mathrm{chain}}\bigr]
\le \frac{\mathbb{E}[|T(\pi)|]}{n - 1},
\tag{$\star$}
$$
where $\mathcal{D}_{\mathrm{chain}}$ is the event that $z_0, \dots, z_k$ are pairwise distinct under $\pi \sim \mathrm{Unif}(S_n)$.

**Theorem 5 (Counterexample refuting Lemma `lem:exit` and $(\star)$).** *For every $n \ge 16$, there exists an exact locality-respecting $(L, H, m, p, s)$-Cache-Transformer solving $\mathrm{PC}_{n,2}$ with $k = 2$, $s = 1$, $L = 2$, $H = 1$, $m = 4$, and $p = \lceil \log_2(n + 1) \rceil$ that satisfies all hypotheses of Conjecture `thm:lower` and Lemma `lem:exit` ($n \ge 4k$, $s \le \sqrt{n}/4$, $L \le n/(8s)$, and window $j = 2$ with $\tau_2 = 2 \ge 2$), yet satisfies:*
1. *$z_1(\pi) = \pi(1) \in T(\pi)$ for **every** $\pi \in S_n$, refuting **Lemma `lem:exit` itself** (since the unique candidate $t^\star \in (\tau_1, \tau_2] = (1, 2]$ is $t^\star = 2$, which requires $z_1(\pi_1) \notin T(\pi_1)$);*
2. *$\Pr[z_1(\pi) \in T(\pi) \mid \mathcal{D}_{\mathrm{chain}}] = 1 > \frac{3 - 1/n}{n - 1} = \frac{\mathbb{E}[|T(\pi)|]}{n - 1}$, refuting step $(\star)$ (at $n = 16$, $1 \le 47/240$ fails).*

*Proof.* Take alphabet $\mathcal{Q}_p = \{0, 1, \dots, n\}$ (so $|\mathcal{Q}_p| = n + 1 \le 2^p$) and $m = 4$. Initialize data tokens to $X^{(0)}_i = (i, \pi(i), 0, 0)$ for $i \in [n]$ and the query token to $X^{(0)}_0 = (0, 0, 0, 0)$. Use zero query/key projections so singleton softmax always places weight $1$ on the sole cached token, identity output projections, residual addition, and identity MLPs:
- **Layer 1:** The controller selects $S^{(1)} = \{1\}$ (depending only on $X^{(0)}_0$). The value projection maps $(x_1, x_2, x_3, x_4) \mapsto (0, 0, x_2, 0)$, writing $\pi(1)$ into the fresh zero coordinate $3$ of every token; in particular, $X^{(1)}_0 = (0, 0, \pi(1), 0)$.
- **Layer 2:** The locality-respecting controller reads coordinate $3$ of $X^{(1)}_0$ and selects $S^{(2)} = \{\pi(1)\}$. The value projection maps $(x_1, x_2, x_3, x_4) \mapsto (0, 0, 0, x_2)$, writing $\pi(\pi(1)) = \pi^2(1)$ into the fresh zero coordinate $4$ of $X^{(2)}_0$ (using that coordinate $2$ of token $\pi(1)$ remained unchanged at $\pi^2(1)$ during layer 1).

All weights lie in $\{0, 1\}$, all intermediate vectors lie in $\mathcal{Q}_p^4$, the circuit computes $\pi^2(1)$ exactly on every $\pi \in S_n$, and it attains $L = \lceil k/s \rceil = 2$ with equality. For every $\pi \in S_n$, the full cache support is $T(\pi) = \{0, 1, \pi(1)\}$.

At window $j = 2$, we have $\tau_1 = 1$, $\tau_2 = 2 \ge 2$, and $\sigma_2 = 1$. Every hypothesis of Lemma `lem:exit` holds for $n \ge 16$: $n \ge 8 = 4k$, $s = 1 \le \sqrt{n}/4$, and $L = 2 \le n/8 = n/(8s)$. Yet for the unique index $t^\star \in (\tau_1, \tau_2] = (1, 2]$, we have $z_{t^\star - 1}(\pi) = z_1(\pi) = \pi(1) \in T(\pi)$ for **every** $\pi \in S_n$. Thus no chain-exiting permutation $\pi_1$ exists, refuting Lemma `lem:exit` itself. Moreover, under $\pi \sim \mathrm{Unif}(S_n)$, $|T(\pi)| = 2$ when $\pi(1) = 1$ (probability $1/n$) and $|T(\pi)| = 3$ otherwise (probability $1 - 1/n$), so $\mathbb{E}[|T(\pi)|] = 3 - 1/n$, whereas $\Pr[z_1(\pi) \in T(\pi) \mid \mathcal{D}_{\mathrm{chain}}] = 1$. Inequality $(\star)$ asks for $1 \le (3 - 1/n)/(n - 1)$, which is false for every $n \ge 4$ and in particular for every $n \ge 16$ ($47/240 < 1$ at $n = 16$). $\square$

---

## 4. Reproducible Verification

Run from the repository root:

```sh
.venv/bin/python -B solutions/kv_cache_depth/explore.py
```

The script [`explore.py`](explore.py) uses exact integer and `fractions.Fraction` arithmetic. All checks passed:

| Check | Scope and exact verified result |
|---|---|
| Output range (Prop. 1) | All 86,544 permutation/hop pairs for $4 \le n \le 8$ and $1 \le k \le \lfloor n/4 \rfloor$ attained all $n$ outputs. |
| Adversarial completions (Lemma 3) | For $(n,k) \in \{(4,1),(8,2),(12,3),(16,4),(20,5)\}$, all 172,957 adaptive probe prefixes of length $< k$ (including repeated probes) admitted both verified permutations $\pi_a, \pi_b$ with $\pi_a^k(1) \neq \pi_b^k(1)$. |
| Parameter grid (Thm. 4) | For $4 \le n \le 64$, all admissible $k, s$, $H \in \{1, 2\}$, $m, p \in \{1, 2, 4\}$: output capacity excluded 257,048 tuples; the bandwidth factor was $1$ on all 2,952 remaining small-cache ($s \le \sqrt{n}/4$) and 128,512 remaining large-cache tuples. |
| Adaptive circuit (Thm. 5) | At $n = 16$, all 226 ordered pairs $(\pi(1), \pi^2(1))$ produced exact readout $\pi^2(1)$ with $\pi(1) \in T(\pi)$; exact probability weighting gave $1 \le 47/240$ (false) in $(\star)$. |

---

## References

- X. Wang, *How Much Cache Does Reasoning Need? Depth–Cache Tradeoffs in KV-Compressed Transformers*, [arXiv:2604.17935](https://arxiv.org/abs/2604.17935) (v1), April 2026.
