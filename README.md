# Conjectures

Proofs, counterexamples, Python/Arb checks, and Lean 4 formalizations for 11 problems across machine learning, neuroscience, biology, and biophysics.

All Lean files compile with Lean `v4.24.0` + Mathlib, zero `sorry`s, and standard axioms only (`propext`, `Classical.choice`, `Quot.sound`).

## Problems

### Open conjectures settled

| Problem | Paper | Result | Proof & Check | Lean 4 |
|---|---|---|---|---|
| ReLU topological expressivity | [arXiv:2610.13931](https://arxiv.org/abs/2610.13931) (Conj. 2.16) | **Proved** Part A ($P(\mathbb{R}^d)=\partial\Delta_0^{d+1}\cong S^d$) and Part B ($(d,d,d+1)$ impossible via straight-ray escape + Alexander duality) | [pdf](solutions/relu_topology/proof.pdf) · [md](solutions/relu_topology/solution.md) · [check](solutions/relu_topology/verify_conjecture.py) | [ReluTopology.lean](lean_proofs/Conjectures/ReluTopology.lean) |
| ReLU bottleneck topology (arbitrary width & tori) | Extends [arXiv:2610.13931](https://arxiv.org/abs/2610.13931) | **Proved** contractibility of clipped convex polyhedra at all $m$; bi-Lipschitz embedding $W^\top : g(\mathbb{R}^d) \xrightarrow{\sim} W^\top\mathbb{R}_{\ge 0}^m$ ruling out $S^d, T^d$ for all $(d,d,m)$; excluded $T^k$ ($k\ge 2$) for $(k,k+1,k+1)$ (on $\mathbb{R}^k$ & compact-image domains) and compactly supported $(k,k+1,m)$. Unrestricted $(2,3,m)$ ($m\ge 4$) remains open. | [pdf](solutions/relu_topology_ext/proof.pdf) · [md](solutions/relu_topology_ext/solution.md) · [check](solutions/relu_topology_ext/explore.py) | [ReluTopology.lean](lean_proofs/Conjectures/ReluTopology.lean) |
| Curto–Morrison CTLN dynamics | [arXiv:2301.12638](https://arxiv.org/abs/2301.12638) (*Notices AMS* 2023, OQ 11, 13, 14) | **Proved** OQ 13 (every proper source enters exact $e^{-t}$ decay, for all $\delta>0, 0<\varepsilon\le 1, \theta>0, x(0)\ge 0$); **Disproved** OQ 11 (Arb-certified Gaudi oscillation $0.00428027$ + constant-sum rigidity no-go theorem); **Disproved** OQ 14 in the strict reading (5-node TLN & Arb-certified 7-node CTLN destabilize shared attractors at midpoint; nearby deformed cycle persists and $n=5$ oriented dictionary untested). | [pdf](solutions/ctln_dynamics/proof.pdf) · [md](solutions/ctln_dynamics/solution.md) · [check](solutions/ctln_dynamics/explore.py) | [CTLNDynamics.lean](lean_proofs/Conjectures/CTLNDynamics.lean) |
| Similarity matching PCA | [arXiv:2506.06134](https://arxiv.org/abs/2506.06134) (*Neural Comput.* 2026, Conj. 1–3) | **Proved** Conj. 1 (uniform singular-value barrier & global existence); **Proved** intended Conj. 2 (corrected stationary set, rank-deficient exclusion, strict saddles; citing Pehlevan et al. 2018 & Lipshutz–Lipshutz 2026); **Disproved** Conj. 3 (kernel rotations for $n-m\ge 2$ + exact iff; trace lemma still holds via Horn log-majorization). | [pdf](solutions/sim_matching_pca/proof.pdf) · [md](solutions/sim_matching_pca/solution.md) · [check](solutions/sim_matching_pca/explore.py) | [SimMatchingPCA.lean](lean_proofs/Conjectures/SimMatchingPCA.lean) |
| ENTP `Count3` | [arXiv:2410.01600](https://arxiv.org/abs/2410.01600) (TMLR 2025, Conj. 6.1) | **Disproved** on read-only RAM via $O(n^{3/2})$-time / $O(\sqrt{n}\log n)$-bit residue blocking (plus $O(n^2\log\log n/\log n)$ / $O(1)$-word and $O(n\log n)$ / $o(n)$-bit word-RAM refinements); proved tight $\Theta(n)$-bit one-pass streaming bound. Lemma 6.2 loses its premise; causal Transformer depth remains open. | [pdf](solutions/entp_count3/proof.pdf) · [md](solutions/entp_count3/solution.md) · [check](solutions/entp_count3/explore.py) | [ENTPCount3.lean](lean_proofs/Conjectures/ENTPCount3.lean) |
| Neural codes Boolean rank | [arXiv:2603.10837](https://arxiv.org/abs/2603.10837) (Conj. & Q2) | **Disproved** via $3\times 4$ counterexample & exponential-gap family; proved positive case for intersection-complete codes; Q2 bound $2\lambda-2$ sharp for all $n\ge 1$. | [pdf](solutions/neural_codes/proof.pdf) · [md](solutions/neural_codes/solution.md) · [check](solutions/neural_codes/verify_conjecture.py) | [NeuralCodes.lean](lean_proofs/Conjectures/NeuralCodes.lean) |
| Multilayer transformer head count | [arXiv:2510.06662](https://arxiv.org/abs/2510.06662) (ICLR 2026, Conj. 1) | **Disproved** for standard bidirectional encoders with residuals ($L=2,h=1$ solves any $D\ge 3$ with PE; NoPE + zero mid-FFNs hits $D=(2L-1)h$). CLS-only and residual-free variants remain open. | [pdf](solutions/transformer_heads/proof.pdf) · [md](solutions/transformer_heads/solution.md) · [check](solutions/transformer_heads/verify_transformer_heads.py) | [TransformerHeads.lean](lean_proofs/Conjectures/TransformerHeads.lean) |

### Extensions, refutations, and formalizations (with recent partial prior work)

| Problem | Paper | Result & Scope | Proof & Check | Lean 4 |
|---|---|---|---|---|
| Hopfield mixed memories | [arXiv:2504.04879](https://arxiv.org/abs/2504.04879) (Conj. 1.4) | **Disproved** ("only if" direction) for dense Hopfield $F(x)=x^p/p$ ($p\ge 3$: exact $n=7$ witness for $p=3,4$, Parseval lift, witnesses verified to $p=500$) and general smooth $F$. Classical $p=2$ half was also posted in Oct 2026 (cited in writeup). | [pdf](solutions/hopfield_mixed_memories/proof.pdf) · [md](solutions/hopfield_mixed_memories/solution.md) · [check](solutions/hopfield_mixed_memories/verify_hopfield.py) | [HopfieldMixedMemories.lean](lean_proofs/Conjectures/HopfieldMixedMemories.lean) |
| Stochastic thermo skewness & kurtosis | [arXiv:2110.14540](https://arxiv.org/abs/2110.14540) (*J. Phys. A* 2022) | **Disproved** Table 1 upper-bound thresholds and the "kink" claim; proved exact $\Omega=2$ upper envelopes ($C^1$-smooth) and continuum thresholds. Lower bounds were independently proved by Li & Yeroushalmi (Nov 2026). | [pdf](solutions/stoch_thermo/proof.pdf) · [md](solutions/stoch_thermo/solution.md) · [check](solutions/stoch_thermo/verify_stoch_thermo.py) | [StochThermo.lean](lean_proofs/Conjectures/StochThermo.lean) |
| Deep softmax self-attention identifiability | [arXiv:2408.17221](https://arxiv.org/abs/2408.17221) (Conj. 3.10) | **Proved** virtual-weight identifiability for all $l\ge 1, t\ge 2$ and $C^{1,1}$ normalizers with $\mathcal{S}'(0)\neq 0$, plus $t=2$ skew collapse. Real-analytic/softmax prose case was also claimed in Konda ([arXiv:2610.09620](https://arxiv.org/abs/2610.09620), Oct 2026); Lean formalization and non-analytic case are new here. | [pdf](solutions/attention_identifiability/proof.pdf) · [md](solutions/attention_identifiability/solution.md) · [check](solutions/attention_identifiability/verify_attention.py) | [AttentionIdentifiability.lean](lean_proofs/Conjectures/AttentionIdentifiability.lean) |
| Phylogenetics OLA–SPR bound | [arXiv:2503.10169](https://arxiv.org/abs/2503.10169) (*Bull. Math. Biol.* 2026) | **Sharpened** the published $2\mathrm{ht}_\rho(T)$ bound to $\|A(p)\cup A(v_{\mathrm{top}})\|-1 \le 2\mathrm{ht}_\rho(T)-4$, with exact sharpness for $2h\le n\le 2^h$ and an $\Omega(n)$ caterpillar lower bound. | [pdf](solutions/phylo_ola/proof.pdf) · [md](solutions/phylo_ola/solution.md) · [check](solutions/phylo_ola/verify_ola.py) | [PhyloOLA.lean](lean_proofs/Conjectures/PhyloOLA.lean) |

## Reproducing

### Lean 4
```bash
cd lean_proofs
lake exe cache get
lake build Conjectures
```
[`lean_proofs/Conjectures.lean`](lean_proofs/Conjectures.lean) imports all 10 modules and prints the axioms for all 106 main theorems. Each `.lean` file header documents exactly what is machine-checked vs. prose-only.

### Python & Arb checks
```bash
pip install numpy mpmath scipy sympy
python solutions/relu_topology/verify_conjecture.py
python solutions/relu_topology_ext/explore.py
python solutions/ctln_dynamics/explore.py all
uv run solutions/ctln_dynamics/explore.py certificate   # 448-bit Arb proofs (python-flint 0.9)
python solutions/sim_matching_pca/explore.py
python solutions/entp_count3/explore.py
python solutions/neural_codes/verify_conjecture.py
python solutions/transformer_heads/verify_transformer_heads.py
python solutions/hopfield_mixed_memories/verify_hopfield.py
python solutions/stoch_thermo/verify_stoch_thermo.py
python solutions/attention_identifiability/verify_attention.py
python solutions/phylo_ola/verify_ola.py
```
