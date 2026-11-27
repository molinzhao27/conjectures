# Conjectures

Proofs, counterexamples, Python checks, and Lean 4 formalizations for 7 conjectures across machine learning, neuroscience, biology, and biophysics.

All Lean files compile with Lean `v4.24.0` + Mathlib, zero `sorry`s, and standard axioms only (`propext`, `Classical.choice`, `Quot.sound`).

## Problems

### Open conjectures settled

| Problem | Paper | Result | Proof & Check | Lean 4 |
|---|---|---|---|---|
| ReLU topological expressivity | [arXiv:2610.13931](https://arxiv.org/abs/2610.13931) (Conj. 2.16) | **Proved** Part A ($P(\mathbb{R}^d)=\partial\Delta_0^{d+1}\cong S^d$) and Part B ($(d,d,d+1)$ impossible via straight-ray escape + Alexander duality) | [solution](solutions/relu_topology/solution.md) · [check](solutions/relu_topology/verify_conjecture.py) | [ReluTopology.lean](lean_proofs/Conjectures/ReluTopology.lean) |
| Neural codes Boolean rank | [arXiv:2603.10837](https://arxiv.org/abs/2603.10837) (Conj. & Q2) | **Disproved** via $3\times 4$ counterexample; proved positive case for intersection-complete codes; Q2 bound $2\lambda-2$ sharp for all $n\ge 1$ | [solution](solutions/neural_codes/solution.md) · [check](solutions/neural_codes/verify_conjecture.py) | [NeuralCodes.lean](lean_proofs/Conjectures/NeuralCodes.lean) |
| Multilayer transformer head count | [arXiv:2510.06662](https://arxiv.org/abs/2510.06662) (ICLR 2026, Conj. 1) | **Disproved** for standard bidirectional encoders with residuals ($L=2,h=1$ solves any $D\ge 3$ with PE; NoPE + zero mid-FFNs hits $D=(2L-1)h$). CLS-only and residual-free variants remain open. | [solution](solutions/transformer_heads/solution.md) · [check](solutions/transformer_heads/verify_transformer_heads.py) | [TransformerHeads.lean](lean_proofs/Conjectures/TransformerHeads.lean) |

### Extensions, refutations, and formalizations (with recent partial prior work)

| Problem | Paper | Result & Scope | Proof & Check | Lean 4 |
|---|---|---|---|---|
| Hopfield mixed memories | [arXiv:2504.04879](https://arxiv.org/abs/2504.04879) (Conj. 1.4) | **Disproved** ("only if" direction) for dense Hopfield $F(x)=x^p/p$ ($p\ge 3$: exact $n=7$ witness for $p=3,4$, Parseval lift, witnesses verified to $p=500$) and general smooth $F$. Classical $p=2$ half was also posted in Oct 2026 (cited in writeup). | [solution](solutions/hopfield_mixed_memories/solution.md) · [check](solutions/hopfield_mixed_memories/verify_hopfield.py) | [HopfieldMixedMemories.lean](lean_proofs/Conjectures/HopfieldMixedMemories.lean) |
| Stochastic thermo skewness & kurtosis | [arXiv:2110.14540](https://arxiv.org/abs/2110.14540) (*J. Phys. A* 2022) | **Disproved** Table 1 upper-bound thresholds and the "kink" claim; proved exact $\Omega=2$ upper envelopes ($C^1$-smooth) and continuum thresholds. Lower bounds were independently proved by Li & Yeroushalmi (Nov 2026). | [solution](solutions/stoch_thermo/solution.md) · [check](solutions/stoch_thermo/verify_stoch_thermo.py) | [StochThermo.lean](lean_proofs/Conjectures/StochThermo.lean) |
| Deep softmax self-attention identifiability | [arXiv:2408.17221](https://arxiv.org/abs/2408.17221) (Conj. 3.10) | **Proved** virtual-weight identifiability for all $l\ge 1, t\ge 2$ and $C^{1,1}$ normalizers with $\mathcal{S}'(0)\neq 0$, plus $t=2$ skew collapse. Real-analytic/softmax prose case was also claimed in Konda ([arXiv:2610.09620](https://arxiv.org/abs/2610.09620), Oct 2026); Lean formalization and non-analytic case are new here. | [solution](solutions/attention_identifiability/solution.md) · [check](solutions/attention_identifiability/verify_attention.py) | [AttentionIdentifiability.lean](lean_proofs/Conjectures/AttentionIdentifiability.lean) |
| Phylogenetics OLA–SPR bound | [arXiv:2503.10169](https://arxiv.org/abs/2503.10169) (*Bull. Math. Biol.* 2026) | **Sharpened** the published $2\mathrm{ht}_\rho(T)$ bound to $\|A(p)\cup A(v_{\mathrm{top}})\|-1 \le 2\mathrm{ht}_\rho(T)-4$, with exact sharpness for $2h\le n\le 2^h$ and an $\Omega(n)$ caterpillar lower bound. | [solution](solutions/phylo_ola/solution.md) · [check](solutions/phylo_ola/verify_ola.py) | [PhyloOLA.lean](lean_proofs/Conjectures/PhyloOLA.lean) |

## Reproducing

### Lean 4
```bash
cd lean_proofs
lake exe cache get
lake build Conjectures
```
[`lean_proofs/Conjectures.lean`](lean_proofs/Conjectures.lean) imports all 7 modules and prints the axioms for all 47 main theorems. Each `.lean` file header documents exactly what is machine-checked vs. prose-only.

### Python checks
```bash
pip install numpy mpmath scipy
python solutions/relu_topology/verify_conjecture.py
python solutions/neural_codes/verify_conjecture.py
python solutions/transformer_heads/verify_transformer_heads.py
python solutions/hopfield_mixed_memories/verify_hopfield.py
python solutions/stoch_thermo/verify_stoch_thermo.py
python solutions/attention_identifiability/verify_attention.py
python solutions/phylo_ola/verify_ola.py
```
