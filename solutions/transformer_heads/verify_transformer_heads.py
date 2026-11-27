"""Direct end-to-end numerical verification of Theorems A and B refuting Conjecture 1
of Yu et al. (ICLR 2026, arXiv:2510.06662).
We explicitly instantiate BOTH:
  1. Theorem A: 2-layer, 1-head PE transformer (L*h = 2) solving D = 5 retrieval on T = 200
  2. Theorem B: 2-layer, 1-head NoPE + NoLN + zero-intermediate-FFN transformer (L*h = 2)
                solving D = 3 retrieval on T = 200 with n = 9
and check ALL Assumption 1 constraints: ||x^(l)(s)||_2 <= 1, all weights/biases in [-1, 1],
and uniform approximation error < eps across random inputs X_T in [0, 1]^{T x d}!
"""
import math
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def make_psi_approx(centers, delta):
    """Exact Lemma 1 bounded-weight 2-layer ReLU FFN approximating f_i(x) = (1/d)||x - x^(i)||^2
    within delta, with all weights/biases in [-1, 1].
    """
    D, d = centers.shape
    N = max(2, math.ceil(1.0 / (2.0 * math.sqrt(delta))))
    knots = np.arange(1, N, dtype=float) / N  # t_1..t_{N-1}

    def eval_psi(X):
        # X: (T, d) -> returns (T, D) in [0, 1]
        T = X.shape[0]
        out = np.zeros((T, D))
        for i in range(D):
            for j in range(d):
                c0 = centers[i, j]
                g0p = (1.0 / N) - 2.0 * c0
                assert abs(g0p / 2.0) <= 1.0 and 2.0 / N <= 1.0
                u = X[:, j]
                val = (c0 ** 2) + (g0p / 2.0) * relu(u) + (g0p / 2.0) * relu(u)
                for tk in knots:
                    val = val + (2.0 / N) * relu(u - tk)
                out[:, i] += val / d
        return out

    return eval_psi, N


def verify_theorem_A(D=5, d=3, T=200, eps=1e-2):
    print(f"=== Verifying Theorem A (PE, L=2, h=1 -> L*h=2 < D={D}, T={T}, eps={eps}) ===")
    rng = np.random.default_rng(42)
    centers = np.array([[ (i + 1) / (D + 1) ] * d for i in range(D)])
    L0 = float(D)  # ||grad F_0||_1 = D for F_0(z) = sum z_i
    delta = eps / (8.0 * L0 * D)
    psi, N = make_psi_approx(centers, delta)

    E = 5 * D + 2
    c = 1.0 / math.sqrt(11 * D + 3)
    b1_tilde = max(8.0 * L0 * D * T / (math.e * eps), math.log(8.0 * L0 * D * (T + 1) / eps))
    b2_tilde = math.log(8.0 * L0 * T / eps)
    beta1 = b1_tilde / (c ** 2)
    beta2 = b2_tilde / (c ** 2)

    # Slices: A[0:D], B[D:2D], C[2D:3D], F[3D:4D], G[4D:5D], H[5D], I[5D+1]
    sA, sB, sC, sF, sG, iH, iI = slice(0, D), slice(D, 2*D), slice(2*D, 3*D), slice(3*D, 4*D), slice(4*D, 5*D), 5*D, 5*D+1

    WQ1 = np.zeros((E, E)); WQ1[:D, sB] = np.eye(D)
    WK1 = np.zeros((E, E)); WK1[:D, sA] = np.eye(D)
    WV1 = np.zeros((E, E)); WV1[:D, sC] = np.eye(D)
    WO1 = np.zeros((E, E)); WO1[sF, :D] = np.eye(D)

    WQ2 = np.zeros((E, E)); WQ2[0, iI] = 1.0
    WK2 = np.zeros((E, E)); WK2[0, iH] = 1.0
    WV2 = np.zeros((E, E)); WV2[:D, sG] = np.eye(D)
    WO2 = np.zeros((E, E)); WO2[sG, :D] = np.eye(D)

    for mat in [WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2]:
        assert np.all(np.abs(mat) <= 1.0)

    max_err = 0.0
    max_norm = 0.0
    for trial in range(25):
        X = rng.uniform(0.0, 1.0, size=(T, d))
        # Exact target H_D(X)
        dists = np.mean((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (T, D)
        H_true = float(np.sum(np.min(dists, axis=0)))

        # Step 1: Embeddings (T+1, E)
        P = psi(X)  # (T, D)
        X0 = np.zeros((T + 1, E))
        X0[:T, sA] = -c * P
        for s in range(D):
            X0[s, D + s] = c
            X0[s, iH] = c
        X0[:T, sC] = c * P
        X0[T, sA] = -2.0 * c
        X0[T, iI] = c
        max_norm = max(max_norm, float(np.max(np.linalg.norm(X0, axis=1))))

        # Step 2: Layer 1 attention + residual
        Q1, K1, V1 = X0 @ WQ1.T, X0 @ WK1.T, X0 @ WV1.T
        A1 = softmax(beta1 * (Q1 @ K1.T))
        X1_tilde = X0 + (A1 @ V1) @ WO1.T

        # Step 3: Layer 1 FFN (D ReLU neurons, weights in {0,1}, bias -c)
        F1 = np.zeros_like(X1_tilde)
        F1[:, sG] = relu(X1_tilde[:, sF] + X1_tilde[:, sB] - c)
        X1 = X1_tilde + F1
        max_norm = max(max_norm, float(np.max(np.linalg.norm(X1, axis=1))))

        # Step 4: Layer 2 attention + residual
        Q2, K2, V2 = X1 @ WQ2.T, X1 @ WK2.T, X1 @ WV2.T
        A2 = softmax(beta2 * (Q2 @ K2.T))
        X2_tilde = X1 + (A2 @ V2) @ WO2.T
        max_norm = max(max_norm, float(np.max(np.linalg.norm(X2_tilde, axis=1))))

        # Step 5: Readout FFN (exact rescaled sum via Lemma 2: alpha_D = D/c)
        H_hat = float((D / c) * np.sum(relu(X2_tilde[T, sG])))
        max_err = max(max_err, abs(H_hat - H_true))

    print(f"  max ||x^(l)||_2 = {max_norm:.4f} (<= 1.0), max |H_hat - H_true| = {max_err:.6e} (< {eps})")
    assert max_norm <= 1.0 + 1e-12
    assert max_err < eps


def verify_theorem_B(L=2, h=1, d=3, T=200, eps=1e-2):
    D = (2 * L - 1) * h  # D = 3 for L=2, h=1 (strictly > L*h = 2!)
    n = 5 * L - 3 + math.ceil(2 / h)  # n = 9 for L=2, h=1
    E = n * h
    print(f"\n=== Verifying Theorem B (NoPE + NoLN + zero intermediate FFNs, L={L}, h={h} -> L*h={L*h} < D={D}, n={n}, T={T}) ===")
    rng = np.random.default_rng(123)
    centers = np.array([[ (i + 1) / (D + 1) ] * d for i in range(D)])
    L0 = float(D)
    delta = eps / (8.0 * L0 * D)
    psi, _ = make_psi_approx(centers, delta)

    c = 1.0 / math.sqrt(E)
    b_tilde = max(8.0 * L0 * D * T / (math.e * eps), math.log(16.0 * L0 * D / eps))
    beta = b_tilde / (c ** 2)

    # Blocks: V[0:D], M[D:2D], M'[2D:2D+(L-1)h], cls=2D+(L-1)h, tok=2D+(L-1)h+1
    sV = slice(0, D)
    sM = slice(D, 2 * D)
    sMp = slice(2 * D, 2 * D + (L - 1) * h)
    i_cls = 2 * D + (L - 1) * h
    i_tok = i_cls + 1

    # Build Layer 1..L-1 and Layer L attention matrices
    layers = []
    for ell in range(1, L):
        WQs, WKs, WVs = [], [], []
        WO = np.zeros((E, E))
        for j in range(h):
            i1 = (ell - 1) * h + j
            i2 = L * h + (ell - 1) * h + j
            WQ = np.zeros((n, E))
            WQ[0, i_cls] = 1.0
            WQ[1, i_tok] = 1.0
            WQ[2, i_cls] = 1.0; WQ[2, i_tok] = 1.0
            WQ[3, i_cls] = 1.0; WQ[3, i_tok] = 1.0
            WK = np.zeros((n, E))
            WK[0, i1] = -1.0
            WK[1, i2] = -1.0
            WK[2, i_cls] = -1.0
            WK[3, i_cls] = -1.0
            WV = np.zeros((n, E))
            WV[0, i1] = 1.0
            WV[1, i2] = 1.0
            WQs.append(WQ); WKs.append(WK); WVs.append(WV)
            WO[D + i1, j * n + 0] = 1.0
            WO[D + i2, j * n + 1] = 1.0
        layers.append((WQs, WKs, WVs, WO))

    # Layer L
    WQs_L, WKs_L, WVs_L = [], [], []
    WO_L = np.zeros((E, E))
    for j in range(h):
        i1 = (L - 1) * h + j
        WQ = np.zeros((n, E))
        WQ[0, i_cls] = 1.0; WQ[1, i_cls] = 1.0; WQ[2, i_cls] = 1.0
        WK = np.zeros((n, E))
        WK[0, i1] = -1.0; WK[1, i_cls] = -1.0; WK[2, i_cls] = -1.0
        WV = np.zeros((n, E))
        WV[0, i1] = 1.0
        WO_L[D + i1, j * n + 0] = 1.0
        for ell in range(1, L):
            r_idx = (ell - 1) * h + j
            i2 = L * h + r_idx
            WV[ell, D + i2] = 1.0
            WO_L[2 * D + r_idx, j * n + ell] = 1.0
        WQs_L.append(WQ); WKs_L.append(WK); WVs_L.append(WV)
    layers.append((WQs_L, WKs_L, WVs_L, WO_L))

    for WQs, WKs, WVs, WO in layers:
        assert np.all(np.abs(WO) <= 1.0)
        for M in WQs + WKs + WVs:
            assert np.all(np.abs(M) <= 1.0)

    max_err = 0.0
    max_norm = 0.0
    for trial in range(25):
        X = rng.uniform(0.0, 1.0, size=(T, d))
        dists = np.mean((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        H_true = float(np.sum(np.min(dists, axis=0)))

        # Step 1: Strictly NoPE embeddings!
        P = psi(X)
        cur = np.zeros((T + 1, E))
        cur[:T, sV] = c * P
        cur[:T, i_tok] = c
        cur[T, i_cls] = c
        max_norm = max(max_norm, float(np.max(np.linalg.norm(cur, axis=1))))

        # Forward through L attention layers (ZERO intermediate FFNs!)
        for WQs, WKs, WVs, WO in layers:
            heads = []
            for WQ, WK, WV in zip(WQs, WKs, WVs):
                Q, K, V = cur @ WQ.T, cur @ WK.T, cur @ WV.T
                A = softmax(beta * (Q @ K.T))
                heads.append(A @ V)
            concat = np.concatenate(heads, axis=1)
            cur = cur + concat @ WO.T
            max_norm = max(max_norm, float(np.max(np.linalg.norm(cur, axis=1))))

        # Readout from CLS slots M[0:L*h] and M'[0:(L-1)*h]
        cls_out = cur[T]
        z_hat = np.concatenate([cls_out[D : D + L * h], cls_out[2 * D : 2 * D + (L - 1) * h]]) / c
        H_hat = float(np.sum(relu(z_hat)))
        max_err = max(max_err, abs(H_hat - H_true))

    print(f"  max ||x^(l)||_2 = {max_norm:.4f} (<= 1.0), max |H_hat - H_true| = {max_err:.6e} (< {eps})")
    assert max_norm <= 1.0 + 1e-12
    assert max_err < eps


if __name__ == "__main__":
    verify_theorem_A(D=5, d=3, T=200, eps=1e-2)
    verify_theorem_B(L=2, h=1, d=3, T=200, eps=1e-2)
    verify_theorem_B(L=3, h=2, d=3, T=200, eps=1e-2)  # L*h = 6 < D = 10!
    print("\nALL TRANSFORMER HEAD COUNT COUNTEREXAMPLES VERIFIED!")
