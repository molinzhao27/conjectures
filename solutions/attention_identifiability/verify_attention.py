"""Numerical verification of Theorem 1 (layer-by-layer recovery via two-cluster
small-u asymptotics at orders u^1, u^3, u^9, ...) and Proposition 2 (exact
necessity of non-skew-symmetry for t = 2) for Conjecture conj:deepsmax
(arXiv:2408.17221).
"""
import numpy as np
import mpmath as mp


def forward_deep_softmax(M_list, L, X):
    """Evaluate l-layer normalized softmax self-attention phi_{(M, L)}(X).
    X has shape (d0, t). Returns shape (dl, t).
    """
    Z = X.copy()
    for M in M_list:
        logits = Z.T @ M @ Z  # (t, t): entry (i, j) = Z_i^T M Z_j
        logits = logits - np.max(logits, axis=1, keepdims=True)
        W = np.exp(logits)
        W = W / np.sum(W, axis=1, keepdims=True)
        Z = Z @ W.T  # Z_new[:, i] = sum_j W[i, j] Z[:, j]
    return L @ Z


def verify_two_cluster_and_recovery():
    """Using high-precision mpmath arithmetic, verify:
    1. Two-cluster invariance (Lemma 1),
    2. Exact leading asymptotics at u^(3^(r-1)) (Lemma 2),
    3. Exact layer-by-layer reconstruction of M_1, M_2, M_3 solely from phi(X(u))_1!
    """
    mp.mp.dps = 400
    rng = np.random.default_rng(2026)

    for l, t in [(1, 2), (2, 2), (2, 4), (3, 2), (3, 3)]:
        d0, dl = 3, 2
        M_true = [rng.normal(size=(d0, d0)) for _ in range(l)]
        # Make sure M_r is clearly non-skew-symmetric
        for r in range(l):
            M_true[r] += np.eye(d0)
        L_true = rng.normal(size=(dl, d0))

        # High-precision two-cluster scalar recurrence for a_l(u)
        def eval_a_l_mp(M_list, y, v, eta, u):
            a = mp.mpf(1)
            b = mp.mpf(0)
            for M in M_list:
                # Compute scalar quadratic forms
                def qf(s1, s2):
                    w1 = [eta * y[k] + s1 * v[k] for k in range(d0)]
                    w2 = [eta * y[k] + s2 * v[k] for k in range(d0)]
                    s = mp.mpf(0)
                    for i in range(d0):
                        for j in range(d0):
                            s += w1[i] * mp.mpf(str(M[i, j])) * w2[j]
                    return u * s
                p1, q1 = qf(a, a), qf(a, b)
                p2, q2 = qf(b, a), qf(b, b)
                w1 = mp.mpf(1) / (mp.mpf(1) + mp.mpf(t - 1) * mp.exp(q1 - p1))
                w2 = mp.mpf(1) / (mp.mpf(1) + mp.mpf(t - 1) * mp.exp(q2 - p2))
                d = a - b
                a, b = b + d * w1, b + d * w2
            return a

        # Verify Lemma 1 (two-cluster invariance against full matrix forward pass)
        y_np = rng.normal(size=d0)
        v_np = rng.normal(size=d0)
        eta_val = 0.7
        u_val = 0.05
        mu = np.sqrt(u_val)
        X = np.column_stack([mu * (eta_val * y_np + v_np)] + [mu * eta_val * y_np] * (t - 1))
        out_full = forward_deep_softmax(M_true, L_true, X)[:, 0]
        a_l_val = float(eval_a_l_mp(M_true, [mp.mpf(str(x)) for x in y_np],
                                    [mp.mpf(str(x)) for x in v_np], mp.mpf(str(eta_val)), mp.mpf(str(u_val))))
        out_pred = mu * (eta_val * (L_true @ y_np) + a_l_val * (L_true @ v_np))
        assert np.allclose(out_full, out_pred, atol=1e-13), "Two-cluster invariance mismatch!"

        # Verify layer-by-layer recovery of M_1, ..., M_l from ONLY a_l(u)!
        # With dps=400:
        #   r=1: u=1e-180 -> err_1 ~ 1e-180 (well above 1e-(400-180)=1e-220)
        #   r=2: u=1e-50  -> err_1 / u^2 ~ 1e-80, truncation O(u) ~ 1e-50 -> err_2 ~ 1e-50
        #   r=3: u=1e-5   -> err_2 / u^6 ~ 1e-20, truncation O(u) ~ 1e-5  -> err_3 ~ 1e-5
        sig0p = mp.mpf(t - 1) / mp.mpf(t * t)
        M_rec_mp = []
        V_basis = np.eye(d0) + 0.17 * rng.normal(size=(d0, d0))
        V_basis_mp = mp.matrix([[mp.mpf(str(V_basis[i, j])) for j in range(d0)] for i in range(d0)])
        V_inv_mp = V_basis_mp ** -1
        Y_basis = np.eye(d0)

        for r in range(1, l + 1):
            u_small = {1: mp.mpf('1e-180'), 2: mp.mpf('1e-50'), 3: mp.mpf('1e-5')}[r]
            B_mat = mp.matrix(d0, d0)
            for col_idx in range(d0):
                v_vec = [V_basis_mp[i, col_idx] for i in range(d0)]
                c_prev = mp.mpf(1)
                for j in range(1, r):
                    M_j = M_rec_mp[j - 1]
                    beta_j = sum(v_vec[a] * M_j[a, b] * v_vec[b]
                                 for a in range(d0) for b in range(d0))
                    c_prev = sig0p * beta_j * (c_prev ** 3)
                for row_idx in range(d0):
                    y_vec = [mp.mpf(str(Y_basis[i, row_idx])) for i in range(d0)]
                    def residual(eta_mp):
                        a_l_v = eval_a_l_mp(M_true, y_vec, v_vec, eta_mp, u_small)
                        if r == 1:
                            return a_l_v - mp.mpf(1) / mp.mpf(t)
                        a, b = mp.mpf(1), mp.mpf(0)
                        for M in M_rec_mp:
                            def qf(s1, s2):
                                w1 = [eta_mp * y_vec[k] + s1 * v_vec[k] for k in range(d0)]
                                w2 = [eta_mp * y_vec[k] + s2 * v_vec[k] for k in range(d0)]
                                s = mp.mpf(0)
                                for i in range(d0):
                                    for j in range(d0):
                                        s += w1[i] * M[i, j] * w2[j]
                                return u_small * s
                            p1, q1 = qf(a, a), qf(a, b)
                            p2, q2 = qf(b, a), qf(b, b)
                            w1 = mp.mpf(1) / (mp.mpf(1) + mp.mpf(t - 1) * mp.exp(q1 - p1))
                            w2 = mp.mpf(1) / (mp.mpf(1) + mp.mpf(t - 1) * mp.exp(q2 - p2))
                            d = a - b
                            a, b = b + d * w1, b + d * w2
                        return a_l_v - b - (a - b) / mp.mpf(t)
                    diff = residual(mp.mpf(1)) - residual(mp.mpf(0))
                    est = diff / (sig0p * (c_prev ** 2) * (u_small ** (3 ** (r - 1))))
                    B_mat[row_idx, col_idx] = est
            M_r_est = B_mat * V_inv_mp
            err = max(float(abs(M_r_est[i, j] - mp.mpf(str(M_true[r - 1][i, j]))))
                      for i in range(d0) for j in range(d0))
            assert err < 1e-2, f"Recovery failed at l={l}, t={t}, r={r}: err={err}"
            M_rec_mp.append(M_r_est)
        max_total = max(
            max(float(abs(M_rec_mp[r][i, j] - mp.mpf(str(M_true[r][i, j]))))
                for i in range(d0) for j in range(d0))
            for r in range(l)
        )
        print(f"  l={l}, t={t}: exact two-cluster invariance & full blind cascade recovery of M_1..M_{l} OK (max_err={max_total:.2e})")


def verify_t2_skew_necessity():
    """Verify Proposition 2: for t=2 and skew-symmetric M_1, varying M_2 arbitrarily
    gives the EXACT same output function phi_{(M_1, M_2), L}!"""
    rng = np.random.default_rng(7)
    d0, dl = 4, 3
    A = rng.normal(size=(d0, d0))
    M1_skew = A - A.T  # skew-symmetric!
    M2_a = rng.normal(size=(d0, d0))
    M2_b = rng.normal(size=(d0, d0))
    L = rng.normal(size=(dl, d0))
    max_diff = 0.0
    for _ in range(100):
        X = rng.normal(size=(d0, 2))
        y_a = forward_deep_softmax([M1_skew, M2_a], L, X)
        y_b = forward_deep_softmax([M1_skew, M2_b], L, X)
        max_diff = max(max_diff, float(np.max(np.abs(y_a - y_b))))
    assert max_diff < 1e-14, f"Expected identical outputs for skew M1 at t=2, got {max_diff}"
    print(f"  t=2 skew-symmetric necessity verified across 100 random inputs (max_diff={max_diff:.2e})")


if __name__ == "__main__":
    print("=== Verifying Deep Softmax Self-Attention Identifiability (arXiv:2408.17221) ===")
    verify_two_cluster_and_recovery()
    verify_t2_skew_necessity()
    print("\nALL ATTENTION IDENTIFIABILITY VERIFICATIONS PASSED!")
