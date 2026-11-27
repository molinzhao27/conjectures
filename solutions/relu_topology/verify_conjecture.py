"""Computational verification of Theorem 1 (Part A) and Theorem 2 (Part B)
for Beshkov's Conjecture (arXiv:2610.13931).
"""
import numpy as np


def P_network(X: np.ndarray) -> np.ndarray:
    """Evaluate the (d, d+1, d+1) ReLU network on batch X of shape (N, d)."""
    d = X.shape[1]
    h_main = np.maximum(0.0, X)
    h_last = np.maximum(0.0, 1.0 - np.sum(X, axis=1, keepdims=True))
    H = np.concatenate([h_main, h_last], axis=1)
    S = np.sum(H, axis=1, keepdims=True)
    return np.maximum(0.0, H + 2.0 * (1.0 - S))


def preimage_on_boundary_simplex(z: np.ndarray) -> np.ndarray:
    """Explicit right-inverse from Theorem 1: given z in d(Delta_0^{d+1}),
    return x in [-1, 2]^d such that P(x) == z.
    """
    d = len(z) - 1
    Sigma = float(np.sum(z))
    if abs(Sigma - 1.0) < 1e-13:
        y = z.copy()
    elif abs(Sigma) < 1e-13:
        y = np.zeros(d + 1)
        y[0] = 2.0
    else:
        A = z > 1e-14
        m = int(np.sum(A))
        assert 1 <= m <= d, f"Expected 1 <= m <= d, got m={m}"
        t = 2.0 * (1.0 - Sigma) / (2 * m - 1)
        y = np.zeros(d + 1)
        y[A] = z[A] + t

    # Now invert h(x) = y using Step 1
    S = float(np.sum(y))
    if abs(S - 1.0) < 1e-13 or abs(y[d]) < 1e-13:
        x = y[:d].copy()
    else:
        zeros = np.where(y[:d] <= 1e-13)[0]
        assert len(zeros) > 0, "Expected at least one zero among y_1..y_d"
        k = zeros[0]
        x = y[:d].copy()
        x[k] = 1.0 - S
    return x


def verify_part_a(rng: np.random.Generator) -> None:
    print("=== Verifying Part A: P(R^d) = d(Delta_0^{d+1}) for d = 1..8 ===")
    for d in range(1, 9):
        # 1. Forward check: random points in [-3, 4]^d map into d(Delta_0^{d+1})
        X = rng.uniform(-3.0, 4.0, size=(20000, d))
        Z = P_network(X)
        assert np.all(Z >= -1e-12), f"Negative output at d={d}"
        sums = np.sum(Z, axis=1)
        assert np.all(sums <= 1.0 + 1e-12), f"Sum > 1 at d={d}"
        # On boundary iff min(z) == 0 OR sum(z) == 1
        on_boundary = (np.min(Z, axis=1) <= 1e-11) | (np.abs(sums - 1.0) <= 1e-11)
        assert np.all(on_boundary), f"Interior point found at d={d}!"

        # Check support subset (-1, 2)^d
        nonzero = np.any(Z > 1e-12, axis=1)
        if np.any(nonzero):
            X_nz = X[nonzero]
            assert np.all(X_nz > -1.0 - 1e-12) and np.all(X_nz < 2.0 + 1e-12), (
                f"Support escaped (-1,2)^d at d={d}"
            )

        # 2. Backward check: sample all faces of d(Delta_0^{d+1}) and verify preimage
        max_err = 0.0
        # Facet sum(z) = 1
        samples = rng.dirichlet(np.ones(d + 1), size=500)
        # Coordinate facets z_k = 0 with sum(z) <= 1
        for k in range(d + 1):
            u = rng.dirichlet(np.ones(d + 1), size=200)  # sum <= 1 by dropping last
            z_face = np.zeros((200, d + 1))
            cols = [j for j in range(d + 1) if j != k]
            z_face[:, cols] = u[:, :d]
            samples = np.vstack([samples, z_face])
        # Include origin and vertices
        samples = np.vstack([samples, np.zeros((1, d + 1)), np.eye(d + 1)])

        for z in samples:
            x = preimage_on_boundary_simplex(z)
            assert np.all(x >= -1.0 - 1e-12) and np.all(x <= 2.0 + 1e-12)
            z_rec = P_network(x[None, :])[0]
            err = float(np.max(np.abs(z_rec - z)))
            max_err = max(max_err, err)
        assert max_err < 1e-12, f"Preimage error {max_err} at d={d}"
        print(f"  d={d}: forward boundary & support OK; backward max_err = {max_err:.2e}")


def verify_straight_ray_escape(rng: np.random.Generator) -> None:
    print("\n=== Verifying Part B Straight-Ray Escape Lemma for d = 1..6 ===")
    for d in range(1, 7):
        for trial in range(30):
            n = rng.normal(size=d + 1)
            alpha = float(rng.normal())
            for _ in range(50):
                p = rng.uniform(-1.0, 2.0, size=d + 1)
                # Check if p in R(H):
                # If p has any negative coord, p not in R(H).
                # If p > 0, p in R(H) iff n.p == alpha.
                if np.any(p < 0):
                    k = int(np.argmin(p))
                    v = np.zeros(d + 1)
                    v[k] = -1.0
                elif np.all(p > 1e-9):
                    if abs(np.dot(n, p) - alpha) < 1e-9:
                        continue
                    nn, aa = (n, alpha) if np.dot(n, p) < alpha else (-n, -alpha)
                    pos_idx = np.where(nn > 0)[0]
                    if len(pos_idx) > 0:
                        k = int(pos_idx[0])
                        v = np.zeros(d + 1)
                        v[k] = -1.0
                    else:
                        neg_idx = np.where(nn < 0)[0]
                        k = int(neg_idx[0])
                        v = np.zeros(d + 1)
                        v[k] = 1.0
                else:
                    continue

                # Verify ray p + t v misses R(H) for sampled t >= 0 (including hitting boundary!)
                ts = np.concatenate([[0.0], np.linspace(1e-6, 5.0, 40)])
                if np.any(v < 0):
                    k_hit = int(np.argmin(v))
                    if p[k_hit] > 0:
                        ts = np.concatenate([ts, [p[k_hit]]])
                for t in ts:
                    q = p + t * v
                    if np.any(q < -1e-12):
                        continue  # outside R_{\ge 0}^{d+1}
                    # Check if there exists u with R(u) = q and n.u = alpha
                    # Free coords: those with q_j == 0 have u_j <= 0; others fixed u_j = q_j
                    fixed = q > 1e-12
                    free = ~fixed
                    rem = alpha - float(np.dot(n[fixed], q[fixed]))
                    if not np.any(free):
                        assert abs(rem) > 1e-9, f"Ray hit R(H) at interior point! d={d}"
                    else:
                        # Need sum_{j in free} n_j u_j = rem with u_j <= 0
                        nf = n[free]
                        can_hit = (
                            (abs(rem) < 1e-12)
                            or (rem > 0 and np.any(nf < 0))
                            or (rem < 0 and np.any(nf > 0))
                        )
                        assert not can_hit, f"Ray hit R(H) at boundary point! d={d}"
        print(f"  d={d}: 30 random hyperplanes x 50 points -> escape ray verified OK")


if __name__ == "__main__":
    rng = np.random.default_rng(20261126)
    verify_part_a(rng)
    verify_straight_ray_escape(rng)
    print("\nALL VERIFICATIONS PASSED!")
