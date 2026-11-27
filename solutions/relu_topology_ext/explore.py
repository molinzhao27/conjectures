"""Reproducible numerical checks for Beshkov (d, d, m) and torus extensions (A1, A2, B1, B2).

Covers:
1. A1: Exact mod-2 Betti numbers of clipped affine polyhedral nerves via LP feasibility.
2. A2: Firm nonexpansiveness, piece-wise injectivity (W^T D_sigma W), bi-Lipschitz inverse
       bound, and exact cone surjectivity W^T g(R^d) = W^T R^m_{>=0} (via KKT / LP check).
3. B1: Coordinate escape rays for (k, k+1, k+1) networks avoiding (W_2 R^{k+1}_{>=0} + b_2)_+.
4. Torus: Width-four (2, 4, 4) product torus triangulation with Betti numbers (1, 2, 1).
"""

from itertools import combinations, product

import numpy as np
from scipy.optimize import linprog, minimize


def feasible(a, b, eq=None, rhs=None):
    result = linprog(
        np.zeros(a.shape[1]),
        A_ub=a,
        b_ub=b,
        A_eq=eq,
        b_eq=rhs,
        bounds=(None, None),
        method="highs",
    )
    if result.status not in (0, 2):
        raise RuntimeError(result.message)
    return result.success


def cells(w, bias, orthant):
    m, d = w.shape
    pieces = []
    for signs in product((-1, 1), repeat=m):
        signs = np.array(signs)
        a, b = -signs[:, None] * w, signs * bias
        if orthant:
            a = np.vstack((a, -np.eye(d)))
            b = np.r_[b, np.zeros(d)]
        result = linprog(
            np.r_[np.zeros(d), -1],
            A_ub=np.c_[a, np.ones(len(a))],
            b_ub=b,
            bounds=[(None, None)] * d + [(0, 1)],
        )
        if result.status == 2:
            continue
        if not result.success:
            raise RuntimeError(result.message)
        if result.x[-1] > 1e-8:
            active = signs > 0
            pieces.append((a, b, w * active[:, None], bias * active))
    return pieces


def intersect(pieces):
    d = pieces[0][0].shape[1]
    m = pieces[0][2].shape[0]
    variables = len(pieces) * d + m
    ub, eq, bounds, targets = [], [], [], []
    for j, (a, b, linear, bias) in enumerate(pieces):
        cols = slice(j * d, (j + 1) * d)
        block = np.zeros((len(a), variables))
        block[:, cols] = a
        ub.append(block)
        bounds.extend(b)
        block = np.zeros((m, variables))
        block[:, cols] = linear
        block[:, -m:] = -np.eye(m)
        eq.append(block)
        targets.extend(-bias)
    return feasible(np.vstack(ub), bounds, np.vstack(eq), targets)


def boundary_rank(lower, upper):
    indices = {face: i for i, face in enumerate(lower)}
    pivots = {}
    for simplex in upper:
        column = sum(1 << indices[f] for f in combinations(simplex, len(simplex) - 1))
        while column:
            pivot = column.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = column
                break
            column ^= pivots[pivot]
    return len(pivots)


def test_a1_nerve(label, rows, offsets, orthant):
    w = np.asarray(rows, dtype=float)
    bias = np.asarray(offsets, dtype=float)
    pieces = cells(w, bias, orthant)
    simplices = [[(i,) for i in range(len(pieces))]]
    for size in (2, 3, 4):
        previous = set(simplices[-1])
        simplices.append(
            [
                c
                for c in combinations(range(len(pieces)), size)
                if all(f in previous for f in combinations(c, size - 1))
                and intersect([pieces[i] for i in c])
            ]
        )
    ranks = [0] + [boundary_rank(a, b) for a, b in zip(simplices, simplices[1:])]
    betti = tuple(len(simplices[j]) - ranks[j] - ranks[j + 1] for j in range(3))
    a, b = w, -bias
    if orthant:
        a = np.vstack((a, -np.eye(w.shape[1])))
        b = np.r_[b, np.zeros(w.shape[1])]
    zero = feasible(a, b)
    print(f"A1 {label}: cells={len(pieces)}, contains_zero={zero}, betti={betti}")
    assert not zero and betti == (1, 0, 0)


def test_a2_embedding():
    rng = np.random.default_rng(261013931)
    max_violation = 0.0
    for _ in range(200):
        d = int(rng.integers(1, 5))
        m = int(rng.integers(1, 8))
        w = rng.normal(size=(m, d))
        if rng.random() < 0.25:
            w[0] = 0.0  # include zero rows
        b = rng.normal(size=m)
        s = rng.normal(size=(25, d))
        t = rng.normal(size=(25, d))
        y = np.maximum(0.0, s @ w.T + b)
        z = np.maximum(0.0, t @ w.T + b)
        diff = y - z
        lhs = np.sum(diff**2, axis=1)
        rhs = np.sum((diff @ w) * (s - t), axis=1)
        max_violation = max(max_violation, float(np.max(lhs - rhs)))
    assert max_violation < 1e-11

    # Check exact cone surjectivity W^T g(R^d) = W^T R^m_{>=0} via KKT/LP on 60 random (W, b, q).
    cone_hits = 0
    for _ in range(60):
        d = int(rng.integers(2, 4))
        m = int(rng.integers(3, 6))
        w = rng.normal(size=(m, d))
        b = rng.normal(size=m)
        z_seed = rng.uniform(0.0, 2.0, size=m)
        if rng.random() < 0.3:
            z_seed[0] = 0.0  # boundary ray
        q = w.T @ z_seed
        # Minimize 0.5 ||z - b||^2 s.t. z >= 0, W^T z = q.
        res = minimize(
            lambda z: 0.5 * float(np.sum((z - b) ** 2)),
            z_seed,
            jac=lambda z: z - b,
            bounds=[(0.0, None)] * m,
            constraints={"type": "eq", "fun": lambda z: w.T @ z - q, "jac": lambda z: w.T},
            method="SLSQP",
            options={"ftol": 1e-13, "maxiter": 500},
        )
        assert res.success
        z_opt = np.maximum(0.0, res.x)
        # Active rows z_opt > 1e-7 determine t via least squares; check z_opt = (W t + b)_+.
        active = z_opt > 1e-7
        if np.any(active):
            t_hat, *_ = np.linalg.lstsq(w[active], z_opt[active] - b[active], rcond=None)
        else:
            t_hat = np.zeros(d)
        # Verify exact KKT / LP feasibility for t in the active/inactive pattern.
        a_ub = w[~active] if np.any(~active) else np.zeros((1, d))
        b_ub = -b[~active] if np.any(~active) else np.ones(1)
        a_eq = w[active] if np.any(active) else None
        b_eq = (z_opt[active] - b[active]) if np.any(active) else None
        lp = linprog(np.zeros(d), A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=(None, None), method="highs")
        assert lp.success
        g_t = np.maximum(0.0, w @ lp.x + b)
        assert np.linalg.norm(w.T @ g_t - q) < 1e-6
        cone_hits += 1

    # Check bi-Lipschitz upper bound max_sigma 1 / s_min(W^T|_{range(D_sigma W)}) on 15 random nets.
    for _ in range(15):
        d, m = 2, 4
        w = rng.normal(size=(m, d))
        b = rng.normal(size=m)
        lip_bound = 0.0
        for mask in product((0, 1), repeat=m):
            active = np.array(mask, dtype=bool)
            if not np.any(active):
                continue
            wa = w[active]
            svals = np.linalg.svd(wa, compute_uv=False)
            pos = svals[svals > 1e-10]
            if len(pos):
                # Since ||W^T D_sigma W v|| / ||D_sigma W v|| >= s_min(wa), inverse slope <= 1 / s_min(wa).
                lip_bound = max(lip_bound, 1.0 / float(pos.min()))
        pts = rng.normal(size=(40, d))
        ys = np.maximum(0.0, pts @ w.T + b)
        qs = ys @ w
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                dq = float(np.linalg.norm(qs[i] - qs[j]))
                dy = float(np.linalg.norm(ys[i] - ys[j]))
                if dq > 1e-10:
                    assert dy / dq <= lip_bound + 1e-7
    print(
        f"A2 checks: 5000 firm-nonexpansive pairs (max viol {max_violation:.2e}), "
        f"{cone_hits} cone-surjectivity KKT/LP hits, 15 bi-Lipschitz nets passed"
    )


def test_b1_escape_rays():
    """Verify B1 hidden-coordinate escape rays avoid (W_2 R^n_{>=0} + b_2)_+."""
    rng = np.random.default_rng(20261126)
    rays_checked = 0
    for _ in range(25):
        n = 3
        w2 = rng.normal(size=(n, n))
        while abs(np.linalg.det(w2)) < 0.2:
            w2 = rng.normal(size=(n, n))
        w2_inv = np.linalg.inv(w2)
        b2 = rng.normal(size=n)
        for _ in range(15):
            y = rng.uniform(0.1, 3.0, size=n)
            a = w2_inv @ (y - b2)
            neg = np.where(a < -1e-6)[0]
            if len(neg) == 0:
                continue
            i = int(neg[0])
            row = w2_inv[i]
            nonpos = np.where(row <= 0)[0]
            if len(nonpos) > 0:
                j = int(nonpos[0])
                direction = np.zeros(n)
                direction[j] = 1.0
            else:
                j = 0
                direction = np.zeros(n)
                direction[j] = -1.0
            # Verify via LP that NO t >= 0 has y + t * direction in (W_2 R^n_{>=0} + b_2)_+.
            # Test both strictly positive segment and zero-crossing point (for negative ray).
            # For any candidate point q = y + t*d, q in (W_2 R^n_{>=0} + b_2)_+ iff exists h >= 0, u >= 0
            # with (W_2 h + b_2)_r = q_r when q_r > 0 and <= 0 when q_r == 0 (never possible for q_r < 0).
            if direction[j] > 0:
                # All t >= 0 keep q > 0, so membership requires W_2 h + b_2 = y + t*d, h >= 0, t >= 0.
                # Variables: (h_0..h_{n-1}, t) >= 0, W_2 h - t*d = y - b_2.
                a_eq = np.c_[w2, -direction]
                res = linprog(np.zeros(n + 1), A_eq=a_eq, b_eq=y - b2, bounds=(0, None), method="highs")
                assert res.status == 2
            else:
                # 0 <= t < y_j: q > 0 -> infeasible; t = y_j: q_j = 0, q_{r!=j} = y_r > 0 -> infeasible.
                a_eq = np.c_[w2, -direction]
                bounds = [(0, None)] * n + [(0, float(y[j]))]
                res1 = linprog(np.zeros(n + 1), A_eq=a_eq, b_eq=y - b2, bounds=bounds, method="highs")
                assert res1.status == 2
                # At t = y_j, require (W_2 h + b_2)_j <= 0 and (W_2 h + b_2)_r = y_r for r != j, h >= 0.
                other = [r for r in range(n) if r != j]
                res2 = linprog(
                    np.zeros(n),
                    A_ub=w2[j : j + 1],
                    b_ub=np.array([-b2[j]]),
                    A_eq=w2[other],
                    b_eq=y[other] - b2[other],
                    bounds=(0, None),
                    method="highs",
                )
                assert res2.status == 2
            rays_checked += 1
    assert rays_checked >= 100
    print(f"B1 checks: {rays_checked} escape rays certified disjoint from (W_2 R^3_{{>=0}} + b_2)_+ via LP")


def test_width_four_torus():
    def factor(t):
        a, b = max(0, t), max(0, 1 - t)
        return (max(0, a + 2 * (1 - a - b)), max(0, b + 2 * (1 - a - b)))

    triangles = set()
    for i, j in product(range(-1, 2), repeat=2):
        a, b = factor(i), factor(i + 1)
        c, d = factor(j), factor(j + 1)
        for vertices in ((a + c, b + c, b + d), (a + c, a + d, b + d)):
            triangles.add(tuple(sorted(vertices)))
    edges = sorted({e for t in triangles for e in combinations(t, 2)})
    vertices = sorted({(v,) for e in edges for v in e})
    rank1, rank2 = boundary_rank(vertices, edges), boundary_rank(edges, triangles)
    betti = (len(vertices) - rank1, len(edges) - rank1 - rank2, len(triangles) - rank2)
    print(f"Torus width-four product: f={(len(vertices), len(edges), len(triangles))}, betti={betti}")
    assert betti == (1, 2, 1)


if __name__ == "__main__":
    rows = [[1, -2], [-2, 1], [1, 1], [-1, 2]]
    offsets = [2, 2, -2, -1]
    for m in (3, 4):
        test_a1_nerve(f"cone d=2 m={m}", rows[:m], offsets[:m], orthant=True)
    test_a1_nerve("line d=1 m=3", [[1], [-1], [2]], [1, 1, -1], orthant=False)
    test_a1_nerve("plane d=2 m=4", rows, offsets, orthant=False)
    test_a2_embedding()
    test_b1_escape_rays()
    test_width_four_torus()
