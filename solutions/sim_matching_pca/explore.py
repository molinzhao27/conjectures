"""Reproduce checks and stress tests for the three conjectures in arXiv:2506.06134."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from scipy.stats import ortho_group


def matrix_power(a, exponent):
    values, basis = np.linalg.eigh((a + a.T) / 2)
    if exponent < 0:
        assert values.min() > 0
    return (basis * np.maximum(values, 0) ** exponent) @ basis.T


def flow(w, c):
    return -4 * w + 4 * matrix_power(w @ c @ w.T, -1 / 3) @ w @ c


def cost(w, c):
    return 2 * np.sum(w**2) - 3 * np.trace(matrix_power(w @ c @ w.T, 2 / 3))


def check_flow():
    rng = np.random.default_rng(61034)
    c = np.diag([4.0, 2.0, 1.0])
    for m in (2, 3):
        basis = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        left = np.linalg.qr(rng.normal(size=(m, m)))[0]
        w = left @ np.diag(np.geomspace(2.0, 0.01, m)) @ basis[:m]
        dw = flow(w, c)
        residual = (
            dw @ w.T
            + w @ dw.T
            + 8 * w @ w.T
            - 8 * matrix_power(w @ c @ w.T, 2 / 3)
        )
        assert_allclose(residual, 0, atol=1e-11)
        sol = solve_ivp(
            lambda t, y: flow(y.reshape(m, 3), c).ravel(),
            (0, 8),
            w.ravel(),
            t_eval=np.linspace(0, 8, 161),
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        assert sol.success
        traj = sol.y.T.reshape(-1, m, 3)
        singular = np.linalg.svd(traj, compute_uv=False)
        assert singular.min() >= 0.01 - 1e-10
        assert singular.max() <= 4 + 1e-9
        print(
            f"C1 {m}x3: Gram residual {np.linalg.norm(residual):.2e}, "
            f"sampled singular values [{singular.min():.8f}, {singular.max():.8f}]"
        )

    # Hard stress tests across dimensions (including square m=n and nearly singular W0).
    worst_lo, worst_hi = np.inf, -np.inf
    for trial in range(20):
        m, n = [(2, 5), (3, 3), (4, 7), (5, 5), (1, 4)][trial % 5]
        v = ortho_group.rvs(n, random_state=rng)
        lam = np.sort(rng.uniform(0.05, 5.0, n))[::-1]
        c_mat = v @ np.diag(lam) @ v.T
        s = np.geomspace(rng.uniform(1.0, 8.0), 10 ** rng.uniform(-5, -2), m)
        u_mat = ortho_group.rvs(m, random_state=rng) if m > 1 else np.eye(1)
        w0 = u_mat @ np.diag(s) @ ortho_group.rvs(n, random_state=rng)[:m]
        lo_b = min(s.min(), lam.min())
        hi_b = max(s.max(), lam.max())
        sol = solve_ivp(
            lambda t, y: flow(y.reshape(m, n), c_mat).ravel(),
            (0, 4),
            w0.ravel(),
            method="DOP853",
            rtol=1e-10,
            atol=1e-13,
            dense_output=True,
        )
        assert sol.success
        ts = np.linspace(0, 4, 401)
        sv = np.array([np.linalg.svd(sol.sol(t).reshape(m, n), compute_uv=False) for t in ts])
        worst_lo = min(worst_lo, (sv.min(1) - lo_b).min() / lo_b)
        worst_hi = max(worst_hi, (sv.max(1) - hi_b).max() / hi_b)
    assert worst_lo > -1e-7 and worst_hi < 1e-7
    print(f"C1 stress (20 hard runs): min rel slack {worst_lo:.2e}, max rel excess {worst_hi:.2e}")


def check_landscape():
    v = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    p = np.eye(2)[::-1]
    d = np.diag([2.0, 1.0])
    c = v @ d @ v.T
    w = (p @ d @ p.T)[:1, :1] @ (p @ v)[:, :1].T
    error = np.linalg.norm(flow(w, c))
    assert_allclose(error, 4 * (2 ** (2 / 3) - 1))
    print(f"C2 printed set: gradient norm {error:.9f}")

    directions = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    tie_costs = [cost(row[None], np.eye(2)) for row in directions]
    assert_allclose(tie_costs, -1)
    print("C2 cutoff tie: costs", np.round(tie_costs, 9))

    w = np.array([[1.0, 0.0]])
    c = np.diag([1.0, 3.0])
    h = 1e-4
    curvature = [
        (cost(w + h * z, c) + cost(w - h * z, c) - 2 * cost(w, c)) / h**2
        for z in np.eye(2)
    ]
    assert_allclose(curvature, [8 / 3, -8], atol=5e-7)
    print("C2 saddle: radial and exchange curvatures", np.round(curvature, 7))

    w = np.diag([2.0, 0.0])
    c = np.diag([2.0, 1.0])
    change = cost(w + np.diag([0.0, 1e-3]), c) - cost(w, c)
    assert_allclose(change, 2e-6 - 3e-4)
    print(f"C2 rank-one boundary: cost change {change:.9f}")

    # 100 random critical points with forced eigenvalue ties.
    rng = np.random.default_rng(250606134)
    for _ in range(100):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, n + 1))
        lam = rng.choice([1.0, 2.0, 3.0, 5.0], size=n)
        v_mat = ortho_group.rvs(n, random_state=rng)
        c_mat = v_mat @ np.diag(lam) @ v_mat.T
        q_mat = np.zeros((n, n))
        for val in np.unique(lam):
            s_idx = np.where(lam == val)[0]
            rot = ortho_group.rvs(len(s_idx), random_state=rng) if len(s_idx) > 1 else np.eye(1)
            q_mat[:, s_idx] = v_mat[:, s_idx] @ rot
        idx = rng.choice(n, size=m, replace=False)
        vm, sig = q_mat[:, idx], lam[idx]
        u_mat = ortho_group.rvs(m, random_state=rng) if m > 1 else np.eye(1)
        w_mat = u_mat @ np.diag(sig) @ vm.T
        assert np.linalg.norm(flow(w_mat, c_mat)) < 1e-9
        assert abs(cost(w_mat, c_mat) + np.sum(sig**2)) < 1e-9
        top = np.sort(lam)[::-1][:m]
        is_glob = abs(np.sum(sig**2) - np.sum(top**2)) < 1e-12
        alpha_i = int(np.argmin(sig))
        alpha = sig[alpha_i]
        comp = [k for k in range(n) if k not in idx]
        if comp:
            beta_k = max(comp, key=lambda k: lam[k])
            beta = lam[beta_k]
            z = np.outer(u_mat[:, alpha_i], q_mat[:, beta_k])
            hz = (cost(w_mat + h * z, c_mat) + cost(w_mat - h * z, c_mat) - 2 * cost(w_mat, c_mat)) / h**2
            assert abs(hz - 4 * (1 - beta / alpha)) < 1e-4
            assert (beta > alpha) == (not is_glob)
        zr = np.outer(u_mat[:, alpha_i], vm[:, alpha_i])
        hr = (cost(w_mat + h * zr, c_mat) + cost(w_mat - h * zr, c_mat) - 2 * cost(w_mat, c_mat)) / h**2
        assert abs(hr - 8 / 3) < 1e-4
    print("C2 stress: 100 random critical points (with ties) matched exact Hessian & global iff")


def check_diagonalization():
    rot = np.array([[0.6, -0.8], [0.8, 0.6]])
    for label, a, u in (
        ("sign", np.array([[1.0, 0.0]]), np.diag([-1.0, 1.0])),
        ("active tie", np.eye(2), rot),
        ("kernel", np.array([[1.0, 0.0, 0.0]]), block_diag(1.0, rot)),
        ("simple", np.diag([2.0, 1.0]), rot),
    ):
        n = u.shape[0]
        d = np.diag(np.arange(n, 0, -1, dtype=float) ** 3)
        b = a @ u
        gram = b.T @ b
        h = b.T @ matrix_power(b @ d @ b.T, -1 / 3) @ b
        root = np.sqrt(d)
        assert_allclose(root @ h @ root, matrix_power(root @ gram @ root, 2 / 3), atol=1e-12)
        off = np.linalg.norm(h - np.diag(h.diagonal()))
        if label == "simple":
            assert off > 0.1
        else:
            assert off < 1e-12
        print(f"C3 {label}: diagonal {h.diagonal().round(7)}, off-diagonal norm {off:.2e}")

    rng = np.random.default_rng(7)
    # Paper's dimensions: kernel rotations give diagonal H for every Lambda_1, Lambda_2.
    for m, n in [(5, 10), (10, 20), (10, 50)]:
        a_vec = np.sort(rng.uniform(1, 5, m))[::-1]
        l1 = np.hstack([np.diag(a_vec), np.zeros((m, n - m))])
        d_mat = np.diag(np.sort(rng.uniform(1, 9, n))[::-1])
        u_mat = np.eye(n)
        u_mat[m:, m:] = ortho_group.rvs(n - m, random_state=rng)
        h_mat = u_mat.T @ l1.T @ matrix_power(l1 @ u_mat @ d_mat @ u_mat.T @ l1.T, -1 / 3) @ l1 @ u_mat
        assert np.linalg.norm(h_mat - np.diag(np.diag(h_mat))) < 1e-11
    # Exact iff test (250 trials) + Horn log-majorization trace bound (500 trials).
    for trial in range(250):
        n = int(rng.integers(2, 6))
        m = int(rng.integers(1, n + 1))
        a_vec = rng.choice([1.0, 2.0, -2.0, 3.0], size=m)
        l1 = np.hstack([np.diag(a_vec), np.zeros((m, n - m))])
        d_mat = np.diag(np.sort(rng.uniform(1, 9, n))[::-1])
        e_mat = l1.T @ l1
        if trial % 2 == 0:
            ev = np.diag(e_mat)
            u_mat = np.zeros((n, n))
            for val in np.unique(ev):
                s_idx = np.where(ev == val)[0]
                r_mat = ortho_group.rvs(len(s_idx), random_state=rng) if len(s_idx) > 1 else np.eye(1)
                u_mat[:, s_idx] = np.eye(n)[:, s_idx] @ r_mat
            u_mat = u_mat[:, rng.permutation(n)] * rng.choice([-1, 1], n)
        else:
            u_mat = ortho_group.rvs(n, random_state=rng)
        h_mat = u_mat.T @ l1.T @ matrix_power(l1 @ u_mat @ d_mat @ u_mat.T @ l1.T, -1 / 3) @ l1 @ u_mat
        g_mat = u_mat.T @ e_mat @ u_mat
        assert (np.linalg.norm(h_mat - np.diag(np.diag(h_mat))) < 1e-9) == (
            np.linalg.norm(g_mat - np.diag(np.diag(g_mat))) < 1e-9
        )
    for _ in range(500):
        n = int(rng.integers(2, 7))
        m = int(rng.integers(1, n + 1))
        a_vec = np.sort(rng.uniform(0.1, 3, m))[::-1]
        l1 = np.hstack([np.diag(a_vec), np.zeros((m, n - m))])
        l2 = np.sort(rng.uniform(0.1, 5, n))[::-1]
        v_mat = ortho_group.rvs(n, random_state=rng)
        val = np.trace(matrix_power(l1 @ v_mat @ np.diag(l2) @ v_mat.T @ l1.T, 2 / 3))
        assert val <= np.sum((a_vec**2 * l2[:m]) ** (2 / 3)) + 1e-9
    print("C3 stress: kernel rotations (5,10)/(10,20)/(10,50), 250 iff trials, 500 Horn trace bounds passed")


if __name__ == "__main__":
    check_flow()
    check_landscape()
    check_diagonalization()
