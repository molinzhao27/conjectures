"""Verify the two-token Riemannian stacks in results.md. Requires NumPy and mpmath."""
import argparse
import mpmath as mp

import numpy as np


def attention(q, k, factors, tau=1, exp=np.exp):
    """Return softmax attention, Euclidean costs and projected metric costs."""
    diff = q[:, None, :] - k[None, :, :]
    euclidean = np.sum(diff**2, axis=-1)
    proj_i = (diff[..., None] * factors[:, None]).sum(axis=2)
    proj_j = (diff[..., None] * factors[None, :]).sum(axis=2)
    metric = 0.5 * np.sum(proj_i**2 + proj_j**2, axis=-1)
    scores = -(euclidean + metric) / tau
    weights = exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True), euclidean, metric


def dobrushin(matrix):
    diff = matrix[:, None, :] - matrix[None, :, :]
    return 0.5 * np.abs(diff).sum(axis=-1).max()


def singular_values(matrix):
    return np.linalg.svd(matrix, compute_uv=False)


def check_one_layer():
    h = np.eye(2)
    q = h @ np.diag([0.0, 2.0])
    k = h @ np.array([[0, 0], [1, 0]])
    for strength in (1.0, 4.0, 16.0):
        factors = np.sqrt(strength) * np.array([np.diag(np.sqrt([5, 1])),
                                                np.diag(np.sqrt([1, 2]))])
        a, e, g = attention(q, k, factors)
        np.testing.assert_allclose(e, [[0, 1], [4, 5]])
        np.testing.assert_allclose(g, strength * np.array([[0, 3], [6, 9]]))
        assert np.isclose(np.var(np.sum(factors**2, axis=(1, 2))), 2.25 * strength**2)
        assert (a > 0).all() and np.allclose(a.astype(float).sum(axis=1), 1)
        assert dobrushin(a) < 2e-15 and singular_values(a @ h)[1] < 2e-15
    print("One-layer checks passed for s = 1, 4, 16.")


def margin_and_product(case, n):
    if case == "constant":
        return mp.mpf(4), mp.tanh(2)**n
    if case == "harmonic":
        return mp.log(2*n + 1), mp.mpf(1) / (n + 1)
    if case == "summable":
        return mp.log(2*(n + 1)**2 - 1), mp.mpf(n + 2) / (2*(n + 1))
    raise ValueError(case)


def check_stack(case, show_all):
    # Use guard digits for open-loop evaluation of the predetermined weights.
    mp.mp.dps = 90
    exp = np.vectorize(mp.exp, otypes=[object])
    identity = np.eye(2, dtype=int)
    mean = np.full((2, 2), mp.mpf("0.5"))
    h = identity.copy()
    product = identity.copy()
    target = np.array([[0, 0], [1, 0]])
    silu_one = 1 / (1 + mp.exp(-1))
    print("case n sigma1(A) sigma2(A) delta(A) delta(P) sigma1(H) sigma2(H) EffRank(H)")
    for n in range(1, 201):
        margin, exact_product = margin_and_product(case, n)
        previous = margin_and_product(case, n - 1)[1]
        decoder = mean + (identity - mean) / previous
        labels = h @ decoder
        q = labels @ target
        strength = (margin - 1) / 2
        target_factors = np.array([np.diag([mp.sqrt(strength), 1]),
                                   np.diag([mp.sqrt(3*strength), 1])])
        hidden = labels / (1 + exp(-labels))
        factors = (hidden @ target_factors.reshape(2, 4) / silu_one).reshape(2, 2, 2)
        a, _, g = attention(q, q, factors, exp=exp)
        product = a @ product
        h = a @ h
        sv_a, sv_h = singular_values(a.astype(float)), singular_values(h.astype(float))
        delta_a, delta_p = float(dobrushin(a)), float(dobrushin(product))
        eff_rank = np.sum((sv_h / sv_h[0])**2)
        assert (a > 0).all() and np.allclose(a.astype(float).sum(axis=1), 1)
        atol = 2e-14
        strength, exact_product = float(strength), float(exact_product)
        np.testing.assert_allclose(g.astype(float), [[0, 2*strength], [2*strength, 0]], atol=atol)
        np.testing.assert_allclose(sv_a, [1, float(mp.tanh(margin / 2))], atol=atol, rtol=0)
        np.testing.assert_allclose(delta_a, sv_a[1], atol=atol, rtol=0)
        np.testing.assert_allclose([*sv_h, delta_p, eff_rank],
                                   [1, exact_product, exact_product, 1 + exact_product**2],
                                   atol=atol, rtol=0)
        if show_all or n in (1, 10, 50, 100, 200):
            numbers = (*sv_a, delta_a, delta_p, *sv_h, eff_rank)
            print(case, n, *(f"{x:.12g}" for x in numbers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="print every verified depth")
    args = parser.parse_args()
    check_one_layer()
    for schedule in ("constant", "harmonic", "summable"):
        check_stack(schedule, args.all)
    print("All 600 stack-depth checks passed.")
