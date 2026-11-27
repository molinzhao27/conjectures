"""Reproduce the exact arithmetic and small selector experiments in results.md."""
from fractions import Fraction as F
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize


def gamma(d, k):
    values = []
    for s in range(k + 1, d + 1):
        q, a = divmod(d, s)
        values.append(F(s - k) / (F(s - a, q) + F(a, q + 1)))
    return max(values)


def arithmetic():
    cells = branches = 0
    for d in range(2, 101):
        for k in range((d + 2) // 3, d):
            assert gamma(d, k) == F(d - k, d)
            cells += 1
    for d in range(5, 202, 2):
        k = (d - 1) // 2
        c = F(k + 1, d)
        threshold = F(1, k * d)
        assert (1 - threshold) * F(k, d - 2) == c
        assert (1 - F(2, d)) * F(k, d - 2) + F(1, d) == c
        even_bound = (1 + F(1, d - 1)) * (1 + F(k, d - 2)) - 1
        assert even_bound * (1 - c) <= c
        for a in range(1, k + 1):
            b = d - 2 * a
            assert 3 * a * threshold * ((b + 1) // 2) <= c
            branches += 1
    return cells, branches


def simplex_extremizer(d, k):
    """Exact shifted axis instance, including omitted axes and arbitrary weights."""
    shift = d + 1
    # Costs multiplied by d: best errors for 0, 1 or 2 selected rows per axis.
    best = min(sum((shift**2, 1, 0)[t] for t in counts)
               for counts in product(range(3), repeat=d)
               if sum(counts) <= d + k)
    assert F(best, d) == F(d - k, d)


def graph_data(d, edges, rng, spread):
    """Positive coordinate rays and negative two-coordinate rays; arbitrary energies."""
    negative = np.zeros((len(edges), d))
    for row, (i, j) in zip(negative, edges):
        row[[i, j]] = -np.exp(rng.uniform(-spread, spread, size=2))
    g = np.vstack((np.diag(-negative.sum(axis=0)), negative))
    assert np.linalg.matrix_rank(g) == d and np.allclose(g.sum(axis=0), 0)
    c = np.exp(rng.uniform(-2 * spread, 2 * spread, size=len(g)))
    c /= c.sum()
    x = g / np.sqrt(c[:, None])
    y = np.sqrt(c)
    assert np.allclose(x.T @ y, 0)
    return x, y


def best_adjacent(x, y):
    """Minimize on the simplex of interpolants of each d+1-row support."""
    n, d = x.shape
    fitted = {}
    for basis in combinations(range(n), d):
        block = x[list(basis)]
        if np.linalg.matrix_rank(block) == d:
            fitted[basis] = x @ np.linalg.solve(block, y[list(basis)])
    best = np.inf
    for support in combinations(range(n), d + 1):
        vertices = [fitted[b] for b in combinations(support, d) if b in fitted]
        if not vertices:
            continue
        table = np.array(vertices)
        gram = table @ table.T
        # Scaling leaves the minimizer unchanged and improves line searches.
        gram /= max(1., np.max(np.diag(gram)))
        count = len(vertices)
        init = np.ones(count) / count
        result = minimize(lambda p: p @ gram @ p, init, jac=lambda p: 2 * gram @ p,
                          bounds=[(0, 1)] * count, method="SLSQP",
                          constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1,
                                        "jac": lambda p: np.ones(count)}],
                          options={"ftol": 1e-13, "maxiter": 200})
        assert result.success and abs(result.x.sum() - 1) < 1e-8
        point = result.x @ table
        best = min(best, point @ point)
    return best


def composition_search():
    """Optimize the hard 2+3 direct sum over its energy allocation."""
    rng = np.random.default_rng(506)
    worst = 0.
    for trial in range(96):
        edges = [(0, 1), (0, 2)]
        # Include a second leaf-to-leaf triangle on half of the trials.
        if trial % 2:
            edges.append((1, 2))
        x, y = graph_data(3, edges, rng, rng.uniform(0.1, 3))
        cost3 = completed_edges(x, y, edges, 0)
        cost4 = best_adjacent(x, y)
        assert -1e-8 <= cost4 <= cost3 + 1e-6
        # Extra row on the two-dimensional simplex or on the 3D component.
        # Equalize the two affine costs to maximize their minimum.
        value = 2 * cost3 / (2 + cost3 - cost4)
        assert value <= 6 / 5 + 1e-7
        worst = max(worst, value)

    # The balanced two-simplex adversary attains the conjectured 5,6 value.
    simplex = np.vstack((np.eye(3), -np.ones(3)))
    labels = np.ones(4) / 2
    cost3 = completed_edges(simplex, labels, [], 0)
    cost4 = best_adjacent(simplex, labels)
    assert abs(cost3 - 3) < 1e-12 and cost4 < 1e-12
    assert abs(2 * cost3 / (2 + cost3 - cost4) - 6 / 5) < 1e-12
    return worst


def completed_edges(x, y, edges, k):
    """Exact-flat witness plus interpolated rows. Enumerate disjoint completed edges."""
    n, d = x.shape
    # Unwhitened coordinates work: calibrate selected vertex coordinates to zero.
    best = np.inf
    for j in range(min(k, d // 2) + 1):
        for matching in combinations(edges, j):
            fixed = {v for edge in matching for v in edge}
            if len(fixed) != 2 * j:
                continue
            free = sorted(set(range(d)) - fixed)
            quotient = x[:, free]
            for rows in combinations(range(n), len(free)):
                if not free:
                    return 0.
                square = quotient[list(rows)]
                if np.linalg.matrix_rank(square) < len(free):
                    continue
                w = np.linalg.solve(square, y[list(rows)])
                pred = quotient @ w
                best = min(best, pred @ pred)
    return best


def experiments():
    rng = np.random.default_rng(260828007)
    reports = []
    for d in (3, 4, 5, 6):
        # Sparse overlapping triangles; basis subsets include singular minors.
        edges = [(i, i + 1) for i in range(d - 1)]
        if d >= 4:
            edges.append((0, d - 1))
        for k in ([1, 2] if d == 3 else [1, 2, 3]):
            if k >= d:
                continue
            costs = []
            for spread in (0.35, 0.9, 1.6):
                x, y = graph_data(d, edges, rng, spread)
                excess = best_adjacent(x, y) if k == 1 else completed_edges(x, y, edges, k)
                assert excess >= -1e-10 and excess <= float(gamma(d, k)) + 1e-7
                costs.append(excess)
            reports.append((d, d + k, max(costs)))
    return reports


if __name__ == "__main__":
    print("Exact Gamma cells and odd packing branches:", arithmetic())
    for d in range(2, 10):
        for k in range(d // 2, d):
            simplex_extremizer(d, k)
    assert gamma(5, 1) == F(6, 5)
    assert gamma(5, 2) == F(3, 5)
    for d, n, value in experiments():
        print(f"d={d} n={n}: largest witness excess across three systems {value:.8f}")
    print(f"Largest excess among 96 coupled 2+3 compositions: {composition_search():.8f}")
    print("All assertions passed.")
