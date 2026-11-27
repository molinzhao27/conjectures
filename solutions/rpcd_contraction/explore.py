"""Reproduce the RPCD searches and exact certificates using Python, NumPy, SciPy and SymPy."""
from functools import lru_cache
from itertools import permutations
from math import factorial, floor

import numpy as np
import sympy as sp

from _certificate import verify_three_dimensions
from _groups import search_group_models


def bound(n, sigma):
    return max((1 - 1 / n)**n, (1 - sigma / n)**(2 * n))


@lru_cache(None)
def orders(n):
    return np.array(list(permutations(range(n))))


def epoch_radius(a):
    """Construct the full adjoint second-moment matrix using every permutation."""
    n = len(a)
    paths = orders(n)
    t = np.tile(np.eye(n), (len(paths), 1, 1))
    rows = np.arange(len(paths))
    for i in paths.T:
        t[rows, i] -= (a[i, None, :] @ t)[:, 0]
    flat = t.reshape(len(paths), n * n)
    gram = (flat.T @ flat / len(paths)).reshape(n, n, n, n)
    adjoint = gram.transpose(1, 3, 0, 2).reshape(n * n, n * n)
    return max(abs(np.linalg.eigvals(adjoint)))


def search_dense():
    rng = np.random.default_rng(20261127)
    for n, trials in [(3, 900), (4, 700), (5, 500), (6, 250), (7, 60), (8, 5)]:
        best = -np.inf
        for _ in range(trials):
            sigma = rng.choice([.001, .01, .1, .3, .5, .52, .6, .8, .98])
            rank = rng.integers(2, n + 1)
            v = rng.normal(size=(n, rank)) * 10**rng.uniform(-2, 1, rank)
            v /= np.linalg.norm(v, axis=1)[:, None]
            g = v @ v.T
            scale = (1 - sigma) / (1 - np.linalg.eigvalsh(g)[0])
            a = np.eye(n) + scale * (g - np.eye(n))
            assert abs(np.linalg.eigvalsh(a)[0] - sigma) < 1e-11
            gap = (epoch_radius(a) - bound(n, sigma)) / sigma
            best = max(best, gap)
        assert best < 0
        print(f"Dense n={n}: {trials} trials, max (rho-bound)/sigma {best:.9g}")


def optimize_rank_two():
    """Target close violations by jointly tuning the angles and strong convexity."""
    from scipy.optimize import differential_evolution

    for n in (3, 4, 5, 6):
        def objective(parameters):
            sigma = parameters[0]
            angles = np.r_[0.0, parameters[1:]]
            gram = np.cos(angles[:, None] - angles[None, :])
            a = sigma * np.eye(n) + (1-sigma) * gram
            return (bound(n, sigma)-epoch_radius(a)) / sigma

        opt = differential_evolution(
            objective, [(0.0001, .99)] + [(0, np.pi)]*(n-1),
            seed=2026+n, maxiter=25, popsize=8, polish=False, tol=0, atol=0)
        assert opt.fun > 0
        print(f"Optimized n={n}: {opt.nfev} evaluations, "
              f"max (rho-bound)/sigma {-opt.fun:.9g}")


def certify_example(n):
    """Return an exact algebraic specification via a rational isolating interval."""
    vectors = np.array([(5, 0), (3, 4), (4, -3), (0, 5), (4, 3), (3, -4),
                        (-5, 0)], dtype=object)[:n]
    numerator = 25 * np.eye(n, dtype=object) + vectors @ vectors.T
    denominator = 50
    # Each column represents one basis element of the symmetric matrix space.
    i, j = np.triu_indices(n)
    lifted = np.zeros((len(i), len(i)), dtype=object)
    for path in permutations(range(n)):
        t = np.eye(n, dtype=object)
        for idx in path:
            change = numerator[idx] @ t
            t *= denominator
            t[idx] -= change
        # For M X M^T, collect the coefficient of each X_kl with k<=l.
        left = t[i[:, None], i] * t[j[:, None], j]
        right = t[i[:, None], j] * t[j[:, None], i]
        lifted += left + right * (i != j)
    divisor = factorial(n) * denominator**(2*n)
    polynomial = sp.Matrix(lifted.tolist()).charpoly().as_poly().to_field()
    rho = epoch_radius(np.array(numerator / denominator, dtype=float))
    decimals = 10**9
    k = floor(rho * decimals)
    lo = sp.Rational(k, decimals) * divisor
    hi = sp.Rational(k+1, decimals) * divisor
    # All derivatives are positive above lo except the negative constant.
    # Exactly one zero lies above lo, and it is below hi.
    shifted = polynomial.shift(lo)
    assert shifted.TC() < 0 and all(c > 0 for c in shifted.all_coeffs()[:-1])
    assert polynomial.eval(hi) > 0
    assert sp.Rational(k+1, decimals) < (1 - sp.Rational(1, 2*n))**(2*n)
    print(f"Exact example n={n}: rho in ({k}/{decimals}, {k+1}/{decimals})")


if __name__ == "__main__":
    verify_three_dimensions()
    for dimension in range(3, 7):
        certify_example(dimension)
    search_dense()
    optimize_rank_two()
    search_group_models(epoch_radius, bound)
