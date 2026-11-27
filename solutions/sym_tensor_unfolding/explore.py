"""Simulate the GOTE in the orthonormal multiset basis. Run with --help."""
import argparse
import math
from functools import lru_cache
from itertools import combinations_with_replacement, product

import numpy as np
from scipy.linalg import eigh


def rank_multiset(indices, n):
    """Colexicographic multiset rank, starting at zero."""
    rank = np.zeros(indices.shape[:-1], dtype=np.int64)
    for j in range(indices.shape[-1]):
        lookup = np.array([math.comb(i + j, j + 1) for i in range(n)])
        rank += lookup[indices[..., j]]
    return rank


@lru_cache(maxsize=32)
def basis(n, order):
    indices = np.array(list(combinations_with_replacement(range(n), order)))
    indices = indices[np.argsort(rank_multiset(indices, n))]
    orbit = []
    for row in indices:
        counts = np.bincount(row)
        orbit.append(math.factorial(order) // math.prod(
            math.factorial(c) for c in counts if c > 1))
    return indices, np.array(orbit)


class Unfolding:
    def __init__(self, n, p, k):
        left, ml = basis(n, k)
        right, mr = basis(n, p - k)
        _, mp = basis(n, p)
        self.index = np.empty((len(left), len(right)), dtype=np.int64)
        for i, row in enumerate(left):
            joined = np.column_stack((np.broadcast_to(row, (len(right), k)), right))
            self.index[i] = rank_multiset(np.sort(joined, axis=1), n)
        self.weights = np.sqrt(ml[:, None] * mr / mp[self.index])

    def __call__(self, coefficients):
        return self.weights * coefficients[self.index]


def leading(matrix):
    last = len(matrix) - 1
    values, vectors = eigh(matrix, subset_by_index=(last, last), check_finite=False)
    return values[0], vectors[:, 0]


def check_exchanges(n, p, k, unfold, rng):
    """Test all block-exchange coefficients against the exact contraction."""
    b = p - k

    def probe(order):
        rows, orbit = basis(n, order)
        compact = rng.normal(size=len(rows))
        compact /= np.linalg.norm(compact)
        ordered = np.array(list(product(range(n), repeat=order)))
        ranks = rank_multiset(np.sort(ordered, axis=1), n)
        return compact, compact[ranks] / np.sqrt(orbit[ranks])

    x, x_full = probe(k)
    y, y_full = probe(k)
    f, f_full = probe(b)
    h, h_full = probe(b)
    partitions = math.comb(p, k)

    def sym_product(first, second):
        values = first[:, None] * second * unfold.weights
        return np.bincount(unfold.index.ravel(), weights=values.ravel())

    exact = partitions * (sym_product(x, f) @ sym_product(y, h))
    expansion = (x @ y) * (f @ h)
    for j in range(1, k + 1):
        width = n**j
        contracted = y_full.reshape(width, -1) @ x_full.reshape(width, -1).T
        term = np.sum(f_full.reshape(width, -1) * (contracted @ h_full.reshape(width, -1)))
        expansion += math.comb(k, j) * math.comb(b, j) * term
    np.testing.assert_allclose(expansion, exact, atol=2e-13)


def verify():
    """Compare compact maps to independently symmetrized dense arrays."""
    rng = np.random.default_rng(129)
    n = 3
    for p in range(2, 7):
        raw = rng.normal(size=(n,) * p)
        sums = {}
        for pos in np.ndindex(raw.shape):
            key = tuple(sorted(pos))
            sums[key] = sums.get(key, 0.) + raw[pos]
        rows, mp = basis(n, p)
        coeff = np.array([sums[tuple(r)] for r in rows]) / np.sqrt(mp)
        dense = np.array([sums[tuple(sorted(pos))] /
                          (math.factorial(p) // math.prod(math.factorial(c)
                           for c in np.bincount(pos)))
                          for pos in product(range(n), repeat=p)])
        for k in range(1, p // 2 + 1):
            unfold = Unfolding(n, p, k)
            check_exchanges(n, p, k, unfold, rng)
            compressed = unfold(coeff)
            np.testing.assert_allclose(np.linalg.svd(compressed, compute_uv=False),
                                       np.linalg.svd(dense.reshape(n**k, -1),
                                                     compute_uv=False)[:len(compressed)],
                                       atol=1e-12)
            expected = math.comb(n + p - 1, p - k) / math.comb(p, k)
            np.testing.assert_allclose((unfold.weights**2).sum(axis=1), expected)


def simulate(n, p, k, reps, rng):
    b = p - k
    square = k == b
    unfold = Unfolding(n, p, k)
    recursive = Unfolding(n, k, 1) if square and k > 1 else None
    sigma = math.sqrt(math.factorial(k) * math.factorial(b) /
                      (n * math.factorial(p - 1)))
    left_scale = sigma * (n**p / (math.factorial(k)*math.factorial(b)))**0.25
    right_scale = sigma * math.sqrt(n**b / math.factorial(b))
    grid = [(scale, tau) for scale in (['L'] if square else ['L', 'R'])
            for tau in (0.6, 1.0, 1.5, 2.0)]
    observations = [[] for _ in grid]
    for _ in range(reps):
        coeff = rng.normal(scale=math.sqrt(p/n), size=math.comb(n+p-1, p))
        noise = unfold(coeff)
        gram = None if square else noise @ noise.T
        for results, (scale, tau) in zip(observations, grid):
            strength = tau * (left_scale if scale == 'L' else right_scale)
            # Orthogonal invariance permits v=e_1. Its power has colex rank 0.
            if square:
                spiked = noise.copy()
                spiked[0, 0] += strength
                _, vec = leading(spiked)
                if recursive is None:
                    recovered = vec[0]**2
                else:
                    flat = recursive(vec)
                    _, direction = leading(flat @ flat.T)
                    recovered = direction[0]**2
                results.append((vec[0]**2, recovered))
            else:
                spiked = gram.copy()
                spiked[:, 0] += strength * noise[:, 0]
                spiked[0, :] += strength * noise[:, 0]
                spiked[0, 0] += strength**2
                eig, vec = leading(spiked)
                right = (vec @ noise[:, 0] + strength * vec[0])**2 / eig
                results.append((vec[0]**2, right))
    for (scale, tau), values in zip(grid, observations):
        data = np.array(values)
        assert np.all((data >= -1e-12) & (data <= 1+1e-12))
        mean = data.mean(axis=0)
        se = data.std(axis=0, ddof=1) / math.sqrt(reps)
        print(f'{p:1} {k:1} {n:3} {scale} {tau:.1f} '
              f'{mean[0]:.5f} {se[0]:.5f} {mean[1]:.5f} {se[1]:.5f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=261105137)
    parser.add_argument('--reps', type=int, default=24)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error('standard errors require at least two repetitions')
    verify()
    print('p k n scale tau lifted_or_left SE recursive_or_right SE')
    for p, sizes in [(2, (80, 160)), (3, (32, 64)), (4, (16, 28)),
                     (5, (10, 16)), (6, (8, 12))]:
        for k in range(1, p // 2 + 1):
            for n in sizes:
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, p, k, n]))
                simulate(n, p, k, args.reps, rng)


if __name__ == '__main__':
    main()
