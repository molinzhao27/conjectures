"""Exact checks for the two KernelSHAP-IQ conjectures. Run with Python and SymPy."""

from fractions import Fraction
from itertools import combinations
from math import comb

from sympy import Matrix, Rational, bernoulli, diag, zeros


def choose(n, r):
    return comb(n, r) if 0 <= r <= n else 0


def sii_weight(d, k, t, overlap):
    return Fraction((-1) ** (k - overlap), (d - k + 1) * comb(d - k, t - overlap))


def check_intersection_formulas(d, k):
    """Use Johnson intersection counts without allocating a power set."""
    for t in range(k, d - k + 1):
        layer_eigenvalue = sum(
            (-1) ** (k - j) * comb(k, j) * choose(d - 2 * k + j, d - t)
            for j in range(k + 1)
        )
        assert layer_eigenvalue == comb(d - 2 * k, t - k)
        for overlap in range(k + 1):
            # Sum conjectured A[S,U] over all k-sets U contained in T.
            projected = sum(
                choose(overlap, j) * choose(t - overlap, k - j) * sii_weight(d, k, k, j)
                for j in range(k + 1)
            )
            assert projected / layer_eigenvalue == sii_weight(d, k, t, overlap)


def masks(d, k):
    return [sum(1 << i for i in indices) for indices in combinations(range(d), k)]


def design(d, k):
    # Bernoulli polynomials at zero use B_1 = -1/2 on all SymPy versions.
    coefficients = [
        sum(comb(j, r) * bernoulli(k - r, 0) for r in range(1, j + 1))
        for j in range(k + 1)
    ]
    sets = masks(d, k)
    return Matrix([[coefficients[(s & t).bit_count()] for s in sets]
                   for t in range(1 << d)])


def check_full_matrices(d, k):
    """Compute the inverse limit independently by compressing to the boundary kernel."""
    sets = masks(d, k)
    x = design(d, k)
    interior = [t for t in range(1 << d) if k <= t.bit_count() <= d - k]
    boundary = sorted(set(range(1 << d)) - set(interior))
    z = Matrix.hstack(*x[boundary, :].nullspace())
    assert z.cols == len(sets) - comb(d, k - 1)
    x_i = x[interior, :]
    weights = diag(*[Rational(1, comb(d - 2 * k, t.bit_count() - k)) for t in interior])
    weighted_transpose = x_i.T * weights
    gram = weighted_transpose * x_i
    restricted = z.T * gram * z
    inverse_limit = z * restricted.inv() * z.T
    a = Matrix([[sii_weight(d, k, k, (s & u).bit_count()) for u in sets] for s in sets])
    assert inverse_limit == a
    assert gram * z == (d - 2 * k + 1) * z

    # Verify the split estimator on the complete basis of point-mass games.
    target = Matrix([[sii_weight(d, k, t.bit_count(), (s & t).bit_count())
                      for t in range(1 << d)] for s in sets])
    split = target.copy()
    fitted = inverse_limit * weighted_transpose
    for column, t in enumerate(interior):
        split[:, t] = fitted[:, column]
    assert split == target

    # Includes every possible lower-order approximation via its unanimity basis.
    for order in range(k):
        lower = masks(d, order)
        games = Matrix([[int(r & t == r) for r in lower] for t in range(1 << d)])
        assert split * games == zeros(len(sets), len(lower))


def main():
    dense_cases = [(d, k) for d in range(2, 9) for k in range(1, d // 2 + 1)]
    for d, k in dense_cases:
        check_full_matrices(d, k)
    print(f"Full design, inverse limit, and split SII matrices: {len(dense_cases)} cases passed.")
    cases = [(d, k) for d in range(2, 33) for k in range(1, d // 2 + 1)]
    for d, k in cases:
        check_intersection_formulas(d, k)
    print(f"Johnson layer and projected-column identities: {len(cases)} cases passed.")


if __name__ == "__main__":
    main()
