"""Exact checks for the three coded block retrieval conjectures (prime fields)."""

from collections import Counter
from fractions import Fraction as Q
from functools import cache
from itertools import combinations
from math import comb
from random import Random


def extend(basis, vector, prime):
    """Canonical reduced row basis for the span after adding one vector."""
    row = list(vector)
    for pivot, old in basis:
        coeff = row[pivot]
        if coeff:
            row = [(x - coeff * y) % prime for x, y in zip(row, old)]
    pivot = next((j for j, x in enumerate(row) if x), None)
    if pivot is None:
        return basis
    inv = pow(row[pivot], -1, prime)
    row = tuple(x * inv % prime for x in row)
    updated = []
    for j, old in basis:
        if old[pivot]:
            old = tuple((x - old[pivot] * y) % prime for x, y in zip(old, row))
        updated.append((j, old))
    return tuple(sorted([*updated, (pivot, row)]))


def axes(k):
    return [tuple(int(i == j) for j in range(k)) for i in range(k)]


def span(columns, prime):
    basis = ()
    for v in columns:
        basis = extend(basis, v, prime)
    return basis


def decoded(basis, targets, prime):
    return all(extend(basis, e, prime) == basis for e in targets)


def expectations(columns, prime, split):
    """Solve the two first-step equations at every reachable span."""
    counts = Counter(columns)
    n = len(columns)
    k = len(columns[0])
    unit = axes(k)
    files = (unit[:split], unit[split:])
    assert len(span(columns, prime)) == k

    @cache
    def solve(basis):
        if len(basis) == k:
            return (Q(0), Q(0))
        successors = Counter()
        for v, weight in counts.items():
            new = extend(basis, v, prime)
            if new != basis:
                successors[new] += weight
        exit_weight = sum(successors.values())
        cost = [Q(n), Q(n)]
        for new, weight in successors.items():
            later = solve(new)
            for i in range(2):
                cost[i] += weight * later[i]
        return tuple(Q(0) if decoded(basis, file, prime) else cost[i] / exit_weight
                     for i, file in enumerate(files))

    result = solve(())
    solve.cache_clear()
    return result


def subset_check(columns, prime, split):
    """Average over the uniform subset of indices after j distinct arrivals."""
    k, n = len(columns[0]), len(columns)
    unit = axes(k)
    times = [Q(0), Q(0)]
    for j in range(n):
        factor = Q(n, (n - j) * comb(n, j))
        for subset in combinations(columns, j):
            basis = span(subset, prime)
            for i, file in enumerate((unit[:split], unit[split:])):
                if not decoded(basis, file, prime):
                    times[i] += factor
    return tuple(times)


def small_fields():
    rng = Random(260317154)
    trials = 0
    for prime in (2, 3, 5):
        for k, split in ((3, 2), (4, 2), (5, 3)):
            for n in (k, k + 1, k + 3):
                for _ in range(3):
                    while True:
                        cols = [tuple(rng.randrange(prime) for _ in range(k))
                                for _ in range(n)]
                        if len(span(cols, prime)) == k:
                            break
                    assert expectations(cols, prime, split) == subset_check(cols, prime, split)
                    trials += 1
    return trials


def main():
    print(f"Small-field subset cross-checks passed: {small_fields()}")
    e1, e2, e3 = axes(3)
    columns = [e3] * 11 + [(1, t, t * t % 11) for t in range(11)]
    pair = expectations(columns, 11, 2)
    assert pair == (Q(937, 210), Q(25, 14))
    score = 2 / pair[0] + 1 / pair[1]
    assert score == Q(23618, 23425) > 1
    for m in (2, 3, 7):
        assert expectations(columns * m, 11, 2) == pair
    print(f"(2,1), GF(11), n=22: {pair[0]}, {pair[1]}; score {score}")

    # Put A in the first two coordinate directions.
    prime = 83
    a_cols = [(1, u, 0, 0) for u in range(61, 81)]
    b_cols = [(t*t % prime, t**3 % prime, 1, t) for t in range(1, 21)]
    pair = expectations(a_cols + b_cols, prime, 2)
    assert pair == (Q(30729, 9139), Q(134404, 27417))
    score = 2 / pair[0] + 2 / pair[1]
    assert score == Q(2070815149, 2065050258) > 1
    print(f"(2,2), GF(83), n=40: {pair[0]}, {pair[1]}; score {score}")

    for prime in (2, 3, 5):
        assert expectations([e1, e2] + [e3] * 7, prime, 2) == (Q(27, 2), Q(9, 7))
        assert expectations([e1, e2] + [e3] * 8, prime, 2) == (15, Q(5, 4))
    assert 1 + Q(3, 10) > Q(9, 7) and 15 > Q(27, 2)
    print("Length 9 to 10 obstruction passed. All assertions passed.")


if __name__ == "__main__":
    main()
