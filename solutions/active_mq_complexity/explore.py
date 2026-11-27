"""Exact finite decision-tree and batch-halving checks for conj:halve."""

from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from math import ceil, log2
from random import Random


class Table:
    """Concepts are row indices. Each query is the bit mask of its positive rows."""

    def __init__(self, rows):
        self.n = len(rows)
        self.full = (1 << self.n) - 1
        self.cols = tuple(
            sum(int(value) << i for i, value in enumerate(col)) for col in zip(*rows)
        )

    def parts(self, rows, queries):
        parts = [rows] if rows else []
        for q in queries:
            mask = self.cols[q]
            parts = [side for p in parts for side in (p & mask, p & ~mask) if side]
        return parts

    def depth_solver(self, sample):
        colors = self.parts(self.full, sample)

        @cache
        def solve(rows, budget):
            count = sum(bool(rows & group) for group in colors)
            if count <= 1:
                return ()
            if count > 1 << budget:
                return None
            for q, mask in enumerate(self.cols):
                zero, one = rows & ~mask, rows & mask
                if not zero or not one:
                    continue
                left = solve(zero, budget - 1)
                if left is not None:
                    right = solve(one, budget - 1)
                    if right is not None:
                        return q, left, right
            return None

        for depth in range((len(colors) - 1).bit_length(), len(sample) + 1):
            tree = solve(self.full, depth)
            if tree is not None:
                return depth, tree
        raise AssertionError("Querying the sample always suffices")

    def batch(self, rows, sample):
        colors = self.parts(rows, sample)
        limit = (len(colors) + 1) // 2
        for t in range(len(sample) + 1):
            for batch in combinations(range(len(self.cols)), t):
                if all(sum(bool(p & c) for c in colors) <= limit
                       for p in self.parts(rows, batch)):
                    return t, batch
        raise AssertionError("Querying the sample always suffices")

    def walk(self, tree, rows):
        if not tree:
            return [rows] if rows else [], []
        q, left, right = tree
        a, qa = self.walk(left, rows & ~self.cols[q])
        b, qb = self.walk(right, rows & self.cols[q])
        return a + b, [q] + qa + qb


def families():
    # Nonzero binary vectors give all points and normals of PG(2,2).
    yield "projective lines PG(2,2)", [
        [(a & x).bit_count() % 2 == 0 for x in range(1, 8)] for a in range(1, 8)
    ]
    plane = list(product(range(3), repeat=2))
    yield "affine lines AG(2,3)", [
        [(u*x + v*y - b) % 3 == 0 for x, y in plane]
        for u, v in [(1, 0), (0, 1), (1, 1), (1, 2)] for b in range(3)
    ]
    yield "Reed-Muller RM(1,3)", [
        [((a & x).bit_count() + b) % 2 for x in range(8)]
        for a in range(8) for b in range(2)
    ]
    yield "radius-one Hamming balls", [
        [(a ^ x).bit_count() <= 1 for x in range(8)] for a in range(8)
    ]
    yield "bipartite edge incidence", [
        [i == a for i in range(3)] + [j == b for j in range(3)]
        for a, b in plane
    ]
    yield "product of two threshold chains", [
        [a <= t for t in range(2)] + [b <= t for t in range(2)] for a, b in plane
    ]
    yield "two disjoint threshold chains", [
        [i == block and a <= t for block in range(2) for t in range(4)]
        for i in range(2) for a in range(4)
    ]
    yield "binary tree subtrees", [
        [leaf >> level == node for level in range(3) for node in range(8 >> level)]
        for leaf in range(8)
    ]
    yield "three gated search blocks", [
        [i == block for block in range(3)]
        + [i == block and bool(j & (1 << bit)) for block in range(3) for bit in range(2)]
        + [i == block and j == leaf for block in range(3) for leaf in range(4)]
        for i in range(3) for j in range(4)
    ]


def search():
    rng = Random(20261127)
    cases = 0
    for name, rows in families():
        table = Table(rows)
        domain = tuple(range(len(table.cols)))
        samples = {domain}
        for m in {1, 2, 3, min(5, len(domain))}:
            samples.update(tuple(sorted(rng.sample(domain, m))) for _ in range(18))
        samples.add(domain[-12:] if name == "three gated search blocks" else domain)
        peak_depth = peak_batch = 0
        for sample in sorted(samples):
            depth, tree = table.depth_solver(sample)
            leaves, _ = table.walk(tree, table.full)
            assert all(len(table.parts(p, sample)) == 1 for p in leaves)
            assert depth >= ceil(log2(len(table.parts(table.full, sample))))
            peak_depth = max(peak_depth, depth)
            versions = {table.full, table.full & 0x55555555, rng.randrange(1, table.full + 1)}
            for version in versions:
                size, batch = table.batch(version, sample)
                total = len(table.parts(version, sample))
                assert all(2 * len(table.parts(p, sample)) <= total + 1
                           for p in table.parts(version, batch))
                peak_batch = max(peak_batch, size)
                cases += 1
        print(f"{name:33s} {len(samples):3d} samples   max depth {peak_depth}   max batch {peak_batch}")
    print(f"{cases} finite sample/version-space checks")


def mixture_check():
    # Each color has two possible positions. The first four queries are color tests.
    table = Table([[p % 4 == c for c in range(4)] + [p < t for t in range(1, 8)]
                   for p in range(8)])
    sample = tuple(range(4))
    groups = table.parts(table.full, sample)
    # The threshold p < 4 halves one hand-picked transversal but leaves every
    # color possible on either answer when both extensions are retained.
    split = (7,)
    assert [len(table.parts(p, sample)) for p in table.parts(0b11000011, split)] == [2, 2]
    assert [len(table.parts(p, sample)) for p in table.parts(table.full, split)] == [4, 4]

    # For an ordered training pair (y,z), the test "color = y" separates it.
    # Its first endpoint is uniform. Aggregate all 4**9 training sequences.
    weights = Counter({0: 1})
    for _ in range(9):
        next_weights = Counter()
        for mask, count in weights.items():
            for c in sample:
                next_weights[mask | (1 << c)] += count
        weights = next_weights
    denominator = sum(weights.values())
    assert denominator == 4**9
    trees = {}
    for mask in weights:
        tests = [c for c in sample if mask & (1 << c)]
        trees[mask] = table.depth_solver(tests)
    slots = 1 << max(depth for depth, tree in trees.values())
    queries = Counter()
    good_weights = Counter()
    transversals = [w for w in range(1 << table.n)
                    if w.bit_count() > 2 and all((w & g).bit_count() <= 1 for g in groups)]
    for mask, (_, tree) in trees.items():
        leaves, nodes = table.walk(tree, table.full)
        for q in nodes + [0] * (slots - len(nodes)):
            queries[q] += weights[mask]
        for w in transversals:
            if all(2 * (w & leaf).bit_count() < w.bit_count() for leaf in leaves):
                good_weights[w] += weights[mask]
    assert min(good_weights[w] for w in transversals) * 2 >= denominator
    total = denominator * slots
    assert sum(queries.values()) == total
    heavy = []
    for i, j in combinations(range(table.n), 2):
        pair = (1 << i) | (1 << j)
        mass = sum(weight for q, weight in queries.items() if (pair & table.cols[q]).bit_count() == 1)
        if 4 * slots * mass >= total:
            heavy.append(pair)
    assert all(any(w & pair == pair for pair in heavy) for w in transversals)
    # Hit the heavy disagreements and check the resulting batch on full fibers.
    for t in range(len(queries) + 1):
        for batch in combinations(queries, t):
            if all(any((pair & table.cols[q]).bit_count() == 1 for q in batch) for pair in heavy):
                assert max(len(table.parts(p, sample)) for p in table.parts(table.full, batch)) <= 2
                print(f"Mixture: {denominator} training sequences, {len(transversals)} transversals, "
                      f"minimum good-tree probability {Fraction(min(good_weights.values()), denominator)}, "
                      f"heavy-pair hitting batch {t}")
                return
    raise AssertionError("Positive-mass ranges are hit by the support")


if __name__ == "__main__":
    search()
    mixture_check()
    print("All assertions passed.")
