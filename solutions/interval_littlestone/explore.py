"""Exact checks for the randomized Littlestone dimension of interval positions."""
import argparse
from fractions import Fraction
from functools import cache
from heapq import heapify, heappop, heappush


def greedy_children(p):
    base = 1 << (p.bit_length() - 2)
    a = max(base, p - 2 * base)
    return a, p - a


def partitions(total, least=2):
    """All multisets of non-unit pieces with the given total measure."""
    if total == 0:
        yield ()
    if total >= least:
        yield (total,)
    for first in range(least, total // 2 + 1):
        for tail in partitions(total - first, first):
            yield (first,) + tail


def explicit_interval_value(m):
    """Generic shattered-tree recursion on actual sets of hypothesis indices."""
    intervals = [0] + [
        ((1 << length) - 1) << start
        for length in range(1, m + 1)
        for start in range(m - length + 1)
    ]
    tests = [sum(1 << i for i, h in enumerate(intervals) if h & (1 << x))
             for x in range(m)]

    @cache
    def solve(version):
        best = Fraction(0)
        for test in tests:
            pos = version & test
            if pos and pos != version:
                best = max(best, (1 + solve(pos) + solve(version ^ pos)) / 2)
        return best

    return solve((1 << len(intervals)) - 1)


def verify(mass, rod_limit):
    # Each immediate reward is an integer multiple of 1/scale.
    scale = 2 << max(mass, rod_limit).bit_length()
    lam = [0]
    for x in range(1, max(mass, rod_limit) + 1):
        t = x.bit_length() - 1
        lam.append(t * scale + (x - (1 << t)) * (scale >> t))

    def greedy_score(state):
        # After n cuts, accumulator = scale * 2**n * discounted value.
        heap = [-p for p in state if p > 1]
        heapify(heap)
        acc = 0
        while heap:
            p = -heappop(heap)
            a, b = greedy_children(p)
            reward = scale + (lam[a] + lam[b]) // 2
            assert reward == lam[p]
            acc = 2 * acc + reward
            for child in (a, b):
                if child > 1:
                    heappush(heap, -child)
        return acc

    cuts = 0

    @cache
    def optimal_score(state):
        nonlocal cuts
        if not state:
            return 0
        # Every negative branch decreases this rank by exactly one.
        remaining = sum(state) - len(state)
        best = 0
        for i, p in enumerate(state):
            if i and p == state[i - 1]:
                continue
            rest = state[:i] + state[i + 1:]
            for u in range(1, p // 2 + 1):
                v = p - u
                child = tuple(sorted(rest + tuple(x for x in (u, v) if x > 1)))
                reward = scale + (lam[u] + lam[v]) // 2
                best = max(best, (reward << (remaining - 1)) + optimal_score(child))
                cuts += 1
        return best

    count = 0
    for total in range(mass + 1):
        for state in partitions(total):
            opt = optimal_score(state)
            assert opt == greedy_score(state), (state, opt)
            count += 1

    @cache
    def R(p):
        if p == 1:
            return Fraction(0)
        return 1 - Fraction(1, 1 << (p - 1)) + (R(p // 2) + R((p + 1) // 2)) / 2

    for p in range(1, rod_limit + 1):
        den = scale << (p - 1)
        assert Fraction(greedy_score((p,)), den) == R(p), p
        if p <= mass:
            assert Fraction(optimal_score((p,) if p > 1 else ()), den) == R(p), p
        if p <= 11:
            assert explicit_interval_value(p - 1) == R(p), p

    print(f"Exact Bellman optimum = greedy on {count:,} multisets of total measure <= {mass}")
    print(f"Distinct legal first cuts evaluated: {cuts:,}")
    print(f"Single rods: Bellman = R through m={mass - 1}, greedy = R through m={rod_limit - 1}")
    print(f"Explicit hypothesis recursion = R through m={min(10, rod_limit - 1)}")
    print("All assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=int, default=50)
    parser.add_argument("--rod-limit", type=int, default=512)
    args = parser.parse_args()
    if not 2 <= args.mass <= args.rod_limit:
        parser.error("require 2 <= mass <= rod-limit")
    verify(args.mass, args.rod_limit)
