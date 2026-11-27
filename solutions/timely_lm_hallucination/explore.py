"""Finite checks for the companion proof. Run with Python 3.10+."""

from collections import Counter
from heapq import merge
from itertools import count, islice
from math import isqrt
from random import Random


def cutoff(t):
    """Power of two within a constant factor of t**(1/phi), exactly."""
    h = t.bit_length() - 1
    return 1 << ((isqrt(5 * h * h) - h) // 2)


def product_order():
    for product in count(1):
        small = [d for d in range(1, isqrt(product) + 1) if product % d == 0]
        large = [product // d for d in reversed(small) if d * d != product]
        for s in small + large:
            yield s, product // s


def pair_count(m):
    """Sum floor(m/d), grouping consecutive equal quotients."""
    total = 0
    left = 1
    while left <= m:
        quotient = m // left
        right = m // quotient
        total += (right - left + 1) * quotient
        left = right + 1
    return total


def block_end(n):
    h = n.bit_length()
    return (n + 1) * h - (1 << h) + 1


def petal(s, k):
    diagonal = s + k - 2
    return diagonal * (diagonal + 1) // 2 + s


def owner(x):
    """Invert the diagonal enumeration of disjoint infinite petals."""
    diagonal = (isqrt(8 * (x - 1) + 1) - 1) // 2
    return x - diagonal * (diagonal + 1) // 2


class Family:
    """Common core plus private petals in a known integer order.

    With finite core the only common point is 0. With infinite core the
    multiples of 7 form the common core. Private points are distinct.
    """

    def __init__(self, infinite):
        self.infinite = infinite

    def private(self, s, k):
        x = petal(s, k)
        return x + (x - 1) // 6 if self.infinite else x

    def label(self, x):
        if x == 0 or (self.infinite and x % 7 == 0):
            return 0
        return owner(x - x // 7 if self.infinite else x)

    def elements(self, key):
        if key == -1:  # universe
            return count(0)
        common = count(0, 7) if self.infinite else iter([0])
        if key == 0:
            return common
        private = (self.private(key, k) for k in count(1))
        return merge(common, private)


def simulate(s, infinite, stream, seed, ranks):
    family = Family(infinite)
    prefix = list(islice(family.elements(s), max(ranks)))
    ordered_input = family.elements(s)
    rng = Random(seed)
    used, first_input, output_time = set(), {}, {}
    cursors, probe_times, changes = {}, set(), []
    forced = Counter()
    visits = Counter()
    rounds = collisions = fallbacks = checks = 0
    previous = None

    def pop_fresh(key):
        if key not in cursors:
            cursors[key] = family.elements(key)
        return next((x for x in cursors[key] if x not in used), None)

    for n, (label, visit) in enumerate(product_order(), 1):
        length = n.bit_length()
        slot = rng.randrange(length)
        visits[label] += 1
        assert visits[label] == visit
        for offset in range(length):
            rounds += 1
            if stream == "delayed" and rounds <= 1200:
                observed = 7 * (rounds - 1) if infinite else 0
            elif stream == "mixed" and rounds % 4:
                observed = family.private(s, rounds * rounds + 12)
            else:
                observed = next(ordered_input)
            known = observed in used
            first_input.setdefault(observed, rounds)
            used.add(observed)
            observed_label = family.label(observed)
            if observed_label:
                forced[observed_label] += 1
            r = cutoff(rounds)
            # A private observation leaves at most one viable candidate.
            if forced:
                assert set(forced) == {s}
                key = s if s <= r else -1
            else:
                key = 1 if r == 1 else 0
            if previous != key:
                changes.append(rounds)
            previous = key
            if offset == slot:
                result = pop_fresh(label)
                probe_times.add(rounds)
            else:
                result = pop_fresh(key)
                if result is None:
                    # The common singleton is exhausted. With no private
                    # observations, the largest usable prefix has length 1.
                    assert key == 0 and not infinite and not forced
                    result = pop_fresh(1)
                    fallbacks += 1
            assert result is not None and result not in used
            used.add(result)
            output_time[result] = rounds
            collisions += int(known)
        assert rounds == block_end(n)
        if label != s or visit not in ranks:
            continue
        i = visit
        assert rounds <= block_end(pair_count(s * i))
        head = prefix[:i]
        assert all(x in used for x in head)
        generated = sum(x in output_time for x in head)
        probe_losses = sum(first_input.get(x) in probe_times for x in head)
        warmup = min(rounds, next(t for t in count(1) if cutoff(t) >= s) - 1)
        assert 2 * generated >= i - warmup - probe_losses - len(changes)
        deadline = lambda j: 1 + isqrt(j**3 - 1)  # ceil(j**1.5)
        timely = sum(output_time.get(x, rounds + deadline(i) + 1) <= deadline(j)
                     for j, x in enumerate(head, 1))
        expired = sum(deadline(j) < rounds for j in range(1, i + 1))
        assert timely >= generated - expired
        checks += 1
        if visit == max(ranks):
            break
    return rounds, checks, fallbacks, collisions


def packing_checks():
    # Give the common target the powers of 2 in the universe N. Place the
    # diagonally indexed petals at the other positions in increasing order.
    def private_position(s, k):
        rank = petal(s, k)
        lo, hi = 1, 4 * rank + 4
        while lo < hi:
            mid = (lo + hi) // 2
            if mid - mid.bit_length() >= rank:
                hi = mid
            else:
                lo = mid + 1
        return lo

    checks = 0
    for size in (2, 5, 13, 32, 80):
        for i in (16, 64, 256, 1024):
            cap = private_position(size, i)
            assert cap <= 4 * (size + i)**2
            # If all targets get i/3 credits, each needs this many
            # private outputs even if it gets credit for every common point.
            private_needed = max(0, (i + 2) // 3 - cap.bit_length())
            budget = 0
            owners = set()
            for s in range(1, size + 1):
                last = private_position(s, i)
                assert last <= cap
                for k in range(1, private_needed + 1):
                    x = private_position(s, k)
                    assert x not in owners and x & (x - 1)
                    owners.add(x)
                budget += private_needed
            assert len(owners) == budget
            checks += 1
    return checks


def main():
    serial = 0
    for m in range(1, 2001):
        for s, k in ((d, m // d) for d in range(1, m + 1) if m % d == 0):
            serial += 1
            assert s * k == m
            assert block_end(serial) - block_end(serial - 1) == serial.bit_length()
        assert serial == pair_count(m)
    expected = sorted(((s, k) for s in range(1, 2001)
                       for k in range(1, 2000 // s + 1)), key=lambda pair: (pair[0] * pair[1], pair[0]))
    assert list(islice(product_order(), serial)) == expected
    ranks = {1, 2, 4, 8, 16, 32, 64, 128}
    totals = Counter()
    for s in (1, 2, 5, 8):
        for infinite in (False, True):
            for stream in ("ordered", "delayed", "mixed"):
                for seed in (17, 2605):
                    values = simulate(s, infinite, stream, seed, ranks)
                    totals.update(dict(zip(("rounds", "prefixes", "fallbacks", "repeat_observations"), values)))
                    totals["games"] += 1
    print(f"Exact schedule: {serial:,} pairs through product 2,000")
    print("Online games:", dict(totals))
    print("Private-petal packing checks:", packing_checks())
    print("All assertions passed.")


if __name__ == "__main__":
    main()
