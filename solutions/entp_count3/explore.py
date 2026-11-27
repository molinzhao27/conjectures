"""Exact checks for the Count3 time/space counterexamples and streaming bounds."""

from bisect import bisect_left
from itertools import product
from math import isqrt
from random import Random


def definition(xs):
    n = len(xs)
    return sum((x + y + xs[-1]) % n == 0 for x in xs for y in xs) % n


def blocked(xs, width):
    """Read-only input. Only width histogram counters and scalar scratch."""
    n = len(xs)
    t = xs[-1]
    total = 0
    for lo in range(0, n, width):
        counts = [0] * min(width, n - lo)
        for x in xs:
            offset = x % n - lo
            if 0 <= offset < len(counts):
                counts[offset] += 1
        for y in xs:
            offset = (-t - y) % n - lo
            if 0 <= offset < len(counts):
                total += counts[offset]
    return total % n


def packed(xs, field_bits=None, width=None):
    """Read-only input. Packed counters fit in one word (n.bit_length() bits)."""
    n = len(xs)
    t = xs[-1] % n
    word_bits = n.bit_length()
    if field_bits is None:
        field_bits = word_bits.bit_length()
    if width is None:
        width = word_bits // field_bits
    saturated = (1 << field_bits) - 1
    total = 0
    rescans = 0
    for lo in range(0, n, width):
        size = min(width, n - lo)
        fields = 0
        for x in xs:
            offset = x % n - lo
            if 0 <= offset < size:
                shift = offset * field_bits
                before = (fields >> shift) & saturated
                if before < saturated:
                    fields += 1 << shift
                    # Verify no carry into any other field.
                    assert ((fields >> shift) & saturated) == before + 1
        assert fields < (1 << (size * field_bits))
        for y in xs:
            offset = (-t - y) % n - lo
            if 0 <= offset < size:
                count = (fields >> (offset * field_bits)) & saturated
                if count < saturated:
                    total += count
        for offset in range(size):
            if ((fields >> (offset * field_bits)) & saturated) == saturated:
                rescans += 1
                a = lo + offset
                b = (-t - a) % n
                left = right = 0
                for x in xs:
                    residue = x % n
                    left += residue == a
                    right += residue == b
                total += left * right
    assert rescans <= n // saturated
    return total % n


def near_linear(xs, width=None, field_bits=None, per_word=None):
    """O(n log n) time and o(n) workspace bits on a word RAM."""
    n = len(xs)
    t = xs[-1] % n
    w = n.bit_length()
    if width is None:
        width = max(1, n // w)
    if field_bits is None:
        field_bits = min(w, (w * w).bit_length())
    saturated = (1 << field_bits) - 1
    if per_word is None:
        per_word = w // field_bits
    total = 0
    searches = 0
    for lo in range(0, n, width):
        size = min(width, n - lo)
        words = [0] * ((size + per_word - 1) // per_word)

        def count_at(offset):
            index, field = divmod(offset, per_word)
            return (words[index] >> (field * field_bits)) & saturated

        for x in xs:
            offset = x % n - lo
            if 0 <= offset < size and count_at(offset) < saturated:
                index, field = divmod(offset, per_word)
                before = count_at(offset)
                words[index] += 1 << (field * field_bits)
                assert count_at(offset) == before + 1
        heavy = [j for j in range(size) if count_at(j) == saturated]
        exact = [0] * len(heavy)
        if heavy:
            for x in xs:
                offset = x % n - lo
                if 0 <= offset < size and count_at(offset) == saturated:
                    searches += 1
                    exact[bisect_left(heavy, offset)] += 1
        for y in xs:
            offset = (-t - y) % n - lo
            if 0 <= offset < size:
                count = count_at(offset)
                if count == saturated:
                    searches += 1
                    count = exact[bisect_left(heavy, offset)]
                total += count
    assert searches <= 2 * n
    return total % n


def streaming_unary(xs):
    """One-pass streaming with a (2n - 2)-bit unary histogram of x_1..x_{n-1}."""
    n = len(xs)
    counts = [0] * n
    for x in xs[:-1]:
        counts[x % n] += 1
    # Stars-and-bars unary code: n - 1 ones (items) and n - 1 zeros (bucket separators).
    bits = []
    for r, c in enumerate(counts):
        bits.extend([1] * c)
        if r < n - 1:
            bits.append(0)
    assert len(bits) == 2 * n - 2

    # Decode after x_n arrives and compute Count3 in O(n) time.
    decoded = [0] * n
    bucket = 0
    for b in bits:
        if b:
            decoded[bucket] += 1
        else:
            bucket += 1
    decoded[xs[-1] % n] += 1
    t = xs[-1] % n
    return sum(decoded[a] * decoded[(-t - a) % n] for a in range(n)) % n


def verify_algorithms():
    exhaustive = 0
    for n in range(1, 7):
        for xs in product(range(1, n + 1), repeat=n):
            expected = definition(xs)
            assert (
                blocked(xs, isqrt(n))
                == packed(xs)
                == near_linear(xs)
                == streaming_unary(xs)
                == expected
            ), xs
            exhaustive += 1

    rng = Random(241001600)
    stress = 0
    for n in (7, 15, 16, 17, 31, 32, 63, 64, 97, 127, 128, 257, 513):
        limit = (1 << n.bit_length().bit_length()) - 1
        for _ in range(4):
            data = [rng.randrange(1, n**3 + 1) for _ in range(n)]
            for copies in (0, limit - 1, limit, min(n, limit + 1), n):
                xs = [2 * n] * copies + data[copies:]
                rng.shuffle(xs)
                expected = definition(xs)
                assert (
                    blocked(xs, isqrt(n))
                    == packed(xs)
                    == near_linear(xs)
                    == streaming_unary(xs)
                    == expected
                )
                stress += 1

    # Explicitly exercise multi-field packing and multi-heavy blocks.
    packed_stress = 0
    for n in (64, 127, 256, 300):
        for _ in range(25):
            # Force many heavy residues inside the same block.
            num_heavy = rng.randint(2, min(20, n // 4))
            heavy_vals = rng.sample(range(1, n + 1), num_heavy)
            xs = []
            for hv in heavy_vals:
                xs.extend([hv] * rng.randint(3, 6))
            while len(xs) < n:
                xs.append(rng.randrange(1, n + 1))
            xs = xs[:n]
            rng.shuffle(xs)
            expected = definition(xs)
            assert packed(xs, field_bits=2, width=8) == expected
            assert packed(xs, field_bits=3, width=5) == expected
            assert near_linear(xs, width=32, field_bits=2, per_word=4) == expected
            assert near_linear(xs, width=40, field_bits=3, per_word=3) == expected
            assert streaming_unary(xs) == expected
            packed_stress += 1

    return exhaustive, stress, packed_stress


def verify_streaming_reduction():
    queries = 0
    for n in (12, 25, 47, 60, 96, 101):
        m = n // 12
        keys = range(m + 1, 2 * m + 1)
        filler = 4 * m + 1
        for bits in product((0, 1), repeat=m):
            prefix = [2 * k if bit else filler for k, bit in zip(keys, bits)]
            prefix.extend([filler] * (n - 1 - m))
            for k, bit in zip(keys, bits):
                assert definition(prefix + [n - k]) == 2 * bit
                queries += 1
    return queries


if __name__ == "__main__":
    exhaustive, stress, packed_stress = verify_algorithms()
    print(f"Exhaustive sequences (n=1..6): {exhaustive}")
    print(f"Seeded and saturation cases: {stress}")
    print(f"Multi-field / multi-heavy packing cases: {packed_stress}")
    print(f"One-pass INDEX queries checked: {verify_streaming_reduction()}")
    print("All assertions passed.")
