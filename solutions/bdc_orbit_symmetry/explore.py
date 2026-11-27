"""Exact polynomial transition and orbit checks for the binary deletion channel."""

from collections import Counter, defaultdict
from itertools import groupby, product
from math import comb


def transition_column(word):
    """Entry at y is counts[y] times d**(n-len(y)) * (1-d)**len(y)."""
    counts = Counter({"": 1})
    for bit in word:
        # Deleting this bit preserves each prefix output. Keeping it appends bit.
        counts.update({y + bit: count for y, count in counts.items()})
    return counts


def orbit(word):
    complement = word.translate(str.maketrans("01", "10"))
    return frozenset((word, complement, word[::-1], complement[::-1]))


def check_length(n):
    bins = defaultdict(set)
    for bits in product("01", repeat=n):
        word = "".join(bits)
        counts = transition_column(word)
        layers = [[] for _ in range(n + 1)]
        for y, count in counts.items():
            layers[len(y)].append(count)
        assert [sum(layer) for layer in layers] == [comb(n, k) for k in range(n + 1)]
        assert counts[word] == 1

        # Deletions give the same output exactly when they occur in one run.
        runs = [len(list(run)) for _, run in groupby(word)]
        assert sorted(layers[n - 1]) == sorted(runs)
        deck = {y: c for y, c in counts.items() if len(y) == n - 1}
        moments = [sum(c for y, c in deck.items() if y[i] == "1") for i in range(n - 1)]
        for i, value in enumerate(moments, start=1):
            assert value == (n - i) * int(bits[i - 1]) + i * int(bits[i])
        if n >= 3:
            recovered = [int(moments[0] >= n - 1)]
            for i, value in enumerate(moments, start=1):
                bit, remainder = divmod(value - (n - i) * recovered[-1], i)
                assert remainder == 0 and bit in (0, 1)
                recovered.append(bit)
            assert "".join(map(str, recovered)) == word

        # Unlabelled multiplicity histograms are invariant under output relabelling
        # within lengths. Omitted zeros are determined by the known layer sizes.
        signature = tuple(tuple(sorted(Counter(layer).items())) for layer in layers)
        bins[signature].add(word)

    sizes = Counter()
    for cell in bins.values():
        assert cell == orbit(next(iter(cell)))
        sizes[len(cell)] += 1
    assert set(sizes) <= {2, 4}
    assert 2 * sizes[2] + 4 * sizes[4] == 2**n
    print(f"{n:2}  {2**n:6}  {sizes[2]:6}  {sizes[4]:6}  {max(sizes):4}")


if __name__ == "__main__":
    print(" n  inputs  size-2  size-4   max")
    for length in range(1, 13):
        check_length(length)
    print("All exact assertions passed.")
