"""Exact shortest-word exclusion search and independent membership checks.

Use python -B explore.py. Tokens -1 and -2 are terminals in the binary
search. Nonnegative integers are variables, named in first-occurrence order.
"""
from collections import Counter
from functools import cache
from itertools import product


def patterns(length, variables=None, alphabet=2):
    def visit(prefix, used):
        remaining = length - len(prefix)
        if variables is not None and not used <= variables <= used + remaining:
            return
        if not remaining:
            yield prefix
            return
        for token in range(-alphabet, used + 1):
            yield from visit(prefix + (token,), max(used, token + 1))
    return visit((), 0)


@cache
def columns(length):
    """Bit position w represents the binary word with coordinate i at bit i."""
    full = (1 << (1 << length)) - 1
    return full, [full // ((1 << (1 << i)) + 1) << (1 << i)
                  for i in range(length)]


def cube(tokens):
    full, bits = columns(len(tokens))
    mask = full
    previous = {}
    for i, token in enumerate(tokens):
        if token == -1:
            mask &= ~bits[i]
        elif token == -2:
            mask &= bits[i]
        elif token in previous:
            mask &= ~(bits[i] ^ bits[previous[token]])
        else:
            previous[token] = i
    return mask


def skeleton(pattern):
    """Group consecutive once-occurring variables into one unrestricted gap."""
    counts = Counter(pattern)
    groups = []
    sizes = {}
    for token in pattern:
        if token >= 0 and counts[token] == 1:
            if groups and isinstance(groups[-1], tuple):
                groups[-1] += (token,)
            else:
                groups.append((token,))
        else:
            groups.append(token)
            if token >= 0:
                sizes[token] = counts[token]
    for group in groups:
        if isinstance(group, tuple):
            sizes[group] = 1
    return groups, sizes


def allocation_bound(sizes, slack, alphabet):
    coeff = [1] + [0] * slack
    for cost in sizes.values():
        for i in range(cost, slack + 1):
            coeff[i] += alphabet * coeff[i - cost]
    return coeff[-1]


def layer(groups, sizes, slack):
    items = list(sizes.items())
    increments = {}
    def expand(i, left):
        if i < len(items):
            key, cost = items[i]
            union = 0
            for value in range(left // cost + 1):
                increments[key] = value
                union |= expand(i + 1, left - value * cost)
            return union
        if left:
            return 0
        tokens = []
        for group in groups:
            if isinstance(group, tuple):
                tokens.extend((group, j) for j in range(len(group) + increments[group]))
            elif group < 0:
                tokens.append(group)
            else:
                tokens.extend((group, j) for j in range(1 + increments[group]))
        return cube(tokens)
    return expand(0, slack)


@cache
def source_cubes(n, variables, zeros, ones):
    return [cube(p) for p in patterns(n, variables)
            if p.count(-1) >= zeros and p.count(-2) >= ones]


def canonical(seq):
    names = {}
    return tuple(names.setdefault(x, len(names)) if x >= 0 else x for x in seq)


def orbit_rep(q):
    flip = tuple(-3 - x if x < 0 else x for x in q)
    return q == min(q, flip, canonical(q[::-1]), canonical(flip[::-1]))


def binary_search(n):
    built = tested = 0
    for slack in range(2, (n - 1) // 2 + 1):
        for q in patterns(n - slack):
            zeros, ones = q.count(-1), q.count(-2)
            v = max(q) + 1 if max(q) >= 0 else 0
            wanted = v + 2 * slack
            if wanted + zeros + ones > n or not orbit_rep(q):
                continue
            groups, sizes = skeleton(q)
            if allocation_bound(sizes, slack, 2) < 4 ** slack:
                continue
            accepted = layer(groups, sizes, slack)
            built += 1
            if accepted.bit_count() < 2 ** wanted:
                continue
            outside = columns(n)[0] ^ accepted
            for source in source_cubes(n, wanted, zeros, ones):
                tested += 1
                assert source & outside, (n, q, wanted, source)
    return built, tested


def matches(pattern, word, erasing=False):
    """Direct backtracking matcher, independent of the layer calculation."""
    env = {}
    def visit(i, pos):
        if i == len(pattern):
            return pos == len(word)
        t = pattern[i]
        if t < 0:
            return pos < len(word) and word[pos] == -1-t and visit(i+1, pos+1)
        if t in env:
            val = env[t]
            return word[pos:pos+len(val)] == val and visit(i+1, pos+len(val))
        for end in range(pos + int(not erasing), len(word) + 1):
            env[t] = word[pos:end]
            if visit(i+1, end):
                return True
            del env[t]
        return False
    return visit(0, 0)


def defect_two():
    """All terminal-free upper patterns of defect two, with shortened gaps."""
    checked = 0
    for core in ((0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0)):
        for gaps in product(range(3), repeat=len(core)+1):
            q = []
            fresh = 2
            for i, size in enumerate(gaps):
                q.extend(range(fresh, fresh+size))
                fresh += size
                if i < len(core):
                    q.append(core[i])
            groups, sizes = skeleton(tuple(q))
            n = len(q) + 2
            assert layer(groups, sizes, 2) != columns(n)[0], q
            checked += 1
    return checked


def short_scan(alphabet):
    """Compare all canonical pattern pairs of length at most three on words up to four."""
    pool = [p for n in (1, 2, 3) for p in patterns(n, alphabet=alphabet)]
    words = [w for n in range(5) for w in product(range(alphabet), repeat=n)]
    weights = [2*len(p) - len({t for t in p if t >= 0}) for p in pool]
    counts = []
    for erasing in (False, True):
        masks = [sum(1 << i for i, w in enumerate(words) if matches(p, w, erasing))
                 for p in pool]
        counts.append(sum(a != b and not a & ~b
                          for i, a in enumerate(masks) for j, b in enumerate(masks)
                          if weights[i] <= weights[j]))
    assert counts[0] == 0 and counts[1] > 0
    print(f's={alphabet}: {len(pool)} patterns, {len(words)} words, '
          f'{counts} bounded strict-inclusion candidates (NE, E)')


def small_checks():
    checks = 0
    for q in patterns(3):
        groups, sizes = skeleton(q)
        for n in range(3, 6):
            mask = layer(groups, sizes, n-3)
            for i in range(1 << n):
                w = tuple((i >> j) & 1 for j in range(n))
                assert bool((mask >> i) & 1) == matches(q, w)
                checks += 1
    for n in range(9):
        for w in product(range(2), repeat=n):
            if matches((-1, 0, -2), w, True):
                assert matches((0, -1, -2, 1), w, True)
            checks += 1
    assert matches((0, -1, -2, 1), (1, 0, 1), True)
    assert not matches((-1, 0, -2), (1, 0, 1), True)
    for n in range(10):
        unary = (0,) * n
        assert matches((0, 1, 2, 3), unary) == (n >= 4)
        assert matches((0, 0, 1), unary) == (n >= 3)
        checks += 2
    for power in (1, 3, 9, 27):
        odd_word = (0,) * power
        assert matches((0, 0) + (1,)*power, odd_word, True)
        assert not matches((0, 0) + (1,)*(3*power), odd_word, True)
        checks += 2
    for s in (2, 3, 4, 5):
        # A square, the universal erasing pattern, and terminal-containing
        # morphic specializations. Include epsilon in the tested ground words.
        for n in range(5):
            for w in product(range(s), repeat=n):
                square = n % 2 == 0 and w[:n//2] == w[n//2:]
                assert matches((0, 0), w, True) == square
                assert matches((0, 1, 2), w, True)
                assert not matches((-1, 0, 0), w) or matches((-1, 1, 2), w)
                assert matches((-1, 0, 0), (0,) + w, True) == square
                assert matches((-1, 0, 1, 2), (0,) + w, True)
                checks += 5
        # All shortest prefixes matter for this terminal-free, nonmorphic
        # inclusion. Only the s+1 selected letters enter its certificate.
        for letters in product(range(s), repeat=s+1):
            assert len(set(letters)) < len(letters)
            i, j = next((i, j) for j in range(1, len(letters))
                        for i in range(j) if letters[i] == letters[j])
            w = [0] * (2*s + 3)
            for h, a in enumerate(letters):
                w[2*h+1] = a
            assert matches((0, 1, 2, 1, 3), tuple(w))
            assert 2*j+1 > 2*i+2
            checks += 1
    return checks


if __name__ == '__main__':
    print(f'{small_checks()} direct checks passed', flush=True)
    print(f'{defect_two()} reduced defect-two cases passed', flush=True)
    for s in (2, 3, 4, 5):
        short_scan(s)
    for n in range(1, 13):
        built, tested = binary_search(n)
        print(f'n={n}: {built} exact target layers, {tested} source comparisons', flush=True)
    print('All assertions passed.')
