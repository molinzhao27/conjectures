"""Exact checks for arXiv:1912.00963v3, Conjectures 3.9 and 3.13.

Run with Python 3.10+ and the standard library. Words are integer bit masks.
"""
from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import combinations, permutations, product
from operator import and_


def word(digits):
    return sum(1 << (int(digit) - 1) for digit in digits)


def words(digit_strings):
    return {word(digits) for digits in digit_strings.split()}


def code(digit_strings):
    return {0} | words(digit_strings)


def subsets(face):
    part = face
    while part:
        yield part
        part = (part - 1) & face
    yield 0


def complex_of(codewords):
    return {part for face in codewords for part in subsets(face)}


def facets(faces):
    return {s for s in faces if not any(s != t and s & t == s for t in faces)}


def facet_intersections(maximal):
    return {reduce(and_, family) for size in range(2, len(maximal) + 1)
            for family in combinations(maximal, size)}


def link(complex_, face):
    return {t ^ face for t in complex_ if t & face == face}


def collapses_to_point(faces):
    """Each removed pair is a free face and its unique strict coface."""
    remaining = set(faces)
    while len(remaining) > 2:
        for face in sorted(remaining - {0}):
            above = [t for t in remaining if t != face and t & face == face]
            if len(above) == 1:
                coface, = above
                assert (coface ^ face).bit_count() == 1
                remaining.difference_update((face, coface))
                break
        else:
            return False  # No certificate found by this order.
    return 0 in remaining and len(remaining) == 2


def check_four_neurons():
    complexes = {frozenset({0})}
    for face in range(1, 16):
        simplex = set(subsets(face))
        complexes |= {k | simplex for k in complexes}
    euler_counts = Counter()
    for k in complexes:
        for s in facet_intersections(facets(k)) - {0}:
            chi = sum((-1) ** (t.bit_count() - 1) for t in link(k, s) if t)
            euler_counts[chi] += 1
    assert len(complexes) == 167 and euler_counts == {0: 8, 2: 222, 3: 36}
    print("n=4: 167 complexes containing the empty face; 266 nonempty "
          "facet-intersection links, all with Euler characteristic different from 1.")


def axis_samples(bounds):
    endpoints = sorted({x for interval in bounds for x in interval})
    midpoints = [Fraction(x + y, 2) for x, y in zip(endpoints, endpoints[1:])]
    return [endpoints[0] - 1, *endpoints, *midpoints, endpoints[-1] + 1]


def box_code(boxes, closed):
    """One exact point per product cell, including all boundary strata."""
    axes = [axis_samples(bounds) for bounds in zip(*boxes)]
    result = set()
    for point in product(*axes):
        active = 0
        for i, box in enumerate(boxes):
            if all((lo <= x <= hi if closed else lo < x < hi)
                   for x, (lo, hi) in zip(point, box)):
                active |= 1 << i
        result.add(active)
    return result


def verify_realizations(boxes, expected, maxima, m1, m2, sigma, missing):
    assert box_code(boxes, closed=False) == box_code(boxes, closed=True) == expected
    assert facets(expected) == maxima and len(maxima) >= 3
    assert {m1, m2} <= maxima and m1 != m2
    assert sigma in expected and sigma != m1 and sigma & m1 == sigma
    assert sigma & m2 not in expected and facet_intersections(maxima) - expected
    k = complex_of(expected)
    assert len(missing) >= 2 and missing <= k - expected
    assert all(collapses_to_point(link(k, s)) for s in missing)
    print(f"n={len(boxes)}: both box realizations give exactly {len(expected)} words.")


def check_first_conjecture():
    check_four_neurons()
    intervals = [(1, 8), (0, 5), (4, 9), (6, 7), (2, 3)]
    verify_realizations([(interval,) for interval in intervals],
                        code("2 3 12 13 125 123 134"), words("125 123 134"),
                        word("125"), word("134"), word("12"), words("1 5"))
    rectangles = [((-5, 1), (-6, 6)), ((-1, 5), (-6, 6)),
                  ((-4, 4), (1, 5)), ((-4, 4), (-5, -1)),
                  ((-3, -2), (2, 4)), ((2, 3), (2, 4))]
    verify_realizations(rectangles, code("123 124 135 236 12 13 14 23 24 1 2"),
                        words("123 124 135 236"), word("123"), word("236"),
                        word("13"), words("3 4"))


NON_OPEN = code("2345 123 134 145 13 14 23 34 45 3 4")
NON_CLOSED = code("123 125 145 234 12 15 23 4")
SEVEN = code("3 4 34 45 123 137 147 236 1236 1237 1347 1457 23456")


def embed(face, labels):
    return sum(1 << (label - 1) for i, label in enumerate(labels) if face & (1 << i))


def common_extension(n, labels):
    left = word("12345")
    right = embed(left, labels)
    target = {embed(t, labels) for t in NON_CLOSED}
    candidate = {t for t in range(1 << n) if t & left in NON_OPEN and t & right in target}
    if {t & left for t in candidate} == NON_OPEN and {t & right for t in candidate} == target:
        return candidate
    return None


def check_second_conjecture():
    labels = (1, 2, 6, 4, 7)
    assert common_extension(7, labels) == SEVEN
    assert {s & word("12345") for s in SEVEN} == NON_OPEN
    assert {s & word("12467") for s in SEVEN} == code("126 127 147 246 12 17 26 4")
    assert {embed(s, labels) for s in NON_CLOSED} == {s & word("12467") for s in SEVEN}

    maximal = facets(SEVEN)
    assert maximal == words("1236 1237 1347 1457 23456")
    exceptional = {"1": "236 237 347 457", "13": "26 27 47",
                   "17": "23 34 45", "23": "456 16 17"}
    assert facet_intersections(maximal) - SEVEN == {word(s) for s in exceptional}
    k = complex_of(SEVEN)
    for s, link_facets in exceptional.items():
        assert facets(link(k, word(s))) == words(link_facets)
    missing = k - SEVEN
    assert len(missing) == 46 and all(collapses_to_point(link(k, s)) for s in missing)
    print("n=7: 14 words, 5 facets, 46 missing-face links collapse to points; "
          "both five-neuron restrictions verified.")

    # Reproduce the overlap search for these two specific obstruction codes.
    for n in (6, 7):
        hits = 0
        for shared in combinations(range(1, 6), 10 - n):
            for mapping in permutations((*shared, *range(6, n + 1))):
                candidate = common_extension(n, mapping)
                if candidate is not None:
                    delta = complex_of(candidate)
                    if all(collapses_to_point(link(delta, s)) for s in delta - candidate):
                        hits += 1
        print(f"n={n}: {hits} labeled gluings with both projections and collapse certificates.")


if __name__ == "__main__":
    check_first_conjecture()
    check_second_conjecture()
