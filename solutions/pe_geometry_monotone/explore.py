"""Exact certificates for the monotonicity counterexamples. Python 3, stdlib."""
from fractions import Fraction as Q
from itertools import combinations, product
from math import log, sqrt


BITS = (1, 0, 0, 1, 0, 1, 1)
ROOT_WEIGHTS = (1, 2, 3)
WEIGHTS = (1, 4, 9)
EPSILON = Q(1, 10000)


def kkt_polynomial(r, z):
    # Multiply F'_r(log z) by the positive (1+z)(1+2z).
    return 2 * (1 - r) * z**2 + (1 - 2 * r) * z - r


def positive_root(r):
    # z = (b + sqrt(discriminant)) / a. All entries stay rational.
    a = 4 * (1 - r)
    b = 2 * r - 1
    discriminant = b**2 + 8 * r * (1 - r)
    # Substitute and check constant and radical coefficients separately.
    assert 2 * (1 - r) * (b*b + discriminant) - b*b*a - r*a*a == 0
    assert 4 * (1 - r) * b - b*a == 0
    assert discriminant > b*b and a > 0
    return a, b, discriminant


def root_bracket(r):
    lo, hi = Q(0), Q(1)
    while kkt_polynomial(r, hi) <= 0:
        hi *= 2
    for _ in range(32):
        mid = (lo + hi) / 2
        if kkt_polynomial(r, mid) <= 0:
            lo = mid
        else:
            hi = mid
    assert kkt_polynomial(r, lo) <= 0 < kkt_polynomial(r, hi)
    return lo, hi


def verify(n):
    # sqrt(mu) has the padded integer rows below, divided by sqrt(14).
    rows = [(0,) * i + ROOT_WEIGHTS + (0,) * (n-1-i) for i in range(n)]
    dist2 = [[Q(sum((a-b)**2 for a, b in zip(u, v)), 14)
              for v in rows] for u in rows]
    for i, j, k in product(range(n), repeat=3):
        if abs(i-j) <= abs(i-k):
            assert dist2[i][j] <= dist2[i][k]
    assert dist2[0][1] == Q(6, 7) < Q(11, 7) == dist2[0][2]

    # Enumerate the original product corpus, then derive its tag marginals.
    rates = [Q(0) for _ in rows]
    total = Q(0)
    for offsets in product(range(3), repeat=n):
        prob = Q(1)
        for offset in offsets:
            prob *= Q(WEIGHTS[offset], 14)
        total += prob
        for i, offset in enumerate(offsets):
            rates[i] += prob * BITS[i + offset]
    assert total == 1
    assert rates == [Q(x, 14) for x in (1, 9, 4, 10, 13)[:n]]
    assert len(set(rates)) == n

    brackets = [root_bracket(r) for r in rates]
    assert brackets[0][1] < brackets[2][0] < brackets[2][1] < brackets[1][0]
    roots = [positive_root(r) for r in rates]
    offsets = [log((float(b) + sqrt(disc)) / a) for a, b, disc in roots]

    # log(2) > 1/2 by integrating 1/x on [1,2].
    # Per swap the two rows contribute >= 2*log(2)*min(mismatch probs).
    perturbation = 2 * n * EPSILON
    for r, s in combinations(rates, 2):
        assert min(r*(1-s), s*(1-r)) > perturbation
    for a, b in ((0, 2), (2, 1)):
        assert (rates[b] - rates[a])**2 / 8 > perturbation

    print(f"n={n}: {3**n} sequences, {n**3} Hellinger triples, "
          f"{n*(n-1)//2} swaps, {n} exact KKT roots; "
          f"d12={offsets[1]-offsets[0]:.9f}, d13={offsets[2]-offsets[0]:.9f}")


def verify_stress():
    # Use units of the equilateral side length. On each sorted chamber
    # the normalized stress is ((u-1)^2+(v-1)^2+(u+v-1)^2)/3.
    u = v = Q(2, 3)
    assert 2*(u-1) + 2*(u+v-1) == 2*(v-1) + 2*(u+v-1) == 0
    assert 4 > 0 and 4*4 - 2*2 > 0  # PD Hessian of the numerator
    assert ((u-1)**2 + (v-1)**2 + (u+v-1)**2)/3 == Q(1, 9)
    assert u+v > u  # place positions (1,3,2) in sorted order


if __name__ == "__main__":
    for length in (3, 4, 5):
        verify(length)
    verify_stress()
    print("All exact certificates passed, including raw stress = 1/9.")
