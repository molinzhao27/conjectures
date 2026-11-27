"""Exact & Monte Carlo verification of the refutation of Conjecture 1.conj.2
in Gayrard (arXiv:2504.04879, Mix-ALL_v2.tex).
"""
from fractions import Fraction
import itertools
import math
import numpy as np


def chow_exact(w):
    """Exact Chow parameters Phi(w) and min margin over r in {-1, 1}^n."""
    n = len(w)
    sums = [Fraction(0) for _ in range(n)]
    min_abs = None
    for r in itertools.product((-1, 1), repeat=n):
        dot = sum(r[i] * w[i] for i in range(n))
        assert dot != 0, f"Tie found at r={r} for w={w}"
        a = abs(dot)
        if min_abs is None or a < min_abs:
            min_abs = a
        sgn = 1 if dot > 0 else -1
        for i in range(n):
            sums[i] += r[i] * sgn
    denom = Fraction(1, 2 ** n)
    return [s * denom for s in sums], min_abs


def verify_classical_hopfield():
    print("=== 1. Classical Hopfield F(x) = x^2 / 2 ===")
    m5 = [Fraction(5, 8), Fraction(3, 8), Fraction(3, 8), Fraction(1, 8), Fraction(1, 8)]
    phi5, margin5 = chow_exact(m5)
    assert phi5 == m5, f"Mismatch: {phi5} != {m5}"
    assert margin5 == Fraction(1, 8)
    print(f"  n=5: m={list(map(str, m5))} is EXACT Chow fixed point! margin={margin5}")

    # Verify lifted n = 7, 9, 11
    for k in [1, 2, 3]:
        alpha = Fraction(math.comb(2 * k, k), 4 ** k)
        mk = [alpha] * (2 * k) + [alpha * x for x in m5]
        phik, margink = chow_exact(mk)
        assert phik == mk, f"Mismatch at n={2*k+5}"
        assert margink == alpha * Fraction(1, 8)
        print(f"  n={2*k+5} (k={k}): EXACT Chow fixed point! alpha={alpha}, margin={margink}")

    # Monte Carlo check of local minimality for N=20000, M=8 at n=5
    # (noise std ~ sqrt(3/20000) = 0.0122, max over N sites ~ 0.054 < margin 0.125)
    rng = np.random.default_rng(2026)
    N, M = 20000, 8
    xi = rng.choice([-1, 1], size=(M, N))
    w_vec = np.array([5, 3, 3, 1, 1] + [0] * (M - 5), dtype=float)
    sigma = np.sign(w_vec @ xi)
    assert np.all(sigma != 0)
    overlaps = (xi @ sigma) / N
    # One Glauber / Hopfield synchronous update T^G(sigma) WITHOUT self-coupling (j != i)
    field = overlaps @ xi - (M / N) * sigma
    sigma_next = np.sign(field)
    assert np.array_equal(sigma, sigma_next), "Not a Hopfield T^G fixed point!"
    print(f"  Monte Carlo (N={N}, M={M}, no self-coupling): EXACT Hopfield T^G fixed point! overlaps[:5]={np.round(overlaps[:5], 4)}")


def verify_dense_hopfield():
    print("\n=== 2. Dense Hopfield F(x) = x^p / p (p >= 3) ===")
    # Direct exhaustive check for p = 3, 4 at n = 7
    m7 = [Fraction(11, 32)] * 3 + [Fraction(9, 32)] * 4
    for p in [3, 4]:
        w = [x ** (p - 1) for x in m7]
        phi, margin = chow_exact(w)
        assert phi == m7, f"Mismatch at p={p}"
        print(f"  p={p}, n=7: m=(11/32^3, 9/32^4) EXACT Chow fixed point! margin={float(margin):.6e}")

    # Check lifted n = 9 for p = 3, 4
    alpha2 = Fraction(1, 2)
    m9 = [alpha2, alpha2] + [alpha2 * x for x in m7]
    for p in [3, 4]:
        w = [x ** (p - 1) for x in m9]
        phi, margin = chow_exact(w)
        assert phi == m9
        print(f"  p={p}, n=9 (lifted k=1): EXACT Chow fixed point! margin={float(margin):.6e}")

    # Verify Substep 3A exact fractions for p = 3..19
    table = [
        ([3, 4], 1, 2, Fraction(11, 9)),
        ([5, 6], 2, 6, Fraction(121, 115)),
        ([7], 2, 5, Fraction(587, 563)),
        ([8, 9, 10], 2, 4, Fraction(235, 228)),
        ([11, 12], 3, 13, Fraction(138977, 137107)),
        (list(range(13, 20)), 2, 3, Fraction(127, 125)),
    ]
    for ps, a, b, q_expected in table:
        x = Fraction(2 * math.comb(2 * b, b - a - 1), math.comb(2 * a + 2 * b, a + b))
        q = (1 + x) / (1 - Fraction(a + 1, b) * x)
        assert q == q_expected, f"q({a},{b})={q} != {q_expected}"
        lo = Fraction(2 * a + 2, 2 * a + 1)
        hi = Fraction(2 * a, 2 * a - 1)
        for p in ps:
            qp = q ** (p - 1)
            assert lo < qp < hi, f"Failed for p={p}, ({a},{b}): {float(qp)} not in ({float(lo)}, {float(hi)})"
    print("  Substep 3A (p = 3..19): ALL exact rational inequalities verified!")

    # Verify Substep 3B constructively for p = 20..500
    for p in range(20, 501):
        found = None
        for a in range(4, 15):
            lo = Fraction(2 * a + 2, 2 * a + 1)
            hi = Fraction(2 * a, 2 * a - 1)
            b0 = {4: 10, 5: 14}.get(a, (a * a) // 2)
            # Check if p-1 is in range for this a
            for b in range(b0, b0 + 5000):
                x = Fraction(2 * math.comb(2 * b, b - a - 1), math.comb(2 * a + 2 * b, a + b))
                q = (1 + x) / (1 - Fraction(a + 1, b) * x)
                # Fast log check then exact Fraction power check
                val = (p - 1) * math.log(float(q))
                if val > math.log(float(hi)):
                    break
                if val > math.log(float(lo)):
                    # Exact rational verification
                    if lo < q ** (p - 1) < hi:
                        found = (a, b)
                        break
            if found:
                break
        assert found is not None, f"No (a, b) found for p={p}!"
    print("  Substep 3B (p = 20..500): explicit exact (a, b) verified for EVERY integer p in [20, 500]!")


def verify_general_smooth():
    print("\n=== 3. General Smooth F(x) Counterexamples ===")
    # 3a. Non-odd F'(x) = e^x - 1 (F(x) = e^x - x) at n = 1 and n = 3
    for label, m_neg in [("n=1", [Fraction(-1)]), ("n=3", [Fraction(-1, 2)] * 3)]:
        w_neg = [math.exp(float(x)) - 1.0 for x in m_neg]
        n = len(m_neg)
        sums = [0] * n
        for r in itertools.product((-1, 1), repeat=n):
            dot = sum(r[i] * w_neg[i] for i in range(n))
            assert abs(dot) > 1e-9
            sgn = 1 if dot > 0 else -1
            for i in range(n):
                sums[i] += r[i] * sgn
        phi = [Fraction(s, 2 ** n) for s in sums]
        assert phi == m_neg, f"Mismatch for e^x - x ({label}): {phi} != {m_neg}"
        print(f"  F(x)=e^x-x, {label}: m={list(map(str, m_neg))} EXACT tie-free Chow fixed point!")

    # 3b. Odd strictly increasing F'(x) = 4x / sqrt(1 + 4x^2) (F(x) = sqrt(1 + 4x^2)) at n = 5
    m5 = [Fraction(7, 8)] + [Fraction(1, 8)] * 4
    w = [4 * float(x) / math.sqrt(1 + 4 * float(x) ** 2) for x in m5]
    # Check ratio in (2, 4)
    ratio = w[0] / w[1]
    assert 2.0 < ratio < 4.0
    # Compute Chow parameters
    sums = [0] * 5
    for r in itertools.product((-1, 1), repeat=5):
        dot = sum(r[i] * w[i] for i in range(5))
        assert abs(dot) > 1e-9
        sgn = 1 if dot > 0 else -1
        for i in range(5):
            sums[i] += r[i] * sgn
    phi = [Fraction(s, 32) for s in sums]
    assert phi == m5
    print(f"  F(x)=sqrt(1+4x^2), n=5: m=(7/8, 1/8^4) EXACT Chow fixed point! ratio={ratio:.6f} in (2,4)")


if __name__ == "__main__":
    verify_classical_hopfield()
    verify_dense_hopfield()
    verify_general_smooth()
    print("\nALL HOPFIELD MIXED MEMORIES REFUTATION VERIFICATIONS PASSED!")
