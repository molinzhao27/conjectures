"""Numerical checks for solution.md. Run with the project's Python interpreter."""
from math import cos, exp, log, radians, sin, sqrt

import numpy as np


TAU = 0.05
H = radians(5.7 / 2)
A, B, S = 0.600, 0.599, 0.001


def potential(dots, k, norms=None):
    """Cosine selection; dots are the actual scores used in the exponent."""
    ranks = dots if norms is None else dots / norms
    selected = dots[np.argpartition(ranks, -k)[-k:]]
    peak = selected.max()
    return 1 - peak - TAU * log(np.exp((selected - peak) / TAU).mean())


def shift(x, y, k, norms=None):
    return abs(potential(cos(H) * x + sin(H) * y, k, norms)
               - potential(cos(H) * x - sin(H) * y, k, norms))


def paired_family():
    rng = np.random.default_rng(260808485)
    count = 0
    for K in (30, 100, 300):
        n = 30_000 // K
        for noisy in (False, True):
            for t in (-S, -S / 10, -S / 10000, 0.0):
                x = np.tile(np.repeat([A, B], n // 2), (K, 1))
                y = np.tile(np.repeat([t, S], n // 2), (K, 1))
                phi = rng.uniform(-0.001, 0.001, (K, n)) if noisy else np.zeros((K, n))
                radius = np.sqrt(1 - x*x - y*y)
                z, u = radius * np.cos(phi), radius * np.sin(phi)
                # z is cluster-specific, u is a shared transverse coordinate.
                r0 = sqrt(1 - A*A - t*t)
                dist2 = (x - A)**2 + (y - t)**2 + (z - r0)**2 + u*u
                assert dist2.max() < 0.004**2
                assert K * 0.004**2 < (0.25 - 0.004)**2
                centers = np.column_stack([c.mean(axis=1) for c in (x, y, z, u)])
                norms = np.linalg.norm(centers, axis=1)
                cx, cy = centers[:, 0], centers[:, 1]
                for k in (1, 10, K):
                    full = shift(x.ravel(), y.ravel(), k)
                    raw = shift(cx, cy, k, norms)
                    unit = shift(cx / norms, cy / norms, k)
                    expected = sin(H) * (t + S)
                    assert abs(full - 2*sin(H)*abs(t)) < 2e-14
                    assert abs(raw - expected) < 2e-14
                    assert expected - 2e-14 <= unit <= expected / 0.996 + 2e-14
                    if not noisy:
                        assert abs(unit - expected / norms[0]) < 2e-14
                    if t == -S / 10:
                        assert abs(raw / full - 4.5) < 1e-9
                    count += 1
    print(f"Paired-family configurations verified: {count}")


def bootstrap():
    rng = np.random.default_rng(8426)
    K, N, k = 300, 30_000, 10
    x = np.full(K, 0.99)
    y = np.full(K, 0.04)
    x[0], y[0] = 1, 0
    anchor = shift(x, y, k)
    def partition(sign):
        gap = -0.01*cos(H) + sign*0.04*sin(H)
        return 1 + (k-1)*exp(gap / TAU)
    predicted = TAU * log(partition(1) / partition(-1))
    assert abs(anchor - predicted) < 1e-14
    trials = 128
    for counts in rng.multinomial(N, np.full(K, 1/K), size=trials):
        assert counts.min() > 0
        assert counts[0] >= k
        assert shift(np.repeat(x, counts), np.repeat(y, counts), k) == 0
    lower = (1 - K*exp(-100))*anchor
    upper = 2*sin(H)*exp(-40.5)
    assert lower / upper > 1e15
    print(f"Bootstrap: {trials} draws, anchor shift {anchor:.12g}, "
          f"proved expectation ratio > {lower / upper:.4g}")


def one_merge():
    k = 10
    for N in (100, 1000, 30_000):
        for t in (0, -1e-7):
            # The first k-1 support sites differ in transverse coordinates.
            x = np.r_[np.full(k-1, A), B, np.full(N-k-1, -A)]
            y = np.r_[np.full(k-1, t), S, np.zeros(N-k-1)]
            raw = shift(np.r_[x, A], np.r_[y, t], k)
            anchor = shift(x, y, k)
            assert abs(raw - 2*sin(H)*abs(t)) < 2e-14
            assert anchor > 9e-6
            if t:
                assert anchor / raw > 900
    print("Near-endpoint checks passed at K = N-1.")


if __name__ == "__main__":
    paired_family()
    bootstrap()
    one_merge()
