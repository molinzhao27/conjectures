"""Numerical checks for the cumulative-average matrix GARIP theorem."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Game:
    name: str
    A: np.ndarray
    row: np.ndarray
    col: np.ndarray


def game(name, matrix, row, col):
    return Game(name, np.array(matrix, float), np.array(row, float), np.array(col, float))


def kl(p, q):
    active = p > 0
    return np.sum(p[active] * np.log(p[active] / q[:, active]), axis=1)


def normalized_step(z, direction, eta, split):
    weighted = z * np.exp(eta * direction)
    for block in (slice(None, split), slice(split, None)):
        weighted[:, block] /= weighted[:, block].sum(axis=1, keepdims=True)
    return weighted


def gradients(z, A, split):
    return np.column_stack((z[:, split:] @ A.T, -z[:, :split] @ A))


def gap(z, A, split):
    g = gradients(z, A, split)
    return g[:, :split].max(axis=1) + g[:, split:].max(axis=1)


def energy(p, z, prev_g, eta, displacement):
    return (kl(p, z) - eta[:, 0] * (prev_g * (z - p)).sum(axis=1)
            + displacement**2 / 16)


def check_game(case, rng, steps=7000):
    A = case.A
    m, n = A.shape
    p = np.r_[case.row, case.col]
    assert abs(gap(p[None, :], A, m)[0]) < 1e-13
    M = max(1, np.max(np.abs(A)))
    support = p > 0
    mu = p[support].min()
    v = case.row @ A @ case.col
    slacks = np.r_[v - A @ case.col, case.row @ A - v]
    gamma = slacks[~support].min() if (~support).any() else np.inf
    eta_max = min(1 / (64 * M), gamma / (1200 * M**2))
    c_max = mu / 8
    beta_max = c_max / (1 + c_max)
    # Two corners with opposite scale separations and one tiny-steps control.
    # The fourth certified pair and the two exploratory pairs test visible motion.
    params = [(beta_max, eta_max / 200), (beta_max / 200, eta_max),
              (1e-7, min(1e-7, eta_max)), (beta_max / 4, eta_max),
              (0.015, 0.18 / M), (0.45, 0.01 / M)]
    configurations = np.repeat(params, 2, axis=0)
    beta = configurations[:, :1]
    eta = configurations[:, 1:]
    c = beta[:, 0] / (1 - beta[:, 0])
    starts = np.column_stack((rng.dirichlet(np.ones(m), 2),
                              rng.dirichlet(np.ones(n), 2)))
    if case.name == "weak boundary":
        # Row two is profitable against this initial column strategy.
        starts[0] = [1 - 1e-7, 1e-7, 0.9, 0.1]
    z = np.tile(starts, (len(params), 1))
    a = z.copy()
    gprev = np.zeros_like(z)
    dprev = np.zeros(len(z))
    practical = slice(8, 10)
    certified = slice(0, 8)
    initial = gap(z, A, m)
    half_increase = -np.inf
    worst_residual = -np.inf
    for t in range(steps):
        alpha = 1 / (t + 2)
        g = gradients(z, A, m)
        h = normalized_step(z, 2 * g - gprev, eta, m)
        w = (1 - beta) * h + beta * a
        anew = a + alpha * (w - a)
        dnext = np.abs(w - z).sum(axis=1)
        H, Hnext = kl(p, a), kl(p, anew)
        E = energy(p, z, gprev, eta, dprev)
        Enext = energy(p, w, g, eta, dnext)
        # Raw half-step KL, the source's convexity inequality, and (6).
        half = kl(p, h)
        half_increase = max(half_increase, (half - kl(p, z))[certified].max())
        assert np.all(kl(p, w) <= (1 - beta[:, 0]) * half + beta[:, 0] * H + 3e-14)
        if t >= 1:
            R = (dprev**2 + dnext**2) / 32
            R += c * mu / 8 * ((w - a)[:, support]**2).sum(axis=1)
            if gamma < np.inf:
                R += eta[:, 0] * gamma / 4 * z[:, ~support].sum(axis=1)
            # Scale the residual of (6) to avoid amplification of roundoff.
            delta = (alpha * (Enext - E + R) + c * (Hnext - H))[certified]
            worst_residual = max(worst_residual, delta.max())
            assert delta.max() < 3e-14
        z, a, gprev, dprev = w, anew, g, dnext
    assert np.all(z > 0) and np.all(a > 0)
    assert np.allclose(z[:, :m].sum(axis=1), 1, atol=2e-14, rtol=0)
    assert np.allclose(z[:, m:].sum(axis=1), 1, atol=2e-14, rtol=0)
    final = gap(z, A, m)
    assert np.all(final >= -2e-14)
    if np.any(A != 0):
        assert np.all(final[practical] < initial[practical])
        assert np.all(final[practical] < 0.003)
    print(f"{case.name:19s} practical gaps {initial[practical]} -> {final[practical]}"
          f"  large-anchor gaps {final[-2:]}  max scaled residual {worst_residual:.2g}")
    return half_increase


def main():
    games = [
        game("matching pennies", [[1, -1], [-1, 1]], [.5, .5], [.5, .5]),
        game("rock-paper-scissors", [[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
             [1/3]*3, [1/3]*3),
        game("asymmetric", [[2, -1], [-2, 1]], [.5, .5], [1/3, 2/3]),
        game("duplicate actions", [[1, -1], [1, -1], [-1, 1]],
             [.25, .25, .5], [.5, .5]),
        game("dominated actions", [[1, -1, 2], [-1, 1, 2], [-2, -2, 0]],
             [.5, .5, 0], [.5, .5, 0]),
        game("weak boundary", [[0, 0], [1, -1]], [1, 0], [.25, .75]),
        game("zero game", np.zeros((2, 3)), [.5, .5], [1/3]*3),
    ]
    rng = np.random.default_rng(260622688)
    np.set_printoptions(precision=7)
    increases = [check_game(example, rng) for example in games]
    assert increases[0] > 1e-7
    print(f"Matching-pennies raw half-step KL increase {increases[0]:.7g}.")
    print("84 trajectories of 7000 updates, including 56 in the proof rectangle. All assertions passed.")


if __name__ == "__main__":
    main()
