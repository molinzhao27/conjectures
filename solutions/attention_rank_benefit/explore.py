"""Fixed-weight rank-one nearest-neighbor transformer. Run with Python and NumPy."""
from itertools import chain

import numpy as np


def unit_sphere(rng, shape):
    points = rng.standard_normal(shape)
    return points / np.linalg.norm(points, axis=-1, keepdims=True)


def directions(count, dimension):
    j = np.arange(count)
    if dimension == 2:
        phi = 2 * np.pi * j / count
        return np.column_stack((np.cos(phi), np.sin(phi)))
    if dimension == 3:
        z = 1 - (2 * j + 1) / count
        phi = j * np.pi * (3 - np.sqrt(5))
        radius = np.sqrt(1 - z**2)
        return np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    raise ValueError("The experiment supplies grids on the circle and two-sphere.")


def softmax(logits, axis):
    weights = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return weights / weights.sum(axis=axis, keepdims=True)


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def last_token(x, y, grid, comparison_gain, gate_gain, beta):
    """Evaluate the three layers at the readout, batching independent sequences."""
    batch, n, _ = x.shape
    count = len(grid)
    if n == 1:
        # All retrieval heads and the cancelling mean head read the same token.
        return x[:, 0].copy()
    log_odds = np.log(((n + 1) // 2) / (n // 2))
    scores = y @ grid.T
    pairwise = scores[:, :, None] - scores[:, None, :]
    wins = sigmoid(comparison_gain * pairwise + log_odds)
    idx = np.arange(count)
    wins[:, idx, idx] = 0
    gates = sigmoid(gate_gain * (wins.sum(axis=2) - (count - 1.5)) + log_odds)
    logits = (x @ grid.T) * (beta * gates[:, None, :])
    weights = softmax(logits, axis=1)
    # Sum directional heads, then subtract (count - 1) uniform heads' worth.
    combined = (weights - 1 / n).sum(axis=2) + 1 / n
    return (x.transpose(0, 2, 1) @ combined.reshape(batch, n, 1))[:, :, 0]


def literal_transformer(x, y, grid, comparison_gain, gate_gain, beta, serial):
    """Evaluate factorized heads on every token, including diagonally masked scores."""
    n, d = x.shape
    count = len(grid)
    const, bit = d, d + 1
    wins = d + 2 + np.arange(count)
    gates = wins + count
    out = d + 2 + 2 * count + np.arange(d)
    state = np.zeros((n + 1, out[-1] + 1))
    basis = np.eye(state.shape[1])
    state[:, :d] = np.vstack((x, y))
    state[:, const] = 1
    state[::2, bit] = 1
    embedded_grid = grid @ basis[:d]

    def layer(heads):
        total = np.zeros_like(state)
        for key, query, value, output in heads:
            logits = np.outer(state @ query, state @ key)
            np.fill_diagonal(logits, -np.inf)
            update = np.outer(softmax(logits, 1) @ (state @ value), output)
            if serial:
                state[:] += update
            else:
                total += update
        state[:] += total

    layer((basis[bit], comparison_gain * (u - v), basis[bit], basis[wins[k]])
          for k, u in enumerate(embedded_grid)
          for j, v in enumerate(embedded_grid) if k != j)
    layer((basis[bit], gate_gain * (basis[w] - (count - 1.5) * basis[const]),
           basis[bit], basis[g]) for w, g in zip(wins, gates))
    search = ((u, beta * basis[g], basis[j], basis[out[j]])
              for u, g in zip(embedded_grid, gates) for j in range(d))
    mean = ((basis[const], basis[const], basis[j], (1 - count) * basis[out[j]])
            for j in range(d))
    layer(chain(search, mean))
    return state[-1, out]


def verify_layers(rng):
    grid = directions(5, 2)
    # Moderate gains exercise fractional gates and odd-length offsets.
    gains = (4.5, 7.0, 9.0)
    max_error = 0
    checks = 0
    for n in (1, 2, 3, 4, 7, 16):
        x = unit_sphere(rng, (n, 2))
        for y in (unit_sphere(rng, (2,)), np.array([np.cos(np.pi/5), np.sin(np.pi/5)])):
            expected = last_token(x[None], y[None], grid, *gains)[0]
            for serial in (False, True):
                actual = literal_transformer(x, y, grid, *gains, serial=serial)
                max_error = max(max_error, np.linalg.norm(actual - expected))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                checks += 1
            np.testing.assert_allclose(last_token(x[::-1][None], y[None], grid, *gains)[0],
                                       expected, rtol=0, atol=1e-12)
    print(f"{checks} literal masked-layer checks, max discrepancy {max_error:.2e}")


def run():
    rng = np.random.default_rng(240716153)
    verify_layers(rng)
    comparison_gain, gate_gain, beta = 30000, 90, 300
    print(" d grid     N  trials   rank-one MSE       SE    full-rank MSE")
    for d, count in ((2, 64), (3, 96)):
        grid = directions(count, d)
        for n in (1, 2, 3, 8, 31, 128, 511, 2048, 4097):
            trials = 1024 if n <= 128 else 256
            batch = min(32, max(1, 900000 // (n * count)))
            rank_one, full_rank = [], []
            for start in range(0, trials, batch):
                size = min(batch, trials - start)
                x = unit_sphere(rng, (size, n, d))
                y = unit_sphere(rng, (size, d))
                dots = (x * y[:, None]).sum(axis=2)
                target = x[np.arange(size), dots.argmax(axis=1)]
                pred = last_token(x, y, grid, comparison_gain, gate_gain, beta)
                full = (x * softmax(beta * dots, 1)[:, :, None]).sum(axis=1)
                rank_one.extend(np.sum((pred - target)**2, axis=1))
                full_rank.extend(np.sum((full - target)**2, axis=1))
            mse = np.mean(rank_one)
            se = np.std(rank_one, ddof=1) / np.sqrt(trials)
            print(f"{d:2d} {count:4d} {n:5d} {trials:7d} {mse:14.7f} {se:8.5f}"
                  f" {np.mean(full_rank):16.7f}")
            assert np.isfinite(mse) and mse < 0.25


if __name__ == "__main__":
    run()
