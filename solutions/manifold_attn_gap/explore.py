"""Run with .venv/bin/python solutions/manifold_attn_gap/explore.py (NumPy)."""

import numpy as np


def sphere_check():
    # Product quadrature for normalized surface area: height and azimuth
    # are independently uniform. Common normalization cancels in Bayes' rule.
    t = 1 / np.tanh(1.0) - 1
    sigma = 1 - t
    radius = sigma / t
    height, weights = np.polynomial.legendre.leggauss(32)
    phi = 2 * np.pi * np.arange(64) / 64
    h, angle = np.meshgrid(height, phi, indexing="ij")
    ring = np.sqrt(1 - h**2)
    nodes = radius * np.stack((ring * np.cos(angle), ring * np.sin(angle), h), -1)
    nodes = nodes.reshape(-1, 3)
    areas = np.repeat(weights, len(phi))

    directions = np.random.default_rng(261110857).normal(size=(97, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    z = t * radius * directions
    dist_sq = np.sum((z[:, None, :] - t * nodes[None, :, :]) ** 2, axis=-1)
    posterior = areas * np.exp(-dist_sq / (2 * sigma**2))
    means = (posterior @ nodes) / posterior.sum(axis=1, keepdims=True)
    # The residual stack with zero output projections returns z exactly.
    error = np.max(np.sum((means - z) ** 2, axis=1))
    print(f"Sphere: t={t:.12f}, R={radius:.12f}, 97 directions, max squared error={error:.3e}")
    assert error < 1e-24


def square_projections(channels, eps):
    # Each head is scalar. Layout: content channels, tag, unit, optional padding.
    width = 2 * channels + 1
    tag, unit = channels, channels + 1
    q, k, v, out = [np.zeros((width, width)) for _ in range(4)]
    b = np.arctanh(0.5)
    slope, curvature = 0.75, -0.75  # tanh derivatives at b
    q[unit, :] = b
    k[tag, :] = 1
    v[tag, :] = 1
    even = 1 / (curvature * eps**2)
    odd = 1 / (2 * slope * eps)
    for r in range(channels):
        plus, minus = 2 * r, 2 * r + 1
        q[r, plus], q[r, minus] = eps, -eps
        out[plus, r], out[minus, r] = even - odd, even + odd
        out[-1, r] = -2 * even
    return q, k, v, out


def attention(stream, projections):
    q, k, v, out = projections
    queries, keys, values = stream @ q, stream @ k, stream @ v
    # batch x query position x key position x scalar head
    scores = queries[:, :, None, :] * keys[:, None, :, :]
    weights = np.exp(scores - scores.max(axis=2, keepdims=True))
    weights /= weights.sum(axis=2, keepdims=True)
    return stream + (weights * values[:, None, :, :]).sum(axis=2) @ out


def polynomial_check():
    grid = np.linspace(-1, 1, 41)
    x, y = np.meshgrid(grid, grid)
    forms = np.stack((x, y, (x - y) / 2), axis=-1).reshape(-1, 3)
    # Every point occurs at both positions, with a different companion patch.
    inputs = np.stack((forms, np.roll(forms, 311, axis=0)), axis=1)
    readout = np.array([[1, 0], [-0.5, 0.25], [0, 1]])
    channels = inputs.shape[-1]
    width = 2 * channels + 1
    base = np.zeros((*inputs.shape[:2], width))
    base[:, :, :channels] = inputs
    base[:, :, channels] = [1, -1]
    base[:, :, channels + 1] = 1
    print(f"Polynomial check: {len(forms)} image pairs, {4 * width**2} attention parameters/layer")
    for depth in (1, 2, 3):
        target = inputs ** (2**depth) @ readout
        volterra = inputs.copy()
        for _ in range(depth):
            volterra = volterra * volterra
        np.testing.assert_allclose(volterra @ readout, target, atol=1e-14)
        errors = []
        for eps in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
            matrices = square_projections(channels, eps)
            assert sum(matrix.size for matrix in matrices) == 196
            stream = base
            for _ in range(depth):
                stream = attention(stream, matrices)
            estimate = stream[:, :, :channels] @ readout
            errors.append(np.max(np.sum((estimate - target) ** 2, axis=-1)))
        print(f"  degree {2**depth}: " + " ".join(f"{e:.3e}" for e in errors))
        # Halving epsilon asymptotically divides the squared error by 16.
        ratios = np.array(errors[:-1]) / errors[1:]
        assert np.all((ratios[-3:] > 15) & (ratios[-3:] < 17))


if __name__ == "__main__":
    sphere_check()
    polynomial_check()
