"""Reproduce the MHA checks with Python 3 and NumPy. All matrices act on rows."""
import numpy as np


def attend(x, a):
    logits = x @ a @ np.swapaxes(x, -1, -2)
    weights = np.exp(logits - logits.real.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)) @ x


def layer(x, projection, heads):
    z = x @ projection
    return np.concatenate([attend(z, a) @ v for a, v in heads], axis=-1)


def close(x, y):
    np.testing.assert_allclose(x, y, atol=2e-11, rtol=2e-11)


def expanding_width(rng):
    eye = np.eye(3)
    projection = np.vstack([eye, eye])
    heads = [(np.array([[a]]), eye[j:j + 1]) for j, a in enumerate([1, 2])]
    g = np.diag([1, 1, 2])
    x = rng.normal(size=(64, 3, 1))
    h = layer(x, np.eye(1), heads)
    y = layer(h, projection, [(eye, eye)])
    close(y, layer(h, projection, [(eye, g)]))
    restricted = np.vstack([np.diag([1, 0, 0]), np.diag([0, 1, 0])])
    close(y, layer(h, restricted, [(eye, eye)]))
    assert [np.linalg.matrix_rank(b) for b in restricted.reshape(2, 3, 3)] == [1, 1]
    assert [np.linalg.matrix_rank(b) for b in projection.reshape(2, 3, 3)] == [3, 3]
    probes = np.array([[[1.], [1.], [1.]], [[1.], [0.], [0.]]])
    probe_outputs = layer(layer(probes, np.eye(1), heads), projection, [(eye, eye)])
    minor = np.linalg.det(probe_outputs[:, 0, :2])
    assert minor > 0.05
    print(f"Output span witness: minor {minor:.9f}; last-layer block ranks (3,3) vs (1,1).")
    assert np.ptp(y[..., 0]) > 1
    assert np.max(np.abs(h[..., 0] - h[..., 4])) > 0.01
    # Constant rows force any hypothesized lift to act on projected rows by g.
    z = np.zeros((3, 3))
    z[0, 2] = 1
    defect = np.max(np.abs(attend(z @ g, eye) - attend(z, eye) @ g))
    assert defect > 0.7
    print(f"Expanding width: 64 equal outputs; forced lift fails by {defect:.9f}.")

    for _ in range(32):
        value_rows = rng.normal(size=(2, 3))
        blocks = rng.normal(size=(2, 3, 3))
        value = rng.normal(size=(3, 3))
        a = rng.normal(size=(3, 3))
        p = np.vstack(blocks)
        projected_rows = np.einsum("hi,hij->hj", value_rows, blocks)
        _, _, vt = np.linalg.svd(projected_rows @ value)
        g = eye + np.outer(vt[-1], vt[-1])
        random_heads = [(rng.normal(size=(1, 1)), row[None]) for row in value_rows]
        h = layer(x, np.eye(1), random_heads)
        close(layer(h, p, [(a, value)]), layer(h, p, [(a, value @ g)]))
        probe_outputs = layer(layer(probes, np.eye(1), random_heads), p, [(a, value)])
        assert np.linalg.matrix_rank(probe_outputs[:, 0, :]) == 2
        restricted_blocks = []
        for row, block in zip(value_rows, blocks):
            restricted_blocks.append(np.outer(row, row @ block) / (row @ row))
        close(layer(h, p, [(a, value)]), layer(h, np.vstack(restricted_blocks), [(a, value)]))
        assert all(np.linalg.matrix_rank(b @ value, tol=1e-10) == 1 for b in restricted_blocks)
        assert all(np.linalg.matrix_rank(b @ value) == 3 for b in blocks)
        r = value @ g @ np.linalg.inv(value)
        close(np.linalg.det(r), 2)
        assert np.linalg.matrix_rank(a) == 3
        assert not np.allclose(r @ a @ r.T, a)
    print("32 random maximal-rank expanding models passed the determinant obstruction.")


def degeneracies(rng):
    eye = np.eye(2)
    x = rng.normal(size=(64, 3, 2))
    a = np.array([[1, 2], [-1, 1]])
    v = [eye[0:1].T, eye[1:2].T]
    w = [v[0] + 2 * v[1], -v[1]]
    p = np.ones((2, 1))
    final = [(np.eye(1), np.eye(1))]
    lifted = [layer(x, eye, [(a, row) for row in values]) for values in [v, w]]
    close(layer(lifted[0], p, final), layer(lifted[1], p, final))
    assert all(abs(np.linalg.det(np.column_stack([old, w[0]]))) > 0 for old in v)
    zero = np.zeros((2, 1))
    close(layer(x, eye, [(a, zero)]), layer(x, eye, [(-a, zero)]))
    # An unused head can also change its function, including on reachable inputs.
    p_unused = np.array([[1], [0]])
    unused = layer(x, eye, [(a, v[0]), (-a, w[0])])
    close(layer(unused, p_unused, final), layer(lifted[0], p_unused, final))

    x = x[:, :2]
    signed = [(eye, eye), (-eye, eye)]
    p = np.vstack([eye, eye])
    heads = layer(x, eye, signed)
    close(heads @ p, np.broadcast_to(2 * x.mean(axis=1, keepdims=True), x.shape))
    close(layer(heads, p, [(eye, eye)]), layer(layer(2*x, eye, signed), p, [(eye, eye/2)]))
    for j in range(2):
        assert not np.allclose(layer(2*x, eye, signed)[..., :2], 2 * heads[..., 2*j:2*j+2])
    print("Tied, zero-value, unused and two-token opposite-head identities passed.")


def balanced_checks(rng, n):
    # Gauge-fixed coordinates for widths 2, heads (2,2,1), including the end map
    # on constant rows. A complex-step Jacobian tests all 32 free coordinates.
    eye = np.eye(2)
    counts = [2, 2, 1]
    coefficients = rng.normal(size=(8, 2, 2))
    coefficients[0] = eye + 0.2 * coefficients[0]
    samples = rng.normal(size=(192, n, 2)) * rng.uniform(0.5, 3, size=(192, 1, 1))

    def unpack(params):
        cursor = 1
        for count in counts:
            scores = params[cursor:cursor + count]
            cursor += count
            free = list(params[cursor:cursor + count - 1])
            cursor += count - 1
            yield list(zip(scores, free + [eye - sum(free)]))

    def model(params, x):
        z = x @ params[0]
        for heads in unpack(params):
            z = sum(attend(z, a) @ b for a, b in heads)
        return z

    cols = []
    for index in np.ndindex(coefficients.shape):
        varied = coefficients.astype(complex)
        varied[index] += 1e-20j
        cols.append(model(varied, samples).imag.ravel() / 1e-20)
    sv = np.linalg.svd(np.column_stack(cols), compute_uv=False)
    assert sv[-1] / sv[0] > 1e-9

    # Equal-function raw realization with head permutations and value basis changes.
    z = samples
    projection = coefficients[0]
    for heads in unpack(coefficients):
        raw_heads, next_blocks = [], []
        for a, b in reversed(heads):
            d = eye if len(heads) == 1 else np.array([[1., 2.], [0., 1.]])
            raw_heads.append((a, d))
            next_blocks.append(np.linalg.solve(d, b))
        z = layer(z, projection, raw_heads)
        projection = np.vstack(next_blocks)
    close(z, model(coefficients, samples))

    # Recover the first-block response from end-to-end samples near a common token.
    m = np.array([0.3, -0.5])
    v = np.array([0.7, 1.2])
    predicted = sum((v @ b) / (1 + (n-1)*np.exp(-m @ a @ v))
                    for a, b in next(unpack(coefficients)))
    t = 0.002
    z = np.broadcast_to(m / t, (n, 2)).copy()
    z[-1] += t*v
    measured = (model(coefficients, z @ np.linalg.inv(coefficients[0])) - m/t)/t
    error = np.max(np.abs(measured - predicted))
    assert error < 2e-5
    print(f"Balanced n={n}: Jacobian rank 32/32, min/max singular value "
          f"{sv[-1]/sv[0]:.3g}; first-block probe error {error:.3g}.")


if __name__ == "__main__":
    rng = np.random.default_rng(21645)
    expanding_width(rng)
    degeneracies(rng)
    for n in (2, 3):
        balanced_checks(rng, n)
