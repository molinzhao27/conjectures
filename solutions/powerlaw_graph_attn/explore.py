"""Deterministic float64 checks for the constructions in solution.md. Requires NumPy."""
import numpy as np


def rms(a):
    return np.sqrt(np.mean(a**2))


def order(a, b):
    delta = rms(a - b)
    return 0.0 if delta == 0 else 2 * delta / (rms(a) + rms(b))


def normalize(a):
    centered = a - a.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(np.mean(centered**2, axis=-1, keepdims=True) + 1e-6)


def normalized_gram(q):
    a = normalize(q.T @ q)
    for _ in range(8):  # zero nonlinear branches in the residual units
        a = normalize(a)
    return a


def interaction(a):
    return a**2 / (1 + np.exp(-a)) + 1e-9


def softmax(scores):
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def masked(scores, cost):
    return softmax(np.where(np.tri(len(scores), dtype=bool), scores, scores - cost))


def rigidity():
    print("width    max m over tensor types       G radius")
    for d in (8, 32, 128, 256):
        t = 1 / d
        v = np.tile([1.0, -1.0], d // 2)[None, :]
        pair = []
        for q in (0 * v, v):
            a = 1 + t * normalized_gram(q)
            b = interaction(a)
            pair.append((a, b, b, b / t))
        stats = [order(x, y) for x, y in zip(*pair)]
        g0, g1 = pair[0][-1], pair[1][-1]
        radius = rms(g1 - g0) / 2
        assert max(stats) < 3 * t and radius > 0.8
        # Choose any desired radius, here inverse-logarithmic in width.
        h = 1 / np.log(d + 1)
        scaled = 1 + h * (g1 - g0) / radius
        np.testing.assert_allclose(rms(scaled - 1) / 2, h)
        print(f"{d:5d}    {max(stats):.9f}                 {radius:.9f}")
    assert order(np.zeros(2), np.zeros(2)) == 0


def stationary(p):
    n = len(p)
    system = p.T - np.eye(n)
    system[-1] = 1
    rhs = np.zeros(n)
    rhs[-1] = 1
    pi = np.linalg.solve(system, rhs)
    np.testing.assert_allclose(pi @ p, pi, atol=2e-14)
    return pi


def reversible_spectra(p):
    pi = stationary(p)
    assert np.min(pi) > 0
    adjoint = (p.T * pi) / pi[:, None]
    results = []
    for r in ((p + adjoint) / 2, adjoint @ p):
        symmetric = np.sqrt(pi[:, None] / pi) * r
        np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-13)
        results.append(np.linalg.eigvalsh(symmetric))
    return results


def causal_checks():
    for n in (4, 16, 64, 128):
        scores = np.zeros((n, n))
        ideal = masked(scores, np.inf)
        np.testing.assert_allclose(stationary(ideal), np.eye(1, n)[0], atol=1e-14)
        np.testing.assert_allclose(np.sort(np.linalg.eigvals(ideal))[::-1],
                                   1 / np.arange(1, n + 1))
        cost = 2.0  # finite, numerically resolvable leakage
        add, mult = reversible_spectra(masked(scores, cost))
        for spec in (add, mult):
            np.testing.assert_allclose(spec[-1], 1)
            assert np.max(np.abs(spec[:-1])) <= 1 - np.exp(-2 * cost) + 1e-13
        print(f"S={n:3d}  finite-mask gaps  {1-add[-2]:.8f}  {1-mult[-2]:.8f}")


def programmable_head():
    # Two tokens differ only in an extra query plane at position zero.
    n, d = 4, 10
    angles = 1e4 ** (-2 * np.arange(d // 2) / d)
    phi = np.arange(n)[:, None] * angles[:-1]
    active = np.stack((np.cos(phi), np.sin(phi)), axis=-1).reshape(n, d - 2)
    right_inverse = np.linalg.pinv(active)
    q0 = np.pad(active, ((0, 0), (0, 2)))
    q1 = q0.copy()
    q1[0, -2] = 1
    cost = 3
    upper = np.triu(np.ones((n, n)), 1)
    for r in (0., 0.5, 0.99):
        target = r * np.eye(n) + (1 - r) / n
        g = np.zeros((d, d))
        g[:-2, :-2] = np.sqrt(d) * right_inverse @ (np.log(target) + cost * upper) @ right_inverse.T
        assert order(g, g.copy()) == 0
        a0, a1 = (normalized_gram(q) for q in (q0, q1))
        delta = np.zeros_like(g)
        delta[:, -2:] = interaction(a1[:, -2:]) - interaction(a0[:, -2:])
        assert rms(delta) > 0
        other = g + 100 * rms(g) * delta / rms(delta)
        assert order(g, other) > 1.9
        for q, form in ((q0, g), (q1, g), (q1, other)):
            p = masked(q @ form @ q0.T / np.sqrt(d), cost)
            np.testing.assert_allclose(p, target, atol=3e-10)
            add, mult = reversible_spectra(p)
            np.testing.assert_allclose(add[:-1], r, atol=3e-10)
            np.testing.assert_allclose(mult[:-1], r*r, atol=3e-10)
        print(f"prescribed eigenvalue {r:.2f}, invisible variation m={order(g, other):.6f}")


def edge_measures():
    # Uniform mass on the within-pair differences. For theta=2 the limiting
    # density is 2(1-lambda), with moment 2/((t+1)(t+2)).
    for pairs in (8, 32, 128):
        quantiles = 1 - np.sqrt((np.arange(pairs) + 0.5) / pairs)
        n = 2 * pairs
        p = np.zeros((n, n))
        for i, r in enumerate(quantiles):
            p[2*i:2*i+2, 2*i:2*i+2] = np.array([[1+r, 1-r], [1-r, 1+r]]) / 2
        p = (1 - 1/n) * p + 1/n**2
        f = np.tile([1., -1.], pairs)
        corr = f @ np.linalg.matrix_power(p, 4) @ f / n
        np.testing.assert_allclose(corr, np.mean(((1 - 1/n) * quantiles)**4))
        # Reset error <= 4/n; midpoint rule for a monotone [0,1]-valued function.
        assert abs(corr - 1/15) <= 4/n + 1/pairs
        print(f"{pairs:3d} pairs, theta=2 prescribed: fourth moment {corr:.8f}")


def transfer():
    d = 8
    j = np.ones((d, d))
    rng = np.random.default_rng(260810288)
    q, k = rng.normal(size=(2, 5, d))
    np.testing.assert_allclose(q @ j @ k.T, q @ (-j) @ (-k).T)
    for _ in range(16):
        s = np.eye(d) + rng.normal(scale=0.1, size=(d, d))
        inv = np.linalg.inv(s)
        assert rms(-j - inv.T @ j @ inv) >= 1 - 1e-13
    bits = np.array([-1., 1.])
    # For each source bit b, index the binary target c' by the columns.
    conditional = softmax(np.log(3) * np.outer(bits, bits) / 2)
    np.testing.assert_allclose(conditional.diagonal(), 0.75)
    np.testing.assert_allclose(conditional.mean(axis=0), 0.5)
    # Gradient step from zero readout weights on balanced data.
    gradient = bits @ (0.5 - conditional) / 2
    prediction = np.argmax(-np.outer(bits, gradient), axis=1)
    np.testing.assert_equal(prediction, [0, 1])

    # Actual one-token residual stream for each of the eight input symbols.
    tokens = np.array([(h, b, c) for h in bits for b in bits for c in bits])
    v = np.tile([1., -1.], d // 2)
    embedded = normalize(np.sqrt(d) * np.outer(tokens[:, 1], v))
    hidden = normalize(normalize(embedded))
    weights = np.log(3) * np.outer(v, tokens[:, 2]) / (2 * abs(hidden[0] @ v))
    target = softmax(hidden @ weights)
    source = softmax((0 * hidden) @ weights)
    np.testing.assert_allclose(source, 1/8)
    matching = tokens[:, 1, None] == tokens[None, :, 2]
    np.testing.assert_allclose((target * matching).sum(axis=1), 0.75)
    np.testing.assert_allclose((source * matching).sum(axis=1), 0.5)
    # One full readout step with the population pair frequencies.
    fitted = hidden.T @ (target - source) / len(tokens)
    prob = softmax(hidden @ fitted)[:, tokens[:, 2] == 1].sum(axis=1)
    np.testing.assert_equal(prob > 0.5, tokens[:, 1] == 1)
    print("binary transfer benchmark: constant accuracy .50, target optimum .75")


if __name__ == "__main__":
    rigidity()
    causal_checks()
    programmable_head()
    edge_measures()
    transfer()
    print("All assertions passed.")
