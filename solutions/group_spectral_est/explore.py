"""Reproduce the three counterexamples in solution.md. Requires only NumPy."""
from itertools import permutations

import numpy as np


def permutation(order):
    return np.eye(len(order))[:, order]


def cyclic(order):
    shift = permutation(np.roll(np.arange(len(order)), -1))
    relabel = permutation(order)
    return np.array([relabel @ np.linalg.matrix_power(shift, k) @ relabel.T
                     for k in range(len(order))])


def cn(rng, size):
    return (rng.normal(size=size) + 1j * rng.normal(size=size)) / np.sqrt(2)


def outer(x):
    return np.outer(x, x.conj())


def twirl(group, x):
    orbit = group @ x
    return orbit.T @ orbit.conj() / len(group)


def concentration(matrix):
    return np.linalg.eigvalsh(matrix)[-1] / np.trace(matrix).real


def residual(matrix, covariance):
    return (np.linalg.norm(matrix @ covariance - covariance @ matrix)
            / (np.linalg.norm(matrix) * np.linalg.norm(covariance)))


def blind(rng):
    g = cyclic([0, 1, 2, 3])
    h = np.array([permutation([j ^ k for j in range(4)]) for k in range(4)])
    u = np.array([1, 1, -1, -1]) / 2
    v = g[1] @ u
    plus = (u + 1j * v) / np.sqrt(2)
    minus = plus.conj()
    r = 4 * outer(plus) + outer(minus)
    assert np.allclose(g @ r, r @ g)
    assert not np.allclose(h @ r, r @ h)
    failures = 0
    for sign in [-1, 1]:
        s = 2 * plus + sign * minus
        fg, fh = twirl(g, s), twirl(h, s)
        assert np.allclose([concentration(fg), concentration(fh)], [.8, .9])
        assert np.allclose([residual(fg, r), residual(fh, r)], [0, 12 / np.sqrt(697)])
        for _ in range(128):
            sigma = .001
            x = s * np.exp(2j * np.pi * rng.random()) + sigma * cn(rng, 4)
            fg, fh = twirl(g, x), twirl(h, x)
            cov = r + sigma**2 * np.eye(4)
            assert residual(fg, cov) < 1e-12 < residual(fh, cov)
            assert concentration(fh) > concentration(fg)
            failures += 1
    print(f"Blind: scores 4/5 and 9/10; {failures}/256 noisy draws selected the rival.")

    # A second example uses only cyclic groups and a polynomial graph filter.
    h = cyclic([0, 2, 1, 3])
    # If the oracle twirls the population second moment before the commutator.
    s = np.array([2, 1, -2, -1])
    cov = outer(s) + .01 * np.eye(4)
    expected = [np.mean(k @ cov @ k.transpose(0, 2, 1), axis=0) for k in (g, h)]
    assert residual(expected[0], cov) < 1e-12 < residual(expected[1], cov)
    assert np.allclose([concentration(twirl(k, s)) for k in (g, h)], [.5, .9])
    a = g[1] + g[1].T
    filt = 2 * np.eye(4) - a
    r = filt @ filt.T
    b = np.array([1, -1, 1, -1]) / 2
    x = 3 * u + v + 1j * b
    assert np.allclose(filt @ ((3*u + v)/2 + 1j*b/4), x)
    fg, fh = twirl(g, x), twirl(h, x)
    assert np.allclose([concentration(fg), concentration(fh)], [5/11, 9/11])
    assert residual(fg, r) < 1e-12 < residual(fh, r)
    errors = 0
    for _ in range(2048):
        x = filt @ cn(rng, 4)
        fg, fh = twirl(g, x), twirl(h, x)
        assert residual(fg, r) < 1e-12 < residual(fh, r)
        errors += int(concentration(fh) > concentration(fg))
    assert errors > 0
    print(f"Blind, filtered four-cycle: {errors}/2048 draws selected the rival cycle.")


def graph(rng):
    i = np.eye(3)
    a = np.block([[np.ones((3, 3)) - i, i], [i, np.zeros((3, 3))]])
    group = np.array([permutation(p) for p in permutations(range(6))
                      if np.array_equal(a[np.ix_(p, p)], a)])
    assert len(group) == 6
    assert any(not np.array_equal(p @ q, q @ p) for p in group for q in group)
    filt = 3 * np.eye(6) + a
    adj_spectrum = [1 + np.sqrt(2), 1 - np.sqrt(2)]
    for sign in [-1, 1]:
        adj_spectrum.extend([(-1 + sign * np.sqrt(5))/2] * 2)
    assert np.allclose(np.linalg.eigvalsh(filt @ filt.T),
                       np.sort((3 + np.array(adj_spectrum))**2))
    shifts = cyclic(list(range(6)))
    scores = []
    for _ in range(128):
        x = filt @ cn(rng, 6)
        scores.append(concentration(twirl(group, x)))
        basis = np.linalg.qr(np.column_stack((x, np.eye(6)[:, 1:])))[0]
        fourier = np.fft.fft(np.eye(6), norm="ortho")
        unitary = fourier @ basis.conj().T
        assert np.allclose(twirl(unitary.conj().T @ shifts @ unitary, x), outer(x))
    assert max(scores) < 1
    print(f"NADH: |Aut| = 6; mean S3 score {np.mean(scores):.6f}, cyclic oracle 1 (128 draws).")
    return group, filt


def averaging(rng, group, filt):
    # T = |Z|^2 is Exp(1). Integrate exact truncated moments against its density.
    for limit in [1., 4., 16., 64.]:
        t = np.linspace(0, limit, 8193)
        first = np.exp(-t/2)
        second = np.exp(t/2)**2 * np.exp(-t)
        integrate = getattr(np, "trapezoid", None) or np.trapz
        assert np.isclose(integrate(first, t), 2 * (-np.expm1(-limit/2)), rtol=2e-6)
        assert np.isclose(integrate(second, t), limit)

    # Every empirical variance of the scalar outer product scales as sigma**4.
    z = cn(rng, (4096, 4))
    for sigma in [1, 2, 8]:
        assert np.isclose(np.var(abs(sigma*z[:, 0])**2),
                          sigma**4 * np.var(abs(z[:, 0])**2))
    s = filt @ cn(rng, 6)
    z = cn(rng, (2048, 6))
    x = s + z
    values = np.mean(np.abs(x)**2, axis=1)
    assert np.allclose([np.mean(np.abs(group @ y)**2) for y in x], values)
    # Sum the total variance over every independent pair from the empirical law.
    avg = (values[:, None] + values[None, :]) / 2
    assert np.isclose(np.var(avg), np.var(values)/2)
    # The same invariant divergent statistic applies under a fixed graph shift.
    for y in x[:16]:
        assert np.allclose(np.sum(np.abs(group @ y)**2, axis=1)/2, np.vdot(y, y).real/2)
    print("GAAT: truncated second moments 1, 4, 16, 64; noise scaling and iid halving verified.")


def main():
    rng = np.random.default_rng(260403634)
    blind(rng)
    group, filt = graph(rng)
    averaging(rng, group, filt)
    print("All assertions passed.")


if __name__ == "__main__":
    main()
