"""Verify scalar one-hidden-layer examples from solution.md. Requires NumPy and SciPy."""

from math import comb, factorial

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import expit, ndtr


def activation(name, z):
    if name == "relu":
        return np.maximum(z, 0), (z > 0).astype(float)
    if name == "gelu":
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        return z * ndtr(z), ndtr(z) + z * phi
    if name == "tanh":
        value = np.tanh(z)
        return value, 1 - value**2
    if name in {"logistic", "centered logistic"}:
        value = expit(z)
        offset = 0.5 if name == "centered logistic" else 0.0
        return value - offset, value * (1 - value)
    raise ValueError(name)


def evaluate(name, x, w, b, a, c):
    """Return values and sample Jacobian, with column blocks (w, b, a, c)."""
    z = x[:, None] * np.asarray(w) + b
    if name == "relu":
        assert np.all(z != 0), "The Jacobian check needs differentiable sample points"
    h, dh = activation(name, z)
    jac = np.column_stack((x[:, None] * dh * a, dh * a, h, np.ones_like(x)))
    return h @ a + c, jac


def check_rank(jac, expected):
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = np.count_nonzero(singular > 1e-10 * singular[0])
    assert rank == expected, (rank, expected, singular)


def check_identity(x):
    for name, rank in [("relu", 4), ("gelu", 5)]:
        cases = 0
        for center in [-1.25, 0, 0.875]:
            for slope in [0.625, 1, 1.75]:
                negative_slopes = [-slope, -1.375] if name == "relu" else [-slope]
                for negative in negative_slopes:
                    for w in [np.array([slope, negative]), np.array([negative, slope])]:
                        y, jac = evaluate(name, x, w, -center * w, 1 / w, center)
                        assert_allclose(y, x, rtol=0, atol=3e-14)
                        tangent = np.r_[np.zeros(2), -w, np.zeros(2), 1]
                        assert_allclose(jac @ tangent, 0, rtol=0, atol=2e-14)
                        check_rank(jac, rank)
                        cases += 1
        print(f"{name} identity: {cases} parameter choices, rank {rank}/7")


def check_tanh_and_logistic(x):
    for name in ["tanh", "centered logistic", "logistic"]:
        y, jac = evaluate(name, x, [1], [0], [1], 0)
        output_bias = 1 if name == "logistic" else 0
        reflected, jac_reflected = evaluate(name, x, [-1], [0], [-1], output_bias)
        assert_allclose(y, reflected, rtol=0, atol=4e-16)
        check_rank(jac, 4)
        check_rank(jac_reflected, 4)
        print(f"{name}: reflected width-one pair, rank 4/4, bias change {output_bias}")

    for w, b, a, rank in [
        ([1, 0.6], [0, 0.4], [1, 0], 5),  # unused unit
        ([1, 0], [0, 0], [1, 0], 4),  # coordinate embedding from width one
        ([1, 1], [0, 0], [0.25, 0.75], 4),  # split coefficient across clones
    ]:
        y, jac = evaluate("tanh", x, w, b, a, 0)
        assert_allclose(y, np.tanh(x), rtol=0, atol=3e-16)
        check_rank(jac, rank)
    print("reducible tanh: unused, padded, cloned ranks 5/7, 4/7, 4/7")

    for w, b, a, expected in [([1], [0], [0], 2), ([0], [0], [2], 2),
                                ([0], [0], [0], 1), ([], [], [], 1)]:
        y, jac = evaluate("tanh", x, w, b, a, 3)
        assert_allclose(y, 3, rtol=0, atol=0)
        check_rank(jac, expected)
    print("constant tanh: positive-width ranks 2/4, 2/4, 1/4; width-zero rank 1/1")


def check_polynomials():
    for degree in range(1, 13):
        def difference(x):
            return sum((-1)**(degree-1-j) * comb(degree-1, j) * (x+j)**degree
                       for j in range(degree))

        offset = difference(0)
        for shift in [-3, 0, 5]:
            # Both sides are polynomials of degree at most degree. Integer arithmetic.
            for x in range(degree + 1):
                assert difference(x-shift) - offset + factorial(degree)*shift == factorial(degree)*x
    print("polynomial identity: 12 degrees x 3 shifts certified by exact interpolation")


if __name__ == "__main__":
    samples = np.linspace(-5.013, 5.017, 257)
    check_identity(samples)
    check_tanh_and_logistic(samples)
    check_polynomials()
