"""Exact RDPG connection, curvature, and Laplacian jet checks; numerical transport.

Run with Python 3, SymPy, NumPy, and SciPy. Writes no files.
"""
from itertools import combinations
from math import comb, pi

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_sylvester


def bracket(a, b):
    return a * b - b * a


def coordinates(matrix):
    return sp.Matrix([matrix[i, j] for i, j in combinations(range(matrix.rows), 2)])


def skew_basis(d):
    result = []
    for i, j in combinations(range(d), 2):
        a = sp.zeros(d)
        a[i, j], a[j, i] = 1, -1
        result.append(a)
    return result


class Connection:
    """Invert the positive Lyapunov operator on skew matrices over the rationals."""

    def __init__(self, x):
        self.x = x
        gram = x.T * x
        self.basis = skew_basis(x.cols)
        operator = sp.Matrix.hstack(
            *[coordinates(gram * a + a * gram) for a in self.basis]
        )
        self.inverse = operator.inv()

    def solve(self, rhs):
        coeffs = self.inverse * coordinates(rhs)
        return sum((v * b for v, b in zip(coeffs, self.basis)), sp.zeros(self.x.cols))

    def form(self, v):
        return self.solve(self.x.T * v - v.T * self.x)

    def curvature(self, v, w):
        h = v - self.x * self.form(v)
        k = w - self.x * self.form(w)
        return self.solve(2 * (h.T * k - k.T * h))


def independent(matrices):
    rows = sp.Matrix.hstack(*map(coordinates, matrices))
    pivots = rows.to_DM().rref()[1]
    return [matrices[i] for i in pivots]


def lie_dimensions(generators):
    basis = independent(generators)
    dimensions = [len(basis)]
    while basis:
        brackets = [bracket(a, b) for a, b in combinations(basis, 2)]
        extended = independent(basis + brackets)
        if len(extended) == len(basis):
            break
        basis = extended
        dimensions.append(len(basis))
    return dimensions


def laplacian(p):
    return sp.diag(*[sum(row) for row in p.tolist()]) - p


def projected_jets(x, order):
    """Exact time derivatives via P' = -LP-PL and the product rule."""
    p = [x * x.T]
    ls = [laplacian(p[0])]
    for k in range(order):
        terms = [comb(k, j) * ls[j] * p[k - j] for j in range(k + 1)]
        lp = sum(terms, sp.zeros(x.rows))
        p.append(-lp - lp.T)
        ls.append(laplacian(p[-1]))
    return [x.T * bracket(ls[0], ell) * x for ell in ls[1:]]


def symmetric_state(d):
    return (sp.ones(d + 1, d) + sp.eye(d).col_join(sp.zeros(1, d))) / (4 * d)


def perturbed_state(d):
    # Rational perturbation of the orthogonal-row equilibrium in the proof.
    x = sp.diag(*[2**i for i in range(d)]).col_join(sp.zeros(1, d))
    for i, j in combinations(range(d + 1), 2):
        weight = 1 + int(j == i + 1)
        x[j, i] = sp.Rational(weight, 17 * 2**i)
    return x / (2 ** (d + 1))


def check_state(x):
    p = x * x.T
    assert x.rank() == x.cols and all(0 < a < 1 for a in p)
    assert len(set(p * sp.ones(x.rows, 1))) > 1


def transport(x):
    """Radial segment, circle, and reversed radial segment in a lifted disk."""
    d = x.cols
    base = np.array(x, dtype=float)
    z = np.array(x.T.nullspace()[0], dtype=float).ravel()
    z /= np.linalg.norm(z)
    u, v = np.zeros(d), np.zeros(d)
    u[:2] = (1, -1)
    v[:3] = (1, 1, -2)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    radius = 1 / (40 * d)
    tau = 2 * pi
    segments = [
        lambda t: (radius * t * u, radius * u),
        lambda t: (radius * (np.cos(tau*t)*u + np.sin(tau*t)*v),
                   tau * radius * (-np.sin(tau*t)*u + np.cos(tau*t)*v)),
        lambda t: (radius * (1-t) * u, -radius * u),
    ]
    q = np.eye(d)
    for curve in segments:
        def rhs(t, flat_q):
            a, da = curve(t)
            y = base + np.outer(z, a)
            dy = np.outer(z, da)
            gram = y.T @ y
            omega = solve_sylvester(gram, gram, y.T @ dy - dy.T @ y)
            return (-omega @ flat_q.reshape(d, d)).ravel()

        sol = solve_ivp(rhs, (0, 1), q.ravel(), method="DOP853", rtol=1e-12, atol=1e-14)
        assert sol.success
        q = sol.y[:, -1].reshape(d, d)

    angle = tau * radius**2 / (2 / (4*d)**2 + radius**2)
    exact = expm(-angle * (np.outer(u, v) - np.outer(v, u)))
    error = np.linalg.norm(q - exact)
    assert max(error, abs(np.linalg.det(q) - 1), np.linalg.norm(q.T @ q - np.eye(d))) < 1e-11
    print(f"  loop angle {angle:.12f}, matrix error {error:.2e}, det {np.linalg.det(q):.12f}")
    print(np.array2string(q, precision=7, suppress_small=True))


def run(d):
    x = symmetric_state(d)
    check_state(x)
    conn = Connection(x)
    z = x.T.nullspace()[0]
    h = [z * sp.eye(d).row(i) for i in range(d)]
    m = d * (d - 1) // 2
    a = conn.basis[0]
    vel = -laplacian(x * x.T) * x
    assert conn.form(x * a) == a and conn.form(vel).is_zero_matrix
    assert conn.curvature(vel, 3 * vel).is_zero_matrix
    assert conn.curvature(h[0] + x * a, h[1]) == conn.curvature(h[0], h[1])
    curvatures = [conn.curvature(v, w) for v, w in combinations(h, 2)]
    assert len(independent(curvatures)) == m
    adjacent = [conn.curvature(h[i], h[i + 1]) for i in range(d - 1)]
    dims = lie_dimensions(adjacent)
    assert dims[-1] == m

    order = d * (d + 1) // 2
    symmetric_jets = projected_jets(x, order)
    assert all(b.is_zero_matrix for b in symmetric_jets)
    generic_example = perturbed_state(d)
    check_state(generic_example)
    jets = projected_jets(generic_example, order)
    rank = len(independent(jets))
    assert rank == m
    print(f"d={d}, ambient curvature rank {m}, adjacent Lie dimensions {dims}")
    print(f"  orders 1..{order}: frozen jet ranks symmetric 0, perturbed {rank}")
    transport(x)


if __name__ == "__main__":
    for dimension in (3, 4, 5):
        run(dimension)
