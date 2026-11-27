"""Exact local certificates and a small periodic continuous-mesh search."""

from collections import Counter
from fractions import Fraction as F
from itertools import product
from math import sqrt

import numpy as np
import sympy as sp
from scipy.optimize import minimize


def area(triangle):
    (x, y), (u, v), (s, t) = triangle
    return abs((u - x) * (t - y) - (v - y) * (s - x)) / 2


def edge_extrema(first, second, errors):
    x, y = first
    u, v = second
    k = (u - x) * (v - y)
    d, e = errors[first], errors[second]
    candidates = [d, e]
    if k:
        t = (e - d + k) / (2 * k)
        if 0 < t < 1:
            candidates.append((1 - t) * d + t * e + t * (1 - t) * k)
    return min(candidates), max(candidates)


def check_period_box():
    width, height = F(21, 10), F(51, 10)
    x_grid = [F(0), F(2), width]
    low = [(x, F(0)) for x in x_grid]
    high = [(x, F(5)) for x in x_grid]
    top = [(x, height) for x in x_grid]
    c = (F(2), F(5, 2))
    triangles = []
    errors = {c: F(0)}
    for i, (a, b, t) in enumerate(zip(low, high, top)):
        errors[a] = errors[t] = F(-2, 3) if i != 1 else F(0)
        errors[b] = -errors[a]
    for side in (0, 2):
        triangles.extend([(low[side], high[side], c),
                          (low[side], low[1], c),
                          (high[side], high[1], c)])
    for i in (0, 1):
        triangles.extend([(high[i], high[i + 1], top[i + 1]),
                          (high[i], top[i + 1], top[i])])
    edges = Counter()
    extremes = []
    for triangle in triangles:
        for i, p in enumerate(triangle):
            q = triangle[(i + 1) % 3]
            extremes.extend(edge_extrema(p, q, errors))
            # Retain winding information when identifying translates.
            def representation(start, end):
                return (start[0] % width, start[1] % height,
                        end[0] - start[0], end[1] - start[1])
            edges[min(representation(p, q), representation(q, p))] += 1
    assert len(triangles) == 10 and set(edges.values()) == {2}
    assert sum(map(area, triangles)) == width * height
    assert max(map(area, triangles)) == 5
    assert max(map(abs, extremes)) == F(169, 180) < 1
    assert len(triangles) / (width * height) == F(1000, 1071)
    print("Periodic example: 10 faces, largest area 5, maximum error 169/180.")


def check_certificates():
    a, m, H, Q = sp.symbols("a m H Q", real=True)
    p, r, w = 2*(a+m+H)**2, 2*(a+m-H)**2, 2*Q
    S, C, h = 1-m*m, (a+m)**2, H**2
    corner = 16*C*h + Q*(Q+4*C+4*h)
    assert sp.expand(((p-r)**2 + 2*w*(p+r) + w*w) / 4 - corner) == 0
    bound = 32*(1-m)*(1+m)**2 - 16*S*(h+(1-a)*(1+a+2*m))
    assert sp.expand(corner.subs(Q, 4*(S-h)) - bound) == 0
    assert sp.expand(sp.diff(32*(1-m)*(1+m)**2, m) + 32*(m+1)*(3*m-1)) == 0
    M2, B = F(1024, 27), F(123, 20)
    for t in (F(29, 100), F(39, 100)):
        assert 32*(1-t)*(1+t)**2 < B**2
    assert M2 - B**2 < F(104, 1000)
    assert F(104, 1000)/(16*F(84, 100)) < F(9, 100)**2 < F(1, 100)
    assert F(1, 1) - F(39, 100)**2 > F(84, 100)
    assert 1-2*F(99, 100)**2 < -F(96, 100)
    assert 1-2*F(48, 100)**2 > F(1, 2)
    assert 1-2*F(1, 5)**2 == F(92, 100)
    assert max((i-j)**2 + 2*k*(i+j) + k*k
               for i, j, k in product((0, F(9, 2)), repeat=3)) < 4*B*B
    assert B*B > M2*(1-F(1, 484))**2
    assert sp.expand(w*(32-3*w) + 3*(w-sp.Rational(16, 3))**2 -
                     sp.Rational(256, 3)) == 0
    M = sqrt(M2)
    eta = 1 - float(B)/M
    lower = (1+eta/(1+6*1.1**2))/M
    assert 1/M < lower < sqrt(3)/8
    print(f"Universal lower bound {lower:.12f}; sharp one-multiset bound {sqrt(3)/8:.12f}.")


def optimize_period(mask, angle):
    """Optimize four vertex orbits and two period vectors on a 2-by-2 grid."""
    def coefficients(i, j):
        out = np.zeros(6)
        out[i % 2 + 2*(j % 2)] = 1
        out[-2:] = i//2, j//2
        return out

    edges, faces = [], []
    for j, i in product(range(2), repeat=2):
        a, b, c, d = [coefficients(i+di, j+dj) for di, dj in
                      ((0, 0), (1, 0), (0, 1), (1, 1))]
        flipped = mask & (1 << (i+2*j))
        edges.extend([(a, b), (a, c), (b, c) if flipped else (a, d)])
        faces.extend([(a, b, c), (b, d, c)] if flipped else [(a, b, d), (a, d, c)])
    start, end = np.array(edges).transpose(1, 0, 2)
    diff = end-start
    both = (start+end)[:, :4]
    first, second, third = np.array(faces).transpose(1, 0, 2)
    side1, side2 = second-first, third-first

    def constraint(z, jacobian=False):
        x, y, theta = np.split(z, [6, 12])
        dx, dy = diff@x, diff@y
        s, c = np.sin(theta), np.cos(theta)
        # d = cos(2 theta), hence sqrt(1-d) = sqrt(2) sin(theta).
        S, C = both@s, both@c
        ux, uy, vx, vy = side1@x, side1@y, side2@x, side2@y
        if jacobian:
            return np.block([
                [-dy[:, None]*diff, -dx[:, None]*diff, 4*S[:, None]*both*c],
                [dy[:, None]*diff, dx[:, None]*diff, -4*C[:, None]*both*s],
                [vy[:, None]*side1-uy[:, None]*side2,
                 ux[:, None]*side2-vx[:, None]*side1, np.zeros((8, 4))],
            ])
        return np.r_[2*S*S-dx*dy, 2*C*C+dx*dy, ux*vy-uy*vx-0.01]

    def objective(z):
        return -(z[4]*z[11]-z[5]*z[10])/8

    def gradient(z):
        grad = np.zeros_like(z)
        grad[[4, 5, 10, 11]] = np.array([-z[11], z[10], z[5], -z[4]])/8
        return grad

    co, si = np.cos(angle), np.sin(angle)
    index = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]])
    xy = 2*index@np.array([[co, si], [-si, co]])
    initial = np.r_[xy[:, 0], xy[:, 1], np.full(4, np.pi/4)]
    result = minimize(objective, initial, jac=gradient, method="SLSQP",
                      constraints=[dict(type="ineq", fun=constraint,
                                        jac=lambda z: constraint(z, True))],
                      bounds=[(None, None)]*12+[(0, np.pi/2)]*4,
                      options=dict(maxiter=300, ftol=1e-10))
    if result.success and constraint(result.x).min() >= -1e-7:
        return -result.fun
    return None


def check_search():
    best = 0
    successes = 0
    for mask, angle in product(range(16), (-0.9, -0.35, 0.2, 0.7)):
        value = optimize_period(mask, angle)
        if value is not None:
            successes += 1
            best = max(best, value)
    assert successes > 0
    assert abs(best - 8/sqrt(3)) < 1e-6
    print(f"Periodic search: {successes}/64 converged and passed feasibility checks; "
          f"best mean area {best:.10f}.")


if __name__ == "__main__":
    check_period_box()
    check_certificates()
    check_search()
    print("All assertions passed.")
