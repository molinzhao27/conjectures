"""Reproduce the moving-reference selection counterexamples. Uses NumPy and SciPy."""
from math import ceil, exp

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import expit, logit, softmax


GAME = np.array([[0., 2.], [0., -1.]])


def finite_mmd(matrix, row, col, eta, temperature, period, blocks):
    """Simultaneous equation (mmd) steps with periodic policy resets, in logits."""
    x, y = np.log(row), np.log(col)
    c = 1 / (1 + eta * temperature)
    for _ in range(blocks):
        anchor_x, anchor_y = x.copy(), y.copy()
        for _ in range(period):
            gx, gy = matrix @ softmax(y), -matrix.T @ softmax(x)
            x = anchor_x + c * ((x - anchor_x) + eta * gx)
            y = anchor_y + c * ((y - anchor_y) + eta * gy)
        x -= x.max()
        y -= y.max()
    return softmax(x), softmax(y)


def exact_block(u, v, h):
    """Solve both anchored QRE equations for the 2x2 counterexample."""
    def col_equation(new_v):
        new_p = expit(u + 3 * h * expit(new_v))
        return (new_v - v) + h * (3 * new_p - 1)

    new_v = brentq(col_equation, v - 2*h - 1e-10, v + h + 1e-10)
    new_u = u + 3*h*expit(new_v)
    assert abs(col_equation(new_v)) < 1e-10
    return new_u, new_v


def flow_block(u, v, temperature, duration):
    """Small-step limit during one fixed-reference epoch."""
    def rhs(_, state):
        a, b = state
        return (3*expit(b) - temperature*(a-u),
                1 - 3*expit(a) - temperature*(b-v))

    sol = solve_ivp(rhs, (0., duration), (u, v), method="DOP853",
                    rtol=1e-11, atol=1e-13)
    assert sol.success
    return sol.y[:, -1]


def report(method, p0, q0, row, col, matrix=GAME):
    gap = np.max(matrix @ col) - np.min(row @ matrix)
    assert abs(gap) < 1e-10 and row[0] > p0
    print(f"{method:29} p0={p0:.2f} q0={q0:.2f}  "
          f"row1={row[0]:.10f}  NashConv={gap:.2e}")


def binary_tests():
    for p0, q0 in [(0.5, 0.5), (0.47, 0.23), (0.73, 0.76)]:
        u0, v0 = logit(p0), logit(q0)
        margin = 3*p0 - 1
        for h in (0.25, 1., 4., 16.):
            u, v = u0, v0
            for _ in range(ceil((36 + abs(v0)) / (h * margin))):
                u, v = exact_block(u, v, h)
            p, q = expit(u), expit(v)
            assert u-u0 <= 3*h*exp(v0)/np.expm1(h*margin) + 1e-12
            report(f"exact h={h:g}", p0, q0, np.array([p, 1-p]), [1-q, q])

        for eta, temp, period in [(0.4, 1., 1), (0.15, 1., 7), (0.2, 2., 20)]:
            h = (1 - (1+eta*temp)**(-period)) / temp
            blocks = ceil((36 + abs(v0))/(margin*h))
            row, col = finite_mmd(GAME, [p0, 1-p0], [1-q0, q0], eta, temp,
                                  period, blocks)
            report(f"discrete eta={eta:g} L={temp:g} m={period}", p0, q0, row, col)
        for temp, duration in [(1., 1.), (2., 3.)]:
            h = -np.expm1(-temp*duration)/temp
            u, v = u0, v0
            for _ in range(ceil((36 + abs(v0))/(margin*h))):
                u, v = flow_block(u, v, temp, duration)
            p, q = expit(u), expit(v)
            report(f"flow L={temp:g} T={duration:g}", p0, q0,
                   np.array([p, 1-p]), [1-q, q])

    u, v = flow_block(0., 0., 0., 100.)
    p, q = expit(u), expit(v)
    predicted = brentq(lambda z: z*(1-z)**2 - 1/64, 0.5, 1.)
    assert abs(p-predicted) < 1e-10
    report("continuously reset", 0.5, 0.5, np.array([p, 1-p]), [1-q, q])


def tilted_tests():
    w = np.array([2., -1., 0.])
    cols = [w, -w, [2., 1., 1.], [4., 3., 2.]]
    for ncols in (3, 4):
        matrix = np.column_stack(cols[:ncols])
        for row0 in (np.full(3, 1/3), np.array([0.2, 0.3, 0.5])):
            col0 = np.full(ncols, 1/ncols)
            r = np.cbrt(row0[0]*row0[1]**2/4) / row0[2]
            t = r/(1+3*r)
            row, col = finite_mmd(matrix, row0, col0, 0.12, 1., 25, 300)
            gap = np.max(matrix @ col) - np.min(row @ matrix)
            assert abs(gap) < 1e-10
            assert abs(row @ w) < 1e-10 and row[0] > t + 1e-4
            assert np.allclose(col[:2], [0.5, 0.5], atol=1e-10, rtol=0)
            print(f"tilted 3x{ncols} row0={row0.round(3)}  row1={row[0]:.10f}  "
                  f"projection={t:.10f}  NashConv={gap:.2e}")


if __name__ == "__main__":
    binary_tests()
    tilted_tests()
    print("All 32 cases passed.")
