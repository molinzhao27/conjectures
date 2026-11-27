"""Reproduce checks for Cascading Minimization. Run with Python, NumPy, and SciPy."""

import numpy as np
from scipy.integrate import solve_ivp


def normalized(a, b):
    # Solve 2 b^2 z^2 + a^2 z - 3 = 0 for z = exp(-2 tau / 3).
    z = 6 / (a**2 + np.hypot(a**2, np.sqrt(24) * b))
    return np.asarray([a * np.sqrt(z), b * z])


def margins(a, b):
    return np.asarray([a * b, a**3])


def descent(t, theta):
    a, b = theta
    w, v = np.exp(-margins(a, b)) / 2
    return [b * w + 3 * a**2 * v, a * w]


def path_slope(b, state):
    (a,) = state
    return [b / a + 3 * a * np.exp(a * b - a**3)]


def check_descent(initial):
    a0, b0 = initial
    assert np.exp(-margins(a0, b0)).mean() < 0.5
    for alpha in [-2, 0.5, 2]:
        actual = margins(a0 * np.exp(alpha / 3), b0 * np.exp(2 * alpha / 3))
        np.testing.assert_allclose(actual, np.exp(alpha) * margins(a0, b0))

    options = dict(method="DOP853", rtol=2e-11, atol=1e-13)
    direct = solve_ivp(descent, (0, 1e6), initial, t_eval=np.geomspace(1, 1e6, 16),
                       **options)
    assert direct.success
    physical_a, physical_b = direct.y
    assert np.all(np.diff(direct.y, axis=1) > 0)
    assert np.all(np.diff(physical_a**2 - physical_b**2) >= -1e-9)

    # b is strictly increasing, so this integrates the same orbit beyond the
    # point where the physical-time field underflows.
    checkpoints = np.array([10.0, 100.0, 1000.0, 10000.0])
    path = solve_ivp(path_slope, (b0, checkpoints[-1]), [a0], dense_output=True,
                     **options)
    assert path.success
    np.testing.assert_allclose(path.sol(physical_b)[0], physical_a, rtol=2e-9)
    a = path.sol(checkpoints)[0]
    a_hat, b_hat = normalized(a, checkpoints)
    np.testing.assert_allclose((a_hat**2 + 2 * b_hat**2) / 3, 1, atol=1e-14)
    raw = margins(a, checkpoints)
    scaled = margins(a_hat, b_hat)
    assert np.all(np.diff(raw, axis=1) > 0)
    assert np.all(np.diff(scaled[0]) < 0)
    assert abs(a_hat[-1] - np.sqrt(3)) < 1e-6 and b_hat[-1] < 3.1e-4
    assert scaled[0, -1] < 5.3e-4
    print(f"start {initial}, b={checkpoints[-1]:g}, raw min margin "
          f"{raw[:, -1].min():.8g}, normalized min margin {scaled[:, -1].min():.8g}")


def check_printed_sign():
    # Both signed margins on the second data set equal ab.
    def ascent(t, state):
        a, b = state
        return -np.exp(-a * b) * np.array([b, a])

    times = np.linspace(0, 16, 33)
    path = solve_ivp(ascent, (0, times[-1]), [1.0, 1.0], t_eval=times,
                     rtol=1e-11, atol=1e-15)
    assert path.success
    a, b = path.y
    np.testing.assert_allclose(a, b, rtol=1e-12)
    assert np.all(np.diff(a) < 0)
    a_hat, b_hat = normalized(a, b)
    assert a_hat[-1] < 0.001 and abs(b_hat[-1] - np.sqrt(1.5)) < 1e-6
    print(f"printed A4, t=16, normalized margin {(a_hat * b_hat)[-1]:.8g}")


if __name__ == "__main__":
    for start in [(1.0, 1.0), (1.5, 1.0), (1.0, 1.25)]:
        check_descent(start)
    check_printed_sign()
    print("All assertions passed.")
