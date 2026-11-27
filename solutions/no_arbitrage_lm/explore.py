"""Reproduce the equilibrium-spread counterexamples. Requires NumPy and SciPy."""

from itertools import product
from math import hypot, log, log1p

import numpy as np
from scipy.optimize import brentq, linprog, minimize
from scipy.stats import beta


SCORES = ("log", "Brier", "spherical")
BUDGETS = (0, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 8)
COST = 1 / 16


def risk(score, q, mean):
    """Expected binary loss, with smaller values preferred."""
    if score == "log":
        return -mean * log(q) - (1 - mean) * log1p(-q)
    if score == "Brier":
        return (q - mean) ** 2 + mean * (1 - mean)
    if score == "spherical":
        return -(mean * q + (1 - mean) * (1 - q)) / hypot(q, 1 - q)
    raise ValueError(score)


def risk_slope(score, q, mean):
    if score == "log":
        return (q - mean) / (q * (1 - q))
    if score == "Brier":
        return 2 * (q - mean)
    if score == "spherical":
        return (q - mean) / hypot(q, 1 - q) ** 3
    raise ValueError(score)


def informed_quote(shape, score, budget):
    """Symmetric Beta(shape,shape), one extra Bernoulli draw for the bettor."""
    cap = 1 / (2 * (2 * shape + 1))
    target = budget - 2 * COST
    if target <= 0:
        radius = 0
    elif target >= risk_slope(score, 0.5 + cap, 0.5):
        radius = cap
    else:
        radius = brentq(lambda d: risk_slope(score, 0.5 + d, 0.5) - target, 0, cap)
    return np.array([0.5 - radius, 0.5 + radius])


def numerical_game(score, mean, budget, signals):
    """Epigraph of every pure contingent trading strategy, unrestricted endpoints.

    signals lists (signal probability, conditional success probability).
    The first two variables are endpoints, the third is maximum expected profit.
    """
    constraints = [[-1, 1, 0]]  # upper >= lower
    bounds_rhs = [0.0]
    for actions in product((-1, 0, 1), repeat=len(signals)):
        bid = ask = constant = 0
        for action, (prob, chance) in zip(actions, signals):
            stake = prob * budget * action
            bid += max(-stake, 0)
            ask -= max(stake, 0)
            constant += stake * chance
        constraints.append([-bid, -ask, 1])
        bounds_rhs.append(constant)
    matrix, rhs = np.array(constraints), np.array(bounds_rhs)

    def loss(z):
        l, u, profit = z
        return (risk(score, l, mean) + risk(score, u, mean)) / 2 + COST * (u - l) + profit

    def gradient(z):
        l, u, _ = z
        return [risk_slope(score, l, mean) / 2 - COST,
                risk_slope(score, u, mean) / 2 + COST, 1]

    result = minimize(loss, [0.2, 0.8, budget], jac=gradient, method="SLSQP",
                      bounds=[(1e-10, 1 - 1e-10)] * 2 + [(0, None)],
                      constraints=[dict(type="ineq", fun=lambda z: matrix @ z - rhs,
                                        jac=lambda z: matrix)],
                      options=dict(ftol=1e-12, maxiter=500))
    assert result.success, result.message
    assert min(matrix @ result.x - rhs) > -1e-8
    return result.x[:2]


def anchor_checks():
    # Columns (A, phi), with phi implying A and P(A)=1/2.
    worlds = np.array([[0, 0], [1, 0], [1, 1]])
    equalities = np.array([np.ones(3), worlds[:, 0]])
    extrema = []
    for sign in (1, -1):
        result = linprog(sign * worlds[:, 1], A_eq=equalities, b_eq=[1, 0.5])
        assert result.success
        extrema.append(sign * result.fun)
    np.testing.assert_allclose(extrema, [0, 0.5])

    # Primal logical trader LP. Variables are buys, sells, guaranteed profit.
    for l, u, expected in [(0, 0.5, 0), (0.25, 0.25, 0), (0.6, 0.7, 0.05)]:
        bids, asks = np.array([0.5, l]), np.array([0.5, u])
        constraints = np.column_stack([asks - worlds, worlds - bids, np.ones(3)])
        constraints = np.vstack([constraints, [1, 1, 1, 1, 0]])
        result = linprog([0, 0, 0, 0, -1], A_ub=constraints, b_ub=[0, 0, 0, 1],
                         bounds=[(0, None)] * 4 + [(None, None)])
        assert result.success
        np.testing.assert_allclose(-result.fun, expected, atol=1e-12)
    print("Anchor LP: natural extension [0, 0.5]; singleton profit 0; disjoint-box profit 0.05")


def main():
    public_cases = informed_cases = 0
    max_error = 0
    print("Public bettor: a b | predictive quote | equilibrium width | 95% equal-tail width")
    for a, b in [(1, 1), (2, 2), (20, 20), (2, 5), (9, 2), (0.5, 0.5)]:
        m = a / (a + b)
        qlo, qhi = beta.ppf([0.025, 0.975], a, b)
        assert qlo < qhi
        np.testing.assert_allclose(beta.cdf(qhi, a, b) - beta.cdf(qlo, a, b), 0.95)
        print(f"{a:g} {b:g} | {m:.8f} | 0 | {qhi-qlo:.9f}")
        for score, budget in product(SCORES, BUDGETS):
            numerical = numerical_game(score, m, budget, [(1, m)])
            max_error = max(max_error, max(abs(numerical - m)))
            public_cases += 1

    print("One private draw: a=b budget | log width | Brier width | spherical width | 95% width")
    for shape, budget in product((1, 2, 20), BUDGETS):
        low = shape / (2 * shape + 1)
        signals = [(0.5, low), (0.5, 1 - low)]
        widths = []
        for score in SCORES:
            endpoints = informed_quote(shape, score, budget)
            numerical = numerical_game(score, 0.5, budget, signals)
            max_error = max(max_error, max(abs(numerical - endpoints)))
            widths.append(endpoints[1] - endpoints[0])
            informed_cases += 1
        credible = beta.ppf(0.975, shape, shape) - beta.ppf(0.025, shape, shape)
        assert max(widths) < credible
        print(f"{shape} {budget:g} | " + " | ".join(f"{w:.9f}" for w in [*widths, credible]))
    assert max_error < 1e-6, max_error
    print(f"Verified {public_cases} public and {informed_cases} private-signal cases; "
          f"max endpoint error {max_error:.2g}.")
    anchor_checks()


if __name__ == "__main__":
    main()
