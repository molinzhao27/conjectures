"""Exact finite-experiment checks for the geometric neutral zone results."""

from fractions import Fraction as F
from itertools import product
from math import acos, cos, exp, isclose, log, sqrt


def score(p, rate):
    vertex = p.index(max(p))
    coefficient = sum(sqrt(value * (j == vertex)) for j, value in enumerate(p))
    return exp(-2 * rate * acos(coefficient))


def check_geometry():
    cases = 0
    for classes in range(2, 9):
        for rate in (0.1, 0.75, 2.0):
            previous = 0
            for step in range(21):
                top = F(1, classes) + F(step, 20) * (1 - F(1, classes))
                p = [(1 - top) / (classes - 1)] * (classes - 1) + [top]
                r = score(p, rate)
                assert previous < r
                assert isclose(cos(-log(r) / (2 * rate)) ** 2, top, abs_tol=1e-14)
                previous = r
                cases += 1
    return cases


def groups(states, ranking, cost):
    """States are (mass, S, U). Pool tied ranks before boundary randomization."""
    totals = {}
    for mass, confidence, correctness in states:
        rank = ranking(confidence, correctness)
        spent, gained = totals.get(rank, (F(0), F(0)))
        totals[rank] = (spent + mass * cost(correctness), gained + mass * correctness)
    return [totals[rank] for rank in sorted(totals, reverse=True)]


def best_threshold(ordered, budget):
    """Largest correct mass from an upper threshold at the given budget."""
    gain = F(0)
    for cost, value in ordered:
        if cost > budget:
            return gain + budget * value / cost
        gain += value
        budget -= cost
    return gain


def frontiers_match(left, right):
    # Both frontiers are piecewise linear. Zero-cost groups can cause jumps
    # at knots, so compare both the knots and two interior points per interval.
    knots = {F(0)}
    for ordered in (left, right):
        total = F(0)
        for cost, _ in ordered:
            total += cost
            knots.add(total)
    points = sorted(knots)
    samples = set(points)
    for a, b in zip(points, points[1:]):
        samples.update(((2 * a + b) / 3, (a + 2 * b) / 3))
    return all(best_threshold(left, t) == best_threshold(right, t) for t in samples)


def by_score(s, u):
    return s


def by_correctness(s, u):
    return u


def error_cost(u):
    return 1 - u


def coverage_cost(u):
    return 1


def check_criteria():
    cases = 0
    for posterior in product((F(0), F(1, 3), F(2, 3), F(1)), repeat=3):
        a, b, c = posterior
        if a == b == c and a in (0, 1):
            continue
        states = [(F(1, 4), F(3, 5), a), (F(1, 4), F(3, 5), b),
                  (F(1, 2), F(4, 5), c)]
        pooled = [(F(1, 2), F(3, 5), (a + b) / 2), states[-1]]
        for cost in (error_cost, coverage_cost):
            threshold = groups(states, by_score, cost)
            full = groups(states, by_correctness, cost)
            score_only = groups(pooled, by_correctness, cost)
            assert frontiers_match(threshold, full) == (a == b and a <= c)
            assert frontiers_match(threshold, score_only) == ((a + b) / 2 <= c)
        cases += 1
    return cases


def check_examples():
    low, high = F(3, 5), F(9, 10)
    states = [(F(1, 2), low, high), (F(1, 2), high, low)]
    error = sum(w * (1 - u) for w, _, u in states)
    cost, gain = groups(states, by_correctness, error_cost)[0]
    assert (cost / error, gain / (1 - error)) == (F(1, 5), F(3, 5))
    assert best_threshold(groups(states, by_score, error_cost), cost) / (1 - error) == F(1, 10)
    assert ((1 - high) / (2 * error), (1 - low) / (2 * error)) == (F(1, 5), F(4, 5))
    for a, b in [(F(7, 10), F(4, 5)), (F(7, 10), F(7, 10))]:
        states = [(F(1, 2), low, a), (F(1, 2), high, b)]
        assert frontiers_match(groups(states, by_score, error_cost),
                               groups(states, by_correctness, error_cost))

    states = [(F(1, 4), low, F(3, 10)), (F(1, 4), low, high),
              (F(1, 2), F(4, 5), F(4, 5))]
    for s in {row[1] for row in states}:
        assert sum(w * (u - s) for w, t, u in states if t == s) == 0
    assert best_threshold(groups(states, by_score, error_cost), F(1, 40)) == F(1, 10)
    assert best_threshold(groups(states, by_correctness, error_cost), F(1, 40)) == F(9, 40)

    states = [(F(1, 2), F(3, 4), F(1)), (F(1, 2), F(3, 4), F(1, 2))]
    prior = sum(w * u for w, _, u in states)
    assert prior == F(3, 4)
    assert best_threshold(groups(states, by_score, error_cost), 0) == 0
    assert best_threshold(groups(states, by_correctness, error_cost), 0) / prior == F(2, 3)

    vectors = [(low, 1 - low), (1 - high, high)]
    class_errors = []
    for y in range(2):
        label_mass = sum(p[y] for p in vectors)
        class_errors.append(sum(p[y] for p in vectors if p.index(max(p)) != y) / label_mass)
    assert class_errors == [F(1, 7), F(4, 13)]
    assert all(p[0] > 0 for p in vectors)
    assert [p.index(max(p)) == 0 for p in vectors] == [True, False]


if __name__ == "__main__":
    geometry = check_geometry()
    experiments = check_criteria()
    check_examples()
    print(f"Passed {geometry} geometric checks, {experiments} finite experiments "
          "under both budgets, and all explicit counterexamples.")
