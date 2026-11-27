"""Numerical checks for the sharp softmax information floor (natural logs).

Run from the project root with .venv/bin/python -B solutions/lm_info_floor/explore.py
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import differential_evolution
from scipy.special import log_softmax, logsumexp, softmax, xlogy

SIZES = (2, 3, 4, 5, 10)
NODES, WEIGHTS = leggauss(64)
NODES = (NODES + 1) / 2
WEIGHTS = WEIGHTS / 2


def model(rows, h, mass):
    """Works with a leading population axis or a single finite law."""
    logits = h @ rows.swapaxes(-1, -2)
    log_q = log_softmax(logits, axis=-1)
    q = np.exp(log_q)
    bar = (mass[..., None] * q).sum(axis=-2)
    info = (mass * (q * (log_q - np.log(bar[..., None, :]))).sum(axis=-1)).sum(axis=-1)
    mean = (mass[..., None] * h).sum(axis=-2)
    disp = (mass * ((h - mean[..., None, :]) ** 2).sum(axis=-1)).sum(axis=-1)
    return info, disp, mean, bar, log_q


def leq(a, b):
    if np.any(a > b + 3e-11 * (1 + np.abs(b))):
        raise AssertionError((a, b))


def audit(rows, h, mass):
    info, disp, mean, bar, log_q = model(rows, h, mass)
    q = np.exp(log_q)
    np.testing.assert_allclose(mass.sum(), 1, atol=1e-14)
    np.testing.assert_allclose(rows.sum(axis=0), 0, atol=1e-12)
    d = h - mean
    diameter = np.linalg.norm(rows[:, None] - rows[None, :], axis=-1).max()
    radius = np.linalg.norm(rows, axis=1).max()
    log_ref = log_softmax(rows @ mean)
    ref_kl = (q * (log_q - log_ref)).sum(axis=1)
    centroid_gap = bar @ (np.log(bar) - log_ref)
    entropy_gap = -xlogy(bar, bar).sum() + mass @ (q * log_q).sum(axis=1)
    np.testing.assert_allclose([entropy_gap, mass @ ref_kl - centroid_gap],
                               info, rtol=1e-9, atol=3e-12)
    leq(0, centroid_gap)
    leq(0, info)

    # Start the Taylor segment at each atom and end it at the mean.
    logits = h @ rows.T
    change = -d @ rows.T
    bregman = logsumexp(rows @ mean) - logsumexp(logits, axis=1) - (q * change).sum(axis=1)
    np.testing.assert_allclose(bregman, ref_kl, atol=3e-12)
    tilt = softmax(logits[:, None, :] + NODES[None, :, None] * change[:, None, :], axis=-1)
    slope = change[:, None, :]
    curvature = (tilt * (slope - (tilt * slope).sum(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    width_sq = np.ptp(change, axis=1) ** 2
    leq(curvature, width_sq[:, None] / 4)
    integral = curvature @ (WEIGHTS * (1 - NODES))
    np.testing.assert_allclose(integral, ref_kl, rtol=1e-8, atol=4e-12)
    leq(ref_kl, width_sq / 8)
    leq(width_sq, diameter ** 2 * (d ** 2).sum(axis=1))
    leq(info, mass @ width_sq / 8)
    leq(info, diameter ** 2 * disp / 8)
    leq(diameter, 2 * radius)
    leq(info, radius ** 2 * disp / 2)
    return info, disp


def search(m, seed):
    """Joint population search over centered unit-radius heads and three-atom laws."""
    def decode(pop):
        raw = pop[:, :2*m].reshape(-1, m, 2)
        centered = raw - raw.mean(axis=1, keepdims=True)
        radius = np.linalg.norm(centered, axis=-1).max(axis=1)
        rows = centered / np.maximum(radius[:, None, None], 1e-100)
        pars = pop[:, 2*m:]
        mean = pars[:, :2]
        spread = np.exp(pars[:, 2])
        mass = softmax(np.column_stack((pars[:, 3:5], np.zeros(len(pop)))), axis=1)
        shape = np.zeros((len(pop), 3, 2))
        shape[:, 1, 0] = 1
        shape[:, 2] = pars[:, 5:]
        shape -= (mass[:, :, None] * shape).sum(axis=1, keepdims=True)
        rms = np.sqrt((mass * (shape ** 2).sum(axis=2)).sum(axis=1))
        h = mean[:, None, :] + (spread / rms)[:, None, None] * shape
        return rows, h, mass

    def objective(x):
        rows, h, mass = decode(x.T)
        info, disp, *_ = model(rows, h, mass)
        return -2 * info / disp

    bounds = [(-1, 1)] * (2*m) + [(-25, 25)] * 2 + [(-4, 1)] + [(-3, 3)] * 2 + [(-2, 2)] * 2
    result = differential_evolution(objective, bounds, rng=np.random.default_rng(seed),
                                    maxiter=160, popsize=6, tol=0, polish=False,
                                    vectorized=True, updating="deferred")
    rows, h, mass = (v[0] for v in decode(result.x[None]))
    info, disp = audit(rows, h, mass)
    return 2 * info / disp


def sharp_family(m):
    for eps in (0.2, 0.1, 0.05, 0.02, 0.01):
        b = eps if m > 2 else 0.
        a = np.sqrt(1 - b*b)
        rows = np.zeros((m, 2))
        rows[:2] = [[a, b], [-a, b]]
        if m > 2:
            rows[2:, 1] = -2*b / (m-2)
        height = eps ** -2 if m > 2 else 0.
        h = np.array([[eps, height], [-eps, height]])
        info, disp = audit(rows, h, np.array([0.5, 0.5]))
        x = a * eps
        active_mass = softmax(rows @ h[0])[:2].sum()
        exact = active_mass * (x * np.tanh(x) - np.log(np.cosh(x)))
        np.testing.assert_allclose(info, exact, rtol=5e-10, atol=2e-14)
    ratio = 2 * info / disp
    assert 0.9998 < ratio <= 1
    return ratio


def main():
    rng = np.random.default_rng(260709487)
    print("m   random laws   best of 3 searches   sharp family    (scores = 2 I / (R^2 D))")
    for m in SIZES:
        for i in range(120):
            dim = (1, 3, 7)[i % 3]
            n = int(rng.integers(2, 13))
            rows = rng.normal(size=(m, dim))
            rows -= rows.mean(axis=0)
            rows /= np.linalg.norm(rows, axis=1).max()
            h = rng.normal(size=(n, dim)) * rng.uniform(0.02, 4)
            h += rng.normal(size=dim) * rng.uniform(0, 7)
            mass = rng.dirichlet(np.full(n, 0.5))
            audit(rows, h, mass)
        # Both zero rows and deterministic contexts have zero information.
        audit(np.zeros_like(rows), h, mass)
        audit(rows, h[:1], np.ones(1))
        best = max(search(m, 900*m + j) for j in range(3))
        print(f"{m:2}      120         {best:.8f}         {sharp_family(m):.8f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
