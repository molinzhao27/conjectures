"""Reproduce the two-head softmax examples in solution.md (NumPy and SciPy)."""

from fractions import Fraction
from itertools import combinations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize
from scipy.special import softmax


DECAY = 0.01
EYE = np.eye(2)


def haar_mean(rank, dimension):
    if not 1 <= rank <= dimension:
        raise ValueError("expected full frames of rank between 1 and dimension")
    return Fraction(rank, dimension)


def overlap(frames):
    """Unordered-pair version of the paper's statistic for full-rank frames."""
    bases = []
    for frame in frames:
        if np.linalg.matrix_rank(frame) != frame.shape[1]:
            raise ValueError("rank-deficient frame")
        bases.append(np.linalg.qr(frame, mode="reduced")[0])
    rank = bases[0].shape[1]
    return np.mean([np.linalg.norm(u.T @ v)**2 / rank
                    for u, v in combinations(bases, 2)])


def query_overlap(theta):
    queries = theta.reshape(4, 2, 2)[0]
    return overlap([q[:, None] for q in queries])


def data(weights, shared):
    queries, memories, targets, sample_weights = [], [], [], []
    for i in range(2):
        for j in range(2):
            queries.append(EYE[i])
            memories.append(EYE[j])
            active = i == (0 if shared else j)
            targets.append(EYE[j] if active else np.zeros(2))
            sample_weights.append(weights[i])
    return tuple(np.asarray(a) for a in (queries, memories, targets, sample_weights))


def objective(theta, samples):
    """Rows of each parameter block are the two heads. Train all 16 entries."""
    q, k, v, o = theta.reshape(4, 2, 2)
    x, m, target, weights = samples
    queries, keys, values = x @ q.T, m @ k.T, m @ v.T
    gates = np.tanh(queries * keys)
    hidden = values * gates
    residual = hidden @ o - target
    error = weights[:, None] * residual
    loss = (np.sum(error * residual) + DECAY * (theta @ theta)) / 2
    hidden_grad = error @ o.T
    logit_grad = hidden_grad * values * (1 - gates**2)
    gradient = np.stack(((logit_grad * keys).T @ x, (logit_grad * queries).T @ m,
                         (hidden_grad * gates).T @ m, hidden.T @ error))
    return loss, gradient.ravel() + DECAY * theta


def scalar_solution(weight):
    delta = DECAY / weight

    def cubic(s):
        return 2*s**3 - delta*s**2 - s + delta

    s = brentq(cubic, 0.5, 1)
    b = np.arctanh(s)
    a = (s - delta) / s**2
    loss = weight * (1 - a*s)**2 / 2 + DECAY * (a + b)
    return a, b, loss


def dedicated_solution(weights, shared):
    blocks = np.zeros((4, 2, 2))
    q, k, v, o = blocks
    bound = 0
    for j in range(2):
        direction = 0 if shared else j
        a, b, loss = scalar_solution(weights[direction])
        q[j, direction] = k[j, j] = np.sqrt(b)
        v[j, j] = o[j, j] = np.sqrt(a)
        bound += loss
    return blocks.ravel(), bound


def check_calculus(samples, rng):
    theta = rng.normal(size=16)
    q, k, v, o = theta.reshape(4, 2, 2)
    x, m, target, weights = samples
    tokens = np.stack((m, -m), axis=1)
    logits = (x @ q.T)[:, None, :] * (tokens @ k.T)
    actual = (softmax(logits, axis=1) * (tokens @ v.T)).sum(axis=1) @ o
    direct = (np.sum(weights[:, None] * (actual - target)**2) + DECAY * theta @ theta) / 2
    loss, grad = objective(theta, samples)
    np.testing.assert_allclose(direct, loss, atol=1e-13)
    eps = 1e-5
    numerical = [(objective(theta + eps * e, samples)[0] -
                  objective(theta - eps * e, samples)[0]) / (2*eps) for e in np.eye(16)]
    np.testing.assert_allclose(grad, numerical, rtol=1e-6, atol=2e-9)


def check_training(weights, shared, rng):
    samples = data(weights, shared)
    check_calculus(samples, rng)
    reference, value = dedicated_solution(weights, shared)
    loss, grad = objective(reference, samples)
    np.testing.assert_allclose(loss, value)
    assert np.linalg.norm(grad) < 1e-11
    expected_r = float(shared)
    np.testing.assert_allclose(query_overlap(reference), expected_r)
    # Use both dense perturbations of the explicit solution and unstructured starts.
    starts = [reference + rng.normal(scale=0.08, size=16) for _ in range(3)]
    starts.extend(rng.normal(scale=0.7, size=16) for _ in range(9))
    hits = 0
    best = np.inf
    for initial in starts:
        fit = minimize(objective, initial, (samples,), jac=True, method="L-BFGS-B",
                       options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 2000})
        best = min(best, fit.fun)
        if not shared:
            assert fit.fun >= value - 1e-10
        if fit.success and fit.fun <= value + 1e-9:
            assert np.linalg.norm(fit.jac) < 2e-6
            assert abs(query_overlap(fit.x) - expected_r) < 1e-8
            hits += 1
    assert hits >= 3
    print(f"weights={weights} shared={shared}: reference={value:.10f} best={best:.10f}"
          f" R={expected_r:g}, {hits}/{len(starts)} reached reference or better")


def check_dynamics():
    samples = data((1, 1), False)
    initial, optimum = dedicated_solution((1, 1), False)
    q = initial.reshape(4, 2, 2)[0]
    q[0, 1] = 1.1 * q[0, 0]
    dq = -objective(initial, samples)[1].reshape(4, 2, 2)[0]
    u, du = q[0], dq[0]
    assert np.linalg.norm(dq[1]) < 1e-11
    derivative = 2*u[1]*du[1]/(u @ u) - 2*u[1]**2*(u @ du)/(u @ u)**2
    run = solve_ivp(lambda t, p: -objective(p, samples)[1], (0, 800), initial,
                    t_eval=[0, 1, 10, 800], rtol=2e-10, atol=1e-12, method="DOP853")
    assert run.success and derivative < 0
    overlaps = [query_overlap(p) for p in run.y.T]
    assert overlaps[0] > 0.5 and overlaps[-1] < 1e-10
    assert np.all(np.diff(overlaps) < 0)
    gap = objective(run.y[:, -1], samples)[0] - optimum
    assert abs(gap) < 1e-6
    checkpoints = ", ".join(f"{r:.9g}" for r in overlaps)
    print(f"flow dR/dt(0)={derivative:.10f}, R(0,1,10,800)=[{checkpoints}], final gap={gap:.3g}")


def main():
    rng = np.random.default_rng(260509861)
    print("Haar means for example, GPT-2 small, Llama-3.2-1B:",
          *(str(haar_mean(r, d)) for r, d in [(1, 2), (64, 768), (64, 2048)]))
    # Check the general projector criterion beyond the rank-one examples.
    frames = rng.normal(size=(5, 7, 3))
    bases = np.linalg.qr(frames)[0]
    total = (bases @ bases.transpose(0, 2, 1)).sum(axis=0)
    excess = np.linalg.norm(total - (15/7)*np.eye(7))**2 - 15*(1 - 3/7)
    np.testing.assert_allclose(excess, 60*(overlap(frames) - 3/7), atol=1e-13)
    for weights, shared in [((1, 1), False), ((2, 1), False), ((2, 1), True)]:
        check_training(weights, shared, rng)
    check_dynamics()


if __name__ == "__main__":
    main()
