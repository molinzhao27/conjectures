"""Exact-gradient associative memory experiments. Run directly with NumPy and SciPy."""
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.special import logsumexp


def zipf(n, alpha):
    weights = np.arange(1, n + 1, dtype=float) ** -alpha
    return weights / weights.sum()


def state(u, v, w, q):
    # Rows are inputs, columns are candidate outputs. Excluding the diagonal
    # explicitly retains small residuals when the correct probability rounds to 1.
    logits = v @ w.T @ u.T
    normalizer = logsumexp(logits, axis=1)
    probabilities = np.exp(logits - normalizer[:, None])
    correct_logits = logits.diagonal().copy()
    np.fill_diagonal(logits, -np.inf)
    margins = correct_logits - logits.max(axis=1)
    wrong = probabilities.copy()
    np.fill_diagonal(wrong, 0)
    residual = wrong.sum(axis=1)[:, None] * u - wrong @ u
    loss = q @ np.logaddexp(0, logsumexp(logits, axis=1) - correct_logits)
    return probabilities, (residual * q[:, None]).T @ v, loss, margins


def spectral(g, lam):
    left, s, right = np.linalg.svd(g, full_matrices=False)
    if lam:
        weights = s / np.hypot(s, lam)
    else:
        weights = (s > max(g.shape) * np.finfo(float).eps * s[0]).astype(float)
    return (left * weights) @ right


def hessian(u, v, probs, q):
    # C-order vectorization of the d x d parameter matrix.
    n, d = u.shape
    means = probs @ u
    second = (probs @ (u[:, :, None] * u[:, None, :]).reshape(n, d*d)).reshape(n, d, d)
    cov = second - means[:, :, None] * means[:, None, :]
    return np.einsum("n,nac,nb,ne->abce", q, cov, v, v, optimize=True).reshape(d*d, d*d)


def newton(u, v, probs, q, g):
    h = hessian(u, v, probs, q)
    direction = np.linalg.solve(h, g.ravel())
    np.testing.assert_allclose(h @ direction, g.ravel(), rtol=1e-8, atol=1e-12)
    return direction.reshape(g.shape)


def prefix(margins):
    missed = np.flatnonzero(margins <= 0)
    return int(missed[0]) if missed.size else len(margins)


def finite_examples():
    v = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    u = v @ np.array([[1., 0.], [2., 1.]])
    p = zipf(4, 2)
    probs, g, _, _ = state(u, v, np.zeros((2, 2)), p)
    dn = newton(u, v, probs, p, g)
    np.testing.assert_allclose(dn, [[2, 0], [-4, 2]], atol=1e-13)
    for lam in (0, 1e-4, 0.2, 1., 100.):
        dm = spectral(g, lam)
        np.testing.assert_array_equal(state(u, v, dm, p)[3] > 0, [False, False, True, True])
    assert prefix(state(u, v, dn, p)[3]) == 4
    print("Four items: Muon 2/4, Newton 4/4")

    u = np.arange(3.)[:, None]
    v = np.ones_like(u)
    p = zipf(3, 2)
    for method in ("polar", "newton"):
        w = np.zeros((1, 1))
        path = []
        for _ in range(10):
            probs, g, _, _ = state(u, v, w, p)
            w += spectral(g, 0) if method == "polar" else newton(u, v, probs, p, g)
            path.append(w.item())
        if method == "polar":
            np.testing.assert_array_equal(path, [-1., -2.] * 5)
        else:
            assert abs(state(u, v, w, p)[1].item()) < 1e-14
        print(method, "scalar final pair", path[-2:])


def trajectories(u, v, p, alpha, steps):
    n, d = u.shape
    paths = {}
    for method in ("polar", "stabilized", "deflation", "newton"):
        w = np.zeros((d, d))
        history = []
        for t in range(steps):
            probs, g, _, _ = state(u, v, w, p)
            cutoff = min(n, int(d ** (2 - (1 - 1/(2*alpha))**t) / 4)) if t else 0
            lam = max(cutoff, 1)**(0.5-alpha) / np.sqrt(d) if t else d**-alpha
            if method == "newton":
                w += newton(u, v, probs, p, g) / (2*np.sqrt(d))
            else:
                if method == "deflation":
                    g = (u[cutoff:] * p[cutoff:, None]).T @ v[cutoff:]
                w += 2*np.sqrt(d) * spectral(g, 0 if method == "polar" else lam)
            _, _, loss, margins = state(u, v, w, p)
            history.append((prefix(margins), int(sum(margins > 0)), loss))
        paths[method] = np.array(history)
    return paths


def population_sweep():
    alpha = 1.5
    print("Population sweep: kappa is the output covariance condition number.")
    print("d kappa method | median prefix by step | median final total, loss")
    for d in (4, 8, 12):
        runs = {k: [] for k in (1, 16, 256)}
        for seed in range(3):
            rng = np.random.default_rng(1000*d + seed)
            n = 4*d*d
            v, base = rng.normal(size=(2, n, d)) / np.sqrt(d)
            p = zipf(n, alpha)
            for kappa, group in runs.items():
                scale = np.geomspace(1, kappa**-0.5, d)
                u = base * (scale / np.sqrt(np.mean(scale**2)))
                if kappa == 1 and seed == 0:
                    # Independent gradient and Hessian checks at a nonzero point.
                    w, delta = rng.normal(size=(2, d, d))
                    probs, g, _, _ = state(u, v, w, p)
                    eps = 1e-5
                    plus, minus = [state(u, v, w+s*eps*delta, p) for s in (1, -1)]
                    np.testing.assert_allclose((plus[2]-minus[2])/(2*eps), -np.sum(g*delta),
                                               rtol=1e-7, atol=1e-9)
                    np.testing.assert_allclose((minus[1]-plus[1]).ravel()/(2*eps),
                                               hessian(u, v, probs, p) @ delta.ravel(), atol=1e-9)
                result = trajectories(u, v, p, alpha, 3)
                if kappa == 1:
                    reference = result["newton"]
                np.testing.assert_allclose(result["newton"], reference, atol=1e-9)
                group.append(result)
        for kappa, group in runs.items():
            for method in group[0]:
                med = np.median([r[method] for r in group], axis=0)
                print(d, kappa, method, med[:, 0].astype(int).tolist(),
                      int(med[-1, 1]), f"{med[-1, 2]:.6f}")


def spectral_estimate_check():
    rng = np.random.default_rng(425)
    m = 8
    defects = []
    for delta in (1e-2, 1e-3, 1e-4):
        factors = []
        for _ in range(2):
            error = rng.uniform(-delta, delta, size=(m, m))
            gram = np.eye(m) + (error + error.T)/2
            factors.append(np.linalg.cholesky(gram).T)
        left, right = factors
        for condition in (1, 1e4, 1e8):
            diagonal = np.diag(np.geomspace(1, 1/condition, m))
            g = left @ diagonal @ right.T
            for lam in (0., 1e-5, 1.):
                output = spectral(g, lam)
                coefficients = np.linalg.solve(right, np.linalg.solve(left, output).T).T
                defect = np.max(np.abs(coefficients - spectral(diagonal, lam)))
                defects.append(defect/delta)
    assert max(defects) < 2
    print("Spectral estimate:", len(defects), "cases, max error/delta", f"{max(defects):.6f}")


def sparse_batches():
    # Exact-gradient rank-deficient steps. The ordinary inverse Hessian does
    # not exist in this regime. Track previously observed items directly.
    rng = np.random.default_rng(380)
    d, n, b = 128, 512, 12
    u, v = rng.normal(size=(2, n, d)) / np.sqrt(d)
    p = zipf(n, 1.5)
    batches = [rng.multinomial(b, p)/b for _ in range(4)]
    for lam in (0., 1/b):
        w = np.zeros((d, d))
        seen = np.zeros(n, dtype=bool)
        retained = []
        for q in batches:
            _, g, _, _ = state(u, v, w, q)
            w += np.sqrt(d) * spectral(g, lam)
            seen |= q > 0
            margins = state(u, v, w, p)[3]
            assert np.all(margins[seen] > 0)
            retained.append(seen.sum().item())
        print("Minibatches, lambda", lam, "all seen retained:", retained)


if __name__ == "__main__":
    finite_examples()
    population_sweep()
    spectral_estimate_check()
    sparse_batches()
