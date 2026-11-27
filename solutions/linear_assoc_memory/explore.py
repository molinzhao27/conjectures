"""Finite-dimensional strict top-1 storage in Gaussian OP and DP.

Run from the project root with .venv/bin/python solutions/linear_assoc_memory/explore.py.
The printed fractions count checked witnesses. A failed factor search is inconclusive.
"""
import argparse
import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.optimize import brentq, linprog, minimize


RANKS = (0.25, 0.5, 0.75, 1.0)


def capacity(kappa):
    # s = 2 cos(theta) in the upper tail of the quarter-circle law.
    theta = brentq(lambda t: (2*t - np.sin(2*t))/np.pi - kappa, 0, np.pi/2)
    return (theta - np.sin(4*theta)/4)/np.pi


def load_count(d, load):
    root = brentq(lambda p: p*np.log(p) - load*d*d, 1, max(4, load*d*d + 2))
    return max(2, round(root))


class Instance:
    def __init__(self, keys, targets, competitors):
        self.keys = keys
        self.targets = targets
        # OP uses the shared target matrix. DP stores p-1 fresh outputs per key.
        self.competitors = competitors

    def deficits(self, w):
        h = self.keys @ w.T
        target = (self.targets * h).sum(axis=1)
        if self.competitors is None:
            scores = h @ self.targets.T
            np.fill_diagonal(scores, -np.inf)
        else:
            scores = np.einsum("ijd,id->ij", self.competitors, h)
        return scores - target[:, None]

    def witness(self, w):
        # Check at unit Frobenius norm so the numerical buffer has a fixed scale.
        norm = np.linalg.norm(w)
        return norm > 0 and np.max(self.deficits(w / norm)) < -1e-8

    def differences(self, rows, cols):
        wrong = self.targets[cols] if self.competitors is None else self.competitors[rows, cols]
        return self.targets[rows] - wrong

    def loss(self, w):
        # Sum of squared margin deficits over all ordered competing pairs.
        residual = np.maximum(1 + self.deficits(w), 0)
        if self.competitors is None:
            output_grad = residual @ self.targets
        else:
            output_grad = np.einsum("ij,ijd->id", residual, self.competitors)
        output_grad -= residual.sum(axis=1)[:, None] * self.targets
        return 0.5 * np.vdot(residual, residual), output_grad.T @ self.keys


def factor_search(data, rank, rng, restarts):
    d = data.keys.shape[1]
    left, s, right = np.linalg.svd(data.targets.T @ data.keys, full_matrices=False)
    scale = np.sqrt(s[:rank]) / np.sqrt(d)
    for attempt in range(restarts):
        if attempt == 0:
            x = np.stack((left[:, :rank]*scale, right[:rank].T*scale))
        else:
            x = rng.normal(size=(2, d, rank)) / np.sqrt(d)

        def objective(params):
            q, r = params.reshape(2, d, rank)
            value, grad = data.loss(q @ r.T)
            return value, np.stack((grad @ r, grad.T @ q)).ravel()

        result = minimize(objective, x.ravel(), jac=True, method="L-BFGS-B",
                          options={"maxiter": 350, "ftol": 1e-12})
        q, r = result.x.reshape(2, d, rank)
        w = q @ r.T
        if data.witness(w):
            return w
    return None


def full_lp(data):
    """Constraint generation for <W,A_ij> >= 1. Check every omitted inequality."""
    p, d = data.keys.shape
    w = np.zeros((d, d))
    active = set()
    blocks = []
    while True:
        deficits = data.deficits(w)
        if data.witness(w):
            return w, "witness"
        cols = deficits.argmax(axis=1)
        new = [(i, j) for i, j in enumerate(cols) if deficits[i, j] > -0.9
               and (i, j) not in active]
        if not new:
            return None, "indeterminate"
        active.update(new)
        i, j = np.array(new).T
        rows = np.einsum("na,nb->nab", data.differences(i, j), data.keys[i])
        blocks.append(rows.reshape(len(new), -1))
        a = np.vstack(blocks)
        result = linprog(np.zeros(d*d), A_ub=-a, b_ub=-np.ones(len(a)),
                         bounds=(None, None), method="highs-ipm")
        if result.status == 2:
            return None, "LP-infeasible"
        if not result.success:
            return None, "indeterminate"
        w = result.x.reshape(d, d)


def sweep(d, p, trials, seed, restarts):
    counts = np.zeros((2, len(RANKS)), dtype=int)
    infeasible = [0, 0]
    for rep in range(trials):
        rng = np.random.default_rng(np.random.SeedSequence([seed, d, p, rep]))
        keys, targets = rng.normal(size=(2, p, d))
        for model, competitors in enumerate((None, rng.normal(size=(p, p-1, d)))):
            data = Instance(keys, targets, competitors)
            w = None
            for j, kappa in enumerate(RANKS):
                rank = int(d*kappa)
                if w is None:
                    w = factor_search(data, rank, rng, restarts)
                if w is None and rank == d:
                    w, status = full_lp(data)
                    infeasible[model] += status == "LP-infeasible"
                if w is not None:
                    assert data.witness(w) and np.linalg.matrix_rank(w) <= rank
                    counts[model, j] += 1
    return counts, infeasible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--loads", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.4])
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260510795)
    args = parser.parse_args()
    if min(args.dims) < 4 or min(args.loads) <= 0 or min(args.trials, args.restarts) < 1:
        parser.error("Use d >= 4 and positive loads, trials and restarts.")
    print("rank fractions", RANKS, "thresholds", [round(float(capacity(k)), 8) for k in RANKS])
    print("d p alpha model witnesses by rank, full-rank LP-infeasible, trials", flush=True)
    for d in args.dims:
        for load in args.loads:
            p = load_count(d, load)
            counts, infeasible = sweep(d, p, args.trials, args.seed, args.restarts)
            for model, label in enumerate(("OP", "DP")):
                print(d, p, f"{p*np.log(p)/d**2:.4f}", label, *counts[model],
                      infeasible[model], args.trials, flush=True)


if __name__ == "__main__":
    main()
