"""Reproduce the tables in solution.md. All entropies are in nats."""
from fractions import Fraction
from itertools import combinations, product
from math import comb, log

import mpmath as mp
import sympy as sp

import numpy as np
from scipy.special import digamma, factorial, xlogy


def entropy(p):
    return float(-xlogy(p, p).sum())


def capped_dirichlet(rng, m, cap):
    for attempts in range(1, 1_000_001):
        q = rng.dirichlet(np.ones(m))
        if entropy(q) <= cap:
            return q, attempts
    raise RuntimeError("Rejection budget exhausted")


def greedy(rows):
    residual = rows.copy()
    indices = np.arange(len(rows))
    rebuilt = np.zeros_like(rows)
    masses = []
    limit = 1 + np.sum(np.count_nonzero(rows, axis=1) - 1)
    for _ in range(limit):
        choices = residual.argmax(axis=1)
        mass = residual[indices, choices].min()
        if mass <= 0:
            break
        residual[indices, choices] -= mass
        rebuilt[indices, choices] += mass
        masses.append(mass)
    assert np.max(np.abs(rebuilt - rows)) < 3e-12
    assert abs(sum(masses) - 1) < 3e-12
    return np.array(masses)


def rational_joint(p, q, f):
    # Treat the sampled floating weights as exact rationals before aggregation.
    x = list(map(Fraction.from_float, p))
    e = list(map(Fraction.from_float, q))
    joint = np.full((len(p), len(p)), Fraction(0))
    for i, row in enumerate(f):
        for j, y in enumerate(row):
            joint[i, y] += x[i] * e[j]
    return joint


def binary_minimum(joint, m):
    """Exact rational optimizer for the supplied binary joint weights.

    With the more probable symbol first in each row and p >= q, the
    aligned endpoint majorizes the other endpoint. Check both endpoints
    for the support constraint. Evaluate entropy only after optimization.
    """
    params = []
    for col in joint.T:
        weights = list(col)
        if sum(weights):
            params.append(max(weights) / sum(weights))
    if len(params) == 1:
        params.append(Fraction(1))
    p, q = sorted(params, reverse=True)
    for t in (q, p + q - 1):
        masses = [v for v in (t, p - t, q - t, 1 - p - q + t) if v]
        if len(masses) <= m:
            return masses
    return None


def evaluate(joint, hx, he):
    py = joint.sum(axis=0)
    rows = (joint[:, py > 0] / py[py > 0]).T
    masses = greedy(rows)
    hg = entropy(masses)
    hy = entropy(py)
    # Sorted partial sums of every marginal dominate those of any coupling.
    prefix = np.cumsum(np.sort(rows, axis=1)[:, ::-1], axis=1).min(axis=0)
    meet = np.maximum(np.diff(prefix, prepend=0), 0)
    lower = entropy(meet)
    assert max(map(entropy, rows)) <= lower + 1e-10 <= hg + 2e-10
    assert hg <= log(joint.shape[0]) + 1
    assert entropy(joint) <= min(hy + hg, hx + he) + 1e-10
    return hx, he, hy, lower, hg, len(masses), hg + hy - hx - he


def trials(rng, n, m, c, count):
    records = []
    proposals = exact_feasible = reversals = 0
    for _ in range(count):
        p = rng.dirichlet(np.ones(n))
        q, attempts = capped_dirichlet(rng, m, log(n) + c)
        proposals += attempts
        f = rng.integers(n, size=(n, m))
        joint = np.array([px * np.bincount(row, weights=q, minlength=n)
                          for px, row in zip(p, f)])
        record = evaluate(joint, entropy(p), entropy(q))
        if n == 2:
            opt = binary_minimum(rational_joint(p, q, f), m)
            if opt is not None:
                exact_feasible += 1
                value = entropy(np.array(opt, dtype=float))
                assert record[3] - 1e-10 <= value
                if record[5] <= m:
                    assert value <= record[4] + 1e-10
                reversals += value + record[2] < record[0] + record[1] - 1e-10
        records.append(record)
    a = np.array(records)
    witnesses = np.sum((a[:, 5] <= m) & (a[:, 6] < -1e-10))
    means = " ".join(f"{v:.4f}" for v in a.mean(axis=0))
    print(n, m, c, count, proposals, means, witnesses,
          (exact_feasible, reversals) if n == 2 else "-")


def vertex_minimum(joint, m):
    """Exact basis enumeration for modest transportation polytopes."""
    rows = [col / sum(col) for col in joint.T if sum(col)]
    supports = [list(np.flatnonzero(row)) for row in rows]
    atoms = list(product(*supports))
    equations = [(i, x) for i, supp in enumerate(supports) for x in supp[:-1]]
    A = sp.Matrix([[1] * len(atoms)] +
                  [[int(atom[i] == x) for atom in atoms] for i, x in equations])
    b = sp.Matrix([1] + [rows[i][x] for i, x in equations])
    rank = len(b)  # full row rank by varying one response coordinate at a time
    size = min(m, rank)
    if comb(len(atoms), size) > 10000:
        raise ValueError("Instance exceeds the small exact-enumeration budget")
    candidates = set()
    for basis in combinations(range(len(atoms)), size):
        if any({atoms[j][i] for j in basis} != set(supp)
               for i, supp in enumerate(supports)):
            continue
        columns = A[:, list(basis)]
        square = columns if size == rank else columns.T * columns
        if square.det() == 0:
            continue
        solution = square.inv() * (b if size == rank else columns.T * b)
        if columns * solution != b:
            continue
        weights = list(solution)
        if min(weights) >= 0 and len(weights) - weights.count(0) <= m:
            candidates.add(tuple(sorted(v for v in weights if v)))
    if not candidates:
        return None
    mp.iv.dps = 100
    intervals = []
    for candidate in candidates:
        w = [mp.iv.mpf(int(v.p)) / int(v.q) for v in candidate]
        intervals.append((-sum(v * mp.iv.ln(v) for v in w), candidate))
    ordered = sorted(intervals, key=lambda item: float(item[0].mid))
    best, weights = ordered[0]
    assert all(best.b < other.a for other, _ in ordered[1:])
    return weights


def ternary_exact(rng):
    # Include sparse instances from three forward alphabet sizes.
    for m in (1, 2, 3):
        feasible = 0
        for _ in range(16):
            p = rng.dirichlet(np.ones(3))
            q, _ = capped_dirichlet(rng, m, log(3))
            f = rng.integers(3, size=(3, m))
            joint = rational_joint(p, q, f)
            opt = vertex_minimum(joint, m)
            if opt is not None:
                feasible += 1
                joint_float = np.asarray(joint, dtype=float)
                joint_float /= joint_float.sum()
                record = evaluate(joint_float, entropy(p), entropy(q))
                value = entropy(np.array(opt, dtype=float))
                assert record[3] <= value + 1e-10
                if record[5] <= m:
                    assert value <= record[4] + 1e-10
        print(f"ternary exact m={m}: {feasible}/16 feasible")


def dyadic_bound():
    iv = mp.iv
    iv.dps = 50
    r, correction = iv.mpf(1), iv.mpf(0)
    for j in range(5, -13, -1):
        z = iv.mpf(2)**j
        remainder = 1 - z / (iv.exp(z) - 1)
        correction -= (r - remainder) * iv.ln(z)
        r = remainder
    bound = correction + r * (1 - iv.ln(r))
    # All omitted terms of the Poisson-Gamma series are positive.
    harmonic, fact, partial = iv.mpf(0), 1, iv.mpf(0)
    for k in range(1, 31):
        harmonic += iv.mpf(1) / k
        fact *= k
        partial += k * (harmonic - iv.euler) / fact
    u = iv.mpf(1) / 1024
    deficit_lower = (1-u)*partial/iv.exp(1) + u*iv.ln(u) + (1-u)*iv.ln(1-u)
    assert (bound + 1 - iv.euler - deficit_lower).b < iv.mpf("-0.07354")
    print(f"dyadic coupling entropy correction {float(bound.mid):.10f}")


def single_map_limit(rng):
    k = np.arange(1, 60)
    deficit = np.sum(k * digamma(k + 1) / factorial(k)) / np.e
    print(f"single-map asymptotic deficit {deficit:.10f}")
    print("surrogate n  mean(HY-log n)  mean(Hgreedy-HX)  mean(reverse-forward)")
    for n in (8, 16, 32, 64):
        sums = []
        for _ in range(32):
            p = rng.dirichlet(np.ones(n))
            s = log(n) / n**5
            joint = np.tile(p[:, None] * (s / n), (1, n))
            joint[np.arange(n), rng.integers(n, size=n)] += (1 - s) * p
            hx, _, hy, _, hg, _, gap = evaluate(joint, entropy(p), log(n))
            sums.append((hy - log(n), hg - hx, gap))
        print(n, *(f"{v:.5f}" for v in np.mean(sums, axis=0)))


if __name__ == "__main__":
    rng = np.random.default_rng(210103501)
    print("n m c trials proposals HX HE HY MEC_lower Hgreedy atoms gap witnesses binary(feasible,reversed)")
    for config in [(2, 1, 0, 100), (2, 2, 0, 100), (2, 4, 0, 100),
                   (2, 4, 2, 100), (16, 2, 2, 64), (32, 32, 0, 32),
                   (4, 16, 2, 64), (8, 64, 2, 64), (10, 100, 2, 64),
                   (12, 144, 2, 64)]:
        trials(rng, *config)
    single_map_limit(rng)
    dyadic_bound()
    ternary_exact(rng)
