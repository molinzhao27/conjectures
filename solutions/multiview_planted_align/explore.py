"""Finite checks for the proof in results.md. Run with Python, NumPy and SciPy.

The default enumerates every anchored alternative of full support on 2--4
rows, modulo simultaneous row relabeling. --max-support changes that limit
(factorial cost). The determinant formula accepts arbitrary finite supports.
"""
import argparse
import itertools as it
import math
from collections import Counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import linear_sum_assignment


def logdet(matrix):
    return 2 * np.log(np.diag(np.linalg.cholesky(matrix))).sum()


def partitions(k):
    if k == 0:
        yield ()
        return
    for p in partitions(k - 1):
        yield p + ((k - 1,),)
        for i in range(len(p)):
            yield p[:i] + (p[i] + (k - 1,),) + p[i + 1:]


def partition_rate(r, blocks):
    """Fixed point for the compatible Gaussian tilt; check KL and Renyi forms."""
    k, q = len(r), len(blocks)
    beta = 1 / q
    alpha = 1 - beta
    omega = np.linalg.inv(r)
    w = np.eye(k)
    for _ in range(4000):
        s = np.linalg.inv(alpha * omega + beta * w)
        updated = np.zeros_like(w)
        for block in blocks:
            ix = np.ix_(block, block)
            updated[ix] = np.linalg.inv(s[ix])
        if np.linalg.norm(updated - w) < 1e-11 * max(1, np.linalg.norm(w)):
            break
        w = updated
    else:
        raise ArithmeticError("Gaussian product-reference iteration did not converge")
    rare = (alpha * logdet(r) - beta * logdet(w) - logdet(s)) / (2 * alpha)
    kl = (np.trace(omega @ s) - k + logdet(r) - logdet(s)) / 2
    tc = (sum(logdet(s[np.ix_(b, b)]) for b in blocks) - logdet(s)) / 2
    assert abs(rare - kl - tc / (q - 1)) < 1e-8
    return rare


def rates(r):
    cuts, many = [], []
    for p in partitions(len(r)):
        if len(p) < 2:
            continue
        value = partition_rate(r, p)
        if len(p) == 2:
            cut = (sum(logdet(r[np.ix_(b, b)]) for b in p) - logdet(r)) / 4
            assert value >= cut - 1e-9
            cuts.append(cut)
        else:
            many.append(value)
    return min(cuts), min([*cuts, *many])


def models(k, rng):
    i = np.arange(k)
    rho = 0.7 - 0.09 * i[:-1]
    chain = np.array([[np.prod(rho[min(a, b):max(a, b)]) for b in i] for a in i])
    def equi(x):
        return (1 - x) * np.eye(k) + x * np.ones((k, k))
    yield "chain", chain
    yield "positive", equi(0.48)
    yield "negative", equi(-1 / k)
    yield "near-boundary", equi(-0.98 / (k - 1))
    for j in range(2):
        a = rng.normal(size=(k, k))
        g = a @ a.T + np.eye(k) * 0.15
        scale = np.sqrt(np.diag(g))
        yield f"random-{j + 1}", g / np.outer(scale, scale)


def pattern_classes(k, m):
    perms = np.array(list(it.permutations(range(m))))
    lookup = {tuple(p): i for i, p in enumerate(perms)}
    conjugates = np.array([[lookup[tuple(h[p[np.argsort(h)]])] for p in perms] for h in perms])
    seen = set()
    for tail in it.product(range(len(perms)), repeat=k - 1):
        if tail in seen:
            continue
        orbit = set(map(tuple, conjugates[:, tail]))
        seen.update(orbit)
        sig = perms[[0, *tail]].T
        if np.any(np.all(sig == np.arange(m)[:, None], axis=1)):
            continue
        yield sig, len(orbit)


def exponents(r, sigs):
    """Chernoff maximum and midpoint via generalized covariance eigenvalues."""
    count, m, k = sigs.shape
    cov = np.kron(np.eye(m), r)
    inv_chol = solve_triangular(np.linalg.cholesky(cov), np.eye(k * m), lower=True)
    index = (sigs * k + np.arange(k)).reshape(count, -1)
    alternative = cov[index[:, :, None], index[:, None, :]]
    roots = np.linalg.eigvalsh(inv_chol @ alternative @ inv_chol.T)
    assert np.max(np.abs(np.log(roots).sum(axis=1))) < 1e-8
    shift = roots - 1
    lo, hi = np.zeros((count, 1)), np.ones((count, 1))
    for _ in range(42):
        t = (lo + hi) / 2
        increasing = (shift / (1 + t * shift)).sum(axis=1, keepdims=True) > 0
        lo = np.where(increasing, t, lo)
        hi = np.where(increasing, hi, t)
    return (np.log1p(t * shift).sum(axis=1) / 2,
            np.log1p(0.5 * shift).sum(axis=1) / 2, t.ravel())


def census(k, designs, max_support):
    largest_shift = 0
    worst = np.full(len(designs), np.inf)
    for m in range(2, max_support + 1):
        cases = list(pattern_classes(k, m))
        raw = sum(mult for _, mult in cases)
        expected = sum((-1)**j * math.comb(m, j) * math.factorial(m-j)**(k-1)
                       for j in range(m + 1))
        assert raw == expected
        print(f"  support {m}: {len(cases)} classes, {raw} labeled, "
              f"{audit_prefix_ranks(cases)} proper subsets checked")
        for offset in range(0, len(cases), 128):
            batch = np.array([sig for sig, _ in cases[offset:offset + 128]])
            for j, (_, r, half_cut, _) in enumerate(designs):
                c, b, t = exponents(r, batch)
                assert np.min(c - b) > -1e-9
                assert np.min(b / m) >= half_cut - 1e-9
                worst[j] = min(worst[j], np.min(c / m))
                largest_shift = max(largest_shift, np.max(np.abs(t - 0.5)))
    for val, (_, _, half_cut, _) in zip(worst, designs):
        assert abs(val - half_cut) < 1e-9
    print(f"  greatest |optimal tilt - 1/2|: {largest_shift:.8f}")


def incidence(sig):
    """Left = candidate, right = source. Retain only the support of edges."""
    k = sig.shape[1]
    return [set(row) for row in sig], [set(row) for row in np.argsort(sig, axis=0)]


def clusters(neighbors, vertices):
    groups = []
    for i in vertices:
        component = set(neighbors[i])
        separate = []
        for group in groups:
            if component.isdisjoint(group):
                separate.append(group)
            else:
                component.update(group)
        groups = [*separate, component]
    return groups


def audit_prefix_ranks(cases):
    """All proper nonempty subsets on each connected small incidence graph."""
    audited = set()
    checks = 0
    for sig, _ in cases:
        for neighbors in incidence(sig):
            key = tuple(sorted(tuple(sorted(edge)) for edge in neighbors))
            if key in audited:
                continue
            audited.add(key)
            m = len(neighbors)
            if len(clusters(neighbors, range(m))) != 1:
                continue
            for size in range(1, m):
                for subset in it.combinations(range(m), size):
                    merged = clusters(neighbors, subset)
                    assert sum(len(cell) - 1 for cell in merged) >= size
                    checks += 1
    return checks


def weighted_charge(neighbors, vertices, weights, floor, gap):
    groups = []
    cost = 0.
    for i in sorted(vertices, key=lambda j: -weights[j]):
        left = set(neighbors[i])
        contacts = [g for g in groups if g & left]
        new = left.difference(*contacts)
        merges = len(contacts) + len(new) - 1
        cost += merges * (weights[i] + gap)
        groups = [g for g in groups if not g & left] + [left.union(*contacts)]
    assert len(groups) == 1
    size = len(groups[0])
    assert cost + 1e-10 >= (size - 1) * (floor + gap) + sum(weights[i] - floor for i in vertices)
    return cost, groups[0]


def check_witness(k, rng, repeats=1000):
    """Check both rank payment and the two-sided budget contradiction."""
    m, threshold, gap = 90, 14, 0.2
    for _ in range(repeats):
        # A cyclic shift forces connectedness. The other colors may have collisions.
        sig = np.array([np.arange(m), np.roll(np.arange(m), 1),
                        *(rng.permutation(m) for _ in range(k - 2))]).T
        fwd, rev = incidence(sig)
        graph = [set(m + j for j in row) for row in fwd] + rev
        weight = rng.random(2 * m)
        active = set()
        for i in np.argsort(-weight):
            active.add(i)
            component, pending = {i}, [i]
            while pending:
                fresh = (graph[pending.pop()] & active) - component
                component.update(fresh)
                pending.extend(fresh)
            if len(component) >= threshold:
                break
        left = {v for v in component if v < m}
        right = {v - m for v in component if v >= m}
        assert max(len(left), len(right)) < m
        a, nr = weighted_charge(fwd, left, weight[:m], weight[i], gap)
        b, nl = weighted_charge(rev, right, weight[m:], weight[i], gap)
        excess = a + b - sum(weight[m + j] for j in nr) - sum(weight[j] for j in nl)
        assert excess + 1e-9 >= gap * (len(nl) + len(nr) - 2) - 2 * weight[i] > 0
    return repeats


def exact_ml(r, n, d_values, rng, trials):
    """Enumerate the joint ML objective, caching all permutation-pair scores."""
    k = len(r)
    perms = np.array(list(it.permutations(range(n))))
    candidates = np.indices((len(perms),) * (k - 1), dtype=np.int16).reshape(k - 1, -1)
    indices = [np.zeros(candidates.shape[1], dtype=np.int16), *candidates]
    omega = np.linalg.inv(r)
    chol = np.linalg.cholesky(r)
    wins = Counter()
    for rep in range(trials):
        x = rng.normal(size=(max(d_values), n, k)) @ chol.T
        for d in d_values:
            scores = np.zeros(candidates.shape[1])
            for a, b in it.combinations(range(k), 2):
                pair = x[:d, :, a].T @ x[:d, :, b]
                table = pair[perms[:, None, :], perms[None, :, :]].sum(axis=2)
                scores -= omega[a, b] * table[indices[a], indices[b]]
            if rep == 0:
                idx = len(scores) // 3
                rows = np.stack([x[:d, perms[group[idx]], a] for a, group in enumerate(indices)], axis=-1)
                direct = -0.5 * np.einsum("dij,jk,dik->", rows, omega, rows)
                true = -0.5 * np.einsum("dij,jk,dik->", x[:d], omega, x[:d])
                assert abs(scores[idx] - scores[0] - direct + true) < 1e-8
            wins[d] += int(np.argmax(scores) == 0)
    return [wins[d] for d in d_values]


def chain_ml(r, n, ds, rng, trials):
    """A path precision factors the exact joint ML into independent edge assignments."""
    k = len(r)
    omega = np.linalg.inv(r)
    assert np.max(np.abs(np.triu(omega, 2))) < 1e-12
    assert np.all(omega.diagonal(1) < 0)
    chol = np.linalg.cholesky(r)
    identity = np.arange(n)
    wins = Counter()
    for _ in range(trials):
        x = rng.normal(size=(n, max(ds), k)) @ chol.T
        for d in ds:
            for a in range(k - 1):
                pair = x[:, :d, a] @ x[:, :d, a + 1].T
                _, assignment = linear_sum_assignment(pair, maximize=True)
                if not np.array_equal(identity, assignment):
                    break
            else:
                wins[d] += 1
    return [wins[d] for d in ds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-support", type=int, default=4)
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--seed", type=int, default=261132892)
    args = parser.parse_args()
    if args.max_support < 2 or args.trials < 1:
        parser.error("need --max-support >= 2 and --trials >= 1")
    for k in (3, 4, 5):
        r_rng, graph_rng, ml_rng = map(np.random.default_rng, np.random.SeedSequence([args.seed, k]).spawn(3))
        designs = [(name, r, *rates(r)) for name, r in models(k, r_rng)]
        print(f"K={k}", flush=True)
        for name, r, half_cut, rate in designs:
            print(f"  {name:14s} I*/2={half_cut:.9f} L={rate:.9f}")
            if name == "negative" and k == 3:
                assert abs(rate - 0.0983531882215) < 1e-10
            if name == "negative" and k == 4:
                assert 9.8511971 < 1 / rate < 9.8512024
            # A fully split cyclic alternative, including K=5 on five rows.
            sig = (np.arange(k)[:, None] + np.arange(k)) % k
            c, b, _ = exponents(r, sig[None, :, :])
            assert c[0] >= b[0] - 1e-9 and b[0] / k >= half_cut - 1e-9
        census(k, designs, args.max_support)
        print(f"  weighted witnesses checked: {check_witness(k, graph_rng)}")
        for name, r, _, rate in designs:
            if name not in {"negative", "random-1"}:
                continue
            n = {3: 6, 4: 4, 5: 4}[k]
            ds = sorted({max(1, round(scale * math.log(n) / rate)) for scale in (0.6, 1, 1.4)})
            recovered = exact_ml(r, n, ds, ml_rng, args.trials)
            print(f"  exact ML {name} n={n}, d={ds}, successes={recovered}/{args.trials}", flush=True)
        name, r, _, rate = designs[0]
        assert name == "chain"
        n = 128
        ds = [round(scale * math.log(n) / rate) for scale in (0.6, 1, 1.4)]
        recovered = chain_ml(r, n, ds, ml_rng, args.trials)
        print(f"  exact ML chain n={n}, d={ds}, successes={recovered}/{args.trials}")
    print("All assertions passed.")


if __name__ == "__main__":
    main()
