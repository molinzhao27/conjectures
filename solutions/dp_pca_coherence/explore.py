"""Reproduce the adjacency and normalized-adjacency checks in solution.md."""
import os

# Avoid oversubscribing small dense matrix products on common BLAS backends.
for variable in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.sparse.linalg import eigsh


P = 1 / 5
RANK = 8


def template():
    identity = np.eye(4)
    return np.block([[np.ones((4, 4)) - identity, identity],
                     [identity, np.zeros((4, 4))]])


def normalized(matrix):
    degrees = matrix.sum(axis=1)
    scale = np.zeros_like(degrees)
    live = degrees > 0
    scale[live] = 1 / np.sqrt(degrees[live])
    return scale[:, None] * matrix * scale


def union(graph, probability, rng):
    n = len(graph)
    sample = np.triu(rng.random((n, n)) < probability, k=1)
    adjacency = np.maximum(graph, sample | sample.T)
    assert np.array_equal(adjacency, adjacency.T)
    assert not adjacency.diagonal().any()
    return adjacency


def leading(matrix, count, iterative=False):
    if iterative:
        values, vectors = eigsh(matrix, k=count, which="LM", tol=1e-10,
                                v0=np.random.default_rng(71).normal(size=len(matrix)))
    else:
        values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-np.abs(values))[:count]
    return values[order], vectors[:, order]


def coherence(basis):
    n, rank = basis.shape
    return n / rank * (basis ** 2).sum(axis=1).max()


def controls(rng):
    n = 96
    graphs = {}
    for name, blocks in (("mismatched cliques", [4, 18]), ("small clique", [6])):
        graph = np.zeros((n, n))
        offset = 0
        for size in blocks:
            graph[offset:offset + size, offset:offset + size] = 1 - np.eye(size)
            offset += size
        graphs[name] = (graph, offset, n / offset)
    for name, a, b in (("star", 1, 95), ("thin biclique", 3, 67),
                       ("balanced biclique", 48, 48)):
        graph = np.zeros((n, n))
        graph[:a, a:a + b] = 1
        graph[a:a + b, :a] = 1
        graphs[name] = (graph, 2, n / (2 * min(a, b)))

    print("control, p, matrix, input coherence, union coherence")
    for name, (graph, rank, expected) in graphs.items():
        for p in (0.08, P, 0.5):
            perturbed = union(graph, p, rng)
            for label, func in (("raw", np.asarray), ("normalized", normalized)):
                original = func(graph)
                assert np.linalg.matrix_rank(original) == rank
                _, basis = leading(original, rank)
                assert np.isclose(coherence(basis), expected)
                _, basis = leading(func(perturbed), rank)
                actual = coherence(basis)
                assert 1 - 1e-10 <= actual <= n / rank + 1e-10
                print(f"{name}, {p:g}, {label}, {expected:.4f}, {actual:.4f}")


def population(t, s, normalize):
    # In an orthonormal basis of nine block indicators. Units are A/t.
    base = np.zeros((9, 9))
    base[:8, :8] = template()
    ratios = np.array([1.0] * 8 + [s / t])
    edge = (1 - P) * base + P
    weights = np.sqrt(ratios)
    if normalize:
        weights /= np.sqrt(edge @ ratios)
    return weights[:, None] * edge * weights


def quotient_checks():
    print("t, matrix, positive outlier mass on isolates, scaled cutoff gap")
    for t in (16, 2**12, 2**24, 2**40):
        s = int(t ** 0.75)
        for norm in (False, True):
            quotient = population(t, s, norm)
            values, vectors = leading(quotient, 9)
            mass = vectors[-1, 7] ** 2
            gap = (abs(values[7]) - abs(values[8])) * t / s
            assert values[7] > 0 > values[8] and gap > 0.01
            assert 0.45 < mass < 0.75
            if t == 2**40:
                assert abs(mass - 0.5) < 0.01
                scales = np.array([1., 1., 1.])
                if norm:
                    scales /= np.sqrt([4.8, 2.4, 1.6])
                c, l, i = scales
                ratio = (2*c*c + l*l) / (4*c*c + l*l)
                predicted_gap = P * i*i * (1 - ratio**2)
                assert np.isclose(gap, predicted_gap, rtol=0.01)
            print(f"{t}, {'normalized' if norm else 'raw'}, {mass:.6f}, {gap:.6f}")


def sampled_witnesses(rng):
    print("t, n, matrix, input coherence, union coherence, eighth vector mass on isolates")
    for t in (32, 128, 384):
        s = int(t ** 0.75)
        n = RANK * t + s
        graph = np.zeros((n, n))
        graph[:8*t, :8*t] = np.kron(template(), np.ones((t, t)))
        perturbed = union(graph, P, rng)
        # Independently check the full n-dimensional construction against its quotient.
        if t == 32:
            cells = np.repeat(np.arange(9), [t] * 8 + [s])
            basis = np.eye(9)[cells] / np.sqrt([t] * 8 + [s])
            looped = (1 - P) * graph + P
            for norm in (False, True):
                full = normalized(looped) if norm else looped / t
                reduced = population(t, s, norm)
                assert np.allclose(full, basis @ reduced @ basis.T)
                source = normalized(graph) if norm else graph
                values, vectors = leading(source, 9)
                assert abs(values[7]) > 0.1 and abs(values[8]) < 1e-10
                assert np.isclose(coherence(vectors[:, :8]), n / (8*t))
        for name, matrix in (("raw", perturbed), ("normalized", normalized(perturbed))):
            values, vectors = leading(matrix, 9, iterative=True)
            assert np.linalg.norm(matrix @ vectors - vectors * values) < 1e-8 * abs(values[0])
            assert abs(values[7]) > abs(values[8])
            mass = np.sum(vectors[-s:, 7] ** 2)
            score = coherence(vectors[:, :8])
            assert score >= n / (RANK * s) * mass - 1e-10
            if t == 384:
                assert mass > 0.4 and score > 3 * n / (RANK * t)
            print(f"{t}, {n}, {name}, {n/(8*t):.4f}, {score:.4f}, {mass:.4f}")


if __name__ == "__main__":
    rng = np.random.default_rng(251026679)
    controls(rng)
    quotient_checks()
    sampled_witnesses(rng)
    print("All assertions passed.")
