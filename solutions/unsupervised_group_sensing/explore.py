"""Exact checks for the group-sensing counterexample and four subspace models.

Run with Python 3.10+ and SymPy. No input files or network are needed.
"""

from collections import Counter
from itertools import permutations, product
from math import ceil
from random import Random

from sympy import Matrix, diag, eye, zeros


def rank(matrix):
    return matrix.to_DM().rank()


def sign_character(x, y):
    return (-1) ** ((x & y).bit_count() % 2)


def check_counterexample():
    # Each half of the six-bit index represents a vector in F_2^3.
    labels = [u for u in range(64) if sign_character(u >> 3, u & 7) == -1]
    table = Matrix([[sign_character(g, u) for u in labels] for g in range(64)])
    n = len(labels)
    assert n == 28 and table.T * table == 64 * eye(n)
    traces = Counter(sum(table.row(g)) for g in range(64))
    assert traces == {28: 1, 4: 28, -4: 35}

    # The first n-1 rows are full spark, with a full-support kernel vector.
    v = Matrix(list(range(1, n)) + [1])
    A = eye(n - 1).row_join(-v[:-1, :])
    assert A * v == zeros(n - 1, 1) and rank(A) == 27
    assert rank(A.col_join(eye(n)[-1, :])) == 28
    for g in range(64):
        T = diag(*table.row(g))
        majority = 1 if sum(table.row(g)) > 0 else -1
        witness_map = T - majority * eye(n)
        y = witness_map * v
        assert sum(bool(entry) for entry in y) == (12 if g else 0)
        assert A * T * v == A * y
        # Jacobian of the consistency residual using these linear witnesses.
        assert A * (T - witness_map) == majority * A
    print(f"Sign group: {n} orthogonal character columns; traces {dict(traces)}.")
    print("All 64 sparse witnesses agree at m=27; completed sensor rank 28.")


def signed_permutations(d, rotations_only):
    group = []
    for order in permutations(range(d)):
        for signs in product((-1, 1), repeat=d):
            matrix = eye(d)[:, list(order)] * diag(*signs)
            if not rotations_only or matrix.det() == 1:
                group.append(matrix)
    return group


def standard_permutations(d):
    # Coordinates on the sum-zero subspace in the basis e_i - e_d.
    basis = eye(d - 1).col_join(-Matrix.ones(1, d - 1))
    return [(eye(d)[:, list(p)] * basis)[:-1, :] for p in permutations(range(d))]


def check_subspace_model(name, group, complex_line, rng):
    d, copies = group[0].rows, 5
    # The finite test matrices span the entire action algebra.
    flat = Matrix.vstack(*(g.reshape(1, d * d) for g in group))
    assert rank(flat) == (2 if complex_line else d * d)
    transforms = [diag(*([g] * copies)) for g in group]
    rows_per_probe = 1 if complex_line else d
    probes = ceil(copies / rows_per_probe)
    B = Matrix(probes + 1, copies * d,
               lambda i, j: rng.randint(-15, 15))
    ranks = []
    for q in range(probes + 2):
        stacked = Matrix.vstack(*(B[:q, :] * T for T in transforms))
        observed = rank(stacked)
        assert observed == d * min(copies, q * rows_per_probe)
        ranks.append(observed)
    print(f"{name}: quotient Jacobian ranks {ranks}; first identifying m={d + probes}.")


def main():
    check_counterexample()
    rng = Random(220312513)
    J = Matrix([[0, -1], [1, 0]])
    check_subspace_model("SO(2)", [eye(2), J], True, rng)
    check_subspace_model("O(2)", signed_permutations(2, False), False, rng)
    check_subspace_model("SO(3)", signed_permutations(3, True), False, rng)
    check_subspace_model("S_4 (standard)", standard_permutations(4), False, rng)


if __name__ == "__main__":
    main()
