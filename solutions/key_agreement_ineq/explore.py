"""Finite-alphabet checks for the two conjectures of Abin and Gohari.

Run with .venv/bin/python -B solutions/key_agreement_ineq/explore.py
from the project root. Entropies are in bits.
"""

from itertools import combinations, product
from math import isclose, log2

import numpy as np
from scipy.special import xlogy


def entropy(masses):
    p = np.asarray(masses, dtype=float).ravel()
    return -xlogy(p, p).sum() / np.log(2)


def grouped_entropy(p, *variables):
    labels = np.column_stack(variables)
    _, idx = np.unique(labels, axis=0, return_inverse=True)
    return entropy(np.bincount(idx.ravel(), weights=p))


def information(p, a, b, *given):
    h = lambda *labels: grouped_entropy(p, *labels) if labels else 0.
    return h(a, *given) + h(b, *given) - h(a, b, *given) - h(*given)


def binary_laws(n):
    for i, j, k in combinations(range(n + 3), 3):
        yield np.array([i, j - i - 1, k - j - 1, n + 2 - k]) / n


X, Y = np.indices((2, 2)).reshape(2, 4)
Z = X ^ Y


def lift(p):
    """Represent p by the four row- or column-supported product laws."""
    s = p[0] + p[3]
    t = p[1] + p[2]
    u = p[0] / s if s else 0.
    v = p[1] / t if t else 0.
    weights = [u*v, u*(1-v), (1-u)*v, (1-u)*(1-v)]
    components = [[s,t,0,0], [s,0,t,0], [0,t,0,s], [0,0,t,s]]
    return np.asarray(weights)[:, None] * components


def conditional_mi_2x2(joint):
    """I(X;Y|J) for joint[x,y,j]."""
    return (entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1))
            - entropy(joint) - entropy(joint.sum(axis=(0, 1))))


def check_xor():
    rng = np.random.default_rng(250200294)
    # Every binary degradation on a 9-by-9 grid, including the constant ones.
    channels = [np.array([[1-a, a], [1-b, b]])
                for a, b in product(np.linspace(0, 1, 9), repeat=2)]
    rows = np.tile(X, 4)
    cols = np.tile(Y, 4)
    parity = rows ^ cols
    label = np.repeat(np.arange(4), 4)
    count = positive = 0
    for p in binary_laws(12):
        count += 1
        base = information(p, X, Y)
        joint = lift(p)
        assert np.allclose(joint.sum(axis=0), p)
        flat = joint.ravel()
        assert abs(information(flat, rows, cols, label)) < 1e-12
        assert abs(information(flat, label, parity)) < 1e-12

        # Both unprimed terminals see T. Check all C1 terms with equal Eves.
        loss = (information(flat, label, 2*cols+parity, rows)
                + information(flat, label, 2*rows+parity, cols)
                + information(flat, label, label, rows, cols, parity))
        common_capacity = grouped_entropy(flat, label, parity) - grouped_entropy(flat, parity)
        assert abs(common_capacity - loss - base) < 1e-12

        intrinsic_grid = min(conditional_mi_2x2((p[:, None]*k[Z]).reshape(2,2,-1))
                             for k in channels)
        assert abs(intrinsic_grid - base) < 1e-12
        for _ in range(5):
            channel = rng.dirichlet([.6]*3, size=4)
            aux = p[:, None] * channel
            q = aux.ravel()
            bound = (conditional_mi_2x2(aux.reshape(2, 2, 3))
                     + information(q, np.repeat(np.arange(4),3),
                                   np.tile(np.arange(3),4), np.repeat(Z,3)))
            assert bound >= base - 1e-12
            # A single feasible (U,V) for EVERY Delta: U=T, V constant.
            tj = joint @ channel
            i_tj = entropy(tj.sum(axis=0)) + entropy(tj.sum(axis=1)) - entropy(tj)
            assert conditional_mi_2x2(aux.reshape(2, 2, 3)) + i_tj >= base - 1e-12

        a, b, c, d = p
        e, o = a*d, b*c
        if e+o:
            rate = (e+o) * (1-entropy([e/(e+o), o/(e+o)]))
            assert rate <= base + 1e-12
            assert (rate > 1e-12) == (abs(e-o) > 1e-12)
            positive += rate > 1e-12

    print(f"{count} XOR laws, {count*len(channels)} degradation checks, "
          f"{count*5} auxiliary-channel checks, {count} lift identities")
    print(f"{positive} dependent laws with a strictly positive two-sample rate")


def check_exact_c1():
    """C1 on joint laws with analytically known capacities and Eve penalty."""
    rng = np.random.default_rng(141)
    x, y, xp, yp, e = np.indices((2,)*5).reshape(5, -1)
    for _ in range(500):
        p = rng.dirichlet([.5]*32)
        # Eve is absent from the unprimed source; primed terminals BOTH see e.
        # Z'=e is deterministic given either primed terminal.
        ap, bp = 2*xp + e, 2*yp + e
        base = information(p, x, y)
        primed = information(p, xp, yp, e)
        penalty = information(p, 2*x+y, e)  # constant Z
        penalty += (information(p, bp, x, ap) + information(p, ap, y, bp)
                    + information(p, x, y, ap, bp, e))
        assert base - primed <= penalty + 1e-12
    print("500 six-variable C1 checks with known operational rates")


def check_triangle():
    p = [1/3, 1/3, 1/3, 0.]
    base = information(p, X, Y)
    assert isclose(base, log2(3)-4/3)
    # On the public outcome (1,1) the accepted first bits are secret complements.
    outcomes = np.zeros((2, 2))
    retained = np.zeros((2, 2))
    for i, j in product(range(4), repeat=2):
        u, v = X[i]^X[j], Y[i]^Y[j]
        outcomes[u,v] += p[i]*p[j]
        if u == v == 1:
            retained[X[i],Y[i]] += p[i]*p[j]
    assert np.allclose(retained, [[0,1/9],[1/9,0]])
    assert isclose(outcomes[0,0], 1/3)
    rate = outcomes[1,1] / (2 - outcomes[0,0])
    assert isclose(rate, 2/15)
    # Pair distinct Alice symbols adaptively. Bob succeeds precisely when
    # the X=0 position carries Y=1 and the X=1 position carries Y=0.
    adaptive = min(sum(p[:2]), sum(p[2:])) * p[1]/sum(p[:2]) * p[2]/sum(p[2:])
    assert isclose(adaptive, 1/6) and adaptive > rate
    # Enumerate the joint laws of the public and secret outputs for each branch.
    trit_branch = np.zeros((8, 3))
    bit_branch = np.zeros((8, 8, 2))
    for pointer in range(3):
        rest = [j for j in range(3) if j != pointer]
        for bits in product(range(2), repeat=2):
            bob_mask = sum(bit << pos for pos, bit in zip(rest, bits))
            eve_mask = bob_mask ^ (1 << pointer)
            mass = 1/12
            if sum(bits) == 2:
                trit_branch[eve_mask, pointer] += mass
            if sum(bits) == 1:
                for j in rest:
                    if bob_mask & (1 << j):
                        pair = (1 << j) | (1 << pointer)
                        bit_branch[eve_mask, pair, int(pointer > j)] += mass/2
    assert np.allclose(trit_branch[7], [1/12]*3)
    assert isclose(trit_branch.sum(), 1/4)
    assert np.allclose(bit_branch[:,:,0], bit_branch[:,:,1])
    assert isclose(bit_branch.sum(), 1/4)
    triple_rate = (trit_branch.sum()*log2(3) + bit_branch.sum()) / 3
    assert adaptive < triple_rate < base
    print(f"Triangle source: one-way 0 in both directions, "
          f"{triple_rate:.12f} <= interactive rate <= {base:.12f}")


if __name__ == "__main__":
    check_xor()
    check_exact_c1()
    check_triangle()
    print("All assertions passed.")
