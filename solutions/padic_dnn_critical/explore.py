"""Small-level checks for the proofs in solution.md. Run with Python -B."""

from itertools import product

import numpy as np
import sympy as sp
from scipy.linalg import det
from scipy.optimize import brentq, root


def projection(p, level, j):
    labels = np.arange(p**level) % p**j
    return (labels[:, None] == labels) / (p ** (level - j))


def symbolic_checks():
    r = sp.symbols("r", positive=True)
    a = 1 / (1 - r**2)
    mid = (1 + r) / 2
    assert sp.factor(1 / (1 - mid**2) - a - a*r) == 1 / (r + 3)
    t = sp.symbols("t")
    assert sp.simplify((1-t)*(1/(1-t**2)-a) - (1/(1+t)-a*(1-t))) == 0
    x, y = sp.symbols("x y")
    assert sp.Matrix([[1, -x], [-y, 1]]).det() == 1-x*y


def tree_test(p, level, rng):
    n = p**level
    a = 2.25
    s = np.arccosh(np.sqrt(a))
    beta = 0.999 * (a*np.tanh(s) - s)
    alpha = brentq(lambda z: z-a*np.tanh(z), s, a)
    projections = [projection(p, level, j) for j in range(level + 1)]
    for j, pj in enumerate(projections):
        for k, pk in enumerate(projections):
            np.testing.assert_allclose(pj @ pk, projections[min(j,k)], atol=2e-15)
    weights = np.arange(1, level+1)
    b = sum(w * pj for w, pj in zip(weights, projections)) * (beta / sum(weights))
    matrix = a*np.eye(n) + b
    if n <= 9:
        patterns = np.array(list(product((-1, 0, 1), repeat=n)))
    else:
        patterns = np.unique(np.vstack((
            rng.integers(-1, 2, size=(192,n)), rng.choice([-1,1], size=(64,n)),
            np.ones(n), np.zeros(n), np.r_[-1, np.ones(n-1)],
        )), axis=0)
    stable = 0
    for batch in np.array_split(patterns, (len(patterns)+255)//256):
        h = batch * alpha
        for frac in (0.5, 1.0):
            m = a*np.eye(n) + frac*b
            for _ in range(15):
                u = np.tanh(h)
                residual = h - u @ m.T
                if np.max(np.abs(residual)) < 2e-13:
                    break
                jac = np.eye(n) - m*(1-u**2)[:,None,:]
                h -= np.linalg.solve(jac, residual[:,:,None])[:,:,0]
            assert np.max(np.abs(h - np.tanh(h) @ m.T)) < 2e-12
        assert np.array_equal((h > s).astype(int) - (h < -s), batch)
        d = np.sqrt(1-np.tanh(h)**2)
        eig = np.linalg.eigvalsh(d[:,:,None]*matrix*d[:,None,:] - np.eye(n))
        assert np.min(np.abs(eig)) > 1e-7
        assert np.array_equal((eig > 0).sum(axis=1), (batch == 0).sum(axis=1))
        stable += int(np.sum(eig[:,-1] < 0))
    print(f"tree p={p} l={level}: checked {len(patterns)} patterns, stable {stable}")
    return len(patterns)


def random_levels(p, level, rng):
    """Use one fixed radial variance profile at each of two amplitudes."""
    n = p**level
    diff = np.abs(np.arange(n)[:,None] - np.arange(n))
    # v(0)=1, v(1)=2, v(p^-j)=1 otherwise. The row sum is 2-1/p.
    base = (1 + (diff % p != 0)) / (2-1/p)
    np.testing.assert_allclose(base.sum(axis=1), n)
    for rho in (0.6, 1.8):
        m = rng.normal(size=(n,n)) * np.sqrt(base/n) * rho
        h0 = np.zeros(n)
        found = [h0]
        for initial in rng.normal(scale=1.5, size=(48,n)):
            fit = root(lambda h: h-m@np.tanh(h), initial,
                       jac=lambda h: np.eye(n)-m*(1-np.tanh(h)**2))
            if np.max(np.abs(fit.x-m@np.tanh(fit.x))) > 1e-8:
                continue
            for h in (fit.x, -fit.x):
                if all(np.max(np.abs(h-old)) > 1e-5 for old in found):
                    found.append(h.copy())
        print(f"random p={p} l={level} rho={rho}: {len(found)} roots found")


def shell_checks(rng):
    counts = []
    for n in (32, 64, 128):
        pair_negative = full_negative = agree = 0
        samples = 128
        logscale = np.log(n)-np.log(np.log(n))
        # An old shell at distance 1 and a new shell at distance 2/n.
        # This is a finite illustration of the induction step.
        ii = np.arange(n)
        old_support = (ii[:,None]-ii) % 2 != 0
        row_variance = 0.002 + 1/logscale
        assert row_variance < 1
        for _ in range(samples):
            old = rng.normal(size=(n,n))*old_support*np.sqrt(0.004/n)
            x, y = rng.normal(size=(2,n//2)) / np.sqrt(logscale)
            matrix = old.copy()
            pair = ii[:n//2]
            matrix[pair, pair+n//2] = x
            matrix[pair+n//2, pair] = y
            sign_pair = (-1)**int(np.sum(x*y > 1))
            sign_full = np.sign(det(np.eye(n)-matrix))
            pair_negative += sign_pair < 0
            full_negative += sign_full < 0
            agree += sign_pair == sign_full
            # Independently evaluate the Schur complement on exceptional pairs.
            exc = np.tile(np.maximum(np.abs(x),np.abs(y)) > 0.5, 2)
            keep = ~exc
            f = np.eye(n) - matrix
            bulk = f[np.ix_(keep,keep)]
            eff = (f[np.ix_(exc,exc)] - f[np.ix_(exc,keep)] @
                   np.linalg.solve(bulk, f[np.ix_(keep,exc)]))
            assert np.sign(det(eff))*np.sign(det(bulk)) == sign_full
        counts.append(full_negative)
        print(f"shell n={n}: det<0 pair/full {pair_negative}/{full_negative}, "
              f"sign agreement {agree}/{samples}")
    assert all(k > 0 for k in counts)


if __name__ == "__main__":
    rng = np.random.default_rng(19070)
    symbolic_checks()
    total = 0
    for prime in (2,3):
        for depth in (1,2,3):
            total += tree_test(prime, depth, rng)
            random_levels(prime, depth, rng)
    shell_checks(rng)
    print(f"All assertions passed ({total} tree patterns).")
