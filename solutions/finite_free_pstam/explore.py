"""Exact polynomial identities and independent floating-point p-Stam checks."""

from math import factorial
import numpy as np
from numpy.polynomial.hermite_e import hermeroots
import sympy as sp
from scipy.linalg import eigh_tridiagonal


def convolution(f, g):
    """Descending monic coefficient lists, signs included."""
    n = len(f) - 1
    fac = [factorial(i) for i in range(n + 1)]
    return [sum(f[i] * g[k-i] * fac[n-i] * fac[n-k+i] for i in range(k+1))
            / (fac[n] * fac[n-k]) for k in range(n+1)]


def exact_spectrum(n):
    x = sp.Symbol("x")
    H = [sp.Poly(sp.hermite_prob(i, x), x, domain=sp.QQ) for i in range(n+1)]
    # The root tangent of a polynomial perturbation P is -P(h)/H_n'(h).
    # Its Gram matrix is the field trace (sum of evaluations at all nodes).
    inv = H[n].diff().invert(H[n])
    basis = [(v*inv).rem(H[n]) for v in H[:-1]]
    for j, u in enumerate(basis):
        for i in range(j+1):
            prod = (basis[i]*u).rem(H[n])
            # Sum of residues of prod H_n'/H_n at the roots.
            trace = (prod * H[n].diff()).rem(H[n]).nth(n-1)
            expected = sp.Rational(factorial(j), factorial(n)) if i == j else 0
            assert trace == expected
    # Subtract the baseline so the numerical convolution formula is tested
    # on genuine monic inputs even for lower-degree tangent directions.
    baseline = convolution(H[n].all_coeffs(), H[n].all_coeffs())
    for j in range(n):
        perturbed = convolution((H[n] + H[j]).all_coeffs(), H[n].all_coeffs())
        diff = sp.Poly.from_list([a-b for a, b in zip(perturbed, baseline)], x)
        assert all(diff.nth(k) == H[j].nth(k) * 2**((j-k)//2)
                   for k in range(j+1))
        assert diff.degree() == j


def score(r):
    gaps = r[:, None] - r[None, :]
    np.fill_diagonal(gaps, np.inf)
    return (1/gaps).sum(axis=1)


def output_roots(a, b):
    roots = np.roots(convolution(np.poly(a), np.poly(b)))
    assert np.max(abs(roots.imag)) < 1e-9
    return np.sort(roots.real)


def log_length(r, p):
    return -np.log(np.sum(abs(score(r))**p)) / (p-1)


def log_ratio(a, b, c, p):
    return log_length(c, p) - np.logaddexp(log_length(a, p), log_length(b, p))


def numerical_checks():
    rng = np.random.default_rng(11922)
    p_values = (1.05, 1.25, 1.5, 1.8, 1.99, 2., 2.1, 3., 6.)
    total = negatives = 0
    for n in range(3, 11):
        h = hermeroots([0]*n+[1])
        c = output_roots(h, h)
        assert np.allclose(c, np.sqrt(2)*h)
        for p in p_values:
            ratio = log_ratio(h, h, c, p)
            assert abs(ratio - (p/(2*(p-1))-1)*np.log(2)) < 1e-9
        for _ in range(64):
            def draw():
                r = np.cumsum(np.exp(rng.uniform(-1.6, 1.6, n)))
                return (r-r.mean())/r.std() * np.exp(rng.uniform(-1, 1))
            a, b = draw(), draw()
            c = output_roots(a, b)
            for p in p_values:
                ratio = log_ratio(a, b, c, p)
                if p <= 2:
                    assert ratio >= -1e-8, (n, p, ratio)
                else:
                    negatives += ratio < 0
                total += 1
    print(f"{total} random pair/exponent checks; {negatives} negative above 2")
    print("72 Hermite sign/formula checks in degrees 3 through 10")


def partial_jacobian(a, b, c):
    n = len(a)
    gprime = np.polyder(np.poly(b)) / n
    derivative = np.polyder(np.poly(c))
    return np.column_stack([
        np.polyval(convolution(np.poly(np.delete(a, j)), gprime), c)
        / np.polyval(derivative, c) for j in range(n)
    ])


def jacobian_checks():
    for n in range(3, 13):
        h = hermeroots([0]*n+[1])
        E = partial_jacobian(h, h, output_roots(h, h))
        assert np.allclose(E, E.T, atol=1e-9)
        singular = np.linalg.svd(E, compute_uv=False)
        assert np.allclose(singular, 2.**(-np.arange(n)/2), atol=1e-9)
        a = np.linspace(-1, 1, n)
        b = np.expm1(np.linspace(-0.7, 1.3, n))
        c = output_roots(a, b)
        A, B = partial_jacobian(a, b, c), partial_jacobian(b, a, c)
        assert min(A.min(), B.min()) > 0
        assert np.allclose(A @ score(a), score(c))
        assert np.allclose(B @ score(b), score(c))
        AP = (A-A[:, ::-1])/2
        eta = np.linalg.norm(AP, 1)
        assert 0 < eta < 1
        assert np.linalg.norm(np.column_stack([AP, B]), 2) < 1+1e-8
        for p in (1.1, 1.5, 1.9):
            lower = np.logaddexp(log_length(a, p)-(2-p)/(p-1)*np.log(eta),
                                 log_length(b, p))
            assert log_length(c, p) >= lower-1e-8
    print("10 direct Jacobian SVDs and 30 improved symmetric-input bounds")


def laguerre_checks():
    # (1-D)^m x^n has monic Laguerre parameters n,m-n. The convolution
    # adds m, giving a skew family even when m is just above n-1.
    checks = 0
    for n in (4, 16, 64, 256, 1024):
        def score_magnitudes(m):
            if m == n-1:
                return np.r_[n-1., np.ones(n-1)]/2
            j = np.arange(n)
            roots = eigh_tridiagonal(2*j+m-n+1, np.sqrt(j[1:]*(m-n+j[1:])),
                                     eigvals_only=True)
            return abs(1-(m-n+1)/roots)/2
        parameters = (n-1, n-.99, n, n*1.01, n*1.5, n*2, n*5, n*20, n*200)
        sums = {a+b for a in parameters for b in parameters}
        scores = {m: score_magnitudes(m) for m in set(parameters) | sums}
        for p in (1.05, 1.5, 1.95):
            lengths = {m: -np.log(np.sum(s**p))/(p-1) for m, s in scores.items()}
            for i, a in enumerate(parameters):
                for b in parameters[i:]:
                    assert lengths[a+b] > np.logaddexp(lengths[a], lengths[b])
                    checks += 1
    print(f"{checks} subcritical Laguerre checks, through degree 1024")


def cubic_checks():
    a = np.linspace(-1, 1, 1001)
    angles = np.arccos(a)[:, None]/3 + np.arange(3)*2*np.pi/3
    for p in (1.01, 1.25, 1.5, 1.9, 1.99, 2.):
        U = (1-a*a) / np.sum(abs(np.sin(2*angles))**p, axis=1)**(2/p)
        assert np.max(np.diff(U, 2)) < 0
        if p >= 1.9:
            # Separable shape grid with an independent evaluation at output.
            q = p/(p-1)
            grid = np.arange(0, len(a), 10)
            sh = (a[grid, None]+a[grid])/(2*np.sqrt(2))
            th = np.arccos(sh)[:, :, None]/3 + np.arange(3)*2*np.pi/3
            out = (1-sh*sh)/np.sum(abs(np.sin(2*th))**p, axis=2)**(2/p)
            F = U[grid]**(q/2)
            deficit = (2*out)**(q/2) - F[:, None] - F
            assert np.unravel_index(np.argmin(deficit), deficit.shape) == (50, 50)
    print("6 cubic curvature grids; 3 fixed-variance deficit grids")


if __name__ == "__main__":
    for degree in range(3, 13):
        exact_spectrum(degree)
    print("Exact Hermite tangent actions and Euclidean Gram matrices: degrees 3 through 12")
    jacobian_checks()
    numerical_checks()
    laguerre_checks()
    cubic_checks()
    print("All assertions passed")
