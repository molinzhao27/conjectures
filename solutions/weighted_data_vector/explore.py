"""Global optimization of normalized fits. All scratch binaries are temporary."""

import argparse
import ctypes
import subprocess
import tempfile
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

TMP = tempfile.TemporaryDirectory()
LIB = Path(TMP.name) / "search.so"
subprocess.run(
    [
        "cc",
        "-O3",
        "-shared",
        "-fPIC",
        str(Path(__file__).with_name("search.c")),
        "-o",
        str(LIB),
    ],
    check=True,
)
native = ctypes.CDLL(str(LIB))
_cost = native.cost
pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_cost.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer, ctypes.c_int]
_cost.restype = ctypes.c_double
native.unmodified.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer, pointer]
native.unmodified.restype = ctypes.c_double


def fit(x, y, k, passes=2):
    if not 2 <= len(x) <= 7 or k not in (3, 4) or x.shape != y.shape or x.shape[1] != 2:
        raise ValueError(
            "Expected matching two-column arrays with 2 to 7 rows and budget 3 or 4."
        )
    q, _ = np.linalg.qr(x)
    residual = y - q @ (q.T @ y)
    return _cost(
        len(x),
        k,
        np.ascontiguousarray(q),
        np.ascontiguousarray(residual / np.linalg.norm(residual)),
        passes,
    )


def search(sizes, k, iterations, seed, passes, refine=False, original=False):
    n, g = sum(sizes), len(sizes)
    groups = np.repeat(np.arange(g), sizes)

    def unpack(v):
        angles = np.r_[0.0, v[: g - 1]]
        weights = np.exp(v[g - 1 : g + n - 1])
        x = weights[:, None] * np.column_stack(
            (np.cos(angles[groups]), np.sin(angles[groups]))
        )
        y = v[g + n - 1 :].reshape(n, 2)
        return x, y

    def original_values(x, y):
        q = np.linalg.qr(x)[0]
        r = y - q @ (q.T @ y)
        costs = np.empty(sum(comb(n, i) for i in range(2, k + 1)))
        native.unmodified(
            n,
            k,
            np.ascontiguousarray(q),
            np.ascontiguousarray(r / np.linalg.norm(r)),
            costs,
        )
        return costs

    def objective(v):
        if original:
            return -min(original_values(*unpack(v)))
        return -fit(*unpack(v), k, passes)

    bounds = (
        [(0.001, np.pi - 0.001)] * (g - 1) + [(-1.3, 1.3)] * n + [(-1, 1)] * (2 * n)
    )
    res = differential_evolution(
        objective, bounds, seed=seed, popsize=3, maxiter=iterations, polish=False, tol=0
    )
    v = res.x
    if refine:
        faces = native.faces
        faces.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
            ctypes.c_int,
            pointer,
        ]
        faces.restype = None

        def constraints(w):
            x, y = unpack(w[:-1])
            if original:
                return np.minimum(original_values(x, y), 2) - w[-1]
            q = np.linalg.qr(x)[0]
            r = y - q @ (q.T @ y)
            vals = np.empty(comb(n, k))
            faces(
                n,
                k,
                np.ascontiguousarray(q),
                np.ascontiguousarray(r / np.linalg.norm(r)),
                max(3, passes),
                vals,
            )
            return np.minimum(vals, 2) - w[-1]

        sol = minimize(
            lambda w: -w[-1],
            np.r_[v, -res.fun],
            method="SLSQP",
            bounds=bounds + [(0, 1)],
            constraints=[{"type": "ineq", "fun": constraints}],
            options={"maxiter": 180, "ftol": 1e-10},
        )
        v = sol.x[:-1]
    x, y = unpack(v)
    final = min(original_values(x, y)) if original else fit(x, y, k, 30)
    print(sizes, k, seed, res.nfev, final, flush=True)
    if original and final > {3: 5 / 8, 4: 1 / 4}[k] + 1e-5:
        print(x.tolist(), y.tolist(), flush=True)
    return x, y


def cyclotomic_checks():
    """KKT certificates for squared distance to every dihedral triangle type."""
    import itertools

    import sympy as sp

    x = sp.Symbol("x")
    root = {
        5: (x * x + x - 1, sp.Rational(61803, 100000)),
        6: (x - 1, sp.Integer(1)),
        7: (x**3 + x * x - 2 * x - 1, sp.Rational(124697, 100000)),
    }
    representatives = {
        5: [(0, 1, 2), (0, 1, 3)],
        6: [(0, 1, 2), (0, 1, 3), (0, 2, 4)],
        7: [(0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 4)],
    }

    def check_polygon(n, poly, lo):
        h = 2 if n == 5 else 3
        hi = lo if n == 6 else lo + sp.Rational(1, 100000)
        if n != 6:
            assert poly.subs(x, lo) < 0 < poly.subs(x, hi)

        def red(v):
            num, den = sp.fraction(sp.cancel(v))
            return sp.rem(num * sp.invert(den, poly, x), poly, x).expand()

        def sign(v):
            coeff = sp.Poly(red(v), x).all_coeffs()
            lower = upper = sp.Integer(0)
            for c in coeff:
                vals = [w * t for w in (lower, upper) for t in (lo, hi)]
                lower, upper = min(vals) + c, max(vals) + c
            if lower > 0:
                return 1
            if upper < 0:
                return -1
            assert lower == upper == 0, (v, lower, upper)
            return 0

        cosine = [red(sp.chebyshevt(j, x / 2)) for j in range(n)]

        def edge(i, j):
            return (
                [(i + j + i * l + j * (h - 2 - l)) % n for l in range(h - 1)],
                [(i * l + j * (h - 1 - l)) % n for l in range(h)],
            )

        target = {5: (1 - x) / 2, 6: sp.Rational(1, 2), 7: sp.Rational(1, 4)}[n]
        reached = False
        covered = set()
        expected = {
            5: [1 + x / 2, (1 - x) / 2],
            6: [sp.Rational(1, 2), sp.Rational(1, 2), sp.Integer(1)],
            7: [1 + x / 2, sp.Rational(1, 4), 2 - (x * x + x) / 2, x * x / 2],
        }[n]
        for S, expected_minimum in zip(representatives[n], expected, strict=True):
            covered.update(
                tuple(sorted((s * a + b) % n for a in S))
                for s in (-1, 1)
                for b in range(n)
            )
            edges = [edge(*p) for p in itertools.combinations(S, 2)]
            G = sp.Matrix(
                [
                    [
                        red(
                            sum(
                                cosine[(a - b) % n]
                                for ei, ej in zip(edge_i, edge_j)
                                for a in ei
                                for b in ej
                            )
                        )
                        for edge_j in edges
                    ]
                    for edge_i in edges
                ]
            )
            for m in (1, 2, 3):
                found = False
                for active in itertools.combinations(range(3), m):
                    H = G.extract(active, active)
                    # Bordered system for stationary barycentric coordinates.
                    K = H.row_join(-sp.ones(m, 1)).col_join(
                        sp.ones(1, m).row_join(sp.zeros(1, 1))
                    )
                    det = red(K.det())
                    if det == 0:
                        continue
                    rhs = sp.zeros(m + 1, 1)
                    rhs[m] = 1
                    sol = [red(v / det) for v in K.adjugate() * rhs]
                    c = sol[-1]
                    alpha = sp.zeros(3, 1)
                    for i, w in zip(active, sol):
                        alpha[i] = w
                    if min(sign(w) for w in alpha) < 0:
                        continue
                    if min(sign(v - c) for v in G * alpha) < 0:
                        continue
                    assert red((alpha.T * G * alpha)[0] - c) == 0
                    assert red(c - expected_minimum) == 0
                    assert sign(c - target) >= 0
                    reached |= sign(c - target) == 0
                    print(n, S, "minimum", c)
                    found = True
                    break
                if found:
                    break
            assert found
        assert reached and len(covered) == comb(n, 3)

    for n, (poly, lo) in root.items():
        check_polygon(n, poly, lo)


def dual_checks():
    """Exact mixed-selection certificates, valid for every compatible test mass."""
    import sympy as sp

    X = sp.Matrix([[1, 0], [0, 1], [1, 1], [1, -1], [2, 1], [1, 2], [2, -1]])
    raw = sp.Matrix([[1, -2], [-1, 3], [2, 1], [0, 3], [-1, -1], [2, -2], [-3, 1]])
    Y = raw - X * (X.T * X).inv() * X.T * raw
    assert X.T * Y == sp.zeros(2)
    for supports, nums, D, bound in (
        ([(1, 5, 6), (3, 5, 6)], [93, 7], [[-22, 24], [1, -20]], sp.Rational(1, 10)),
        (
            [(1, 3, 4, 5), (1, 3, 5, 6)],
            [33, 67],
            [[-14, 15], [2, -12]],
            sp.Rational(1, 16),
        ),
    ):
        assert sum(nums) == 100
        second = sp.zeros(2)
        for S, prob in zip(supports, nums):
            x, y = X.extract(S, (0, 1)), Y.extract(S, (0, 1))
            B = (x.T * x).inv() * x.T * y
            second += sp.Rational(prob, 100) * B * B.T
        ratios = []
        for i in range(X.rows):
            xi, yi = X.row(i), Y.row(i)
            ratios.append(
                ((xi * second * xi.T)[0] - (yi * sp.Matrix(D) * xi.T)[0] / 100)
                / yi.dot(yi)
            )
        assert (
            max(ratios)
            == {3: sp.Rational(560659, 7066875), 4: sp.Rational(381447541, 7312728225)}[
                len(supports[0])
            ]
            < bound
        )
        print("fixed-atom dual bound", len(supports[0]), max(ratios))


def original_weight_certificate():
    """Enumerate the second-axis supports of the seven-atom obstruction."""
    from fractions import Fraction as F
    from itertools import combinations

    # The vertical coordinate is stored in units of s = sqrt(56)/9.
    atoms = [(-1, F(1, 100), 5), (-1, -F(1, 100), 5), (F(5, 9), 1, 9), (F(5, 9), -1, 9)]
    vertical_scale = F(56, 81)

    def squared_mean(rows):
        total = sum(mass for _, _, mass in rows)
        a = sum(F(mass, total) * x for x, _, mass in rows)
        b = sum(F(mass, total) * y for _, y, mass in rows)
        return a * a + vertical_scale * b * b

    e = F(20, 21)
    other_energy = sum(F(m, 28) * (a * a + vertical_scale * b * b) for a, b, m in atoms)
    assert squared_mean(atoms) == 0 and other_energy == 1 + F(1, 40500)
    first = {1: e, 2: e / 4, 3: F(0)}
    second = {
        k: min(squared_mean(rows) for rows in combinations(atoms, k))
        for k in range(1, 5)
    }
    for k, target in ((3, F(700205, 1107014)), (4, F(295205, 1107014))):
        best = min(
            a + b for i, a in first.items() for j, b in second.items() if i + j <= k
        )
        assert best / (e + other_energy) == target
        assert target > {3: F(5, 8), 4: F(1, 4)}[k]
    d = squared_mean([(*atoms[0][:2], 1), (*atoms[3][:2], 1)])
    assert (e + d) / (e + other_energy) == F(3320249, 5535070) < F(5, 8)
    assert (e / 4 + d) / (e + other_energy) == F(1295249, 5535070) < F(1, 4)


def identity_checks():
    import sympy as sp

    a = sp.symbols("a")

    def f(t):
        return t / (1 - t) ** 2

    plus, minus = f((1 + a) / 3), f((1 - a) / 3)
    equal_pair_bound = (plus + minus - a * (plus - minus)) / 6
    assert (
        sp.factor(
            equal_pair_bound
            - sp.Rational(1, 4)
            + a * a * (5 * a * a + 4) / (4 * (4 - a * a) ** 2)
        )
        == 0
    )
    s = sp.symbols("s")
    # These polynomial identities give the averaged identities using E[s]=1.
    assert sp.factor((s - 1) ** 2 / (s + 2) - (s - 4 + 9 / (s + 2))) == 0
    assert sp.factor(s * s / (s + 2) - (s - 2 + 4 / (s + 2))) == 0


def sanity_checks():
    # The reference equality system and three nonaxial harmonic systems.
    tri = np.array([[1.0, 0.0], [-0.5, np.sqrt(0.75)], [-0.5, -np.sqrt(0.75)]])
    x = np.repeat(np.eye(2), 3, axis=0)
    y = np.vstack((tri, tri))
    assert abs(fit(x, y, 3) - 5 / 8) < 1e-11
    assert abs(fit(x, y, 4) - 1 / 4) < 1e-11
    for n, h, target in ((5, 2, (3 - np.sqrt(5)) / 4), (6, 3, 0.5), (7, 3, 0.25)):
        theta = np.arange(n) * np.pi / n
        x = np.column_stack((np.cos(theta), np.sin(theta)))
        y = np.column_stack((np.cos((2 * h - 1) * theta), np.sin((2 * h - 1) * theta)))
        assert abs(fit(x, y, 3) - target) < 1e-11
        assert fit(x, y, 4, 30) < 1 / 4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--passes", type=int, default=1)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--sizes", help="Run one search, e.g. 3,2,1")
    p.add_argument("--k", type=int, choices=[3, 4], default=3)
    p.add_argument("--refine", action="store_true")
    p.add_argument(
        "--suite", action="store_true", help="Reproduce all 84 global searches"
    )
    p.add_argument(
        "--original",
        action="store_true",
        help="Restrict selections to their original row multipliers in a single search.",
    )
    args = p.parse_args()
    if args.sizes:
        sizes = list(map(int, args.sizes.split(",")))
        if not (2 <= len(sizes) and 5 <= sum(sizes) <= 7 and min(sizes) > 0):
            p.error("Use at least two direction groups totaling 5 to 7 atoms.")
        search(
            sizes,
            args.k,
            args.iterations,
            args.seed,
            args.passes,
            args.refine,
            args.original,
        )
    elif args.suite:
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from itertools import product

        shapes = (
            "1,1,1,1,1",
            "1,1,1,1,1,1",
            "1,1,1,1,1,1,1",
            "2,2,1",
            "3,1,1",
            "2,2,2",
            "3,2,1",
            "2,2,2,1",
            "3,3,1",
            "3,2,2",
        )

        def run_one(job):
            shape, k, seed = job
            cmd = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--sizes",
                shape,
                "--k",
                str(k),
                "--iterations",
                "100",
                "--seed",
                str(seed),
                "--refine",
            ]
            output = subprocess.check_output(cmd, text=True).strip()
            print(output, flush=True)
            count, cost = output.split()[-2:]
            return sum(map(int, shape.split(","))), k, int(count), float(cost)

        with ThreadPoolExecutor(8) as executor:
            more_shapes = (
                "2,1,1,1",
                "4,1,1",
                "3,1,1,1",
                "2,2,1,1",
                "2,1,1,1,1",
                "5,1,1",
                "4,2,1",
                "4,1,1,1",
                "3,2,1,1",
                "3,1,1,1,1",
                "2,2,1,1,1",
                "2,1,1,1,1,1",
            )
            jobs = [
                *product(shapes, (3, 4), (11, 37, 83)),
                *product(more_shapes, (3, 4), (23,)),
            ]
            runs = list(executor.map(run_one, jobs))
        print("Evolution evaluations", sum(row[2] for row in runs))
        for n, k in product((5, 6, 7), (3, 4)):
            print(
                "Largest witness cost",
                n,
                k,
                max(c for size, b, _, c in runs if size == n and b == k),
            )
    else:
        cyclotomic_checks()
        dual_checks()
        original_weight_certificate()
        identity_checks()
        sanity_checks()
        print("All assertions passed.")


if __name__ == "__main__":
    main()
