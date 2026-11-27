"""Finite strict-argmax checks. Run with the project venv from the repository root."""

from dataclasses import dataclass

import numpy as np
import scipy
from scipy.optimize import linprog
import sympy as sp


def basis(family, i, j):
    """Pair features. Flatten absolute bilinear kernels in row-major order."""
    if family == "ALiBi":
        return np.array([float(j - i)])  # recency magnitude is nonnegative
    if family == "RoPE":
        return np.array([np.cos(i - j), np.sin(i - j)])  # one rotary plane
    if family == "sinusoidal":
        def phi(k):
            return [np.cos(k), np.sin(k)]
    else:
        degree = int(family.removeprefix("monomial"))
        def phi(k):
            return (k / 10.0) ** np.arange(degree + 1)
    return np.outer(phi(i), phi(j)).ravel()


def targets(task, n):
    if task == "copy":
        offset = (n + 1) // 2
        return [(i, i - offset) for i in range(offset, n)]
    if task == "prev":
        return [(i, i - 1) for i in range(1, n)]
    if task == "first":
        return [(i, 0) for i in range(n)]
    if task == "self":
        return [(i, i) for i in range(n)]
    if task == "changed prefix":
        return [(i, 0 if i == 1 and n > 3 else i) for i in range(n)]
    raise ValueError(task)


def comparisons(family, rows):
    dim = len(basis(family, 0, 0))
    gaps = [basis(family, i, t) - basis(family, i, j)
            for i, t in rows for j in range(i + 1) if j != t]
    return np.asarray(gaps).reshape(-1, dim)


def certified_feasible(a, recency):
    """Check a unit-gap witness or obtain and check an independent dual witness."""
    count, dim = a.shape
    if count == 0:
        return True
    bounds = (0, None) if recency else (None, None)
    result = linprog(np.zeros(dim), A_ub=-a, b_ub=-np.ones(count),
                     bounds=bounds, method="highs")
    tol = 1e-7
    if result.success:
        assert np.min(a @ result.x) >= 1 - tol
        assert not recency or np.min(result.x) >= -tol
        return True
    if result.status != 2:
        raise RuntimeError(result.message)

    # For free w, require A.T @ y = 0. For w >= 0, A.T @ y <= 0 suffices.
    total = np.ones((1, count))
    eq = total if recency else np.vstack([total, a.T])
    rhs = [1] if recency else np.r_[1, np.zeros(dim)]
    dual = linprog(np.zeros(count), A_eq=eq, b_eq=rhs,
                   A_ub=a.T if recency else None, b_ub=np.zeros(dim) if recency else None,
                   bounds=(0, None), method="highs")
    assert dual.success, dual.message
    y = dual.x
    assert np.min(y) >= -tol and abs(y.sum() - 1) <= tol
    balance = a.T @ y
    assert (balance.max() if recency else np.abs(balance).max()) <= tol
    return False


@dataclass(frozen=True)
class Case:
    family: str
    task: str
    old: int
    long: int
    expected: str  # old, new, old + new, full two-length system


CASES = [
    Case("RoPE", "prev", 4, 10, "YYYY"),
    Case("RoPE", "copy", 4, 6, "YYNN"),
    Case("sinusoidal", "prev", 4, 10, "YYYY"),
    Case("sinusoidal", "copy", 4, 6, "YYYN"),
    Case("ALiBi", "self", 4, 10, "YYYY"),
    Case("ALiBi", "first", 4, 10, "NNNN"),
    Case("ALiBi", "prev", 1, 4, "YNNN"),
    Case("monomial1", "first", 4, 10, "YYYY"),
    Case("monomial1", "prev", 2, 5, "YNNN"),
    Case("monomial2", "prev", 4, 10, "YYYY"),
    Case("monomial4", "copy", 5, 10, "YYYY"),
    Case("monomial1", "changed prefix", 3, 5, "YYYN"),
]


def check_cases():
    print("family      task             lengths   old/new/old+new/full")
    for case in CASES:
        old = targets(case.task, case.old)
        long = targets(case.task, case.long)
        new = [(i, t) for i, t in long if i >= case.old]
        flags = []
        for rows in (old, new, old + new, old + long):
            ok = certified_feasible(comparisons(case.family, rows), case.family == "ALiBi")
            flags.append("Y" if ok else "N")
        signature = "".join(flags)
        print(f"{case.family:11} {case.task:15} {case.old:2} -> {case.long:2}   {signature}")
        assert signature == case.expected, (case, signature)


def exact_copy_example():
    phi = sp.Matrix([[i**k for k in range(5)] for i in range(10)])
    # For the exact old logit interpolant, extend the target by zeros on unscored rows.
    old = sp.zeros(5)
    for i, t in targets("copy", 5):
        old[i, t] = 1
    inv = phi[:5, :].inv()
    interpolant = phi * inv * old * inv.T * phi.T
    assert interpolant[:5, :5] == old

    def score(i, j):
        d = i - j
        return d - 2 * (d - 3)**2 * (d - 5)**2

    def gaps(rows, kernel):
        return [min(kernel(i, t) - kernel(i, j) for j in range(i + 1) if j != t)
                for i, t in rows]

    old_gaps = gaps(targets("copy", 5), score)
    new_gaps = gaps(targets("copy", 10), score)
    selected = gaps(targets("copy", 10), lambda i, j: interpolant[i, j])
    assert (min(old_gaps), min(new_gaps)) == (1, 2)
    assert selected == [-15, 15, -35, -70, -126]
    print("Copy rescue margins (old, new):", min(old_gaps), min(new_gaps))
    print("Copy interpolant new-row margins:", selected)

    # A fifth forward difference annihilates all possible degree-four score rows.
    diff = sp.Matrix([(-1)**(5-k) * sp.binomial(5, k) for k in range(6)])
    assert diff.T * phi[:6, :] == sp.zeros(1, 5)
    assert diff[4] == -5  # row 9 demands key 4
    print("Fifth difference of the last target row:", diff[4])


if __name__ == "__main__":
    print(f"NumPy {np.__version__}, SciPy {scipy.__version__}, SymPy {sp.__version__}")
    check_cases()
    exact_copy_example()
    print("All checks passed.")
