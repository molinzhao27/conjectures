"""Exact fiber checks. Run with Python 3 and SymPy from any directory."""

from itertools import combinations, combinations_with_replacement, product
from random import Random

import sympy as sp


def veronese(rows):
    """Square coefficients divided by the multinomial coefficient in each column."""
    return sp.Matrix([[row[i] * row[j] for i, j in
                       combinations_with_replacement(range(len(row)), 2)]
                      for row in rows])


def projective_zeros(equations, coords):
    """Search disjoint charts by setting the first nonzero coordinate to one."""
    points = set()
    for i in range(len(coords)):
        prefix = (0,) * i + (1,)
        substitutions = dict(zip(coords, prefix))
        polys = [p.subs(substitutions) for p in equations]
        variables = coords[i + 1:]
        if any(p.is_number and p != 0 for p in polys):
            continue
        if not variables:
            points.add(prefix)
            continue
        basis = sp.groebner(polys, *variables, domain=sp.QQ, order="lex")
        if basis.contains(sp.Integer(1)):
            continue
        assert basis.is_zero_dimensional
        roots = sp.solve_poly_system(list(basis), *variables, strict=True)
        points.update(prefix + tuple(root) for root in roots)
    return sorted(points)


def square_directions(coefficients, input_width):
    coords = sp.symbols(f"a:{input_width}")
    monomials = veronese([coords])
    equations = [(monomials * normal)[0] for normal in coefficients.nullspace()]
    return projective_zeros(equations, coords)


def integer_invertible(rng, n):
    while True:
        matrix = sp.Matrix(n, n, lambda i, j: rng.randint(-3, 3))
        if matrix.det() != 0:
            return matrix


def shallow_and_deep(rows, rng, *, expand_deep=False):
    original = veronese(rows)
    middle = integer_invertible(rng, 7)
    outer = integer_invertible(rng, 7)
    target = middle * original
    points = square_directions(target, 4)
    assert len(points) == 8 and set(rows) < set(points)
    assert all(sp.sympify(entry).is_Rational for p in points for entry in p)
    extra, = set(points) - set(rows)

    variables = sp.symbols("x:4")

    def evaluate_quartics(hidden, weights):
        linear = sp.Matrix(hidden) * sp.Matrix(variables)
        quadrics = weights * linear.applyfunc(lambda form: form**2)
        quartics = [sp.Poly(q, *variables)**2 for q in quadrics]
        return [sum((c * f for c, f in zip(row, quartics)),
                    sp.Poly(0, *variables)) for row in outer.tolist()]

    # Equality before the second activation certifies all deep extensions.
    # Expand the cube example independently through the final linear layer.
    if expand_deep:
        quartic_target = evaluate_quartics(rows, middle)
    bases = list(combinations(points, 7))
    for hidden in bases:
        basis = veronese(hidden)
        pivots = list(basis.rref()[1])
        assert len(pivots) == 7
        weights = target[:, pivots] * basis[:, pivots].inv()
        assert weights * basis == target
        if expand_deep:
            assert evaluate_quartics(hidden, weights) == quartic_target
    print(f"7 of 8 directions: {len(bases)} exact bases, residual {extra}")


def binary_controls():
    rows = [(1, 2), (1, -1)]
    assert square_directions(veronese(rows), 2) == sorted(rows)

    # The quartic pencil in coordinates u,v has only even monomials.
    # A square root is either A*u**2+C*v**2 or B*u*v.
    for a, b, c, d, mixed_square in [(1, 2, 3, 4, False), (1, 1, 1, -1, True)]:
        pencil = sp.Matrix([[a*a, 2*a*b, b*b], [c*c, 2*c*d, d*d]])
        normal, = pencil.nullspace()
        A, C = sp.symbols("A C")
        eq = (sp.Matrix([[A**2, 2*A*C, C**2]]) * normal)[0]
        assert projective_zeros([eq], (A, C)) == sorted(
            [(1, sp.Rational(b, a)), (1, sp.Rational(d, c))])
        assert (normal[1] == 0) == mixed_square
    print("Binary controls: 2 hidden directions; 2 and 3 square members in the quartic pencils.")


if __name__ == "__main__":
    binary_controls()
    rng = Random(17093)
    cube = [(1,) + p for p in product((-1, 1), repeat=3)]
    shallow_and_deep(cube[:-1], rng, expand_deep=True)
    for _ in range(4):
        # Distinct final coordinates make lexicographic elimination inexpensive.
        rows = [(1, rng.randint(-9, 9), rng.randint(-9, 9), z)
                for z in rng.sample(range(-12, 13), 7)]
        shallow_and_deep(rows, rng)
    print("All exact checks passed.")
