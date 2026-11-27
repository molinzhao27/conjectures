"""Exact finite search for Conjecture 12. Run with Python 3.10 or later."""

from fractions import Fraction
from functools import cache
from itertools import combinations, product
from random import Random


def make_graph(vertices, sees):
    """Entry t is a bit mask encoding the visible set S_t."""
    return tuple(
        sum(1 << s for s, u in enumerate(vertices) if u != v and sees(v, u))
        for v in vertices
    )


def circulant(n, steps):
    return make_graph(range(n), lambda t, s: (t - s) % n in steps)


def triangle_power(k):
    def sees(v, u):
        return next((a - b) % 3 for a, b in zip(v, u) if a != b) == 1

    return make_graph(list(product(range(3), repeat=k)), sees)


def wrapped_butterfly(r):
    # Translate all bit strings by XOR, or rotate layer and bit positions together.
    vertices = list(product(range(r), range(2**r)))

    def sees(v, u):
        i, x = u
        j, y = v
        return j == (i + 1) % r and (x ^ y) in (0, 1 << i)

    return make_graph(vertices, sees)


def largest_order(visible):
    @cache
    def solve(candidates):
        best = 0
        remaining = candidates
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            # Make t the last vertex. All previous vertices must be in S_t.
            t = bit.bit_length() - 1
            best = max(best, 1 + solve(candidates & visible[t]))
        return best

    return solve((1 << len(visible)) - 1)


def best_independent_efficiency(visible):
    n = len(visible)
    observers = [
        sum(1 << t for t, row in enumerate(visible) if row & (1 << s))
        for s in range(n)
    ]
    best = Fraction(0)

    def visit(candidates, size, hit):
        nonlocal best
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            t = bit.bit_length() - 1
            new_hit = hit | observers[t]
            assert new_hit, "A singleton with no observers has unbounded ILB"
            best = max(best, Fraction(size + 1, new_hit.bit_count()))
            visit(candidates & ~(observers[t] | visible[t]), size + 1, new_hit)

    visit((1 << n) - 1, 0, 0)
    return best


def squared_values(visible):
    """Orbit formula from solution.md. Caller supplies a vertex-transitive graph."""
    n = len(visible)
    return Fraction(n**2, largest_order(visible)), n**2 * best_independent_efficiency(visible)


def main():
    total = 0
    for n in range(3, 9):
        best = Fraction(0)
        for mask in range(1, 1 << (n - 1)):
            steps = [s for s in range(1, n) if mask & (1 << (s - 1))]
            ub2, ilb2 = squared_values(circulant(n, steps))
            if len(steps) == n - 1:
                assert (ub2, ilb2) == (n, Fraction(n**2, n - 1))
            best = max(best, ub2 / ilb2)
            total += 1
        print(f'n={n} circulants, largest squared ratio {best}')
    print(f'{total} nonempty circulant step sets checked')

    rng = Random(240700571)
    for n in (11, 13, 17, 19):
        # A random Cayley tournament chooses one step from each opposite pair.
        steps = [rng.choice((s, n - s)) for s in range(1, (n + 1) // 2)]
        ub2, ilb2 = squared_values(circulant(n, steps))
        assert ilb2 == Fraction(2 * n**2, n - 1)
        print(f'random circulant tournament n={n}: UB^2={ub2}, ILB^2={ilb2}')

    # Permutations of the five elements act transitively on the two-subsets.
    pairs = list(combinations(range(5), 2))
    kneser = make_graph(pairs, lambda v, u: set(v).isdisjoint(u))
    layered = circulant(12, {1, 4, 7, 10})  # Directed triangle of four-vertex layers.
    bipartite = circulant(10, {1, 3, 5, 7, 9})
    for name, graph in [('Kneser KG(5,2)', kneser), ('three layers', layered),
                        ('complete bipartite', bipartite), ('wrapped butterfly', wrapped_butterfly(3))]:
        ub2, ilb2 = squared_values(graph)
        if name == 'three layers':
            assert (ub2, ilb2) == (72, 144)
        if name == 'complete bipartite':
            assert (ub2, ilb2) == (50, 100)
        print(f'{name}: UB^2={ub2}, ILB^2={ilb2}')

    for k in range(1, 6):
        n = 3**k
        visible = triangle_power(k)
        degree = (n - 1) // 2
        assert all(row.bit_count() == degree for row in visible)
        assert all(bool(visible[t] & (1 << s)) != bool(visible[s] & (1 << t))
                   for s, t in combinations(range(n), 2))
        ub2, ilb2 = squared_values(visible)
        assert (ub2, ilb2) == (Fraction(n**2, 2**k), Fraction(n**2, degree))
        print(f'triangle power k={k}: UB^2={ub2}, ILB^2={ilb2}, ratio^2={ub2 / ilb2}')
    print('All assertions passed.')


if __name__ == '__main__':
    main()
