"""Exact finite checks for solution.md. Run with Python 3.9+ (without -O)."""

from functools import reduce
from itertools import combinations, product
from math import gcd


def words(labels):
    return {sum(1 << (int(i) - 1) for i in s) for s in labels.split(",")}


def trunk(code, mask):
    return frozenset(c for c in code if c & mask == mask)


def trunks(code, n):
    return {trunk(code, mask) for mask in range(1 << n)}


def project(word, requirements):
    return sum(1 << j for j, mask in enumerate(requirements) if word & mask == mask)


def check_morphism(mapping, source_trunks, n):
    target = set(mapping.values())
    for t in trunks(target, n):
        if t and t != target:
            preimage = frozenset(c for c, v in mapping.items() if v in t)
            assert preimage in source_trunks and len(preimage) < len(mapping)
    assert mapping[0] == 0
    return target


def feasible(rows, d):
    """Integer Fourier–Motzkin for affine forms a.x + b >= 0 (or > 0)."""
    rows = list(rows)
    for _ in range(d):
        positive = [(a, s) for a, s in rows if a[0] > 0]
        negative = [(a, s) for a, s in rows if a[0] < 0]
        reduced = {}
        for a, strict in rows:
            if a[0] == 0:
                reduced[a[1:]] = reduced.get(a[1:], False) or strict
        for a, s in positive:
            for b, t in negative:
                row = tuple(a[0] * y - b[0] * x for x, y in zip(a[1:], b[1:]))
                divisor = reduce(gcd, row, 0) or 1
                row = tuple(v // divisor for v in row)
                reduced[row] = reduced.get(row, False) or s or t
        rows = list(reduced.items())
    return all(a[0] > 0 if strict else a[0] >= 0 for a, strict in rows)


def arrangement(forms):
    """Test every proposed positive support, treating zero as inactive."""
    d = len(forms[0]) - 1
    code = set()
    for word in range(1 << len(forms)):
        rows = []
        for i, form in enumerate(forms):
            active = bool(word & (1 << i))
            rows.append((form if active else tuple(-v for v in form), active))
        if feasible(rows, d):
            code.add(word)
    return code


def interval_checks():
    intervals = list(combinations(range(5), 2))
    left = [(1, -a, 0) for a in range(4)]
    right = [(-1, b, 0) for b in range(1, 5)]
    # Coordinates (x,t). Gate t>0; final zero in each form is its constant.
    source = arrangement(left + right + [(0, 1, 0)])
    source_trunks = trunks(source, 9)
    count = 0
    for n in range(1, 7):
        for family in combinations(intervals, n):
            req = [(1 << a) | (1 << (b + 3)) | (1 << 8) for a, b in family]
            mapping = {c: project(c, req) for c in source}
            target = check_morphism(mapping, source_trunks, n)
            # Endpoints, intervening midpoints, and both exterior rays.
            direct = {sum(1 << i for i, (a, b) in enumerate(family) if 2*a < k < 2*b)
                      for k in range(-1, 10)}
            assert target == direct
            count += 1
    return count


def three_neuron_checks():
    positive = negative = maps = 0
    for encoding in range(128):
        code = {0} | {i+1 for i in range(7) if encoding & (1 << i)}
        facets = sorted(c for c in code if not any(c != b and c & b == c for b in code))
        base = {0}
        for k in range(1, len(facets) + 1):
            base.update(reduce(int.__and__, v) for v in combinations(facets, k))
        if base <= code:
            # The simplex construction in the report. Source on the positive
            # chart has group tests followed by one strict cut per extra word.
            extras = sorted(code - base)
            m = len(facets)
            all_cuts = ((1 << len(extras)) - 1) << m
            gate = 1 << (m + len(extras))
            sources = {v | all_cuts | gate for v in range((1 << m) - 1)}
            for j, extra in enumerate(extras):
                parent = next(h for h, f in enumerate(facets) if extra & f == extra)
                v = ((1 << m) - 1) ^ (1 << parent)
                sources.add(gate | v | (all_cuts ^ (1 << (m+j))))
            # All off-chart words collapse to 0. Record only gate there.
            sources.add(0)
            req = []
            for i in range(3):
                missing = sum(1 << h for h, f in enumerate(facets) if not f & (1 << i))
                cuts = sum(1 << (m+j) for j, e in enumerate(extras) if not e & (1 << i))
                req.append(gate | missing | cuts)
            assert {project(c, req) for c in sources} == code
            positive += 1
        else:
            # Two disjoint, nonempty open subsets would cover U_i.
            assert any(1 << i not in code and 7 not in code and
                       all((1 << i) | (1 << j) in code for j in range(3) if j != i)
                       for i in range(3))
            negative += 1

        ts = trunks(code, 3)
        proper = sorted((t for t in ts if t and t != code), key=lambda t: sorted(t))
        for selection in range(1 << len(proper)):
            chosen = [t for j, t in enumerate(proper) if selection & (1 << j)]
            mapping = {c: sum(1 << j for j, t in enumerate(chosen) if c in t) for c in code}
            check_morphism(mapping, ts, len(chosen))
            maps += 1
    return positive, negative, maps


def boundary_examples():
    # Closed triangular fan 0 <= y <= 1, -y <= x <= y.
    forms = [(0,1,0), (0,-1,1), (1,1,0), (-1,1,0), (1,0,0)]
    signed = forms + [tuple(-a for a in f) for f in forms]
    closed = set()
    for w in arrangement(signed):
        neg = [bool(w & (1 << (5+i))) for i in range(5)]
        zero = [not n and not w & (1 << i) for i, n in enumerate(neg)]
        inside = not any(neg[:4])
        labels = [zero[1], zero[2], neg[4] or zero[4], not neg[4], zero[3]]
        closed.add(sum(1 << i for i, v in enumerate(labels) if inside and v))
    assert closed == words(",2345,123,134,145,13,14,23,34,45,3,4")

    normals = [(1,0,0),(1,1,0),(0,1,0),(-1,0,0),(-1,-1,0),(0,-1,0)]
    open_code = arrangement(normals)
    assert open_code == words(",123,126,156,456,345,234,12,16,56,45,34,23")

    # Six 3-balls centred at +/- r e_j with radii 1,...,6.
    axes = [(j, sign) for j in range(3) for sign in (1, -1)]
    forms = [tuple(sign * int(k == j) for k in range(3)) + (0,) for j, sign in axes]
    poly = arrangement(forms)
    ball = set()
    for v in product((-1,0,1), repeat=3):
        # Witness x=v/4. Multiplying the ball test by 16 gives this strict test.
        norm = sum(u*u for u in v)
        ball.add(sum(1 << i for i, (j,s) in enumerate(axes) if norm < 8*(i+1)*s*v[j]))
    assert ball == poly and len(poly) == 27
    # Opposite balls cannot meet, so these 3^3 witnesses exhaust the ball code.

    # Downward cover from the source's example (a,b,c,d).
    source = words(",1,12,23,13,123")
    mapping = {c: project(c, [2,4,3,5]) for c in source}
    target = check_morphism(mapping, trunks(source, 3), 4)
    assert target == words(",13,12,24,1234")
    assert (len(trunks(source, 3)), len(trunks(target, 4))) == (8,7)
    return len(closed), len(open_code), len(ball)


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Assertions must be enabled.")
    print("3-neuron positive / obstructed codes / trunk maps:", three_neuron_checks())
    print("Interval families agreeing with homogenized arrangement:", interval_checks())
    print("Words in closed fan / open six-cycle / six 3-balls:", boundary_examples())
    print("Paper's downward cover: 8 -> 7 trunks. All checks passed.")
