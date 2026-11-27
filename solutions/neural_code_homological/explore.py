"""Exact Hochster computations over F_2 for polarized neural ideals."""
from collections import Counter, defaultdict
from functools import cache
from itertools import combinations, permutations
from math import factorial


def subsets(mask):
    part = mask
    while True:
        yield part
        if not part:
            return
        part = (part - 1) & mask


def atoms(mask):
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


@cache
def cubes(n):
    """All consistent literal masks and the words in the subcube they specify."""
    low = (1 << n) - 1
    return [(s | (t << n), sum(1 << (s | w) for w in subsets(low ^ (s | t))))
            for s in subsets(low) for t in subsets(low ^ s)]


def generators(n, code):
    missing = {s for s, words in cubes(n) if not words & code}
    return tuple(sorted(s for s in missing if all((s ^ b) not in missing for b in atoms(s))))


def binary_rank(columns):
    basis = {}
    for column in columns:
        while column:
            pivot = column.bit_length()
            if pivot not in basis:
                basis[pivot] = column
                break
            column ^= basis[pivot]
    return len(basis)


def hochster(n, code):
    """Map (homological degree, literal mask) to nonzero multiplicity."""
    gens = generators(n, code)
    if not gens:
        return {}
    if gens == (0,):
        return {(0, 0): 1}
    faces = {s for s in range(1 << (2*n)) if all(s & g != g for g in gens)}
    supports = {0}
    for g in gens:
        supports.update([s | g for s in supports])
    result = {}
    for w in supports - {0}:
        levels = defaultdict(list)
        for s in subsets(w):
            if s in faces:
                levels[s.bit_count()].append(s)
        prev_rank = 0
        prev_dim = 0
        for size in range(w.bit_count()+2):
            # Use each face mask as a row index in a binary boundary column.
            rank = binary_rank(sum(1 << (s ^ b) for b in atoms(s)) for s in levels[size])
            dim = prev_dim - prev_rank - rank
            if dim:
                result[w.bit_count()-size, w] = dim
            prev_dim, prev_rank = len(levels[size]), rank
    return result


def taylor_table(n, code):
    """Compute (2) using the relative boundary matrices."""
    gens = generators(n, code)
    if gens == (0,):
        return {(0, 0): 1}
    union = [0] * (1 << len(gens))
    columns = defaultdict(list)
    for family in range(1, len(union)):
        first = family & -family
        union[family] = union[family ^ first] | gens[first.bit_length()-1]
        degree = family.bit_count()-1
        boundary = sum(1 << (family ^ b) for b in atoms(family)
                       if union[family ^ b] == union[family])
        columns[degree, union[family]].append(boundary)
    ranks = {cell: binary_rank(cols) for cell, cols in columns.items()}
    table = {}
    for (i, w), cols in columns.items():
        multiplicity = len(cols) - ranks[i,w] - ranks.get((i+1,w), 0)
        if multiplicity:
            table[i,w] = multiplicity
    return table


def check_duality(n, code):
    """Compare both sides of (4) with the Alexander-dual support test."""
    mask = (1 << n)-1
    missing = ((1 << (1 << n))-1) ^ code
    gens = generators(n, code)
    comp_gens = generators(n, missing)
    wordsets = dict(cubes(n))
    exact_dual = True
    for h in range(1 << (2*n)):
        pos, neg = h & mask, h >> n
        doubles, singles = pos & neg, pos ^ neg
        unused = mask ^ (doubles | singles)
        ones = singles & pos
        reversed_h = (pos << n) | neg
        covered = all(g & reversed_h for g in gens)
        contained = any(h & g == g for g in comp_gens)
        def cylinder(fixed, values):
            return wordsets[values | ((fixed ^ values) << n)]
        projects = all(cylinder(singles | unused, ones | u) & code for u in subsets(unused))
        lifts = any(not cylinder(singles | doubles, ones | d) & missing for d in subsets(doubles))
        assert covered == projects and contained == lifts
        exact_dual &= (covered == contained)
    return exact_dual


def pair(table):
    return (max(i for i, w in table), max(w.bit_count()-i for i, w in table))


def orbits(n):
    permutations_of_words = [
        [sum(((w >> j) & 1) << i for i, j in enumerate(perm)) for w in range(1 << n)]
        for perm in permutations(range(n))]
    unseen = set(range(1, (1 << (1 << n))-1))
    while unseen:
        c = min(unseen)
        words = [b.bit_length()-1 for b in atoms(c)]
        orbit = {sum(1 << (p[w] ^ t) for w in words)
                 for p in permutations_of_words for t in range(1 << n)}
        unseen.difference_update(orbit)
        yield c, len(orbit)


# A polynomial represents an integer linear combination of e_A, where A is
# a set of cube words. Multiplication takes unions since e_A tests avoidance.
ONE = {0: 1}


def difference(a, b):
    out = a.copy()
    for mask, coeff in b.items():
        out[mask] = out.get(mask, 0) - coeff
        if not out[mask]:
            del out[mask]
    return out


def multiply(a, b):
    out = Counter()
    for s, x in a.items():
        for t, y in b.items():
            out[s | t] += x*y
    return {s: x for s, x in out.items() if x}


def product(polys):
    out = ONE
    for p in polys:
        out = multiply(out, p)
        if not out:
            break
    return out


def face_poly(n, face):
    coords = (1 << n)-1
    return product(difference(ONE, {words: 1}) for pattern, words in cubes(n)
                   if pattern & face == pattern
                   and (pattern | pattern >> n) & coords == (face | face >> n) & coords)


def rank_poly(faces, flags):
    """Rank of the active columns, via independent minors over F_2."""
    result = {}
    columns = {s: sum(1 << (s ^ b) for b in atoms(s)) for s in faces}
    for t in range(1, binary_rank(columns.values())+1):
        inactive = product(difference(ONE, product(flags[s] for s in family))
                           for family in combinations(faces, t)
                           if binary_rank(columns[s] for s in family) == t)
        result = difference(result, difference(inactive, ONE))
    return result


def vanishing_poly(beta, bound):
    # Integer-valued zero indicator for 0 <= beta <= bound.
    out = product(difference({0: j}, beta) for j in range(1, bound+1))
    div = factorial(bound)
    assert all(v % div == 0 for v in out.values())
    return {m: v//div for m,v in out.items()}


def region_formula(n):
    """All-n avoidance-polynomial formula; intended for very small n."""
    flags = [face_poly(n,s) for s in range(1 << (2*n))]
    zero = defaultdict(lambda: ONE)
    for w in range(1, 1 << (2*n)):
        levels = [[s for s in subsets(w) if s.bit_count() == t]
                  for t in range(w.bit_count()+1)]
        ranks = [rank_poly(level,flags) for level in levels]
        for t,level in enumerate(levels[:-1]):
            # t is the number of vertices in a chain basis face.
            beta = {}
            for s in level:
                beta = difference(beta, difference({}, flags[s]))
            beta = difference(difference(beta, ranks[t]), ranks[t+1])
            cell = (w.bit_count()-t-1, t+1)
            zero[cell] = multiply(zero[cell], vanishing_poly(beta,len(level)))
    @cache
    def cdf(p, r):
        flag = product(v for (i,j),v in zero.items() if i > p or j > r)
        return sum(c * (1 << ((1 << n)-s.bit_count())) for s,c in flag.items())-2
    counts = {}
    for p in range(2*n):
        for r in range(1,2*n+1):
            m = cdf(p,r)-cdf(p-1,r)-cdf(p,r-1)+cdf(p-1,r-1)
            if m:
                counts[p,r] = m
    return counts


def common_literal_components(n, code, w):
    words = [b.bit_length()-1 for b in atoms(code)]
    mask = (1 << n)-1
    traces = [w & (c | ((mask ^ c) << n)) for c in words]
    reached = {0}
    while True:
        more = {j for j, t in enumerate(traces) if any(t & traces[k] for k in reached)}
        if more <= reached:
            break
        reached.update(more)
    return 1 if len(reached) == len(traces) else 2


def gated_code(n, t):
    """The family with regularity 2n-2 and freely tunable t."""
    m = n-1
    low = (1 << m)-1
    return sum(1 << w for w in range(1 << n)
               if (w & low) != low
               and (w != 0)
               and not ((w >> m) and (w & ((1 << t)-1))))


def private_code(n, k, s):
    """Omit k rows, with s nonconstant columns and k private literals."""
    missing = [(1 << i) for i in range(k)]
    missing[0] |= ((1 << s)-1) ^ ((1 << k)-1)
    return ((1 << (1 << n))-1) ^ sum(1 << w for w in missing)


def main():
    expected_sizes = [1,4,12,26]
    for n, expected in enumerate(expected_sizes,1):
        counts = Counter()
        reps = 0
        alexander_duals = 0
        for c, count in orbits(n):
            reps += 1
            betti = hochster(n,c)
            assert taylor_table(n,c) == betti
            p, r = pair(betti)
            if check_duality(n,c):
                comp = ((1 << (1 << n))-1) ^ c
                assert pair(hochster(n,comp)) == (r-1,p+1)
                alexander_duals += count
            counts[p,r] += count
            if n >= 3:
                antipodal = c.bit_count() == 2 and (
                    (c.bit_length()-1) ^ ((c & -c).bit_length()-1)) == (1 << n)-1
                assert (p == 2*n-3) == antipodal
            for w in range(1,1 << (2*n)):
                d = (w & (w >> n)).bit_count()
                if d:
                    beta = betti.get((w.bit_count()-3,w), 0)
                    assert beta == (d-1)*(common_literal_components(n,c,w)-1)
        assert len(counts) == expected and sum(counts.values()) == (1 << (1 << n))-2
        assert alexander_duals == [2,12,126,5528][n-1]
        predicted = {(0,r) for r in range(1,n+1)}
        predicted.update((p,1) for p in range(n))
        predicted.update((p,2) for p in range(n-1))
        predicted.update((p,r) for p in range(1,2*n) for r in range(3,2*n-p+1))
        if n == 4:
            predicted.update({(3,6), (4,5)})
        assert set(counts) == predicted
        if n <= 2:
            assert region_formula(n) == counts
        print(f'n={n}: {sum(counts.values())} codes, {reps} orbits, {len(counts)} pairs')
        print(' ', sorted(counts))
    for n in range(3,7):
        for t in range(1,n):
            assert pair(hochster(n,gated_code(n,t))) == (max(t,2),2*n-2)
        for k in range(2,n+1):
            for s in range(k,n+1):
                c = private_code(n,k,s)
                betti = hochster(n,c)
                comp = ((1 << (1 << n))-1) ^ c
                assert pair(hochster(n,comp)) == (n+s-k-1,k+1)
                assert pair(betti) == (k-1,n+s-k+1)
                assert sum(betti.values()) == (1 << k)-1
    for c, expected in [(0x7f,(2,2)),(0xff0,(1,3))]:
        assert pair(hochster(4,c)) == (1,3)
        assert pair(hochster(4,0xffff ^ c)) == expected
    assert hochster(2,0) == {(0,0):1} and hochster(2,15) == {}
    print('All assertions passed.')


if __name__ == '__main__':
    main()
