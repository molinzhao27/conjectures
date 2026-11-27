"""Exact certificates and numerical search for the cubic Rademacher inequality."""
from math import comb, gcd, lcm, isqrt
from fractions import Fraction
from functools import reduce
from itertools import product
import numpy as np
from scipy.optimize import minimize
import mpmath as mp


def flat(n):
    total = sum(comb(n, k) * abs(n - 2*k)**3 for k in range(n+1))
    return mp.mpf(total) / (2**n * n * mp.sqrt(n))


def sphere_search(n, rng):
    signs = 1.0 - 2*((np.arange(2**(n-1))[:, None] >> np.arange(n)) & 1)
    target = float(flat(n))

    def objective(v):
        length = np.linalg.norm(v)
        a = v / length
        sums = signs @ a
        f = np.mean(abs(sums)**3)
        d = np.sum((a*a - 1/n)**2)
        if d < 1e-12:
            return 2., np.zeros(n)
        ratio = (target - f) / d
        grad = -(3 * ((sums*abs(sums)) @ signs) / len(sums) + 4*ratio*a**3)/d
        return ratio, (grad - (grad @ a)*a)/length

    candidates = []
    for k in range(1,n):
        a = np.zeros(n)
        a[:k] = 1
        candidates.append((objective(a)[0], a))
    for _ in range(32):
        fit = minimize(objective, rng.normal(size=n), jac=True, method='BFGS',
                       options={'gtol': 1e-9, 'maxiter': 120})
        candidates.append((fit.fun, fit.x))
    ratio, finalist = min(candidates, key=lambda candidate: candidate[0])
    # Each binary float is an exact dyadic rational. Clear the common
    # denominator and sum the cubes as integers before normalizing.
    dyadics = [float(value).as_integer_ratio() for value in finalist]
    denominator = max(d for _, d in dyadics)
    weights = [v * (denominator // d) for v, d in dyadics]
    norm_squared = sum(v*v for v in weights)
    signed_sums = [weights[0]]
    for weight in weights[1:]:
        signed_sums = [v + shift for v in signed_sums for shift in (-weight, weight)]
    moment = (sum(abs(v)**3 for v in signed_sums)
              / (len(signed_sums) * norm_squared * mp.sqrt(norm_squared)))
    delta = (mp.mpf(n * sum(v**4 for v in weights) - norm_squared**2)
             / (n * norm_squared**2))
    checked = (flat(n) - moment) / delta
    assert abs(checked - ratio) < mp.mpf("1e-8")
    return checked, np.count_nonzero(abs(finalist) > 1e-5)


def chamber_certificate(n, k):
    """Prove the flat-k quotient is optimal by exact polynomial certificates."""
    from sympy.polys.rings import ring
    from sympy import ZZ

    patterns = [(1, *tail) for tail in product((-1, 1), repeat=n-1)]
    count = len(patterns)
    h = sum(abs(sum(s))**3 for s in patterns)
    # sqrt(n) times the target optimal constant is A + B sqrt(n*k).
    gap = Fraction(1,k) - Fraction(1,n)
    A = Fraction(h, count*n) / gap
    B = -Fraction(sum(comb(k,i)*abs(k-2*i)**3 for i in range(k+1)), 2**k*k*k) / gap
    d = lcm(A.denominator, B.denominator)
    A, B = int(A*d), int(B*d)
    radical = n*k

    _, *z = ring(','.join(f'z{i}' for i in range(n)), ZZ)
    initial = tuple(tuple(int(j<=i) for j in range(n)) for i in range(n))
    leaves = 0

    def nonnegative(x,y):
        if y < 0:
            return x >= 0 and x*x >= radical*y*y
        return x >= 0 or radical*y*y >= x*x

    def check(rays, depth=0):
        nonlocal leaves
        a = [sum(v*t for v,t in zip(col,z)) for col in zip(*rays)]
        norm = sum(v**2 for v in a)
        q = sum(v**4 for v in a)
        m3 = 0
        cut_pair = None
        for s in patterns:
            vals = [sum(x*y for x,y in zip(s,ray)) for ray in rays]
            pos = sum(v*t for v,t in zip(vals,z) if v>0)
            neg = sum(-v*t for v,t in zip(vals,z) if v<0)
            m3 += pos**3 + neg**3
            if pos and neg and cut_pair is None:
                j, k1 = vals.index(min(vals)), vals.index(max(vals))
                cut_pair = j, k1, vals[k1], -vals[j]
        quad = norm**2
        excess = count*(quad - n*q)
        U = d*h*quad + A*excess
        V = B*excess
        rational = U**2 + radical*V**2 - d**2*n**3*norm*m3**2
        irrational = 2*U*V
        if all(nonnegative(rational.get(e,0),irrational.get(e,0))
               for e in rational.keys() | irrational.keys()):
            leaves += 1
            return
        assert depth < 80
        if cut_pair is None:
            # Bisect the longest edge on the section sum(a)=1.
            def distance(j,k1):
                return sum((Fraction(x,sum(rays[j]))-Fraction(y,sum(rays[k1])))**2
                           for x,y in zip(rays[j],rays[k1]))
            _,j,k1 = max((distance(j,k1),j,k1) for j in range(n) for k1 in range(j))
            cut_pair = j,k1,sum(rays[k1]),sum(rays[j])
        j,k1,alpha,beta = cut_pair
        cut = tuple(alpha*x + beta*y for x,y in zip(rays[j],rays[k1]))
        g = reduce(gcd,cut)
        cut = tuple(v//g for v in cut)
        for m in (j,k1):
            check(rays[:m]+(cut,)+rays[m+1:], depth+1)

    check(initial)
    return leaves


# Interval endpoints represent integer multiples of 1/SCALE.
SCALE = 10**6


def near_certificate(cap, zlo, zhi):
    """Certify the odd-dimensional smoothing bound on the full parameter box."""

    s = SCALE
    def up(num, den):
        return -(-num//den)
    def sqlo(x):
        return x*x//s
    def sqhi(x):
        return up(x*x,s)
    def rootup(x):
        return isqrt(x*s)+1
    def dev(lo,hi):
        return sqlo(max(0,sqlo(lo)-s,s-sqhi(hi)))

    # d, B, A, z, where z is the reciprocal residual dimension.
    todo = [((0,s),(s//5,3*s//2),(s,8*s//5),(zlo,zhi))]
    visits = 0
    while todo:
        box = todo.pop()
        visits += 1
        (d,D),(b,B),(a,A),(z,Z) = box
        if d>B or b>A:
            continue
        # Enclose the residual squared deviations and mean square.
        rem = cap-dev(d,D)-dev(b,B)-dev(a,A)
        if rem < 0:
            continue
        high = 3*s-sqlo(d)-sqlo(b)-sqlo(a)
        low = 3*s-sqhi(D)-sqhi(B)-sqhi(A)
        w_hi = s+up(max(z*high,Z*high),s)
        w_lo = s+min(z*low,Z*low)//s
        if sqlo(b)>w_hi or sqhi(A)<w_lo:
            continue
        # The two variance bounds in (12) both hold on the entire box.
        v = up(rem*s**3, (s-Z)*(s+b)**2)
        if z>0:
            v = min(v,up((A-b)**2*s,4*z*(s-Z)))
        y0 = rootup(sqhi(D)+v)
        y2 = rootup(sqhi(max(0,2*rootup(w_hi)-d))+up(v*(s*s-4*z*z),s*s))
        def kernel(y):
            h = a+b
            ell = max(0,A-b)
            return max(0, max(0,h-y)**2//h -
                       (up(max(0,ell-y)**2,ell) if ell else 0))
        # Downward bounds for the mass factor and both kernel values in (13).
        mass = isqrt((797**2 * (s+3*z)*s*s)//(1000**2*(s+z)))
        lower = kernel(y0) + kernel(y2)*s//(s+2*Z)
        if 3*mass*lower*40 >= 7*8*A*B:
            continue
        idx = max(range(4),key=lambda i:(box[i][1]-box[i][0])*(3 if i==3 else 1))
        lo,hi = box[idx]
        mid = (lo+hi)//2
        assert lo < mid < hi, (cap,box,3*mass*lower/(8*A*B))
        for part in ((lo,mid),(mid,hi)):
            todo.append(box[:idx]+(part,)+box[idx+1:])
    return visits


def far_certificate():
    """Check both endpoints of the three concave scalar bounds."""
    from fractions import Fraction as Q
    constant = Q(7, 40)

    def curvature(x):
        return Q(1595,1000)*(1-x/4)*((1-x*x)/4-x/32)

    global_tangent = (5*curvature(Q(3,7))+4*Q(16,100))/9
    assert (global_tangent-constant)*6 > Q(234,1000)
    assert (5*curvature(Q(4,9))+4*Q(16,100))/9 > constant
    endpoints = 0
    for n, start, penalty in [(7,175,234),(9,145,225),(11,124,220)]:
        assert Q(1596,1000)*(Q(n,8*(n-1))+Q(1,64*(n-1)**2)) < Q(penalty,1000)
        for j in (start,600):
            root = Q(isqrt(j*10**10)+1,10**6)
            x = (1+root)/n
            assert (curvature(x)-constant)*Q(j,100) > Q(penalty,1000)
            endpoints += 1
    return endpoints


def main():
    mp.mp.dps = 75
    constant = 5*mp.sqrt(3)-6*mp.sqrt(2)
    assert abs(6*(flat(3)-flat(2))-constant) < mp.mpf('1e-72')
    assert constant < mp.mpf(7)/40
    assert 1-2*sum(Fraction((-1)**(j-1), (j+1)**2) for j in range(1,26)) > Fraction(64,100)
    for n,k,expected_leaves in [(2,1,1),(3,2,2),(4,2,3),(5,4,41)]:
        assert chamber_certificate(n,k) == expected_leaves
        optimum = (flat(n)-flat(k))/(mp.mpf(1)/k-mp.mpf(1)/n)
        assert optimum >= constant-mp.mpf('1e-72')
        print(f'n={n} exact cone certificate, {expected_leaves} leaves, optimum {float(optimum):.12f}')
    print(f'{far_certificate()} exact far-region endpoints')
    for args,expected in [((1750000,250000,250000),435),
                          ((1450000,166666,166667),427),
                          ((1240000,0,125000),5747)]:
        assert near_certificate(*args) == expected
        print(f'near region {args}, {expected} boxes')
    rng = np.random.default_rng(260817802)
    for n in range(2,14):
        ratio,k = sphere_search(n,rng)
        assert ratio >= constant-mp.mpf("1e-65")
        print(f'n={n:2d}, smallest quotient found {float(ratio):.12f}, support {k}')
    print('All assertions passed.')


if __name__ == '__main__':
    main()
