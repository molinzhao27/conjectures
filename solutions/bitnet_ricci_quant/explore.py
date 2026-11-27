"""Exact Fisher-volume curvature checks. Run with Python and SymPy."""
from functools import cache
from itertools import combinations, product
from math import comb, prod

import sympy as s
from sympy.polys.rings import ring

H = s.S.Half


def dirichlet_mean(poly, cells):
    """Dirichlet(1/2,1/2,1/2,1/2) moments, with the last cell eliminated."""
    return sum(c * prod(s.rf(H, k) for k in powers) / s.rf(2, sum(powers))
               for powers, c in s.Poly(poly, *cells).terms())


def two_margin_trees():
    x, y, z = cells = s.symbols('x y z')
    a, b = x + y, x + z
    u, v = a*(1-a), b*(1-b)
    w = x-a*b
    weight = dirichlet_mean(u*v, cells)
    cross = dirichlet_mean(w*(1-2*a)*(1-2*b), cells)
    assert (weight, dirichlet_mean(u, cells), cross) == (s.Rational(7,240), 1/s.Integer(6), 1/s.Integer(120))
    six = dirichlet_mean(39*H*u*v - 3*H*(u+v) - 2*w*(1-2*a)*(1-2*b), cells) / weight

    def moment(t):
        return s.gamma(1+t)**2/s.gamma(2+2*t)

    weight = (moment(H)+2*moment(3*H))/8
    inv_u = (moment(-H)+2*moment(H))/8
    cross = (moment(H)-4*moment(3*H))/4
    five = s.simplify(14 - (inv_u/2+3*moment(H)/2+cross)/weight)
    assert (five, six) == (s.Rational(14,11), s.Rational(25,14))
    print(f"Five-node tree {five}; six-node tree {six}")

    # Two of A's children also have the same additional independent root.
    # That root contributes 2. On the multinomial base the multiplicities are 4,2.
    mass = dirichlet_mean(u**2*v, cells)
    weighted = ((3*H + 26 + 9)*u**2*v - 5*u*v - 3*H*u**2
                - 4*u*w*(1-2*a)*(1-2*b))
    integral = dirichlet_mean(weighted, cells)
    loop = 2 + integral/mass
    assert (mass, integral, loop) == (s.Rational(1,168), s.Rational(1,60), s.Rational(24,5))
    print(f"Seven-node graph with one chordless four-cycle {loop}")


def shared_roots(n, children):
    """Each of the children depends on all n independent binary roots."""
    k = children * 2**(n-1)
    r = s.symbols('r', positive=True)
    u = r*(1-r)
    # Angular derivative is sqrt(u) d/dr. Compute -2 Delta log F,
    # -|grad log F|^2 and the individual log-warp gradient squares.
    h = k*s.log(u)/2
    scalar = -2*(u*h.diff(r,2) + u.diff(r)*h.diff(r)/2) - u*h.diff(r)**2
    scalar -= k*u*(1/r**2+1/(1-r)**2)/4
    const, coeff = s.Rational(k*(2*k+1),2), s.Rational(k*(k-1),4)
    assert s.cancel(scalar - const + coeff/u) == 0
    if k == 1:
        return n*const
    beta_shape = (k+1)*H
    inverse_mean = (s.gamma(beta_shape-1)**2/s.gamma(2*beta_shape-2)
                    / (s.gamma(beta_shape)**2/s.gamma(2*beta_shape)))
    return s.simplify(n*(const-coeff*inverse_mean))


def direct_scalar(metric, coords, point):
    """Independent diagonal-metric computation via coordinate sectional curvatures."""
    at = dict(zip(coords, point, strict=True))
    g = [v.subs(at) for v in metric]
    # L[i,j] is partial_j log(sqrt(g_ii)).
    logdiff = [[v.diff(t)/(2*v) for t in coords] for v in metric]
    L = [[e.subs(at) for e in row] for row in logdiff]
    total = s.S.Zero
    for i, j in combinations(range(len(coords)), 2):
        for p, q in ((i,j), (j,i)):
            total -= (logdiff[q][p].diff(coords[p]).subs(at)
                      + L[q][p]*(L[q][p]-L[p][p]))/g[p]
        total -= sum(L[i][k]*L[j][k]/g[k] for k in range(len(coords)) if k not in (i,j))
    return s.factor(2*total)


def tree_metric_check(left, right):
    t = s.symbols(f't:{3+2*(left+right)}')
    a, b0, b1 = t[:3]
    b = (1-a)*b0+a*b1
    weights = [1, 1-a, a] + [1-a,a]*left + [1-b,b]*right
    metric = [h/(p*(1-p)) for h,p in zip(weights,t,strict=True)]
    point = [s.Rational(i+2,2*i+7) for i in range(len(t))]
    subs = dict(zip(t,point))
    u, v, w = a*(1-a), b*(1-b), a*(b1-b)
    scalar = 3*H
    for k,h in ((left,u),(right,v)):
        scalar += k*(k+2) + k*H - k*(k+1)/(4*h)
    scalar -= left*right*w*(1-2*a)*(1-2*b)/(2*u*v)
    assert s.cancel(scalar.subs(subs) - direct_scalar(metric,t,point)) == 0


@cache
def arcsine_moment(k):
    return s.QQ(comb(2*k,k),4**k)


def diamond_series():
    """Return an exact coefficient function for the two diamond integrals."""
    pol, r, b0, b1, c0, c1 = ring('r,b0,b1,c0,c1', s.QQ)
    one, half = pol.one, s.QQ(1,2)
    coords = (r,b0,b1,c0,c1)
    u = r*(1-r)
    probs = []
    for i,j in product(range(2), repeat=2):
        f = lambda p, state: p if state else 1-p
        probs.append((1-r)*f(b0,i)*f(c0,j)+r*f(b1,i)*f(c1,j))
    # In the fork metric, the inverse entry of a child is t(1-t)/parent.
    # Its factor parent already divides the derivative of each q.
    parents = (one, 1-r, r, 1-r, r)
    def gamma(f,g):
        return sum(t*(1-t)*f.diff(t).exquo(p)*g.diff(t)
                   for t,p in zip(coords,parents))

    def lap(f):
        # Each cell probability is affine separately in all five coordinates.
        return (3*half*(1-2*r)*f.diff(r)
                + sum(half*(1-2*t)*f.diff(t).exquo(p)
                      for t,p in zip(coords[1:],parents[1:])))

    ordinary = (1,1,1,1)  # twice the inverse powers in the binomial expansion
    Q = prod(probs)
    volume_terms = [(8*u*Q, ordinary)]
    scalar_terms = [((40*u-4)*Q, ordinary)]
    for i,f in enumerate(probs):
        others = prod(probs[:i]+probs[i+1:])
        scalar_terms.append((-8*u*others*lap(f), ordinary))
        powers = tuple(3 if i == j else 1 for j in range(4))
        scalar_terms.append((4*u*others*gamma(f,f), powers))
    for i,j in combinations(range(4),2):
        others = prod(p for t,p in enumerate(probs) if t not in (i,j))
        scalar_terms.append((-4*u*others*gamma(probs[i],probs[j]), ordinary))

    @cache
    def moment(powers):
        return prod(arcsine_moment(n) for n in powers)

    def coefficient(indices):
        """Coefficient indexed by four nonnegative integers; sum by total degree."""
        factor = prod((1-p)**k for p,k in zip(probs,indices))
        def integrate(terms):
            total = 0
            for polynomial,powers in terms:
                multiplier = prod(s.rf(n*H,k)/s.factorial(k) for n,k in zip(powers,indices))
                total += multiplier*sum(c*moment(m) for m,c in (polynomial*factor).items())
            return s.Rational(total)
        return integrate(scalar_terms), integrate(volume_terms)
    return coefficient


def main():
    two_margin_trees()
    for counts in ((1,2),(2,2)):
        tree_metric_check(*counts)
    checks = [(1,1,3*H),(1,2,1),(2,1,2),(2,2,4),(3,1,6),(4,1,16),(5,1,40)]
    for n,m,expected in checks:
        actual = shared_roots(n,m)
        assert actual == expected
        print(f"{n} shared root(s), {m} sink(s): {actual}")
    coefficient = diamond_series()
    leading = coefficient((0,0,0,0))
    print("Diamond exact series, constant numerator and volume coefficients:", leading)
    assert leading == (s.Rational(-5347,2097152), s.Rational(1017,1048576))
    for i in range(4):
        first = coefficient(tuple(int(j == i) for j in range(4)))
        assert first == (s.Rational(-394549,536870912), s.Rational(3051,8388608))
    print("All assertions passed.")


if __name__ == '__main__':
    main()
