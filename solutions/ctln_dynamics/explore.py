# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.5", "scipy==1.18.1", "sympy==1.14.0", "mpmath==1.3.0", "python-flint==0.9.0"]
# ///
"""Reproducible checks for Curto--Morrison questions 11, 13 and 14.

Run the experiments with the project Python or `uv run explore.py all`.
Run `uv run explore.py certificate` for both interval proofs. Run the wording
subcommand to print the full original two-paragraph question directly from TeX.
All node indices inside the code are zero-based. Each W row indexes the receiver.
"""
import argparse
from itertools import combinations
from pathlib import Path

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def ctln(n, edges, eps=0.25, delta=0.5):
    w = np.full((n, n), -1.0 - delta)
    np.fill_diagonal(w, 0)
    for sender, receiver in edges:
        w[receiver, sender] = -1 + eps
    return w


def integrate(w, x, duration, **options):
    def rhs(t, v):
        return np.maximum(0, 1 + w @ v) - v
    result = solve_ivp(rhs, (0, duration), x, method="DOP853",
                       rtol=1e-12, atol=2e-14, **options)
    assert result.success, result.message
    return result


def gaudi_edges():
    return [(j, (j + k) % 5) for j in range(5) for k in (1, 2)]


def gaudi():
    """Check both branches at the parameters printed inside Figure 8.

    Rigidity: if a solution has constant sum, take a time interval in one
    closed chamber (a finite closed cover has a member with interior).
    Its affine continuation z also has constant sum by analyticity. Relative
    to this chamber, write F(x)=Ax+b+q(x) with q>=0 componentwise. Where
    x(t)=z(t), the sum of x(t+s)-z(t+s) is the integral of
    1^T exp(A*(s-v)) q(x(t+v)). For small positive s all kernel entries
    are positive, so equality of the sums forces q=0. Iterate to extend
    equality. For a periodic solution, the orbit therefore stays in one
    closed convex chamber. Homotheties toward the orbit's time average
    give arbitrarily close smaller periodic orbits of the same affine flow
    inside that chamber. Hence no isolated periodic orbit has constant sum.
    """
    w = ctln(5, gaudi_edges(), 0.1, 0.12)
    for seed in (9, 10):
        x = np.random.default_rng(seed).uniform(0, 0.4, 5)
        y = integrate(w, x, 5000, max_step=0.6).y[:, -1]

        def peak(t, v):
            return max(0, 1 + w[0] @ v) - v[0]
        peak.direction = -1
        r = integrate(w, y, 510, events=peak, max_step=0.18, dense_output=True)
        peaks = r.y_events[0][:, 0]
        times = r.t_events[0][peaks > (0.43 if seed == 10 else 0.39)]
        start, end = times[:2]
        grid = np.linspace(start, end, 8193)

        def speed(t):
            v = r.sol(t)
            return (np.maximum(0, 1 + w @ v) - v).sum()
        speeds = np.array([speed(t) for t in grid])
        roots = [brentq(speed, grid[i], grid[i + 1]) for i in range(len(grid) - 1)
                 if speeds[i] * speeds[i + 1] < 0
                 and max(abs(speeds[i]), abs(speeds[i + 1])) > 1e-12]
        # On long all-active intervals the true speed can be below machine
        # precision. The separate multiprecision shooting calculation resolves it.
        totals = r.sol(np.r_[grid, roots]).sum(axis=0)
        closure = np.max(abs(r.sol(end) - r.sol(start)))
        print(f"seed {seed}, T={end-start:.12f}, Smin={min(totals):.14f}, "
              f"Smax={max(totals):.14f}, closure={closure:.3g}")
        assert closure < 1e-9
        if seed == 10:
            assert abs(end - start - 238.590461946) < 1e-6
            assert abs(np.ptp(totals) - 0.0042802720331) < 1e-10

    # Coprimality with p(-z) excludes both zero and pure imaginary roots.
    z = sp.symbols("z")
    w = sp.Matrix(w).applyfunc(lambda a: sp.Rational(str(float(a))).limit_denominator(100))
    counts = []
    for k in range(1, 6):
        polys = {(w - sp.eye(5)).extract(q, q).charpoly(z).as_expr()
                 for q in combinations(range(5), k)}
        assert all(sp.gcd(p, p.subs(z, -z)) == 1 for p in polys)
        counts.append(len(polys))
    print("different active-block characteristic polynomials by size:", counts)


def shooting(certify=False):
    # One fifth of the rotating wave, starting when node 4 switches on.
    # Unknowns: five initial activities and six successive flight times.
    mp.mp.dps = 78
    w = mp.matrix(5, 5)
    for i in range(5):
        for j in range(5):
            if i != j:
                w[i, j] = -mp.mpf(9)/10 if (j, i) in gaudi_edges() else -mp.mpf(28)/25
    switch = [1, 1, 3, 3, 0, 0]
    off = [None, 1, None, 3, None, 0]
    rotation = mp.matrix(5)
    for j in range(5):
        rotation[(j + 1) % 5, j] = 1
    flows = []
    for inactive in off:
        a = mp.matrix(6)
        for i in range(5):
            a[i, i] = -1
            if i != inactive:
                a[i, 5] = 1
                for j in range(5):
                    a[i, j] += w[i, j]
        flows.append(a)

    def evaluate(u):
        state = mp.matrix(list(u[:5]) + [1])
        deriv = mp.matrix(6, 11)
        deriv[:5, :5] = mp.eye(5)
        jac = mp.matrix(11)
        residual = mp.matrix(11, 1)
        starts = []
        transition = mp.eye(5)
        for i, (a, j) in enumerate(zip(flows, switch)):
            starts.append(state)
            flow = mp.expm(a * u[5 + i])
            transition = flow[:5, :5] * transition
            state = flow * state
            deriv = flow * deriv
            deriv[:, 5 + i] = a * state
            residual[i] = 1 + (w[j, :] * state[:5])[0]
            jac[i, :] = w[j, :] * deriv[:5, :]
        residual[6:, :] = state[:5] - rotation * u[:5]
        jac[6:, :] = deriv[:5, :]
        jac[6:, :5] -= rotation
        return residual, jac, starts, transition

    # Use a short forward flight from the largest neuron-0 peak only to seed Newton.
    peak = mp.matrix([.436611663, .064016966, .221528276, .270362740, .000279111, 1])
    # At that peak node 4 is off, unlike the last flight (node 0 off).
    a = mp.matrix(flows[0])
    a[4, :] = mp.matrix(1, 6)
    a[4, 4] = -1
    seed_state = (mp.expm(a * mp.mpf("1.28337692")) * peak)[:5]
    durations = [3.03069628, 11.11617752, 4.39398471, 6.42526694, 18.62740680, 4.12456015]
    u = mp.matrix(list(seed_state) + durations)
    for _ in range(10):
        residual, jac, starts, transition = evaluate(u)
        if mp.norm(residual, mp.inf) < mp.mpf("1e-68"):
            break
        u -= mp.lu_solve(jac, residual)
    assert mp.norm(residual, mp.inf) < mp.mpf("1e-68")
    print("shooting residual", mp.nstr(mp.norm(residual, mp.inf), 4))
    print("Gaudi T", mp.nstr(5 * sum(u[5:]), 65))
    print("six flight times", [mp.nstr(t, 16) for t in u[5:]])
    multipliers = [abs(v)**5 for v in mp.eig(rotation.T * transition, left=False, right=False)]
    print("largest transverse full-period Floquet modulus", mp.nstr(sorted(multipliers)[-2], 10))

    vals = []
    for i, (a, initial) in enumerate(zip(flows, starts)):
        duration = u[5 + i]
        # Independent numerical itinerary check. The exponential action is
        # used only for the mesh; critical values use multiprecision expm.
        from scipy.sparse.linalg import expm_multiply
        states = expm_multiply(np.array(a.tolist(), float), np.array(list(initial), float),
                               start=0, stop=float(duration), num=201)[1:-1, :5]
        fields = 1 + states @ np.array(w.tolist(), float).T
        active = np.ones(5, dtype=bool)
        if off[i] is not None:
            active[off[i]] = False
        assert (fields[:, active] > 0).all() and (fields[:, ~active] < 0).all()
        vals.append(sum(initial[:5]))
        if off[i] is None:
            continue  # S' = 5 - (126/25) S throughout.
        def velocity(t):
            return sum((a * (mp.expm(a*t) * initial))[:5])
        # Each off flight starts with a minimum very close to the switch,
        # followed by a maximum well inside. Confirm derivative sign on a mesh.
        speed_mesh = (np.maximum(fields, 0) - states).sum(axis=1)
        assert np.count_nonzero(np.diff(np.sign(speed_mesh))) == 1
        for bracket in [(0, mp.mpf("1e-6")), (duration/4, duration/2)]:
            t = mp.findroot(velocity, bracket, tol=mp.mpf("1e-70"))
            assert 0 < t < duration and abs(velocity(t)) < mp.mpf("1e-68")
            vals.append(sum((mp.expm(a*t)*initial)[:5]))
    print("Gaudi Smin", mp.nstr(min(vals), 65))
    print("Gaudi Smax", mp.nstr(max(vals), 65))
    print("Gaudi Smax-Smin", mp.nstr(max(vals) - min(vals), 65))

    if certify:
        weights = [[0 if i == j else ("-9/10" if (j,i) in gaudi_edges() else "-28/25")
                    for j in range(5)] for i in range(5)]
        inactive = [set() if k is None else {k} for k in off]
        certify_orbit(u, weights, [4, 0, 1, 2, 3], inactive, switch, (min(vals), max(vals)))


def certify_orbit(seed, weights, permutation, off, switches, gaudi_bounds=None):
    """Rigorous ball enclosures. Optional dependency: python-flint >= 0.9."""
    from flint import arb, arb_mat, ctx

    ctx.prec = 448
    n = len(weights)
    size = n + len(off)
    def identity(dim):
        return arb_mat([[int(i == j) for j in range(dim)] for i in range(dim)])
    def col(values):
        return arb_mat([[v] for v in values])
    def block(m, rows, cols):
        return arb_mat([[m[i, j] for j in cols] for i in rows])
    def row_norm(v):
        return sum(abs(v[0, j]) for j in range(v.ncols()))
    def uniform_norm(v):
        return max(abs(a).upper() for a in v.entries())
    def enclosure(v):
        return arb(mp.nstr(v, 77))
    def exp_time(a, t):
        # Splitting off time uncertainty avoids wrapping a wide perturbation
        # through scaling and squaring on the long all-active flights.
        mid = t.mid()
        return (a * mid).exp() * (a * (t - mid)).exp()

    nodes = range(n)
    w = arb_mat([[arb(entry) for entry in row] for row in weights])
    inputs = [arb_mat([[w[i, j] for j in nodes] + [1]]) for i in nodes]
    sum_row = arb_mat([[1]*n + [0]])
    rotation = arb_mat([[int(permutation[i] == j) for j in nodes] for i in nodes])
    flows = []
    for inactive in off:
        a = arb_mat(n+1, n+1)
        for i in nodes:
            a[i, i] = -1
            if i not in inactive:
                for j in range(n+1):
                    a[i, j] += inputs[i][0, j]
        flows.append(a)
    def evaluate(v):
        x = col(v[:n] + [1])
        derivative = arb_mat(n+1, size)
        for i in nodes:
            derivative[i,i] = 1
        starts = []
        transition = identity(n)
        residual = []
        jac_rows = []
        for i, (a, j) in enumerate(zip(flows, switches)):
            starts.append(x)
            flow = exp_time(a, v[n+i])
            transition = block(flow, nodes, nodes) * transition
            x = flow*x
            derivative = flow*derivative
            dx = a*x
            for k in range(n+1):
                derivative[k, n+i] = dx[k, 0]
            residual.append((inputs[j]*x)[0, 0])
            jac_rows.append((inputs[j]*derivative).entries())
        for i in nodes:
            j = permutation[i]
            residual.append(x[i,0] - v[j])
            derivative[i,j] -= 1
            jac_rows.append([derivative[i,k] for k in range(size)])
        return col(residual), arb_mat(jac_rows), starts, transition

    center = [enclosure(v).mid() for v in seed]
    for _ in range(8):
        f, jac, _, _ = evaluate(center)
        if uniform_norm(f) < arb("1e-75"):
            break
        center = [v.mid() for v in (col(center) - jac.solve(f)).entries()]
    radius = arb("1e-61")
    box = [v + arb(0, radius) for v in center]
    assert all(value > 0 for value in box[:n])
    f, jac, _, _ = evaluate(center)
    y = jac.inv().mid()
    assert not y.det().contains(0)
    _, bound_jac, starts, transition = evaluate(box)
    remainder = identity(size) - y*bound_jac
    lip = max(sum(abs(remainder[i,j]) for j in range(size)).upper() for i in range(size))
    displacement = uniform_norm(y*f)
    assert lip < arb(1)/2 and displacement + lip*radius < radius
    print("certified shooting contraction, Lipschitz bound", lip.str(4))

    # Divide charpoly by (z-1), the time-translation multiplier, by synthetic
    # division. The Schur recursion places all remaining roots in |z|<1.
    coeffs = list(reversed((rotation.transpose()*transition).charpoly()))
    reduced = [coeffs[0]]
    for c in coeffs[1:-1]:
        reduced.append(reduced[-1]+c)
    assert (coeffs[-1]+reduced[-1]).contains(0)
    while len(reduced) > 1:
        a0, an = reduced[0], reduced[-1]
        assert abs(an) < abs(a0)
        reduced = [a0*c - an*d for c, d in zip(reduced[:-1], reversed(reduced[1:]))]

    # Bound the activity on every flight. Find witness times separately so
    # these interval tests enclose the actual extrema on both sides.
    tolerance = arb("1e-50")
    if gaudi_bounds:
        low, high = map(enclosure, gaudi_bounds)
    min_witness = False
    max_witness = False
    count = 0
    for i, (a, x) in enumerate(zip(flows, starts)):
        duration = box[n+i]
        assert duration > 0
        end = exp_time(a, duration)*x
        # The sum on an all-on flight is monotone between its endpoints.
        endpoint_sums = [(sum_row*v)[0,0] for v in (x, end)]
        if gaudi_bounds:
            if not off[i]:
                assert all(low-tolerance < value < high+tolerance for value in endpoint_sums)
            min_witness |= any(value < low+tolerance for value in endpoint_sums)
            max_witness |= any(value > high-tolerance for value in endpoint_sums)
        matrix_norm = max(sum(abs(a[j,k]) for k in range(n+1)).upper() for j in range(n+1))
        entry_switch = switches[i-1] if i else permutation[switches[-1]]
        def visit(left, right, depth):
            nonlocal min_witness, max_witness, count
            assert depth < 160
            mid = (left+right)/2
            h = ((right-left)*duration/2).upper()
            state = exp_time(a, mid*duration)*x
            derivatives = [state]
            for _ in range(4):
                derivatives.append(a*derivatives[-1])
            # For |s| <= h, Taylor at midpoint with homogeneous affine matrix
            # A bounds the fourth-order vector remainder.
            tail = uniform_norm(derivatives[4]) * (matrix_norm*h).exp()*h**4/24
            def span(row):
                value = (row*state)[0,0]
                error = row_norm(row)*tail
                for k, factor in ((1, h), (2, h*h/2), (3, h**3/6)):
                    error += abs((row*derivatives[k])[0,0])*factor
                return value + arb(0, error.upper())
            good = True
            for node, row in enumerate(inputs):
                sign = -1 if node in off[i] else 1
                if sign*span(row) > 0:
                    continue
                # A prescribed zero at a flight endpoint is simple.
                if left == 0 and node == entry_switch and sign*span(row*a) > 0:
                    continue
                if right == 1 and node == switches[i] and sign*span(row*a) < 0:
                    continue
                good = False
                break
            if gaudi_bounds and off[i]:
                bound = span(sum_row)
                good &= low-tolerance < bound < high+tolerance
                witness = (sum_row*state)[0,0]
                min_witness |= witness < low+tolerance
                max_witness |= witness > high-tolerance
            if good:
                count += 1
            else:
                visit(left, mid, depth+1)
                visit(mid, right, depth+1)
        for j in range(32):
            visit(arb(j)/32, arb(j+1)/32, 0)
    print("certified itinerary and transverse attraction;", count, "intervals")
    if gaudi_bounds:
        assert min_witness and max_witness
        print("certified absolute error of each printed extremum < 1.01e-50")
    return box[n:]


def sources():
    """A permanent decay barrier and a comparison proof for arbitrary graphs.

    Write S=sum(x), c=1+delta, h=delta+eps and D=theta-c*S. Summing the
    unrectified inputs gives S' >= n*theta - (1+(n-1)*c)*S, so eventually
    D <= -eta < 0. A source's input is D+c*x_j. Once x_j <= eta/c after
    that time, x_j decays exactly as exp(-t). Otherwise x_j >= m=eta/c,
    and E={t: D+c*x_j>0} has infinite measure by x_j' <= -m+theta*1_E.

    For j->k, input_k >= D+c*x_k+h*x_j. On E, (x_k-x_j)' is at least
    delta*x_k+eps*x_j. Off E, it is at least -(x_k-x_j). Thus x_k catches
    x_j in finite time if initially smaller, and stays above it. Thereafter
    r=x_k/x_j has r' >= h+(1-r)*D/x_j >= h on E, and r' >= 0 off E.
    This contradicts boundedness since x_j >= m and x_k' <= theta-x_k.
    """
    rng = np.random.default_rng(117)
    total = 0
    largest_entry_time = 0
    for n in (2, 3, 5, 8, 16):
        for eps, delta in ((.25, .5), (.01, .02), (.08, 2), (.6, 4)):
            for trial in range(6):
                edges = [(i, j) for i in range(n) for j in range(1, n)
                         if i != j and rng.random() < 0.3]
                edges.append((0, 1))
                w = ctln(n, edges, eps, delta)
                x = rng.uniform(0, 1.5, n)
                if trial % 2 == 0:
                    x[:] = 0
                    x[0] = 2
                c = 1 + delta
                rate = 1 + (n - 1)*c
                limit = n / rate
                floor = (limit + 1/c) / 2
                # For any nonnegative initial state S >= limit*(1-exp(-rate*t)).
                delay = np.log(limit / (limit - floor)) / rate + 1
                margin = (c*floor - 1) / c
                y = integrate(w, x, delay).y[:, -1]
                def entry(t, state):
                    return state[0] - margin
                entry.direction = -1
                entry.terminal = True
                elapsed = delay
                if y[0] > margin:
                    r = integrate(w, y, 10000, events=entry, max_step=1)
                    assert len(r.t_events[0]), (n, eps, delta, x)
                    elapsed += r.t_events[0][0]
                largest_entry_time = max(largest_entry_time, elapsed)
                total += 1
    print(f"{total} source trials entered the permanent decay region; "
          f"latest certified-barrier entry detected by t={largest_entry_time:.6f}")


def seven_node(certify=False):
    # The internal nonedges on vertices 0..4 are exactly the ones of P or Q.
    # All other internal edges are present. The three groups form a directed
    # cycle, with all possible edges between successive groups.
    p = np.eye(5)[[1, 2, 0, 0, 1]]
    q = np.eye(5)[[2, 4, 3, 0, 3]]
    def network(nonedge):
        edges = [(j, i) for i in range(5) for j in range(5) if i != j and not nonedge[i,j]]
        edges += [(j, 5) for j in range(5)] + [(5, 6)] + [(6, j) for j in range(5)]
        return ctln(7, edges, .3, 1.5)
    left, right = network(p), network(q)
    weights = [["-23/5", "-5/2", "-7/10"], ["-7/2", 0, "-5/2"], ["-25/2", "-7/10", 0]]
    w = np.array([[float(sp.Rational(entry)) for entry in row] for row in weights])
    lift = np.eye(3)[[0, 0, 0, 0, 0, 1, 2]]
    for graph in [left, right]:
        assert np.allclose(graph @ lift, lift @ w)
    def on(t, x):
        return 1 + w[0]@x
    r = integrate(w, [.11,.2,.03], 220, events=on, max_step=.08)
    times = r.t_events[0]
    rising = [(w[0]@(np.maximum(0, 1+w@v)-v)) > 0 for v in r.y_events[0]]
    start, end = times[rising][-2:]
    stop = times[(times > start) & (times < end)][0]
    active_fraction = (stop-start)/(end-start)
    assert .5074 < active_fraction < .5076
    print(f"7-node common orbit T={end-start:.12f}, active fraction={active_fraction:.12f}")
    # At the same phase, measure the splitting of the five clone rates under
    # a small generic perturbation, using the original seven-dimensional IVP.
    direction = np.array([1., -1., 2., -2., .5, 0, 0])
    initial = lift @ r.y_events[0][np.asarray(rising)][-1] + 1e-5*direction
    for s in (0., .5, 1.):
        graph = (1-s)*left + s*right
        observed = integrate(graph, initial, 12*(end-start), max_step=.10).y[:5, -1]
        spread_ratio = np.ptp(observed)/np.ptp(initial[:5])
        if s == .5:
            assert spread_ratio > 1
        else:
            assert spread_ratio < .001
        print(f"s={s:g}, clone spread ratio over 12 periods={spread_ratio:.8g}")
        lam = min(np.linalg.eigvals((1-s)*p+s*q).real)
        exponent = -1 + active_fraction*(.7-1.8*lam)
        print(f"s={s:g}, largest normal exponent={exponent:.12f}, "
              f"normal multiplier modulus={np.exp(exponent*(end-start)):.10f}")
    if certify:
        from flint import arb
        # Start with cell 1 switching off (zero-based quotient coordinates).
        off = [{0,1}, {1}, {1,2}, {2}, {0,2}, {0}]
        switches = [0, 2, 1, 0, 2, 1]
        seed = [.00806232, .34815294, .38871275, .30350681, 1.12513642,
                .54710472, 1.18756978, .45467706, 2.01723242]
        lengths = certify_orbit(seed, weights, [0, 1, 2], off, switches)
        fraction = sum(t for t, inactive in zip(lengths, off) if 0 not in inactive)/sum(lengths)
        assert arb(".5074") < fraction < arb(".5076")
        z = sp.symbols("z")
        for graph in (p, q):
            exact = sp.Matrix(graph.astype(int))
            assert sp.expand(exact.charpoly(z).as_expr() - z**2*(z-1)*(z**2+z+1)) == 0
        polynomial = 16*z**4 + 16*z**3 + 8*z**2 + 2*z - 1
        midpoint = sp.Matrix(p.astype(int) + q.astype(int))/2
        assert sp.expand(midpoint.charpoly(z).as_expr() - (z-1)*polynomial/16) == 0
        assert polynomial.subs(z, sp.Rational(-716,1000)) > 0
        assert polynomial.subs(z, sp.Rational(-715,1000)) < 0
        assert -1 + fraction*arb("1.6") < 0
        assert -1 + fraction*(arb(".7") + arb("1.8")*arb(".715")) > 0
        print("certified endpoint attraction and midpoint transverse instability")
        print("certified quotient period", sum(lengths).str(45))
        print("certified block-on fraction", fraction.str(45))


def interpolation():
    # Figure 14, att 4. The optional edges leave node 5.
    core = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]
    left = ctln(5, core + [(4, 0)])
    right = ctln(5, core + [(4, j) for j in (0, 1, 2)])
    x = np.array([.3, .28, .27, 0, 0])
    r = integrate(left, x, 500, max_step=.3)
    section = r.y[:, -1]
    times = np.linspace(0, 60, 3001)
    ref = integrate(left, section, 60, t_eval=times, max_step=.15).y
    error = 0
    for s in np.linspace(0, 1, 21):
        w = (1-s)*left + s*right
        data = integrate(w, section, 60, t_eval=times, max_step=.15).y
        error = max(error, np.max(abs(data - ref)))
    print(f"att 4, 21 matrices: max trajectory discrepancy {error:.3g}, "
          f"max fifth preactivation {np.max(1+left[4]@ref):.9f}, "
          f"max fifth activity {np.max(ref[4]):.3g}")
    assert error < 1e-8

    # Non-isomorphic CTLN endpoints, opposite orientations of the same
    # three-cycle up to relabeling, with a fourth (source) vertex.
    cycle = [(0, 1), (1, 2), (2, 0)]
    left = ctln(4, cycle + [(3, 0)])
    right = ctln(4, [(b, a) for a, b in cycle] + [(3, 0), (3, 1)])
    for s in (0., .5, 1.):
        w = (1-s)*left + s*right
        r = integrate(w, [.42, .15, .31, 0], 300, t_eval=np.linspace(250, 300, 501))
        print(f"relabeling example s={s:g}, neuron 1 range={np.ptp(r.y[0]):.9f}, "
              f"final state={r.y[:, -1]}")

    # Same exact, strictly positive fixed point at both endpoints of a
    # competitive TLN segment; its transverse stability is lost inside.
    p = sp.eye(5)[[1, 2, 0, 0, 1], :]
    q = sp.eye(5)[[2, 4, 3, 0, 3], :]
    z = sp.symbols("z")
    for s in (0, sp.Rational(1, 2), 1):
        r = (1-s)*p + s*q
        a = -sp.eye(5) - sp.Rational(3, 2)*r
        fixed = sp.ones(5, 1)*sp.Rational(2, 5)
        assert a*fixed + sp.ones(5, 1) == sp.zeros(5, 1)
        print(f"TLN s={s}, charpoly(R)={sp.factor(r.charpoly(z).as_expr())}, "
              f"spectral abscissa={max(np.linalg.eigvals(np.array(a, float)).real):.12f}")


def wording():
    path = (Path(__file__).resolve().parents[2] / "papers/curto_morrison_2301.12638"
            / "Notices-extended-jan29-final.tex")
    source = path.read_text()
    print(source.split(r'\item[14.] ', 1)[1].split(r'\end{enumerate}', 1)[0].strip())


if __name__ == "__main__":
    checks = dict(gaudi=gaudi, shooting=shooting, sources=sources,
                  interpolation=interpolation, seven=seven_node, wording=wording)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", nargs="?", choices=[*checks, "certificate", "all"], default="all")
    selection = parser.parse_args().check
    if selection == "certificate":
        shooting(certify=True)
        seven_node(certify=True)
    for name, check in checks.items():
        if selection in (name, "all"):
            check()
