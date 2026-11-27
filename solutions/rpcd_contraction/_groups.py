"""All-permutation second moments for two exchangeable coordinate groups."""
import numpy as np


def group_operator(m, l, a, b, c):
    """Return the epoch map on the five quadratic statistics (acting on row vectors).

    The batch dimension enumerates choices of the three off-diagonal entries.
    The covariance axes refer to [S1,S2,R1,R2], the current group sums and
    remaining original group sums. Each of five inputs is propagated at once.
    """
    a, b, c = np.broadcast_arrays(*[np.atleast_1d(x) for x in (a, b, c)])
    batch = len(a)
    coupling = np.stack([a, b, b, c], axis=-1).reshape(batch, 2, 2)
    identity = np.eye(4)
    prev = np.zeros((1, batch, 5, 4, 4))
    for j in range(4):
        for k in range(4):
            stat = j%2 if j%2 == k%2 else 2
            prev[:, :, stat, j, k] = 1
    units = np.eye(5)
    prev_sq = np.broadcast_to(units[:, 3:], (1, batch, 5, 2)).copy()
    sizes = (m, l)
    for t in range(1, m+l+1):
        counts = np.arange(max(0, t-l), min(m, t)+1)
        cur = np.zeros((len(counts), batch, 5, 4, 4))
        cur_sq = np.zeros((len(counts), batch, 5, 2))
        for g in (0, 1):
            used = counts if g == 0 else t-counts
            dst = np.flatnonzero(used > 0)
            src = counts[dst] - (g == 0) - max(0, t-1-l)
            remaining = (sizes[g]-used[dst]+1)[:, None, None]
            prob = (used[dst]/t)[:, None, None]
            cov = prev[src]
            sq = prev_sq[src].copy()
            h = np.zeros((batch, 4))
            h[:, :2] = coupling[:, g]
            diag = coupling[:, g, g]
            update = np.broadcast_to(identity, (batch, 4, 4)).copy()
            update[:, g] -= h
            v = np.zeros((batch, 4))
            v[:, g] = diag-1
            v[:, g+2] = -1

            # New z = U z + v theta. E[z theta] = E[z R_g]/remaining.
            u = update[None, :, None]
            cross = (u @ cov[..., g+2, None]) / remaining[..., None, None]
            term = cross * v[None, :, None, None, :]
            old_sq = units[:, 3+g] / sizes[g]
            vv = v[:, :, None] * v[:, None, :]
            out = (u @ cov @ u.swapaxes(-1, -2) + term + term.swapaxes(-1, -2)
                   + old_sq[None, None, :, None, None] * vv[None, :, None])
            cur[dst] += prob[..., None, None] * out

            # Replace theta^2 by (diag * theta - h^T z)^2.
            hz = (h[None, :, None, None, :] @ cov)[..., 0, :]
            sq[..., g] += ((hz * h[None, :, None, :]).sum(axis=-1)
                           - 2*diag[None, :, None]*hz[..., g+2]/remaining
                           + (diag**2-1)[None, :, None]*old_sq)
            cur_sq[dst] += prob[..., None] * sq
        prev, prev_sq = cur, cur_sq
    op = np.stack([cur[0, ..., 0, 0], cur[0, ..., 1, 1], cur[0, ..., 0, 1],
                   cur_sq[0, ..., 0], cur_sq[0, ..., 1]], axis=-1)
    if m == 1:
        # For a singleton the square of the sum equals the sum of squares.
        op[:, 0] += op[:, 3]
        keep = [0, 1, 2, 4]
        op = op[:, keep, :][:, :, keep]
    return op


def search_group_models(full_radius, bound):
    for m, l in [(1, 2), (2, 3), (4, 3)]:
        for a, b, c in [(.3, .23, .54), (.8, .75, .8), (-.12, .15, .2)]:
            matrix = np.block([[np.full((m, m), a), np.full((m, l), b)],
                               [np.full((l, m), b), np.full((l, l), c)]])
            np.fill_diagonal(matrix, 1)
            radius = abs(np.linalg.eigvals(group_operator(m, l, a, b, c))).max()
            assert abs(radius - full_radius(matrix)) < 1e-11
    sizes = [(2, 2), (2, 5), (3, 13), (2, 30), (1, 64), (3, 64), (10, 25),
             (2, 256), (16, 64), (2, 60), (4, 150), (24, 24), (60, 60)]
    for index, (m, l) in enumerate(sizes):
        rng = np.random.default_rng(100*m+l if index < 9 else 1)
        if index < 9:
            sigma = 10**rng.uniform(-3, -.02, 160)
            a, c = [(1-sigma)*((q*rng.uniform(0, 1, len(sigma))-1)/(q-1)
                                if q > 1 else 0) for q in (m, l)]
            fraction = 1
        else:
            sigma = np.repeat([.0001, .005, .03, .06, .1, .2, .4, .5, .55, .7], 12)
            a = 1-sigma
            c = a * np.where(np.arange(len(sigma))%2 == 0, 1, rng.uniform(size=len(sigma)))
            fraction = 1-10**rng.uniform(-6, 0, len(sigma))
        b = fraction * np.sqrt((1-sigma+(m-1)*a)*(1-sigma+(l-1)*c)/(m*l))
        radius = abs(np.linalg.eigvals(group_operator(m, l, a, b, c))).max(axis=1)
        gap = (radius - np.array([bound(m+l, s) for s in sigma]))/sigma
        assert gap.max() < 0
        print(f"Groups ({m},{l}): {len(sigma)} trials, max (rho-bound)/sigma {gap.max():.9g}")
