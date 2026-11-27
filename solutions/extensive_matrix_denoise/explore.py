"""Seeded OAMP with population calibration. Run with the project .venv Python."""
import argparse
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, log_ndtr, roots_hermitenorm

ROOT2PI = np.sqrt(2 * np.pi)
GH, GW = roots_hermitenorm(128)
GW /= ROOT2PI
PRIORS = ("rademacher", "laplace", "student3", "gaussian")


def draw(name, rng, size):
    if name == "rademacher":
        return rng.choice([-1., 1.], size)
    if name == "student3":
        return rng.standard_t(3, size) / np.sqrt(3)
    if name == "laplace":
        return rng.laplace(0, 1 / np.sqrt(2), size)
    return rng.standard_normal(size)


def residual_density(name, r, v):
    """Return r-E[S|r] and density of S+N(0,v), for unit signal variance."""
    r = np.asarray(r)
    if name == "rademacher":
        density = (np.exp(-(r-1)**2/(2*v)) + np.exp(-(r+1)**2/(2*v))) / (2*ROOT2PI*np.sqrt(v))
        return r - np.tanh(r/v), density
    if name == "gaussian":
        return r*v/(1+v), np.exp(-r*r/(2*(1+v))) / (ROOT2PI*np.sqrt(1+v))
    if name == "laplace":
        rate = np.sqrt(2.)
        pos = -rate*r + log_ndtr((r-rate*v)/np.sqrt(v))
        neg = rate*r + log_ndtr((-r-rate*v)/np.sqrt(v))
        return rate*v*(2*expit(pos-neg)-1), (rate/2)*np.exp(v + np.logaddexp(pos, neg))

    # Gaussian convolution of p(s)=2/[pi(1+s^2)^2]. Batch the quadrature.
    shift = np.sqrt(v)*GH
    residual = np.empty(r.size)
    density = np.empty(r.size)
    for start in range(0, r.size, 1024):
        part = slice(start, start+1024)
        likelihood = (2/np.pi) / (1+(r.ravel()[part, None]-shift)**2)**2
        weights = likelihood * GW
        density[part] = weights.sum(axis=1)
        residual[part] = (weights @ shift) / density[part]
    return residual.reshape(r.shape), density.reshape(r.shape)


@lru_cache(None)
def scalar_mse(name, v):
    if name == "gaussian":
        return v/(1+v)

    def integrand(r):
        bias, density = residual_density(name, r, v)
        return float(bias*bias*density)

    # Total variance of the additive Gaussian noise.
    return v - 2*quad(integrand, 0, np.inf, epsabs=1e-10)[0]


def scalar_update(name, r, v, a):
    bias, _ = residual_density(name, r, v)
    return r - bias/(1-a)


def haar_coordinate(u, v):
    return v*u/(1-u*u) - u


@lru_cache(None)
def haar_profile(v):
    # On the band, u is the optimal shrinkage and the observation is s(u).
    c = v + v*v/4
    d_high = (c+np.sqrt((c-v)*(c+v)))/2
    d_low = v*v/(4*d_high)
    lo = np.sqrt(max(0., 1-d_high))
    hi = np.sqrt(1-d_low)

    def integrals(u):
        d = 1-u*u
        density = 2*np.sqrt(max(0., (d_high-d)*(d-d_low)))/(np.pi*v*d)
        jacobian = v*(1+u*u)/d**2-1
        return density*jacobian

    mass = quad(integrals, lo, hi, epsabs=1e-10)[0]
    assert abs(mass-1) < 1e-7, mass
    mse = 1-quad(lambda u: u*u*integrals(u), lo, hi, epsabs=1e-10)[0]
    return lo, hi, mse


def shrink(signal_model, s, v, extension="bounded", beta=1.):
    if signal_model == "mp":
        left = np.sqrt(1+v)*(1-np.sqrt(beta))
        right = np.sqrt(1+v)*(1+np.sqrt(beta))
        if extension == "hilbert":
            disc = np.maximum(0., (s*s-left*left)*(s*s-right*right))
            if beta < 1 and np.any(s == 0):
                raise ValueError("the rectangular Hilbert formula has a pole at zero")
            shift = np.divide(np.sqrt(disc), s, out=np.zeros_like(s), where=s > 0)
            shift *= np.sign(s*s-(1+beta)*(1+v))
            return (s+v*shift)/(1+v)
        return np.minimum(s, right)/(1+v)
    lo, hi, _ = haar_profile(v)
    lower = np.full_like(s, lo)
    upper = np.full_like(s, hi)
    for _ in range(50):
        mid = (lower+upper)/2
        left = haar_coordinate(mid, v) < s
        lower = np.where(left, mid, lower)
        upper = np.where(left, upper, mid)
    result = (lower+upper)/2
    edge = haar_coordinate(lo, v)
    if edge > 0:
        result = np.where(s < edge, lo*s/edge, result)
    if extension == "hilbert":
        # Outside the bands the real inverse of z(q) on its physical branch
        # gives f=2q-s. Below the gap use 0<=q<=q_min, above it q>=q_max.
        turning = np.sqrt(2*v+v*v/4)
        for mask, a, b in (
            (s < edge, np.zeros_like(s), np.full_like(s, np.sqrt(max(0., 1+v/2-turning)))),
            (s > haar_coordinate(hi, v), np.full_like(s, np.sqrt(1+v/2+turning)), s)
        ):
            target = s[mask]
            left, right = a[mask], b[mask]
            for _ in range(48):
                mid = (left+right)/2
                small = mid+v*mid/(mid*mid-1) < target
                left = np.where(small, mid, left)
                right = np.where(small, right, mid)
            result[mask] = left+right-target
    return result


def square_haar(rng, n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return np.sqrt(n)*q*np.sign(np.diag(r))


def joint_bound(s, r, ext, ref_s, ref_r, ref_ext):
    # Valid coupling in R^3. Match sorted prior quantile blocks, then sort
    # effective noise within each block. Reference and test have the same number of entries.
    width = s.shape[0]

    def order(x, y):
        blocks = np.argsort(x.ravel(), kind="stable").reshape(-1, width)
        within = np.argsort((y-x).ravel()[blocks], axis=1)
        return np.take_along_axis(blocks, within, axis=1).ravel()

    first, second = order(s, r), order(ref_s, ref_r)
    cost = sum(np.mean((x.ravel()[first]-y.ravel()[second])**2)
               for x, y in ((s, ref_s), (r, ref_r), (ext, ref_ext)))
    return np.sqrt(cost)


def trial(rng, n, name, model, steps, excess=0., extension="bounded", aspect=1.):
    m = n if model == "haar" else max(1, round(n*aspect))
    signal = square_haar(rng, n) if model == "haar" else rng.standard_normal((m, n))
    noise = draw(name, rng, signal.shape)
    y = signal+noise
    estimate = np.zeros_like(y)
    v = 1.
    for step in range(1, steps+1):
        sm = scalar_mse(name, v)
        a = sm/v
        w = sm/(1-a)
        r = y-estimate
        ext = scalar_update(name, r, v, a)
        inp = y-ext
        left, sv, right = np.linalg.svd(inp, full_matrices=False)
        ref_sv = np.linalg.svd(signal+np.sqrt(w)*rng.standard_normal(y.shape), compute_uv=False)
        spec_w2 = np.sqrt(np.mean((sv-ref_sv)**2)/n)
        denoised = (left*(np.sqrt(n)*shrink(model, sv/np.sqrt(n), w, extension, m/n))) @ right
        lm_bulk = haar_profile(w)[2] if model == "haar" else w/(1+w)
        lm = lm_bulk + excess
        b = lm/w
        if b >= 1:
            raise ValueError("expected-risk calibration has left the SE domain")
        estimate = (denoised-b*inp)/(1-b)
        next_v = lm/(1-b)
        ref_s = draw(name, rng, y.shape)
        ref_r = ref_s+np.sqrt(v)*rng.standard_normal(y.shape)
        ref_ext = scalar_update(name, ref_r, v, a)
        triple = joint_bound(noise, r, ext, ref_s, ref_r, ref_ext)
        metrics = [triple, spec_w2, np.mean((denoised-signal)**2)-lm,
                   np.mean((ext-noise)**2)-w, np.mean((estimate-signal)**2)-next_v,
                   np.mean(noise*(r-noise))]
        assert np.all(np.isfinite(metrics))
        yield step, np.array(metrics)
        v = next_v


def rare_check(rng, n, delta):
    # Conditional Monte Carlo, with exact mixture weights. The rare component
    # is n sqrt(delta) times sqrt(n) O(n), with probability 1/n^2.
    v = scalar_mse("rademacher", 1.)
    w = v/(1-v)
    bulk = w/(1+w)
    q = 1/n**2
    risks = []
    for signal in (rng.standard_normal((n, n)), n*np.sqrt(delta)*square_haar(rng, n)):
        noisy = signal+np.sqrt(w)*rng.standard_normal(signal.shape)
        u, sv, vt = np.linalg.svd(noisy, full_matrices=False)
        f = (u*(np.sqrt(n)*shrink("mp", sv/np.sqrt(n), w))) @ vt
        risks.append(np.mean((f-signal)**2))
    b = (bulk+delta)/w
    k = (1/(1+w)-b)/(1-b)
    return np.array([risks[0], (1-q)*risks[0]+q*risks[1], bulk, bulk+delta, -k*w])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--mp-aspect", type=float, default=1., help="aspect ratio of ordinary Gaussian signals")
    parser.add_argument("--extension", choices=("bounded", "hilbert"), default="bounded",
                        help="off-support spectral rule for the ordinary ensembles")
    parser.add_argument("--seed", type=int, default=261129179)
    args = parser.parse_args()
    if not 0 < args.mp_aspect <= 1:
        parser.error("--mp-aspect must be in (0,1]")
    rng = np.random.default_rng(args.seed)
    # Independently check the Student posterior quadrature at typical variances.
    for v in (0.3, 1.):
        for r in (0., 1., 5.):
            def kernel(z):
                return np.exp(-z*z/2)/(1+(r-np.sqrt(v)*z)**2)**2
            exact = np.sqrt(v)*quad(lambda z: z*kernel(z), -12, 12)[0]/quad(kernel, -12, 12)[0]
            assert abs(residual_density("student3", r, v)[0]-exact) < 2e-7
    print("model prior n t triple_coupling spectral_W2 risk_gap S_ext_gap L_ext_gap input_cross")
    for n in args.sizes:
        for model in ("haar", "mp"):
            for name in PRIORS:
                rows = [list(trial(rng, n, name, model, args.steps, extension=args.extension, aspect=args.mp_aspect)) for _ in range(args.reps)]
                for t in range(args.steps):
                    means = np.mean([row[t][1] for row in rows], axis=0)
                    print(model, name, n, t+1, *[f"{x:.6f}" for x in means])
        delta = 0.05
        stats = np.mean([rare_check(rng, n, delta) for _ in range(args.reps)], axis=0)
        # The ordinary branch has probability tending to one.
        rows = [list(trial(rng, n, "rademacher", "mp", 2, delta)) for _ in range(args.reps)]
        cross = np.mean([row[1][1][-1] for row in rows])
        print("rare n usual_risk mixture_risk bulk_target mixture_target predicted_cross observed_cross")
        print("rare", n, *[f"{x:.6f}" for x in [*stats, cross]])


if __name__ == "__main__":
    main()
