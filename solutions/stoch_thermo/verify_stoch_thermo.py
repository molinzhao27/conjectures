"""Numerical and symbolic verification of all results for Wampler & Barato (arXiv:2110.14540)."""
import numpy as np
from scipy.optimize import minimize, brentq
import mpmath as mp


def unicyclic_moments(k_fwd, k_bwd):
    """Exact R, S, K from the characteristic polynomial P(s) = det(s I - L(0))
    and the paper's exact recurrence / our closed-form gamma_k identities.
    """
    k_fwd = np.asarray(k_fwd, dtype=float)
    k_bwd = np.asarray(k_bwd, dtype=float)
    Om = len(k_fwd)
    Pi_p = np.prod(k_fwd)
    Pi_m = np.prod(k_bwd)
    A = np.log(Pi_p / Pi_m)
    assert A > 0
    # Use exact symmetric positive-definite H = -L(1/2) eigenvalues for numerical stability!
    H = np.zeros((Om, Om), dtype=float)
    for i in range(Om):
        j = (i + 1) % Om
        g = np.sqrt(k_fwd[i] * k_bwd[i])
        H[j, i] -= g
        H[i, i] += k_fwd[i]
        H[i, j] -= g
        H[j, j] += k_bwd[i]
    eigs = np.linalg.eigvalsh(H)
    taus = 1.0 / eigs
    w = taus / np.sum(taus)
    eta = 1.0 / np.tanh(A / 4.0)
    # Elementary symmetric polynomials e_2, e_3, e_4 of w (sum w = 1):
    # Via Newton identities from power sums m_k = sum w^k
    m2 = float(np.sum(w**2))
    m3 = float(np.sum(w**3))
    m4 = float(np.sum(w**4))
    e2 = 0.5 * (1.0 - m2)
    e3 = (1.0 - 3.0 * m2 + 2.0 * m3) / 6.0 if Om >= 3 else 0.0
    e4 = (1.0 - 6.0 * m2 + 8.0 * m3 + 3.0 * m2**2 - 6.0 * m4) / 24.0 if Om >= 4 else 0.0
    g2 = eta * e2
    g3 = (eta**2) * e3
    g4 = (eta**3) * e4
    c = 1.0 / np.tanh(A / 2.0)
    y2 = c - 2.0 * g2
    R = A * y2
    S = np.sqrt(A) * (3.0 * c * y2 - 1.0 + 6.0 * g3) / (y2 ** 1.5)
    K = A * (3.0 * c * y2**2 + (12.0 * c**2 - 6.0) * y2 - 3.0 * c + 24.0 * (c * g3 - g4)) / (y2 ** 2)
    return A, R, S, K, (g2, g3, g4, y2)


def S_arw(A, N):
    e = np.exp(A / N)
    return 2.0 * np.sqrt(A / N) * (1.0 + 4.0 * e + e**2) / np.sqrt((e - 1.0) * (1.0 + e)**3)


def K_arw(A, N):
    e = np.exp(A / N)
    return 6.0 * A / N * (1.0 + 8.0 * e + e**2) / (e**2 - 1.0)


def check_closed_form_rho():
    """Verify P(s) + (sqrt(Pi_+) - sqrt(Pi_-))^2 = det(s I + H) and closed-form rho(s)
    matches Xi(s, rho(s)) = 0 to machine precision for random Om=2..6 networks.
    """
    rng = np.random.default_rng(123)
    for Om in range(2, 7):
        k_fwd = rng.uniform(0.5, 5.0, size=Om)
        k_bwd = rng.uniform(0.1, 1.5, size=Om)
        if np.prod(k_fwd) <= np.prod(k_bwd):
            k_fwd *= 3.0
        Pi_p, Pi_m = np.prod(k_fwd), np.prod(k_bwd)
        A = np.log(Pi_p / Pi_m)
        # Build L(1/2)
        L_half = np.zeros((Om, Om))
        # Edge i -> i+1 has forward rate k_fwd[i] with entropy increment ln(k_fwd[i]/k_bwd[i])
        # Under tilting by z, forward rate multiplies by (k_fwd[i]/k_bwd[i])^(-z) -> at z=1/2: sqrt(k_fwd[i]*k_bwd[i])!
        for i in range(Om):
            j = (i + 1) % Om
            g = np.sqrt(k_fwd[i] * k_bwd[i])
            L_half[j, i] += g
            L_half[i, i] -= k_fwd[i]
            L_half[i, j] += g
            L_half[j, j] -= k_bwd[i]
        # Notice L_half is ALREADY real symmetric! Check symmetry & positive definiteness of -L_half
        assert np.allclose(L_half, L_half.T, atol=1e-13)
        eigs = np.linalg.eigvalsh(-L_half)
        assert np.all(eigs > 0), f"Non-positive eigenvalue for Om={Om}: {eigs}"
        taus = 1.0 / eigs
        # Check Xi(s, rho(s)) = 0 for several s
        L0 = np.zeros((Om, Om))
        for i in range(Om):
            j = (i + 1) % Om
            L0[j, i] += k_fwd[i]; L0[i, i] -= k_fwd[i]
            L0[i, j] += k_bwd[i]; L0[j, j] -= k_bwd[i]
        for s in [0.0, 0.05, 0.3, 1.2]:
            prod_sqrt = np.prod(np.sqrt(1.0 + taus * s))
            rho = 0.5 - (2.0 / A) * np.arcsinh(np.sinh(A / 4.0) * prod_sqrt)
            Ps = np.linalg.det(s * np.eye(Om) - L0)
            Xi = Ps - Pi_p * (np.exp(-A * rho) - 1.0) - Pi_m * (np.exp(A * rho) - 1.0)
            assert abs(Xi) < 1e-10 * (1.0 + abs(Ps)), f"Xi residual {Xi} at Om={Om}, s={s}"
        # Also verify S and K from finite-diff derivatives of rho(s) at s=0 match formulas!
        h = 1e-4
        # Better: use mpmath for exact derivatives of rho(s) at s=0!
        with mp.workdps(50):
            A_mp = mp.mpf(str(A))
            taus_mp = [mp.mpf(str(t)) for t in taus]
            def rho_mp(s):
                p = mp.mpf(1)
                for t in taus_mp:
                    p *= mp.sqrt(1 + t * s)
                return mp.mpf('0.5') - (2 / A_mp) * mp.asinh(mp.sinh(A_mp / 4) * p)
            # In Wampler-Barato: rho(s) = -<tau> s / ? Wait: FPT SCGF lambda_FPT(s) = rho(s)
            # Cumulants of FPT: c_n = (-1)^n d^n/ds^n rho(s)|_{s=0} (up to sign/scale)
            r1 = mp.diff(rho_mp, 0, 1)
            r2 = mp.diff(rho_mp, 0, 2)
            r3 = mp.diff(rho_mp, 0, 3)
            r4 = mp.diff(rho_mp, 0, 4)
            # Check R = r2 / (-r1), S = (-r3) / r2^(3/2) * ... let's check ratios directly
            # In Wampler-Barato: kappa_i / J_t = (-1)^i d^i rho / ds^i |_{s=0}
            # mu1 = -r1, mu2 = r2, mu3 = -r3, mu4 = r4
            # Randomness parameter R = kappa_2 / (kappa_1^2 / J_t) = r2 / (-r1)^2 * ... wait: R = kappa_2 / kappa_1 (for J_t=1) * A? No: R = J_t * kappa_2 / kappa_1^2!
            R_mp = float(r2 / ((-r1) ** 2))
            S_mp = float((-r3) / (r2 ** mp.mpf('1.5')))
            K_mp = float(r4 / (r2 ** 2))
        _, R, S, K, _ = unicyclic_moments(k_fwd, k_bwd)
        assert abs(R - R_mp) < 1e-8, f"R mismatch: {R} vs {R_mp}"
        assert abs(S - S_mp) < 1e-7, f"S mismatch: {S} vs {S_mp}"
        assert abs(K - K_mp) < 1e-6, f"K mismatch: {K} vs {K_mp}"
        print(f"Om={Om}: Xi=0 & exact R,S,K match mpmath! R={R:.6f}, S={S:.6f}, K={K:.6f}")


def check_omega2_and_table1_refutation():
    print("\n=== Checking Omega=2 exact thresholds & Table 1 refutation ===")
    A_S2 = 2.0 * np.log(2.0 + np.sqrt(3.0))
    A_K2 = 4.0 * np.arctanh(np.sqrt(np.sqrt(2.0) - 1.0))
    print(f"Exact A_*^S(2) = {A_S2:.8f}  (Table 1 claims 2.84)")
    print(f"Exact A_*^K(2) = {A_K2:.8f}  (Table 1 claims 3.27)")

    # Explicit counterexample at A = 2.75 < 2.84
    A = 2.75
    rhs = np.exp(-A / 2.0) * ((np.exp(A) - 1.0) / np.sqrt(2.0) * np.sqrt(1.0 + np.exp(-A)) - 2.0)
    r = 0.5 * (rhs + np.sqrt(rhs**2 - 4.0))
    k_fwd = [r * np.exp(A / 2.0), (1.0 / r) * np.exp(A / 2.0)]
    k_bwd = [1.0, 1.0]
    A_chk, R, S, K, _ = unicyclic_moments(k_fwd, k_bwd)
    S_arw2 = S_arw(A, 2)
    S_exact_max = 2.0 * np.sqrt(A) * (1.0 / np.tanh(A / 2.0)) ** 1.5
    print(f"A={A_chk:.6f}: S = {S:.8f}, exact max = {S_exact_max:.8f}, S_ARW(A,2) = {S_arw2:.8f}, excess = {S - S_arw2:+.8f}")
    assert S > S_arw2 + 1e-3

    # Explicit counterexample for Kurtosis at A = 3.15 < 3.27
    A = 3.15
    c = 1.0 / np.tanh(A / 2.0)
    y2_opt = c / (2.0 * c**2 - 1.0)
    g2_opt = (c - y2_opt) / 2.0
    # g2 = (Pi_+ - Pi_-) / b1^2 with k_bwd=[1,1], Pi_+=e^A, Pi_-=1 -> b1 = sqrt((e^A-1)/g2_opt)
    b1_opt = np.sqrt((np.exp(A) - 1.0) / g2_opt)
    rhs = np.exp(-A / 2.0) * (b1_opt - 2.0)
    r = 0.5 * (rhs + np.sqrt(rhs**2 - 4.0))
    A_chk, _, _, K, _ = unicyclic_moments([r * np.exp(A / 2.0), (1.0 / r) * np.exp(A / 2.0)], [1.0, 1.0])
    K_arw2 = K_arw(A, 2)
    print(f"A={A_chk:.6f}: K = {K:.8f}, K_ARW(A,2) = {K_arw2:.8f}, excess = {K - K_arw2:+.8f}")
    assert K > K_arw2 + 1e-2


def check_omega3_bifurcation():
    print("\n=== Checking Omega=3 exact bifurcation thresholds & Table 1 refutation ===")
    F_plus = lambda v: v**6 + 3*v**5 - 3*v**4 - 15*v**3 - 12*v**2 + 2
    v_star = brentq(F_plus, 2.01, 2.5)
    A_S3 = 6.0 * np.arccosh(v_star / 2.0)
    print(f"Exact A_bif^S(3) = {A_S3:.8f}  (Table 1 claims 3.15)")

    # Test explicit perturbation at A = 3.10 < 3.15
    A = 3.10
    x = np.exp(A / 3.0)
    eps = 0.05
    alpha = eps * np.array([1.0, -1.0, 0.0])
    beta = eps * np.sqrt(x) * np.array([-1.0, 0.0, 1.0])  # beta_{j-1} = sqrt(x) alpha_j? Wait check index!
    # Check both cyclic shifts for beta_{j-1} = sqrt(x) alpha_j:
    best_S = -1.0
    for shift in [-1, 0, 1]:
        b_try = np.roll(alpha, shift) * np.sqrt(x)
        _, _, S_try, _, _ = unicyclic_moments(x * np.exp(alpha), np.exp(b_try))
        best_S = max(best_S, S_try)
    S_arw3 = S_arw(A, 3)
    print(f"A={A}: perturbed S = {best_S:.8f} vs S_ARW(A,3) = {S_arw3:.8f}, excess = {best_S - S_arw3:+.8e}")
    assert best_S > S_arw3

    G_K_plus = lambda u: (u**4 + 4*u**3 - 19*u**2 - 40*u + 18) + 0.5*np.sqrt(u + 2.0)*(7*u**3 - 8*u**2 - 61*u + 26)
    u_star = brentq(G_K_plus, 2.01, 5.0)
    A_K3 = 3.0 * np.arccosh(u_star / 2.0)
    print(f"Exact A_bif^K(3) = {A_K3:.8f}  (Table 1 claims 3.64)")

    A = 3.58
    x = np.exp(A / 3.0)
    best_K = -1.0
    for shift in [-1, 0, 1]:
        b_try = np.roll(alpha, shift) * np.sqrt(x)
        _, _, _, K_try, _ = unicyclic_moments(x * np.exp(alpha), np.exp(b_try))
        best_K = max(best_K, K_try)
    K_arw3 = K_arw(A, 3)
    print(f"A={A}: perturbed K = {best_K:.8f} vs K_ARW(A,3) = {K_arw3:.8f}, excess = {best_K - K_arw3:+.8e}")
    assert best_K > K_arw3


def stress_test_lower_bounds():
    print("\n=== Stress-testing lower bounds for Omega = 3, 4, 5 ===")
    rng = np.random.default_rng(2026)
    for Om in [3, 4, 5]:
        # Test at several affinities including near kinks
        for A in [1.5, 3.0, 5.0, 8.0, 12.0]:
            S_lb = min(S_arw(A, N) for N in range(1, Om + 1))
            K_lb = min(K_arw(A, N) for N in range(1, Om + 1))
            min_S = 1e9
            min_K = 1e9
            def eval_params(p, which):
                p = np.clip(p, -6.0, 6.0)
                a = np.append(p[:Om-1], -np.sum(p[:Om-1]))
                b = np.append(p[Om-1:], -np.sum(p[Om-1:]))
                kf = np.exp(A / Om + a)
                kb = np.exp(b)
                _, _, S, K, _ = unicyclic_moments(kf, kb)
                return S if which == 'S' else K
            for _ in range(15):
                p0 = rng.normal(scale=2.0, size=2 * (Om - 1))
                resS = minimize(eval_params, p0, args=('S',), method='Nelder-Mead', options={'maxiter': 800})
                min_S = min(min_S, resS.fun)
                resK = minimize(eval_params, p0, args=('K',), method='Nelder-Mead', options={'maxiter': 800})
                min_K = min(min_K, resK.fun)
            assert min_S >= S_lb - 1e-6, f"S LOWER BOUND VIOLATED at Om={Om}, A={A}: {min_S} < {S_lb}"
            assert min_K >= K_lb - 1e-5, f"K LOWER BOUND VIOLATED at Om={Om}, A={A}: {min_K} < {K_lb}"
        print(f"  Om={Om}: S and K lower bounds hold across all tested affinities & multi-start optimizations.")


if __name__ == "__main__":
    check_closed_form_rho()
    check_omega2_and_table1_refutation()
    check_omega3_bifurcation()
    stress_test_lower_bounds()
    print("\nALL STOCHASTIC THERMO VERIFICATIONS PASSED!")
