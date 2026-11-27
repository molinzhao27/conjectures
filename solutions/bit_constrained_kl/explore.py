"""Numerical checks for the exact one-round Gaussian reverse-channel coefficients.

Run with Python 3, NumPy and SciPy. All calculations are deterministic.
"""

from math import exp, inf, pi, sqrt, tanh

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, rel_entr, roots_hermitenorm, softmax, xlog1py


def coefficients(snr, dimension):
    """Evaluate the common exact integral proved in results.md."""
    if dimension < 1 or snr < 0:
        raise ValueError("Require a positive dimension and nonnegative SNR")
    if snr == 0:
        return 0.0, 0.0

    def integrand(z):
        posterior = tanh(snr + sqrt(snr) * z)
        return posterior**2 * exp(-0.5 * z**2) / sqrt(2 * pi)

    value, _ = quad(integrand, -inf, inf, epsabs=1e-15, epsrel=2e-13)
    return value, value


def sign_divergence(mean):
    """KL from a sign with the given mean to a fair sign, in nats."""
    return (xlog1py(1 + mean, mean) + xlog1py(1 - mean, -mean)) / 2


def check_scalar(snr, exact):
    nodes, weights = roots_hermitenorm(256)
    law = np.tile(weights / (2 * sqrt(2 * pi)), 2)
    w = np.concatenate([np.tanh(v * snr + sqrt(snr) * nodes) for v in (-1, 1)])
    np.testing.assert_allclose(law @ (w**2), exact, rtol=3e-12)
    # A local KL witness with density 1 + eps*w. Symmetry pairs w and -w.
    eps = 0.001
    local = sign_divergence(eps * exact) / (law @ sign_divergence(eps * w))
    np.testing.assert_allclose(local, exact, rtol=2e-6)

    # The entropy-minimizing alternatives at fixed output mean are exponential
    # tilts in w. Compute both divergences directly at nonzero tilts.
    for scale in np.geomspace(0.02, 8.0, 81):
        tilt = scale / sqrt(exact)
        alternative = softmax(np.log(law) + tilt * w)
        mean = alternative @ w
        input_kl = rel_entr(alternative, law).sum()
        assert sign_divergence(mean) <= exact * input_kl * (1 + 1e-8)
        assert mean <= tanh(exact * tilt) + 1e-13


def check_product(snr, dimension, exact, rng):
    # Coarsen the real observation into six bins. Lifting a bin likelihood
    # ratio back to Y preserves its input KL and its output distribution.
    edges = np.array([-inf, -2.0, -0.8, 0.0, 0.8, 2.0, inf])
    conditional = np.diff(ndtr(edges[:, None] - sqrt(snr) * np.array([-1, 1])), axis=0)
    marginal = conditional.mean(axis=1)
    reverse = conditional / (2 * marginal[:, None])
    scalar_chi2 = marginal @ (reverse[:, 1] - reverse[:, 0])**2
    law, channel = marginal, reverse
    for _ in range(1, dimension):
        law = np.kron(law, marginal)
        channel = np.kron(channel, reverse)

    reference_output = law @ channel
    np.testing.assert_allclose(reference_output, 2.0**(-dimension), atol=1e-14)
    weighted = np.sqrt(law[:, None] / reference_output) * channel
    singular = np.linalg.svd(weighted, compute_uv=False)
    np.testing.assert_allclose(singular[1]**2, scalar_chi2, rtol=1e-10)
    assert scalar_chi2 <= exact + 1e-14

    for strength in (0.05, 0.5, 2.0, 8.0):
        for _ in range(5):
            score = rng.normal(size=law.size)
            alternative = softmax(np.log(law) + strength * score)
            output_kl = rel_entr(alternative @ channel, reference_output).sum()
            input_kl = rel_entr(alternative, law).sum()
            assert output_kl <= exact * input_kl + 2e-14


def main():
    snrs = (1e-6, 1e-4, 0.01, 0.1, 0.25, 0.5, 1.0)
    dimensions = (1, 2, 3, 4)
    rng = np.random.default_rng(260600703)
    print("SNR       dimensions   eta_KL          eta_chi2        eta_KL/SNR")
    for snr in (0.0, *snrs):
        for d in dimensions:
            kl, chi2 = coefficients(snr, d)
            assert 0 <= kl <= snr
            if snr > 0:
                check_product(snr, d, kl, rng)
        if snr > 0:
            check_scalar(snr, kl)
        ratio = f"{kl / snr:.12f}" if snr else "undefined"
        print(f"{snr:<9g} 1,2,3,4      {kl:.12f}  {chi2:.12f}  {ratio}")
    assert coefficients(1e-6, 1)[0] / 1e-6 > 0.999998
    print("Passed 567 scalar tilts, 28 product spectra and 560 joint KL checks.")


if __name__ == "__main__":
    main()
