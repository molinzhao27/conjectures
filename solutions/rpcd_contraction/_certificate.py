"""Integer coefficient certificate for all 3-by-3 normalized Hessians."""
from functools import lru_cache
from itertools import permutations, product
from math import comb, factorial

import numpy as np


def angle(v):
    # Denominator, cosine numerator, sine numerator for tan(alpha/2)=v/(1+v).
    return (1+v)**2 + v*v, 1+2*v, 2*v*(1+v)


def invariants(u, v, w):
    b, c, d = angle(v)
    f, g, h = angle(w)
    den = (1+u)*b*f
    a = np.array([[den, c*f, c*g-d*h], [c*f, den, b*g], [c*g-d*h, b*g, den]],
                 dtype=object)
    identity = np.eye(3, dtype=object)
    gram_sum = np.zeros((3, 3), dtype=object)
    for i, j, k in permutations(range(3)):
        inv = identity * den**2
        inv[j, i] = -a[j, i]*den
        inv[k, j] = -a[k, j]*den
        inv[k, i] = a[k, j]*a[j, i] - a[k, i]*den
        gram_sum += inv.T @ inv
    loss = (2*(1+u))**4 - (2+u)**4
    matrix = 16*(gram_sum @ a) - (6*loss*(1+u)*(b*f)**5)*identity
    trace = matrix.trace()
    determinant = sum(
        matrix[0, i] * (matrix[1, (i+1)%3]*matrix[2, (i+2)%3]
                        - matrix[1, (i+2)%3]*matrix[2, (i+1)%3])
        for i in range(3))
    return trace, (trace**2 - (matrix @ matrix).trace())//2, determinant


@lru_cache(None)
def scaled_inverse_vandermonde(degree):
    """Integer matrix of power coefficients of d! times the Lagrange bases."""
    matrix = np.empty((degree+1, degree+1), dtype=object)
    for i in range(degree+1):
        polynomial = [1]
        for j in range(degree+1):
            if i != j:
                polynomial = [a-j*b for a, b in zip([0]+polynomial, polynomial+[0])]
        scale = comb(degree, i) * (-1)**(degree-i)
        matrix[:, i] = [scale*v for v in polynomial]
    return matrix


def verify_three_dimensions():
    grid = np.empty((16, 31, 31, 3), dtype=object)
    for u, v, w in product(range(16), range(31), range(31)):
        # Swapping v and w just exchanges the first and last coordinates.
        grid[u, v, w] = grid[u, w, v] if v > w else invariants(u, v, w)
    for k, expected_positive in enumerate((726, 4848, 14412), start=1):
        degrees = (5*k, 10*k, 10*k)
        coeff = grid[:degrees[0]+1, :degrees[1]+1, :degrees[2]+1, k-1]
        for axis, degree in enumerate(degrees):
            coeff = np.moveaxis(coeff, axis, -1)
            scaled = coeff @ scaled_inverse_vandermonde(degree).T
            assert np.all(scaled % factorial(degree) == 0)
            coeff = np.moveaxis(scaled // factorial(degree), -1, axis)
        assert np.all(coeff >= 0)
        assert np.count_nonzero(coeff > 0) == expected_positive
        print(f"Dimension-three certificate e{k}: {expected_positive} positive,"
              f" {np.count_nonzero(coeff == 0)} zero, 0 negative coefficients")
