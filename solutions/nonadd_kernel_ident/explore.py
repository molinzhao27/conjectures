"""Exact checks for the two scaled-uniform candidate families."""

from fractions import Fraction as R


def base_width(s):
    return 4 - min(s, 3)


def split_width(s):
    return s - 2 if 2 < s <= 3 else base_width(s)


def conditional_cdf(x, theta, s, width):
    return min(R(1), max(R(0), (x - theta - s) / width(s)))


def tile_mixture_cdf(x, theta):
    """Sum tile contributions, including the whole infinite tail exactly."""
    if x <= theta + 2:
        return R(0)
    value = R(0)
    weight = R(1, 2)
    while True:
        s = 2 + 2 * weight
        if x >= theta + s + split_width(s):
            return value + 2 * weight  # current and all smaller tiles
        value += weight * conditional_cdf(x, theta, s, split_width)
        weight /= 2


def check_collision(theta):
    offsets = {R(i, 61) for i in range(-61, 306)}
    for n in range(1, 81):
        weight = R(1, 2**n)
        s = 2 + 2 * weight
        assert split_width(s) == 2 * weight
        offsets.update((s, s + weight, s + split_width(s)))
    for offset in offsets:
        x = theta + offset
        assert tile_mixture_cdf(x, theta) == conditional_cdf(
            x, theta, R(2), split_width
        )
    return len(offsets)


def check_local_recovery():
    # Latent density c * (4-s) * s^k on (0,1), remaining mass at 2.
    # The observed density at theta+t is the integral of f(s)/w(s).
    c = R(1, 8)
    count = 0
    for k in range(5):
        for t in (R(1, 64), R(1, 4), R(1, 2), R(3, 4)):
            assert split_width(t) == 4 - t
            true_mass = c * (4 * t ** (k + 1) / (k + 1) - t ** (k + 2) / (k + 2))
            observed_density = c * t ** (k + 1) / (k + 1)
            integrated_density = c * t ** (k + 2) / ((k + 1) * (k + 2))
            recovered_mass = (4 - t) * observed_density + integrated_density
            assert recovered_mass == true_mass
            # Recover the latent density from (4-t) times the derivative.
            recovered_density = (4 - t) * c * t**k
            assert recovered_density / t**k == 4 * c - c * t
            count += 1
    return count


def check_global_recovery(theta):
    atoms = {R(0): R(1, 11), R(1, 2): R(3, 11), R(2): R(2, 11), R(3): R(5, 11)}
    jump_sizes = {}
    for s, mass in atoms.items():
        left = theta + s
        right = left + base_width(s)
        height = mass / base_width(s)
        jump_sizes[left] = jump_sizes.get(left, 0) + height
        jump_sizes[right] = jump_sizes.get(right, 0) - height
    recovered = {
        x - theta: (4 - (x - theta)) * jump
        for x, jump in jump_sizes.items()
        if x < theta + 4
    }
    assert recovered == atoms
    assert sum(jump_sizes.values()) == 0


def main():
    count = 0
    parameters = (R(1), R(3, 2), R(2))
    for theta in parameters:
        count += check_collision(theta)
        check_global_recovery(theta)
    local_count = check_local_recovery()
    print(f"Passed {count} exact mixture CDF comparisons across {len(parameters)} edges.")
    print(f"Passed {local_count} local recovery checks and all capped atomic inversions.")


if __name__ == "__main__":
    main()
