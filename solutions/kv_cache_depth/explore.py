"""Exact finite checks for the product depth bound. Run with Python 3."""
from fractions import Fraction
from itertools import permutations, product


def endpoint(pi, k):
    """Use one-based labels and an unused entry at index zero."""
    current = 1
    for _ in range(k):
        current = pi[current]
    return current


def check_outputs():
    cases = 0
    for n in range(4, 9):
        labels = set(range(1, n + 1))
        for k in range(1, n // 4 + 1):
            outputs = set()
            for perm in permutations(sorted(labels)):
                outputs.add(endpoint((0,) + perm, k))
                cases += 1
            assert outputs == labels
    return cases


def check_adversary(n, k):
    """Branch over every address at every step, including repeated probes."""
    labels = set(range(1, n + 1))

    def visit(path, answers, depth):
        fresh = sorted(labels - set(answers) - set(path))
        missing = k + 1 - len(path)
        assert missing >= 1 and len(fresh) >= missing + 1
        prefix = path + fresh[:missing - 1]
        outputs = []
        for last in fresh[missing - 1:missing + 1]:
            cycle = prefix + [last]
            pi = list(range(n + 1))
            for source, target in zip(cycle, cycle[1:] + [1]):
                pi[source] = target
            assert set(pi[1:]) == labels
            assert all(pi[i] == value for i, value in answers.items())
            outputs.append(endpoint(pi, k))
        assert outputs[0] != outputs[1]

        count = 1
        if depth < k - 1:
            for i in sorted(labels):
                if i in answers:
                    count += visit(path, answers, depth + 1)
                elif i == path[-1]:
                    target = fresh[0]
                    count += visit(path + [target], {**answers, i: target}, depth + 1)
                else:
                    count += visit(path, {**answers, i: i}, depth + 1)
        return count

    return visit([1], {}, 0)


def check_grid():
    """Check the two necessary inequalities and the product for each tuple."""
    small = large = infeasible = 0
    for n in range(4, 65):
        label_bits = (n - 1).bit_length()
        for k, s, heads, width, precision in product(
            range(1, n // 4 + 1), range(1, n + 1), (1, 2), (1, 2, 4), (1, 2, 4)
        ):
            if 2 ** (width * precision) < n:
                infeasible += 1
                continue
            bits = heads * width * precision
            bandwidth = (label_bits + bits - 1) // bits
            layers = (k + s - 1) // s
            assert bandwidth == 1
            assert (layers - 1) * s < k <= layers * s
            assert layers * bandwidth == layers
            if 16 * s * s <= n:
                small += 1
            else:
                large += 1
    return small, large, infeasible


def two_layer_circuit(pi):
    """Singleton attention with identity residual and two scratch coordinates."""
    states = [(i, value, 0, 0) for i, value in enumerate(pi)]
    trace = {0}
    for slot in (2, 3):
        address = 1 if slot == 2 else states[0][2]
        trace.add(address)
        value = states[address][1]
        states = [state[:slot] + (state[slot] + value,) + state[slot + 1:]
                  for state in states]
    return states[0][3], trace


def check_trace_example(n):
    """Each pair of first two destinations determines the circuit trace."""
    labels = set(range(1, n + 1))
    patterns = 0
    good_mass = hit_mass = expected_size = Fraction(0)
    for first in sorted(labels):
        seconds = [1] if first == 1 else sorted(labels - {first})
        for second in seconds:
            specified = {1: first, first: second}
            remaining = iter(sorted(labels - set(specified.values())))
            pi = [0] + [specified[i] if i in specified else next(remaining)
                        for i in sorted(labels)]
            assert set(pi[1:]) == labels
            output, trace = two_layer_circuit(pi)
            assert output == endpoint(pi, 2) == second
            weight = Fraction(1, n * len(seconds))
            expected_size += weight * len(trace)
            if len({1, first, second}) == 3:
                good_mass += weight
                hit_mass += weight * (first in trace)
            patterns += 1
    assert good_mass == Fraction(n - 2, n)
    lhs = hit_mass / good_mass
    rhs = expected_size / (n - 1)
    assert lhs == 1 and rhs == Fraction(3 * n - 1, n * (n - 1)) < lhs
    return patterns, lhs, rhs


def main():
    print("Permutation/hop cases with full output range:", check_outputs())
    prefixes = sum(check_adversary(n, k) for n, k in
                   ((4, 1), (8, 2), (12, 3), (16, 4), (20, 5)))
    print("Adversarial probe prefixes with two verified completions:", prefixes)
    small, large, infeasible = check_grid()
    print("Output-capacity-compatible tuples (small cache, large cache):", small, large)
    print("Tuples excluded by output cardinality:", infeasible)
    patterns, lhs, rhs = check_trace_example(16)
    print(f"Two-layer circuit: {patterns} destination patterns; star compares {lhs} <= {rhs} (false)")


if __name__ == "__main__":
    main()
