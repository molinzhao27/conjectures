"""Run with Python 3. No downloads, checkpoint, or third-party dependencies."""
from collections import Counter
from fractions import Fraction as F
from itertools import permutations, product
from math import acos, cos, erf, exp, hypot, isclose, log2, pi, sin, sqrt
from random import Random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def axis(i, n):
    return tuple(int(i == j) for j in range(n))


def mi_bits(observations):
    joint = Counter(observations)
    first = Counter(x for x, y in observations)
    second = Counter(y for x, y in observations)
    n = len(observations)
    return sum(count / n * log2(count * n / (first[x] * second[y]))
               for (x, y), count in joint.items())


def codebooks():
    n = 5
    states = [axis(i, n) for i in range(n)]
    provenance = Counter()
    truth = Counter()
    for perm in permutations(range(n)):
        decoder = {}
        for i, tag in enumerate(perm):
            decoder[tag, "blue"] = states[i]
            decoder[tag, "red"] = states[i]
            decoder[tag, "green"] = states[(i + 1) % n]
        assert isclose(mi_bits(list(enumerate(perm))), log2(n))
        for i, tag in enumerate(perm):
            h = states[i]
            target = dot(h, decoder[tag, "blue"])
            assert target == 1
            for other in perm:
                if other != tag:
                    provenance[target - dot(h, decoder[other, "blue"])] += 1
            for word in ("red", "green"):
                truth[word, target - dot(h, decoder[tag, word])] += 1
    assert provenance == {1: 2400}
    assert truth == {("red", 0): 600, ("green", 1): 600}
    print("120 codebooks: 2400/2400 provenance wins; 600 red ties, 600 green wins.")

    # Balanced candidate experiment at each fixed target radius.
    for radius in (1, 2):
        for keep_radius in (False, True):
            labels = [(int(r == radius), r if keep_radius else 0) for r in (1, 2)]
            assert mi_bits(labels) == int(keep_radius)
        assert mi_bits([(1, "blue"), (0, "red")]) == 1
    print("Conditional provenance information: 0 or 1 bit; candidate truth information: 1 bit.")


def radial_truth():
    # A direction and an independent fair magnitude bit. The true candidate is
    # presented independently of the AV wording in this Setting 2 variant.
    for direction in (-1, 1):
        observations = []
        for large, candidate_large in product((0, 1), repeat=2):
            h = direction * (1 + large)
            z = 1 if h > 0 else -1
            assert h * z / abs(h) == 1
            observations.append((int(large == candidate_large), (z, candidate_large)))
        assert mi_bits(observations) == 0
    print("Independent magnitude queries: zero distortion and zero answer information.")


def gaussian():
    rng = Random(261018727)
    n = 200_000
    alpha = pi / 24
    wins = retrieved = 0
    score_total = 0.
    for _ in range(n):
        x, y = rng.gauss(0, 1), rng.gauss(0, 1)
        r = hypot(x, y)
        sign = 1 if x >= 0 else -1
        correct = sign * x / r
        rival = sign * (cos(alpha) * x + sin(alpha) * y) / r
        score_total += correct
        retrieved += correct > -correct
        wins += correct > rival
    expected_auc = 0.5 + alpha / (2 * pi)
    assert retrieved == n
    assert abs(wins / n - expected_auc) < 0.005
    assert abs(score_total / n - 2 / pi) < 0.005
    print(f"2-symbol Gaussian: {retrieved}/{n} provenance; truth {wins/n:.6f} "
          f"(exact 25/48); loss {1-score_total/n:.6f} (exact 1-2/pi).")

    # Numerical integration of the pointwise optimal encoder for two AR outputs.
    angles = [2 * pi * (k + 0.5) / 4096 for k in range(4096)]
    for j in range(33):
        separation = pi * j / 32
        average = sum(max(cos(t), cos(t - separation)) for t in angles) / len(angles)
        assert abs(average - 2 / pi * sin(separation / 2)) < 1.e-6

    # External Gaussian readout noise. Integrate the density independently of erf.
    cases = 0
    for theta, offset, sigma in product((0.001, 0.02, 0.2, 1.0),
                                         (0., 0.3, 1.1, 2.7), (0.1, 0.5, 2.0)):
        margin = cos(offset) - cos(offset - theta)
        chord = 2 * sin(theta / 2)
        assert abs(margin) <= chord + 1.e-15
        x = margin / sigma
        chance = (1 + erf(x / sqrt(2))) / 2
        dx = x / 2048
        integral = 0.5 + dx * sum(exp(-((k + 0.5) * dx)**2 / 2)
                                  for k in range(2048)) / sqrt(2 * pi)
        assert abs(chance - integral) < 1.e-7
        assert abs(chance - 0.5) <= chord / (sigma * sqrt(2 * pi)) + 1.e-15
        cases += 1
    print(f"Gaussian quadrature: 33 decoder separations and {cases} noisy margins passed.")


def form_bounds():
    eps = F(1, 8)
    grid = [k * eps for k in range(-5, 6)]
    checks = 0
    for gap, e0, ek in product(grid, (-eps, 0, eps), (-eps, 0, eps)):
        win = gap + e0 - ek > 0
        assert (gap > 2 * eps) <= win <= (gap > -2 * eps)
        checks += 1
    for sign in (-1, 1):
        gap = sign * eps
        assert (gap > 0) != (gap - 2 * sign * eps > 0)
    print(f"Form bounds: {checks} exact margin checks, including both endpoints.")

    original = ["blue"] + ["filler"] * 49
    def edited(index, value):
        tokens = original.copy()
        tokens[index] = value
        assert sum(x != y for x, y in zip(original, tokens)) == 1
        return tokens

    def rewrite_score(tokens):
        return {"red": F(1, 5), "scarlet": F(4, 5)}.get(tokens[0], F(3, 5))

    for replacement in ("native", "external"):
        assert rewrite_score(edited(1, replacement)) == rewrite_score(original)
    native = rewrite_score(edited(0, "red"))
    foreign = rewrite_score(edited(0, "scarlet"))
    target = rewrite_score(original)
    assert native < target < foreign and foreign - native == F(3, 5)
    print("One edit in 50: zero control damage, rival form shift 3/5, ranking reversed.")


def audit_gap():
    words = [("text", "indeed", "today"), ("text", "certainly", "today")]
    def lexical_receiver(message):
        return int("certainly" in message)
    def best_accuracy(messages, rules):
        return max(F(sum(rule(m) == b for b, m in enumerate(messages)), 2) for rule in rules)
    constants = [lambda m: 0, lambda m: 1]
    def gain(messages, rules):
        return best_accuracy(messages, rules) - best_accuracy([(), ()], rules)
    assert gain(words, constants + [lexical_receiver]) - gain(words, constants) == F(1, 2)
    for msg in words:
        paraphrase = msg[:-1] + ("now",)
        assert lexical_receiver(msg) == lexical_receiver(paraphrase)
    ordered = [("Alice", "leads", "Bob"), ("Bob", "leads", "Alice")]
    rules = constants + [lambda m: int(bool(m) and m[0] == "Bob")]
    assert gain(ordered, rules) == F(1, 2)
    for msg in ordered:
        shuffled = tuple(reversed(msg))
        damage = acos(dot(axis(rules[-1](msg), 2), axis(rules[-1](shuffled), 2))) / (pi / 2)
        assert isclose(damage, 1.)
    print("Audit examples: lexical damage 0 with gap 1/2; order damage 1 with gap 0.")


if __name__ == "__main__":
    codebooks()
    radial_truth()
    gaussian()
    form_bounds()
    audit_gap()
