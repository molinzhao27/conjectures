"""Exact rational checks for the CTL counterexamples. Run with Python 3."""

from fractions import Fraction as Q
from itertools import product


def mix(a, b, t):
    return tuple((1 - t) * x + t * y for x, y in zip(a, b, strict=True))


def relu(z):
    return max(z, 0)


def features(weights, x, baseline):
    u, v = weights
    return relu(u * x), relu(v * x), baseline


def train_step(weights, examples):
    """One full-batch step of size 2 on half squared one-hot loss."""
    grad = [0, 0]
    for x, label in examples:
        for i, w in enumerate(weights):
            assert w * x > 0  # all preactivations at initialization are positive
            grad[i] += (w * x - int(i + 1 == label)) * x
    return tuple(w - 2 * Q(g, len(examples)) for w, g in zip(weights, grad))


def check_classifier():
    checkpoint = (1, 1)
    assert train_step(checkpoint, [(Q(1, 2), 1), (Q(1, 2), 2)]) == checkpoint
    models = [train_step(checkpoint, data) for data in (
        [(1, 1)], [(Q(3, 5), 3), (Q(4, 5), 3)], [(1, 2)]
    )]
    assert models == [(1, -1), (0, 0), (-1, 1)]
    for scale, x, baseline in product((1, 8, 1000), (Q(1), Q(2)), (Q(1, 10), Q(1, 1000))):
        a, b, c = [tuple(scale * w for w in p) for p in models]
        feat = lambda p: features(p, x, baseline)
        for n in range(65):
            t = Q(n, 64)
            for p, q in ((a, b), (b, c)):
                assert feat(mix(p, q, t)) == mix(feat(p), feat(q), t)
        actual = feat(mix(a, c, Q(1, 2)))
        target = mix(feat(a), feat(c), Q(1, 2))
        assert actual == (0, 0, baseline)
        assert target == (scale * x / 2, scale * x / 2, baseline)
        dot = sum(z * y for z, y in zip(actual, target))
        target_sq = sum(y * y for y in target)
        assert dot / target_sq == baseline**2 / (baseline**2 + (scale * x)**2 / 2)
        assert dot**2 / (baseline**2 * target_sq) == dot / target_sq


def check_scalar_gates():
    # The only possible interior extremum of the one-gate chord gap is its root.
    for a, b in product(range(-5, 6), repeat=2):
        samples = [Q(0), Q(1)]
        if a * b < 0:
            samples.append(Q(a, a - b))
        gap = max((1 - t) * relu(a) + t * relu(b) - relu((1 - t) * a + t * b)
                  for t in samples)
        assert (gap == 0) == (a * b >= 0)
        if gap:
            assert gap == Q(abs(a * b), abs(a) + abs(b))


def deep_features(weights, x):
    u, v = weights
    h = relu(u * x)
    return h, relu(v * h)


def check_positive_layers():
    for r, s, x in product((1, 3, 10), (1, 4, 20), (1, 3)):
        a, b, c = (1 + r, 1), (1, 1), (1, 1 + s)
        for n in range(65):
            t = Q(n, 64)
            for p, q in ((a, b), (b, c), (a, c)):
                actual = deep_features(mix(p, q, t), x)
                target = mix(deep_features(p, x), deep_features(q, x), t)
                assert actual[0] == target[0]
                assert actual[1] - target[1] == (t * (1 - t) * r * s * x
                                                 if (p, q) == (a, c) else 0)


if __name__ == "__main__":
    check_classifier()
    check_scalar_gates()
    check_positive_layers()
    print("All assertions passed: shared-checkpoint classifier, 121 scalar gate pairs, "
          "and 18 strictly active two-layer settings.")
