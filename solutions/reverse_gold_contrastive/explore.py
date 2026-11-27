"""Exact finite checks for the constructions in results.md. Uses Python 3.9+."""
import re
from itertools import product
from math import exp, log, log1p

# ABC and XYZ encode the left and right values 0, 1, 2 respectively.
LEFT, RIGHT = "ABC", "XYZ"
LINK = re.compile(r"([ABC])[abch]*([XYZ])")


def moves(word):
    """Return the training edits and the co-substitutions as separate sets."""
    training, joint = set(), set()
    for link in LINK.finditer(word):
        left, right = link.start(), link.end() - 1
        i, j = LEFT.index(word[left]), RIGHT.index(word[right])
        for step in (1, 2):
            l = LEFT[(i + step) % 3]
            r = RIGHT[(j + step) % 3]
            training.add(word[:left] + l + word[left + 1:])
            training.add(word[:right] + r + word[right + 1:])
            joint.add(word[:left] + l + word[left + 1:right] + r + word[right + 1:])
    return training, joint


def closure(seed, operator=moves):
    seen, pending = {seed}, [seed]
    while pending:
        training, joint = operator(pending.pop())
        new = (training | joint) - seen
        seen.update(new)
        pending.extend(new)
    return seen


def unary_moves(word):
    training, joint = set(), set()
    for i in range(len(word) - 6):
        if word[i:i + 7] != "a" * 7:
            continue
        head, tail = word[:i], word[i + 7:]
        for shorter in ("a", "aa"):
            training.add(head + shorter + "a" + "aaa" + tail)
            training.add(head + "aaa" + "a" + shorter + tail)
        joint.add(head + "a" + "a" + "a" + tail)
    return training, joint


def check_unary():
    parts = ("a", "aa", "aaa")
    target = {u + "a" + u for u in parts}
    seed = "a" * 7
    train, joint = unary_moves(seed)
    assert {len(w) for w in train & target} == {5}
    assert {len(w) for w in train - target} == {6}
    assert {len(w) for w in joint} == {3}
    assert {len(w) for w in closure(seed, unary_moves)} == {3, 5, 6, 7}
    for n in range(65):
        w = "a" * n
        assert len(closure(w, unary_moves)) <= n + 1
    print("Unary MCFG: train 2, in 1, out 1, joint 1, closure 4; checked lengths 0..64")


def endpoint_pairs(offsets):
    return {(LEFT[i], RIGHT[(i + d) % 3]) for i in range(3) for d in offsets}


def mcfg_slices(limit):
    # P(a x b, c y) <- P(x,y), with base P(epsilon,epsilon).
    x, y = "", ""
    for _ in range(limit + 1):
        yield {l + x + y + r for l, r in endpoint_pairs((0, 1))}
        x, y = "a" + x + "b", "c" + y


def tag_slices(limit):
    # Initial S trees have T(h) between the two endpoints.
    # Adjoining the T auxiliary with frontier a T* b wraps the T yield.
    t = "h"
    for _ in range(limit + 1):
        yield {l + t + r for l, r in endpoint_pairs((0, 1))}
        t = "a" + t + "b"


def check_grammar(slices, middle_at_depth):
    checked = 0
    for n, positives in enumerate(slices):
        middle = middle_at_depth(n)
        all_pairs = {l + middle + r for l, r in product(LEFT, RIGHT)}
        # Independent membership predicate for the yields at this depth.
        expected = {w for w in all_pairs if (RIGHT.index(w[-1]) - LEFT.index(w[0])) % 3 < 2}
        assert positives == expected and len(positives) == 6
        for word in sorted(positives):
            train, joint = moves(word)
            assert len(train) == 4 and len(joint) == 2 and train.isdisjoint(joint)
            assert len(train & positives) == len(train - positives) == 2
            assert joint <= positives and closure(word) == all_pairs
            for neighbor in train:
                assert word in moves(neighbor)[0]
            checked += 1
    return checked


def check_capacity(max_n):
    assignments = 0
    for n in range(1, max_n + 1):
        for offsets in product((0, 1), repeat=n):
            target, support, probes = set(), set(), []
            for t, d in enumerate(offsets, start=1):
                middle = "a" * t
                target.update(l + middle + r for l, r in endpoint_pairs((d,)))
                support.add(LEFT[0] + middle + RIGHT[d])
                probes.append(LEFT[0] + middle + RIGHT[0])
            negatives = set().union(*(moves(w)[0] for w in support))
            assert target.isdisjoint(negatives)
            assert all(w in (negatives if d else support) for w, d in zip(probes, offsets))
            assignments += 1
    return assignments


def main():
    check_unary()
    limit = 40
    for name, slices, middle in [
        ("MCFG", mcfg_slices(limit), lambda n: "a" * n + "b" * n + "c" * n),
        ("TAG", tag_slices(limit), lambda n: "a" * n + "h" + "b" * n),
    ]:
        count = check_grammar(slices, middle)
        print(f"{name}: {count} positives; train 4, in 2, out 2, joint 2, closure 9")
    assert moves("ABC") == (set(), set())
    assert len(closure("AaXBbY")) == 81  # two independent links off the target
    checked = check_capacity(8)
    print(f"Capacity: {checked} binary assignments checked across N = 1..8")
    b = 2
    for bound in (4, 6):
        margin = log(2) - log1p(bound * exp(-2 * b))
        assert 0 < margin < log1p(2)
        print(f"B = {b}, M0 = {bound}: margin {margin:.9f}, collision term {log(3):.9f}")


if __name__ == "__main__":
    main()
