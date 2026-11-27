"""
Exact computational verification for the Conjecture in Section 7.3
and Question 2 in Section 7.2 of arXiv:2603.10837:
  J. Geraci, A. B. Kunin, A. Seceleanu,
  "Adjoints of Morphisms of Neural Codes" (March 10, 2026).
"""

import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def trunk_mask(C: np.ndarray, sigma) -> np.ndarray:
    m = C.shape[0]
    if len(sigma) == 0:
        return np.ones(m, dtype=bool)
    mask = np.ones(m, dtype=bool)
    for j in sigma:
        mask &= (C[:, j] == 1)
    return mask


def root_of_mask(C: np.ndarray, mask: np.ndarray):
    assert mask.any(), "Root of empty trunk is undefined"
    sub = C[mask]
    return tuple(j for j in range(C.shape[1]) if np.all(sub[:, j] == 1))


def analyze_redundancy(C: np.ndarray):
    m, n = C.shape
    col_masks = [C[:, i] == 1 for i in range(n)]
    all_ones = np.ones(m, dtype=bool)

    trivial = []
    redundant = []
    simply_redundant = []
    non_simply_redundant = []

    for i in range(n):
        mi = col_masks[i]
        if not mi.any():
            trivial.append(i)
            continue

        rt = root_of_mask(C, mi)
        rt_minus_i = [j for j in rt if j != i]
        is_red = np.array_equal(mi, trunk_mask(C, rt_minus_i))

        if is_red:
            redundant.append(i)
            is_simple = np.array_equal(mi, all_ones) or any(
                j != i and np.array_equal(mi, col_masks[j]) for j in range(n)
            )
            if is_simple:
                simply_redundant.append(i)
            else:
                non_simply_redundant.append(i)

    return {
        "trivial": trivial,
        "redundant": redundant,
        "simply_redundant": simply_redundant,
        "non_simply_redundant": non_simply_redundant,
    }


def reduce_matrix(C: np.ndarray) -> np.ndarray:
    M = C.copy()
    keep = [i for i in range(M.shape[1]) if M[:, i].any()]
    M = M[:, keep]
    while M.shape[1] > 0:
        info = analyze_redundancy(M)
        if not info["redundant"]:
            break
        idx = info["redundant"][-1]
        M = np.delete(M, idx, axis=1)
    return M


def is_intersection_complete(C: np.ndarray) -> bool:
    rows = {tuple(row) for row in C}
    row_list = list(rows)
    for i in range(len(row_list)):
        for j in range(i, len(row_list)):
            inter = tuple(a & b for a, b in zip(row_list[i], row_list[j]))
            if inter not in rows:
                return False
    return True


def maximal_rectangles(C: np.ndarray):
    m, n = C.shape
    ones = [(i, j) for i in range(m) for j in range(n) if C[i, j] == 1]
    if not ones:
        return [], 0

    one_idx = {pos: k for k, pos in enumerate(ones)}
    row_bits = [0] * m
    for i in range(m):
        bits = 0
        for j in range(n):
            if C[i, j]:
                bits |= (1 << j)
        row_bits[i] = bits

    col_bits = [0] * n
    for j in range(n):
        bits = 0
        for i in range(m):
            if C[i, j]:
                bits |= (1 << i)
        col_bits[j] = bits

    rects = set()
    if m <= n:
        intents = set()
        for i in range(m):
            r = row_bits[i]
            if r == 0:
                continue
            new_intents = {r}
            for x in intents:
                inter = x & r
                if inter:
                    new_intents.add(inter)
            intents |= new_intents
        for col_mask in intents:
            row_mask = 0
            for i in range(m):
                if (row_bits[i] & col_mask) == col_mask:
                    row_mask |= (1 << i)
            rects.add((row_mask, col_mask))
    else:
        extents = set()
        for j in range(n):
            c = col_bits[j]
            if c == 0:
                continue
            new_extents = {c}
            for x in extents:
                inter = x & c
                if inter:
                    new_extents.add(inter)
            extents |= new_extents
        for row_mask in extents:
            col_mask = 0
            for j in range(n):
                if (col_bits[j] & row_mask) == row_mask:
                    col_mask |= (1 << j)
            rects.add((row_mask, col_mask))

    result = []
    for row_mask, col_mask in rects:
        cov = 0
        for i in range(m):
            if (row_mask >> i) & 1:
                for j in range(n):
                    if (col_mask >> j) & 1:
                        cov |= (1 << one_idx[(i, j)])
        result.append((row_mask, col_mask, cov))

    return result, len(ones)


def brank_ilp(C: np.ndarray, return_factors: bool = False):
    m, n = C.shape
    if not C.any():
        if return_factors:
            return 0, np.zeros((m, 0), dtype=int), np.zeros((0, n), dtype=int)
        return 0

    rects, num_ones = maximal_rectangles(C)
    num_rects = len(rects)

    A = np.zeros((num_ones, num_rects), dtype=float)
    for r_idx, (_, _, cov) in enumerate(rects):
        for k in range(num_ones):
            if (cov >> k) & 1:
                A[k, r_idx] = 1.0

    c = np.ones(num_rects, dtype=float)
    constraints = LinearConstraint(A, lb=np.ones(num_ones), ub=np.full(num_ones, np.inf))
    integrality = np.ones(num_rects)
    bounds = Bounds(lb=np.zeros(num_rects), ub=np.ones(num_rects))
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    assert res.success, f"MILP failed: {res.message}"

    chosen = [idx for idx, val in enumerate(res.x) if val > 0.5]
    r = len(chosen)

    V = np.zeros((m, r), dtype=int)
    H = np.zeros((r, n), dtype=int)
    for t, idx in enumerate(chosen):
        row_mask, col_mask, _ = rects[idx]
        for i in range(m):
            if (row_mask >> i) & 1:
                V[i, t] = 1
        for j in range(n):
            if (col_mask >> j) & 1:
                H[t, j] = 1
    prod = (V @ H > 0).astype(int)
    assert np.array_equal(prod, C), "Factorization verification failed!"

    if return_factors:
        return r, V, H
    return r


def boolean_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A @ B) > 0).astype(int)


def residual_colon(C: np.ndarray, H: np.ndarray) -> np.ndarray:
    not_C = 1 - C
    prod = boolean_product(not_C, H.T)
    return 1 - prod


def brank_bruteforce_residual(C: np.ndarray) -> int:
    m, n = C.shape
    if not C.any():
        return 0
    candidates = []
    for vec in itertools.product((0, 1), repeat=n):
        if not any(vec):
            continue
        arr = np.array(vec, dtype=int)
        if any(np.all(arr <= C[i]) for i in range(m)):
            candidates.append(arr)

    for r in range(1, min(m, n) + 1):
        for comb in itertools.combinations(candidates, r):
            H = np.vstack(comb)
            V = residual_colon(C, H)
            if np.array_equal(boolean_product(V, H), C):
                return r
    return min(m, n)


def test_matrix(C: np.ndarray, name: str = "", cross_check: bool = True):
    info = analyze_redundancy(C)
    rho_C = reduce_matrix(C)
    r_C, V_C, H_C = brank_ilp(C, return_factors=True)
    r_rho, V_rho, H_rho = brank_ilp(rho_C, return_factors=True)
    if cross_check and C.shape[1] <= 5:
        assert brank_bruteforce_residual(C) == r_C, "Brute-force mismatch on C!"
        assert brank_bruteforce_residual(rho_C) == r_rho, "Brute-force mismatch on rho(C)!"

    k = len(info["redundant"])
    ell = len(info["non_simply_redundant"])
    pred = r_rho + ell
    diff = r_C - pred
    print(f"=== {name} ===")
    print(f"Shape: {C.shape}, rho(C) shape: {rho_C.shape}")
    print(f"trivial={info['trivial']}, redundant={info['redundant']} (k={k}), "
          f"simply_red={info['simply_redundant']}, non_simply_red={info['non_simply_redundant']} (ell={ell})")
    print(f"brank(rho(C)) = {r_rho}, brank(C) = {r_C}, conjectured brank(rho(C)) + ell = {pred}  [diff = {diff:+d}]")
    print(f"intersection-complete? {is_intersection_complete(C)}")
    print("C =\n", C)
    print("V_C =\n", V_C)
    print("H_C =\n", H_C)
    print()
    return r_C, r_rho, k, ell


def search_smallest_counterexamples():
    print("=" * 70)
    print("EXHAUSTIVE SEARCH FOR SMALLEST CLEAN COUNTEREXAMPLES")
    print("=" * 70)
    for m in range(2, 6):
        for n0 in range(2, 5):
            if m == 5 and n0 == 4:
                continue
            all_cols = [
                np.array(col, dtype=int)
                for col in itertools.product((0, 1), repeat=m)
                if 0 < sum(col) < m
            ]
            found_for_dim = 0
            first_ex = None
            for col_tuple in itertools.combinations(all_cols, n0):
                B = np.column_stack(col_tuple)
                if len({tuple(r) for r in B}) < m:
                    continue
                info_B = analyze_redundancy(B)
                if info_B["trivial"] or info_B["redundant"]:
                    continue
                existing = {tuple(B[:, j]) for j in range(n0)}
                nsr_cols = set()
                for sz in range(2, n0 + 1):
                    for subset in itertools.combinations(range(n0), sz):
                        prod_col = np.all(B[:, list(subset)] == 1, axis=1).astype(int)
                        t = tuple(prod_col)
                        if prod_col.any() and not prod_col.all() and t not in existing:
                            nsr_cols.add(t)
                if not nsr_cols:
                    continue
                r_B = brank_ilp(B)
                for c_tup in nsr_cols:
                    C = np.column_stack([B, np.array(c_tup, dtype=int)])
                    info_C = analyze_redundancy(C)
                    assert len(info_C["non_simply_redundant"]) == 1
                    assert len(info_C["simply_redundant"]) == 0
                    r_C = brank_ilp(C)
                    if r_C != r_B + 1:
                        found_for_dim += 1
                        if first_ex is None:
                            first_ex = (B, C, r_B, r_C)
            print(f"m={m}, n0={n0} (C is {m}x{n0+1}, ell=1): found {found_for_dim} counterexamples")
            if first_ex is not None and m <= 4 and n0 <= 3:
                B, C, r_B, r_C = first_ex
                assert brank_bruteforce_residual(B) == r_B
                assert brank_bruteforce_residual(C) == r_C
                print(f"  First example (brank(rho(C))={r_B}, brank(C)={r_C}, ell=1):\n{C}")


def verify_infinite_families():
    print("\n" + "=" * 70)
    print("VERIFICATION OF INFINITE COUNTEREXAMPLE FAMILIES")
    print("=" * 70)
    print("\n--- Family 1: brank(C_m) = brank(rho(C_m)) = m+1, ell = 2^m - m - 1 ---")
    for m in range(2, 6):
        I_m = np.eye(m, dtype=int)
        J_minus_I = np.ones((m, m), dtype=int) - I_m
        top = np.hstack([I_m, J_minus_I])
        bot = np.hstack([np.zeros((1, m), dtype=int), np.ones((1, m), dtype=int)])
        rho_C = np.vstack([top, bot])

        extra_cols = []
        for sz in range(2, m + 1):
            for subset in itertools.combinations(range(m, 2 * m), sz):
                col = np.all(rho_C[:, list(subset)] == 1, axis=1).astype(int)
                extra_cols.append(col)
        C = np.hstack([rho_C, np.column_stack(extra_cols)])

        info_rho = analyze_redundancy(rho_C)
        info_C = analyze_redundancy(C)
        assert not info_rho["trivial"] and not info_rho["redundant"]
        assert not info_C["trivial"] and not info_C["simply_redundant"]
        assert len(info_C["non_simply_redundant"]) == 2**m - m - 1

        red_C = reduce_matrix(C)
        assert red_C.shape == rho_C.shape

        r_rho = brank_ilp(rho_C)
        r_C = brank_ilp(C)
        assert r_rho == m + 1 and r_C == m + 1
        print(f"m={m}: C shape={C.shape}, brank(rho(C))={r_rho}, brank(C)={r_C}, "
              f"ell={len(info_C['non_simply_redundant'])}, predicted={r_rho + len(info_C['non_simply_redundant'])}, "
              f"gap={len(info_C['non_simply_redundant'])}")

        C_zero = np.vstack([np.zeros((1, C.shape[1]), dtype=int), C])
        rho_zero = np.vstack([np.zeros((1, rho_C.shape[1]), dtype=int), rho_C])
        info_Cz = analyze_redundancy(C_zero)
        assert len(info_Cz["non_simply_redundant"]) == 2**m - m - 1
        assert len(info_Cz["simply_redundant"]) == 0
        assert brank_ilp(rho_zero) == m + 1 and brank_ilp(C_zero) == m + 1

    print("\n--- Family 2: rho(C) = J_n - I_n, brank(rho) = O(log n), brank(C) = n, ell = 2^n - n - 2 ---")
    for n in range(3, 7):
        rho_C = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        extra_cols = []
        for sz in range(2, n):
            for subset in itertools.combinations(range(n), sz):
                col = np.all(rho_C[:, list(subset)] == 1, axis=1).astype(int)
                extra_cols.append(col)
        C = np.hstack([rho_C, np.column_stack(extra_cols)])
        info_C = analyze_redundancy(C)
        assert not info_C["trivial"] and not info_C["simply_redundant"]
        assert len(info_C["non_simply_redundant"]) == 2**n - n - 2
        r_rho = brank_ilp(rho_C)
        r_C = brank_ilp(C)
        assert r_C == n
        print(f"n={n}: C shape={C.shape}, brank(rho(C))={r_rho}, brank(C)={r_C}, "
              f"ell={len(info_C['non_simply_redundant'])}, predicted={r_rho + len(info_C['non_simply_redundant'])}")


def test_intersection_complete_theorem():
    print("\n" + "=" * 70)
    print("VERIFICATION OF POSITIVE THEOREM: INTERSECTION-COMPLETE CODES")
    print("=" * 70)
    for n in [1, 2, 3]:
        all_words = list(itertools.product((0, 1), repeat=n))
        ic_count = 0
        for mask in range(1, 1 << (1 << n)):
            words = [all_words[i] for i in range(1 << n) if (mask >> i) & 1]
            C = np.array(words, dtype=int)
            if not is_intersection_complete(C):
                continue
            ic_count += 1
            r_C = brank_ilp(C)
            distinct_nz_cols = len({tuple(C[:, j]) for j in range(n) if C[:, j].any()})
            assert r_C == distinct_nz_cols, f"Failed for IC code:\n{C}"

            info = analyze_redundancy(C)
            if not info["trivial"] and not info["simply_redundant"]:
                rho_C = reduce_matrix(C)
                r_rho = brank_ilp(rho_C)
                assert r_C == r_rho + len(info["non_simply_redundant"])
        print(f"n={n}: Verified ALL {ic_count} intersection-complete codes! brank(C) == #distinct nonzero columns.")

    rng = np.random.default_rng(42)
    for n in [4, 5, 6]:
        tested = 0
        for _ in range(100):
            k_words = int(rng.integers(2, min(10, 2**n)))
            raw = rng.integers(0, 2, size=(k_words, n))
            word_set = {tuple(r) for r in raw}
            changed = True
            while changed:
                changed = False
                wlist = list(word_set)
                for a in wlist:
                    for b in wlist:
                        inter = tuple(x & y for x, y in zip(a, b))
                        if inter not in word_set:
                            word_set.add(inter)
                            changed = True
            C = np.array(sorted(word_set), dtype=int)
            if C.shape[0] > 24:
                continue
            r_C = brank_ilp(C)
            distinct_nz_cols = len({tuple(C[:, j]) for j in range(n) if C[:, j].any()})
            assert r_C == distinct_nz_cols, f"Failed for n={n} IC code:\n{C}"
            tested += 1
        print(f"n={n}: Verified {tested} random intersection-complete codes! brank(C) == #distinct nonzero columns.")


def check_question2_covering_lambda():
    """
    Also investigate Question 2 in Section 7.2:
      What is max_{C in Code} { lambda(C^{(i)}) - lambda(C) }?
      Does any code achieve the upper bound lambda(C^{(i)}) = 2*lambda(C) - 2?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATING QUESTION 2 (Section 7.2): lambda(C^{(i)}) <= 2*lambda(C) - 2")
    print("=" * 70)

    def covered_code_reduced_lambda(C: np.ndarray, i: int) -> int:
        """Compute lambda(C^{(i)}) for a reduced code C and neuron i (0-indexed)."""
        m, n = C.shape
        tk_i = (C[:, i] == 1)
        cols = []
        # First n trunks: tk_C(j) if tk_C(j) != tk_C(i)
        for j in range(n):
            tk_j = (C[:, j] == 1)
            if not np.array_equal(tk_j, tk_i):
                cols.append(tk_j.astype(int))
        # Second n trunks: tk_C(j) & tk_C(i) if != tk_C(i)
        for j in range(n):
            tk_ij = (C[:, j] == 1) & tk_i
            if tk_ij.any() and not np.array_equal(tk_ij, tk_i):
                cols.append(tk_ij.astype(int))
        if not cols:
            return 0
        D = np.column_stack(cols)
        # Deduplicate rows (since C^{(i)} is a set of codewords)
        D = np.unique(D, axis=0)
        red_D = reduce_matrix(D)
        return red_D.shape[1], red_D

    # Wait: let's construct a code with lambda(C) = n that achieves lambda(C^{(i)}) = 2n - 2!!
    # Think mathematically FIRST!
    # In C^{(i)}, the non-trivial columns before reduction are:
    #   A_j = tk_C(j)  for j != i  (n-1 columns)
    #   B_j = tk_C(i) & tk_C(j)  for j != i  (at most n-1 columns, non-empty and != tk_C(i))
    # For ALL 2n-2 columns to survive reduction (none trivial, none redundant to an intersection of others!):
    # Notice B_j = tk_C(i) & tk_C(j) is ALWAYS a subset of A_j = tk_C(j)!
    # When is B_j NOT redundant to A_j and some other A_k or B_k?
    #Wait: if every B_j is a subset of tk_C(i), what if we just have codewords:
    #   For each j != i:
    #     - codeword {j} (not containing i) -> ensures A_j is not contained in union of B's or other A's!
    #     - codeword {i, j} -> ensures B_j = {{i, j}, ...} contains {i, j}, while for k != j, {i, j} not in A_k and not in B_k!
    # WAIT A SECOND!!
    # Check C_n = { {j} : j in {2..n} } U { {1, j} : j in {2..n} }  (with i = 1)!!
    # Let's check EVERY column of C_n^{(1)}!
    # For each j in {2..n}:
    #   - A_j = tk(j) = { {j}, {1, j} }
    #   - B_j = tk(1) & tk(j) = { {1, j} }
    # Wait: in C_n^{(1)}:
    #   - Does any intersection of OTHER columns equal B_j = {{1, j}}?
    #     NO! Because {1, j} is ONLY in A_j and B_j! So any intersection of columns NOT containing B_j either omits A_j (then doesn't contain {1, j}) or includes ONLY A_j (which equals {{j}, {1, j}} != {{1, j}})!
    #   - Wait: what if we take an intersection of A_j and A_k (k != j)? That's EMPTY (not {{1, j}})!
    #   - So B_j is NEVER redundant!!
    #   - What about A_j = {{j}, {1, j}}?
    #     {j} is ONLY in A_j! So NO other column contains {j}! Hence A_j is NEVER redundant!!
    #   - WAIT A SECOND! Is C_n itself reduced?!
    #     In C_n:
    #       tk(1) = { {1, 2}, {1, 3}, ..., {1, n} }
    #       tk(j) = { {j}, {1, j} } for j = 2..n
    #     None is empty; none is all words; no intersection of other columns equals tk(1) (any other col has size 2, intersection has size <= 1, whereas |tk(1)| = n-1 >= 2 for n >= 3!); and for j >= 2, {j} is ONLY in tk(j), so tk(j) is not an intersection of other columns!
    #   - SO FOR EVERY n >= 3, C_n IS REDUCED WITH lambda(C_n) = n,
    #     AND lambda(C_n^{(1)}) = 2n - 2!!
    # WAIT A SECOND!! CHECK THIS RIGHT NOW IN PYTHON!!
    for n in range(3, 9):
        words = []
        # Optionally include empty word 0...0 (still works!)
        words.append([0] * n)
        for j in range(1, n):
            w1 = [0] * n
            w1[j] = 1
            words.append(w1)
            w2 = [0] * n
            w2[0] = 1
            w2[j] = 1
            words.append(w2)
        C = np.array(words, dtype=int)
        info = analyze_redundancy(C)
        assert not info["trivial"] and not info["redundant"], f"C_{n} not reduced!"
        lam_Ci, red_Ci = covered_code_reduced_lambda(C, 0)
        print(f"n={n}: lambda(C) = {n}, lambda(C^(1)) = {lam_Ci}, upper bound 2n-2 = {2*n - 2}, "
              f"diff = {lam_Ci - n} (achieves bound? {lam_Ci == 2*n - 2})")


if __name__ == "__main__":
    print("=" * 70)
    print("EXPLICIT SMALLEST COUNTEREXAMPLES")
    print("=" * 70)

    C_3x4 = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
    ], dtype=int)
    test_matrix(C_3x4, "Counterexample 1A: Clean 3x4, ell=1, brank(C) < brank(rho(C)) + ell")

    C_4x4_zero = np.vstack([np.zeros((1, 4), dtype=int), C_3x4])
    test_matrix(C_4x4_zero, "Counterexample 1B: Clean 4x4 with empty codeword, ell=1")

    C_all1_2x2 = np.array([
        [0, 1],
        [1, 1],
    ], dtype=int)
    test_matrix(C_all1_2x2, "Counterexample 2A: 2x2 with all-1 column, k=1, ell=0, brank > brank(rho)+ell")

    C_all1_3x3 = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=int)
    test_matrix(C_all1_3x3, "Counterexample 2B: 3x3 with all-1 column, k=1, ell=0, brank > brank(rho)+ell")

    C_dup = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 1, 1, 1],
    ], dtype=int)
    test_matrix(C_dup, "Counterexample 3: Duplicate non-simple redundancy (4x4), k=2, ell=0, brank > brank(rho)+ell")

    search_smallest_counterexamples()
    verify_infinite_families()
    test_intersection_complete_theorem()
    check_question2_covering_lambda()
