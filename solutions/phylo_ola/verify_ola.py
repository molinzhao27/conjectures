"""Exhaustive and asymptotic verification of Theorem 3, Proposition 4, and Proposition 5
for the OLA-SPR Conjecture (arXiv:2503.10169, Conjecture 11).
"""
import itertools


class Node:
    __slots__ = ("id", "children", "parent", "leaf_label", "cf", "cs", "canon_label")

    def __init__(self, nid, leaf_label=None):
        self.id = nid
        self.children = []
        self.parent = None
        self.leaf_label = leaf_label
        self.cf = None
        self.cs = None
        self.canon_label = None

    def is_leaf(self):
        return self.leaf_label is not None


class RootedBinaryTree:
    def __init__(self, n, rho, nodes):
        self.n = n
        self.rho = rho
        self.nodes = nodes

    def copy(self):
        new_rho = Node(-1)
        new_nodes = {nid: Node(nid, nd.leaf_label) for nid, nd in self.nodes.items()}
        for nid, nd in self.nodes.items():
            nnd = new_nodes[nid]
            if nd.parent is self.rho:
                nnd.parent = new_rho
                new_rho.children.append(nnd)
            else:
                nnd.parent = new_nodes[nd.parent.id]
            nnd.children = [new_nodes[c.id] for c in nd.children]
        return RootedBinaryTree(self.n, new_rho, new_nodes)

    @property
    def top(self):
        return self.rho.children[0]

    def height_r(self):
        def dfs(v):
            return 0 if v.is_leaf() else 1 + max(dfs(c) for c in v.children)
        return dfs(self.top)

    def compute_canonical_labels(self):
        order, stack = [], [self.top]
        while stack:
            v = stack.pop()
            order.append(v)
            stack.extend(v.children)
        for v in reversed(order):
            if v.is_leaf():
                v.cf = v.canon_label = v.leaf_label
            else:
                v.cf = min(v.children[0].cf, v.children[1].cf)
        for v in order:
            if not v.is_leaf():
                v.cs = max(v.children[0].cf, v.children[1].cf)
                v.canon_label = -v.cs

    def ola_encode(self):
        t = self.copy()
        t.compute_canonical_labels()
        leaves = {nd.leaf_label: nd for nd in t.nodes.values() if nd.is_leaf()}
        code = [0] * (t.n - 1)
        for i in range(t.n - 1, 0, -1):
            v = leaves[i]
            p = v.parent
            s = p.children[0] if p.children[1] is v else p.children[1]
            code[i - 1] = s.canon_label
            gp = p.parent
            s.parent = gp
            gp.children = [s if c is p else c for c in gp.children]
            del t.nodes[v.id]
            del t.nodes[p.id]
        return tuple(code)

    def clade_set(self):
        clades = set()
        def dfs(v):
            s = frozenset([v.leaf_label]) if v.is_leaf() else (dfs(v.children[0]) | dfs(v.children[1]))
            clades.add(s)
            return s
        dfs(self.top)
        return frozenset(clades)


def ola_decode(code):
    n = len(code) + 1
    rho = Node(-1)
    v0 = Node(0, leaf_label=0)
    v0.parent = rho
    rho.children = [v0]
    nodes = {0: v0}
    label_to_node = {0: v0}
    next_id = 1
    for idx, a_i in enumerate(code):
        i = idx + 1
        v = label_to_node[a_i]
        p = v.parent
        w = Node(next_id)
        u = Node(next_id + 1, leaf_label=i)
        next_id += 2
        w.parent = p
        p.children = [w if c is v else c for c in p.children]
        v.parent = u.parent = w
        w.children = [v, u]
        nodes[w.id], nodes[u.id] = w, u
        label_to_node[-i], label_to_node[i] = w, u
    return RootedBinaryTree(n, rho, nodes)


def ola_distance(c1, c2):
    return sum(x != y for x, y in zip(c1, c2))


def all_spr_neighbors(tree):
    orig = tree.clade_set()
    neighbors = {}
    for u_id in list(tree.nodes):
        if u_id == tree.top.id:
            continue
        u_orig = tree.nodes[u_id]
        anc_p = set()
        cur = u_orig.parent
        while cur is not tree.rho:
            anc_p.add(cur.id)
            cur = cur.parent
        sub_u = set()
        stack = [u_orig]
        while stack:
            x = stack.pop()
            sub_u.add(x.id)
            stack.extend(x.children)
        rem_ids = [nid for nid in tree.nodes if nid not in sub_u and nid != u_orig.parent.id]
        for v_bot_id in rem_ids:
            t_new = tree.copy()
            u_n = t_new.nodes[u_id]
            p_n = u_n.parent
            gp_n = p_n.parent
            sib_n = p_n.children[0] if p_n.children[1] is u_n else p_n.children[1]
            sib_n.parent = gp_n
            gp_n.children = [sib_n if c is p_n else c for c in gp_n.children]
            v_bot_n = t_new.nodes[v_bot_id]
            v_top_n = v_bot_n.parent
            anc_vtop = set()
            cur = v_top_n
            while cur is not t_new.rho:
                anc_vtop.add(cur.id)
                cur = cur.parent
            p_n.parent = v_top_n
            v_top_n.children = [p_n if c is v_bot_n else c for c in v_top_n.children]
            v_bot_n.parent = u_n.parent = p_n
            p_n.children = [v_bot_n, u_n]
            cs = t_new.clade_set()
            if cs == orig:
                continue
            b_size = len(anc_p | anc_vtop)
            if cs not in neighbors or b_size < neighbors[cs][1]:
                neighbors[cs] = (t_new, b_size)
    return list(neighbors.values())


def verify():
    # 1. Paper examples
    for code in [(0, 0, 2, 1), (0, -1, 1, -3), (0, -1, 2, 1, -3, -3, 5)]:
        assert ola_decode(code).ola_encode() == code
    print("Paper roundtrip examples OK.")

    # 2. Exhaustive verification for n = 3, 4, 5 (all 3, 15, 105 trees and all SPR pairs)
    for n in [3, 4, 5]:
        ranges = [range(-i + 1, i) for i in range(1, n)]
        total_pairs = 0
        max_gap_B, max_gap_2h2 = -999, -999
        for code in itertools.product(*ranges):
            t = ola_decode(code)
            h = t.height_r()
            for t2, b_size in all_spr_neighbors(t):
                total_pairs += 1
                d = ola_distance(code, t2.ola_encode())
                assert d <= b_size - 1, f"Bound |B|-1 failed: d={d}, |B|={b_size}"
                assert d <= 2 * h - 2, f"Bound 2h-2 failed: d={d}, h={h}"
                max_gap_B = max(max_gap_B, d - (b_size - 1))
                max_gap_2h2 = max(max_gap_2h2, d - (2 * h - 2))
        print(f"n={n}: ALL {total_pairs} SPR pairs verified! max(d-(|B|-1))={max_gap_B}, max(d-(2h-2))={max_gap_2h2}")

    # 3. Exact sharpness construction for h = 2..20
    for h in range(2, 21):
        code_T = tuple([0] * h + [1] * (h - 1))
        code_Tp = tuple([0, -1] + list(range(2, h)) + [-1] * (h - 1))
        t = ola_decode(code_T)
        assert t.height_r() == h
        assert ola_distance(code_T, code_Tp) == 2 * h - 2
        if h <= 8:
            t2_clades = ola_decode(code_Tp).clade_set()
            assert any(nb.clade_set() == t2_clades for nb, _ in all_spr_neighbors(t))
    print("Exact sharpness 2h-2 verified for h=2..20 (and rSPR adjacency for h=2..8)!")

    # 4. Expected value lower bound on caterpillar T_n = (0,...,0)
    for n in [6, 10, 16, 24]:
        code = (0,) * (n - 1)
        t = ola_decode(code)
        nbrs = all_spr_neighbors(t)
        exp_d = sum(ola_distance(code, nb.ola_encode()) for nb, _ in nbrs) / len(nbrs)
        print(f"Caterpillar n={n:2d} (h={n-1:2d}): |SPR|={len(nbrs):4d}, E[d]={exp_d:.3f}, E[d]/h={exp_d/(n-1):.4f}")

    print("\nALL VERIFICATIONS PASSED!")


if __name__ == "__main__":
    verify()
