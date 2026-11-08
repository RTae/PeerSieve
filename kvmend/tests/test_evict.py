"""After a repair, KVMend drops from the prefix cache only the KV that may have read the damage (CPU only)."""
import types

import torch

from kvmend import hook

PAGE = 16


class Node:
    def __init__(self, parent, page, lock=0):
        self.parent, self.children, self.lock_ref = parent, {}, lock
        self.value = torch.arange(page * PAGE, (page + 1) * PAGE)
        self.key = list(range(PAGE))
        if parent is not None:
            parent.children[len(parent.children)] = self


def test_only_suspect_cached_kv_is_dropped():
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(16 * PAGE, 1, 64, dtype=torch.bfloat16)],
                                 v_buffer=[torch.zeros(16 * PAGE, 1, 64, dtype=torch.bfloat16)])
    st = hook.PoolState(pool)
    root = Node(None, 0, lock=1)
    root.value = torch.zeros(0, dtype=torch.int64)
    a = Node(root, 2)          # holds the damaged page 2, written before the damage
    b = Node(a, 3)             # a reader's KV written after the damage ...
    c = Node(b, 4)             # ... and its continuation
    d = Node(a, 5)             # written before the last good check: kept
    f = Node(a, 7, lock=1)     # written after, but a running request holds it: kept
    e = Node(root, 6)          # written after, but not below a damaged page: kept
    for page, step in ((2, 5), (3, 20), (4, 30), (5, 3), (7, 40), (6, 25)):
        st.wstep[page] = step
    freed = []
    tree = types.SimpleNamespace(root_node=root,
                                 token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda v: freed.append(int(v[0]) // PAGE)))

    def delete_leaf(node):
        k = next(k for k, v in node.parent.children.items() if v is node)
        del node.parent.children[k]

    tree._delete_leaf = delete_leaf
    assert hook.evict_suspect(st, tree, {2}, since=10) == 2 * PAGE
    assert freed == [4, 3]  # the leaf first, then its parent, which became a leaf
    assert list(a.children.values()) == [d, f] and list(root.children.values()) == [a, e]
    assert hook.evict_suspect(st, tree, set(), since=10) == 0


def test_a_repaired_page_stays_cached():
    """The damaged page itself was just recomputed (written after the last good check), but it is repaired, not
    suspect: its node stays in the prefix cache."""
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(16 * PAGE, 1, 64, dtype=torch.bfloat16)],
                                 v_buffer=[torch.zeros(16 * PAGE, 1, 64, dtype=torch.bfloat16)])
    st = hook.PoolState(pool)
    root = Node(None, 0, lock=1)
    root.value = torch.zeros(0, dtype=torch.int64)
    leaf = Node(root, 9)
    st.wstep[9] = 50  # recomputed at step 50, after the last good check at step 10
    freed = []
    tree = types.SimpleNamespace(root_node=root, _delete_leaf=lambda node: None,
                                 token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda v: freed.append(1)))
    assert hook.evict_suspect(st, tree, {9}, since=10) == 0 and not freed and list(root.children.values()) == [leaf]
