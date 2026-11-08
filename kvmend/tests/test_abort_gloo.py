"""The release rule's abort option and the rollback that waits a step, on 2 CPU processes with gloo (no SGLang, no
GPU). The prefix cache is a stand-in with the parts of SGLang 0.5.1's RadixCache (page size 16) that KVMend touches:
nodes with keys, slot values and lock counts, _split_node, _delete_leaf, inc/dec_lock_ref (copied from 0.5.1), and an
allocator whose free moves pages to a new generation, as the patched allocator does.

  abort     on_fail "abort" (rid:every:abort): a check finds a late write in a shared prompt page; the request and the
            other reader of that page are aborted, nothing is mended, and their unchecked tokens stay held. Once both
            have ended (cache_finished_req), the damaged pages leave the prefix cache with everything cached on top of
            them; the clean head of the prefix stays. A request whose check at finish fails ends with an abort.
  deferred  strict release (mend): a request that has just finished its prefill on top of the damage finds it. Its
            rollback must wait a step, so the eviction after the repair keeps its own prompt pages, which it computed
            from the damaged ones; they are evicted when the rollback lets them go."""
import os
import socket
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS, POOL_PAGES = 16, 1, 64, 2, 2, 48
SHARED = list(range(1000, 1064))  # 4 prefix pages, shared by R and O (and by X in the deferred case)


# ---------------------------------------------------------------------------------------------------------------
# The stand-in prefix cache

class Node:
    def __init__(self):
        self.children, self.parent, self.key, self.value, self.lock_ref = {}, None, None, None, 0


class Allocator:
    def __init__(self, pool):
        self.pool, self.freed = pool, []

    def get_kvcache(self):
        return self.pool

    def free(self, idx):
        idx = torch.as_tensor(idx, dtype=torch.int64)
        self.freed += idx.tolist()
        if idx.numel():
            self.pool._kvm.drop(torch.unique(idx // PAGE))  # what patch_allocator does: a new generation


class Tree:
    """RadixCache's methods that KVMend calls, as in SGLang 0.5.1, plus add() to build a tree."""

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator
        self.root_node = Node()
        self.root_node.key, self.root_node.value, self.root_node.lock_ref = [], [], 1
        self.evictable_size_ = self.protected_size_ = 0
        self.finished = []

    def add(self, parent, key, value):
        n = Node()
        n.parent, n.key, n.value = parent, list(key), torch.as_tensor(list(value), dtype=torch.int64)
        parent.children[tuple(key[:PAGE])] = n
        self.evictable_size_ += len(n.value)
        return n

    def inc_lock_ref(self, node):
        while node is not self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node):
        while node is not self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
            node.lock_ref -= 1
            node = node.parent

    def _split_node(self, key, child, split_len):
        new_node = Node()
        new_node.children = {tuple(key[split_len:][:PAGE]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[tuple(key[:PAGE])] = new_node
        return new_node

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v is node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def cache_finished_req(self, req):
        """A finished request lets go of its pages (its KV is already in the tree here)."""
        self.finished.append(req.rid)
        self.dec_lock_ref(req.last_node)


def _nodes(tree):
    out, todo = [], [tree.root_node]
    while todo:
        for c in todo.pop().children.values():
            out.append(c)
            todo.append(c)
    return out


def _cached_pages(tree):
    return sorted({int(s) // PAGE for n in _nodes(tree) for s in n.value.tolist()})


# ---------------------------------------------------------------------------------------------------------------
# Requests, the scheduler and the store path

def _batch(reqs, seq_lens):
    b = types.SimpleNamespace(reqs=list(reqs), seq_lens=torch.tensor(seq_lens, dtype=torch.int64))

    def filter_batch(keep_indices):
        b.reqs = [b.reqs[j] for j in keep_indices]
        b.seq_lens = b.seq_lens[torch.tensor(keep_indices, dtype=torch.int64)]

    b.filter_batch = filter_batch
    return b


def _req(rid, row, prompt, out0, last_node):
    r = types.SimpleNamespace(rid=rid, req_pool_idx=row, origin_input_ids=list(prompt), output_ids=[out0],
                              prefix_indices=list(range(len(prompt))), last_node=last_node, send_token_offset=0,
                              to_abort=False, to_abort_message=None, is_chunked=0, finished_reason=None)
    r.finished = lambda r=r: r.finished_reason is not None
    r.reset_for_retract = lambda r=r: setattr(r, "prefix_indices", [])
    return r


def _setup(rank, rows):
    """A pool, its state, the stand-in tree and a scheduler whose token table has `rows` (lists of slots)."""
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    st.reset()
    pool._kvm = st
    tree = Tree(Allocator(pool))
    table = torch.zeros(len(rows), 256, dtype=torch.int32)
    for i, slots in enumerate(rows):
        table[i, :len(slots)] = torch.tensor(slots, dtype=torch.int32)
    world = dist.group.WORLD
    queued = []
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE, running_batch=_batch([], []), cur_batch=None,
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=table, free=lambda i: None),
        token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator, tree_cache=tree,
        _extend_requests_to_queue=lambda reqs, is_retracted: queued.extend(r.rid for r in reqs))
    st.sched = sched
    return st, pool, tree, sched, queued


def _store(st, pool, sched, req, positions, seed, step=None):
    """Write the KV of the request's positions through the store path, keyed by its tokens, at batch step `step`."""
    if step is not None:
        st.step_t.fill_(step)
    ids = list(req.origin_input_ids) + list(req.output_ids)
    loc = sched.req_to_token_pool.req_to_token[req.req_pool_idx, torch.tensor(positions)].to(torch.int64)
    tk = T.token_keys(torch.tensor([ids[p] for p in positions]), torch.tensor(positions))
    g = torch.Generator().manual_seed(seed)
    for layer in range(LAYERS):
        pool.k_buffer[layer][loc] = torch.randn(len(positions), HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][loc] = torch.randn(len(positions), HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, loc, tk)


def _decode(st, pool, sched, reqs, seed):
    """One decode step: each request stores its newest token's KV and chooses the next token."""
    for k, req in enumerate(reqs):
        pos = len(req.origin_input_ids) + len(req.output_ids) - 1
        _store(st, pool, sched, req, [pos], seed=seed + k)
        req.output_ids.append(req.output_ids[-1] + 1)


def _fake_recompute(st, sched, req, s, e):
    tgt, clean = st.saved_span
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, s:e].to(torch.int64)
    ids = (list(req.origin_input_ids) + list(req.output_ids))[s:e]
    for layer in range(st.layers):
        st.pool.k_buffer[layer][row] = clean[layer, 0, :e - s]
        st.pool.v_buffer[layer][row] = clean[layer, 1, :e - s]
        st.update(layer, row, T.token_keys(torch.tensor(ids), torch.arange(s, e)))


def _stub_sglang():
    """FINISH_ABORT, the one SGLang name the release rule imports, for a finishing check."""
    class FINISH_ABORT:
        def __init__(self, message=None, status_code=None, err_type=None):
            self.message = message

    names = ("sglang", "sglang.srt", "sglang.srt.managers", "sglang.srt.managers.schedule_batch")
    for name in names:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.managers.schedule_batch"].FINISH_ABORT = FINISH_ABORT


def _records(op, **match):
    return [x for x in hook._BUF if x.get("op") == op and all(x.get(k) == v for k, v in match.items())]


# ---------------------------------------------------------------------------------------------------------------
# abort

def _abort_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    recomputed = []
    hook._recompute_range = lambda *args: recomputed.append(args[3:])
    hook.STRICT, hook.ON_FAIL = hook._rules("rf_*:2:abort")
    _stub_sglang()
    hook.patch_radix_cache(types.SimpleNamespace(RadixCache=Tree))
    try:
        # R and O share the prefix (pool pages 1-4), then each has its own question (R: 5-6, O: 7-8); U has its own
        # prefix (9-12) and question (13-14). Decode slots: R from page 15, O from 25, U from 35.
        rows = [list(range(16, 112)) + list(range(240, 400)),
                list(range(16, 80)) + list(range(112, 144)) + list(range(400, 560)),
                list(range(144, 240)) + list(range(560, 720))]
        st, pool, tree, sched, _ = _setup(rank, rows)
        P = tree.add(tree.root_node, SHARED, range(16, 80))
        QR = tree.add(P, range(2000, 2032), range(80, 112))
        QO = tree.add(P, range(3000, 3032), range(112, 144))
        PU = tree.add(tree.root_node, range(4000, 4064), range(144, 208))
        QU = tree.add(PU, range(5000, 5032), range(208, 240))
        R = _req("rf_R", 0, SHARED + list(range(2000, 2032)), 7000, QR)
        O = _req("rf_O", 1, SHARED + list(range(3000, 3032)), 8000, QO)
        U = _req("rf_U", 2, list(range(4000, 4064)) + list(range(5000, 5032)), 9000, QU)
        for node in (QR, QO, QU):
            tree.inc_lock_ref(node)
        sched.running_batch = _batch([R, O, U], [97, 97, 97])
        _store(st, pool, sched, R, list(range(96)), seed=10 + rank)  # the prefill (R writes the shared prefix)
        _store(st, pool, sched, O, list(range(64, 96)), seed=20 + rank)
        _store(st, pool, sched, U, list(range(96)), seed=30 + rank)
        hook.release_checks(st, sched)  # output 1: the check after the prefill
        _decode(st, pool, sched, [R, O, U], seed=40 + 10 * rank)
        hook.release_checks(st, sched)  # output 2
        _decode(st, pool, sched, [R, O, U], seed=50 + 10 * rank)  # output 3: no check is due
        res = {"checked_before": dict(st.checked_upto)}
        clean = [pool.k_buffer[l][48:80].clone() for l in range(LAYERS)]
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=32, pages=2, donor=0), R)  # shared pages 3-4
        _decode(st, pool, sched, [R, O, U], seed=60 + 10 * rank)  # output 4: R, O and U are due
        hook.release_checks(st, sched)
        found = _records("release_check", rid="rf_R")
        held = hook.held_back(st, [R, O, U])
        res["after_check"] = dict(
            to_abort={r.rid: r.to_abort for r in (R, O, U)}, checked=dict(st.checked_upto), recomputed=recomputed,
            still_damaged=not all(torch.equal(pool.k_buffer[l][48:80], clean[l]) for l in range(LAYERS)),
            poisoned=sorted(st.poisoned), cached_pages=_cached_pages(tree), lock_P=P.lock_ref,
            sent={r.rid: list(r.output_ids) for r in (R, O, U)},
            record={k: found[0].get(k) for k in ("aborted", "poisoned_pages", "other_readers", "dropped_tokens")}
            if found else None)
        for r, tail in held:
            r.output_ids.extend(tail)
        _decode(st, pool, sched, [R, O, U], seed=70 + 10 * rank)
        _decode(st, pool, sched, [R, O, U], seed=80 + 10 * rank)  # output 6: due, but R and O are being aborted
        n_checks = len(_records("release_check"))
        hook.release_checks(st, sched)
        res["aborting_not_checked"] = (len(_records("release_check")) == n_checks
                                       and st.checked_upto["rf_R"] == 2 and st.checked_upto["rf_U"] == 6)
        # R and O end with their abort at their next step: SGLang caches them and they let go of their pages
        gen_before = int(st.gen[3])
        for r in (R, O):
            r.finished_reason = "abort"
            tree.cache_finished_req(r)
        res["after_finish"] = dict(
            finished=list(tree.finished), finish_checks=len(_records("release_check", finishing=True)),
            root=sorted(n.value.tolist()[0] for n in tree.root_node.children.values()),
            head=[(n.value.tolist(), n.lock_ref, len(n.children)) for n in tree.root_node.children.values()
                  if int(n.value[0]) == 16],
            cached_pages=_cached_pages(tree), freed=sorted(tree.token_to_kv_pool_allocator.freed),
            evictable=tree.evictable_size_, protected=tree.protected_size_,
            drops=[(x["after"], x["dropped_tokens"]) for x in _records("drop_poisoned")])
        hook.drop_poisoned(st, tree)
        res["poisoned_after"] = sorted(st.poisoned)
        # a damaged page that was freed and given to a new owner is not dropped again
        head = next(n for n in tree.root_node.children.values() if int(n.value[0]) == 16)
        reused = tree.add(head, range(6000, 6016), range(48, 64))
        st.poisoned[3] = gen_before
        res["reused"] = dict(dropped=hook.drop_poisoned(st, tree), kept=reused in head.children.values(),
                             poisoned=sorted(st.poisoned))
        # U finishes with damage its check at finish finds: it ends with an abort, and its damaged page goes
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=16, pages=1, donor=0), U)  # U's prefix page 10
        _decode(st, pool, sched, [U], seed=90 + rank)  # output 7, not checked yet
        tree.cache_finished_req(U)
        fin = U.finished_reason
        res["finish"] = dict(reason=type(fin).__name__, message=getattr(fin, "message", None),
                             checked=st.checked_upto["rf_U"], cached_pages=_cached_pages(tree),
                             record=[(x["finishing"], x.get("aborted"), x.get("poisoned_pages"))
                                     for x in _records("release_check", rid="rf_U")])
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_abort_on_fail_drops_the_damaged_pages_on_two_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_abort_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    for rank in range(RANKS):
        r = res[rank]
        assert r["checked_before"] == {"rf_R": 2, "rf_O": 2, "rf_U": 2}
        a = r["after_check"]
        # R's check finds the shared pages 3-4; R and O (which reads them too) are aborted, U is not
        assert a["to_abort"] == {"rf_R": True, "rf_O": True, "rf_U": False}
        assert a["record"] == dict(aborted=True, poisoned_pages=[3, 4], other_readers=[dict(rid="rf_O", aborted=True)],
                                   dropped_tokens=0)
        # nothing is mended, and R's and O's tokens after their last passing check stay held
        assert a["recomputed"] == [] and a["still_damaged"]
        assert a["checked"] == {"rf_R": 2, "rf_O": 2, "rf_U": 4}
        assert a["sent"] == {"rf_R": [7000, 7001], "rf_O": [8000, 8001], "rf_U": [9000, 9001, 9002, 9003]}
        # while R and O run, they hold the damaged pages, which stay in the cache
        assert a["poisoned"] == [3, 4] and a["lock_P"] == 2 and a["cached_pages"] == list(range(1, 15))
        assert r["aborting_not_checked"]
        f = r["after_finish"]
        assert f["finished"] == ["rf_R", "rf_O"] and f["finish_checks"] == 0  # an abort releases nothing more
        # once both let go: pages 3-4 and everything cached on top of them (5-8) leave; pages 1-2 stay, unlocked
        assert f["drops"] == [("rf_O", 96)]
        assert f["head"] == [(list(range(16, 48)), 0, 0)] and f["root"] == [16, 144]
        assert f["cached_pages"] == [1, 2] + list(range(9, 15))
        assert f["freed"] == list(range(48, 144))
        assert (f["evictable"], f["protected"]) == (32, 96)  # the head is evictable; U's pages are held
        assert r["poisoned_after"] == []  # the dropped pages moved to a new generation
        assert r["reused"] == dict(dropped=0, kept=True, poisoned=[])
        fin = r["finish"]
        assert (fin["reason"], fin["message"]) == ("FINISH_ABORT", "KV damage found; resubmit the request")
        assert fin["checked"] == 6 and fin["record"][-1] == (True, True, [10])
        assert fin["cached_pages"] == [1, 2, 3, 9]  # U's prefix keeps its clean first page; page 3: the new owner


# ---------------------------------------------------------------------------------------------------------------
# deferred

def _deferred_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    hook.STRICT, hook.ON_FAIL = hook._rules("sf_*:2")
    try:
        # R and X share the prefix (pages 1-4); R's question on 5-6, X's on 7-8; decode slots from 15 and 25
        rows = [list(range(16, 112)) + list(range(240, 400)),
                list(range(16, 80)) + list(range(112, 144)) + list(range(400, 560))]
        st, pool, tree, sched, queued = _setup(rank, rows)
        P = tree.add(tree.root_node, SHARED, range(16, 80))
        QR = tree.add(P, range(2000, 2032), range(80, 112))
        R = _req("sf_R", 0, SHARED + list(range(2000, 2032)), 7000, QR)
        tree.inc_lock_ref(QR)
        sched.running_batch = _batch([R], [97])
        _store(st, pool, sched, R, list(range(64)), seed=10 + rank, step=0)  # the prefix, cached before the run
        _store(st, pool, sched, R, list(range(64, 96)), seed=20 + rank, step=1)
        st.step = 1
        hook.release_checks(st, sched)  # R's check after its prefill
        _decode(st, pool, sched, [R], seed=30 + rank)
        st.step = 2
        sched.running_batch.seq_lens[0] = 98
        hook.release_checks(st, sched)  # output 2
        _decode(st, pool, sched, [R], seed=40 + rank)
        sched.running_batch.seq_lens[0] = 99
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=32, pages=2, donor=0), R)  # shared pages 3-4
        # X arrives: its prefill reuses the cached prefix, damage and all, and computes its question on top of it
        QX = tree.add(P, range(3000, 3032), range(112, 144))
        X = _req("sf_X", 1, SHARED + list(range(3000, 3032)), 8000, QX)
        tree.inc_lock_ref(QX)
        _store(st, pool, sched, X, list(range(64, 96)), seed=50 + rank, step=4)
        sched.cur_batch = _batch([X], [97])  # just prefilled: not in the running batch yet
        st.step = 4
        hook.release_checks(st, sched)  # X's check after its prefill finds the damage
        res = {"step4": dict(cached_pages=_cached_pages(tree), pending=dict(st.pending_redo),
                             suspect=sorted(st.pending_suspect), queued=list(queued))}
        st.step = 5
        sched.running_batch, sched.cur_batch = _batch([X], [97]), None  # X joins the running batch
        hook.release_checks(st, sched)
        redo = _records("pending_redo", rid="sf_X")
        res["step5"] = dict(cached_pages=_cached_pages(tree), pending=dict(st.pending_redo),
                            suspect=sorted(st.pending_suspect), queued=list(queued),
                            evicted=redo[0].get("evicted_tokens") if redo else None, P_children=len(P.children))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_a_rollback_that_waits_evicts_its_suspect_prompt_pages():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_deferred_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    for rank in range(RANKS):
        s4, s5 = res[rank]["step4"], res[rank]["step5"]
        # X found the damage; the pages were mended; R rolled back and its question (pages 5-6) left the cache; X's
        # rollback waits, and its question pages (7-8, computed on the damage) stay while it holds them
        assert s4["pending"] == {"sf_X": 0} and s4["suspect"] == ["sf_X"] and s4["queued"] == ["sf_R"]
        assert s4["cached_pages"] == [1, 2, 3, 4, 7, 8]
        # the rollback happens; X's question pages go, so its re-prefill cannot reuse them; the mended prefix stays
        assert s5["pending"] == {} and s5["suspect"] == [] and s5["queued"] == ["sf_R", "sf_X"]
        assert s5["evicted"] == 32 and s5["cached_pages"] == [1, 2, 3, 4] and s5["P_children"] == 0


def test_rules_name_requests_by_id_or_prefix():
    saved = hook.STRICT, hook.ON_FAIL
    try:
        hook.STRICT, hook.ON_FAIL = hook._rules("sc_*:4,rc_*:4:abort,rc_q1*:2,R,X:8")
        rule = {}
        for rid in ("sc_q1", "rc_q1", "rc_q2.1", "R", "X", "fc_q1", "scw_g0", "kvmend-recompute-sc_q1-16"):
            q = types.SimpleNamespace(rid=rid)
            rule[rid] = (hook.strict_every(q), hook.on_fail(q)) if hook.is_strict(q) else None
        assert rule == {"sc_q1": (4, "mend"), "rc_q1": (2, "mend"), "rc_q2.1": (4, "abort"), "R": (1, "mend"),
                        "X": (8, "mend"), "fc_q1": None, "scw_g0": None, "kvmend-recompute-sc_q1-16": None}
        hook.STRICT, hook.ON_FAIL = hook._rules("*:3")
        assert hook.is_strict(types.SimpleNamespace(rid="anything"))
        try:
            hook._rules("R:1:fix")
            raise AssertionError("an unknown action must be refused")
        except ValueError:
            pass
    finally:
        hook.STRICT, hook.ON_FAIL = saved
