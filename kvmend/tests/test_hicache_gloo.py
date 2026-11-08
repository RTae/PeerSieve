"""Loads from host memory (SGLang's hierarchical cache, HiCache) on 2 CPU processes with gloo. A fill request's KV is
backed up to host memory, its device copy is evicted, and request R2 with the same prompt loads it back into new
device pages. The tags travel with the host copy; a load that does not land on one GPU (a partial load) is found by
the check on arrival or by R2's reader check, on that GPU only, and repaired by a copy from the host copy, or, when the
host copy is damaged too, by recompute in place. No SGLang and no GPU: the controller, the host pool, the allocator
and the scheduler are stand-ins with the fields the patches read, and the real patches are applied to them; the
write and load threads copy with plain indexing, the load thread in a thread of its own as in SGLang."""
import os
import queue
import socket
import threading
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS = 16, 1, 64, 2, 2
PROMPT_PAGES, POOL_PAGES, HOST_PAGES = 6, 16, 24
PROMPT = PROMPT_PAGES * PAGE  # R2 reads positions 0-95: it has one output token, so it stores the prompt only
HOST_FIRST = 3  # host pages 0-2 are taken, so host page numbers differ from device page numbers
OLD = list(range(1, 7))  # the fill request's device pages
NEW = list(range(7, 13))  # the device pages the load gets (they hold an earlier owner's KV)
STRUCK = ["11", "12"]  # a partial load of 2 pages strikes the load's last two pages: positions 64-95


class Op:  # HiCache's CacheOperation
    def __init__(self, host_indices, device_indices, node_id, priority=None):
        self.host_indices, self.device_indices, self.node_ids, self.priority = (host_indices, device_indices,
                                                                                [node_id], priority)


class HostBase:  # HostKVCache: frees host slots
    def free(self, indices):
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


class Host(HostBase):  # MHATokenToKVPoolHost with the layer_first layout
    def __init__(self, device_pool):
        self.device_pool, self.size, self.layout = device_pool, HOST_PAGES * PAGE, "layer_first"
        self.kv_buffer = torch.zeros(2, LAYERS, self.size, HEADS, DIM, dtype=torch.bfloat16)
        self.free_slots = torch.arange(HOST_FIRST * PAGE, self.size)

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def alloc(self, n):
        out, self.free_slots = self.free_slots[:n], self.free_slots[n:]
        return out

    def protect_load(self, indices):
        pass

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        device_pool.k_buffer[layer_id][device_indices] = self.k_buffer[layer_id][host_indices]
        device_pool.v_buffer[layer_id][device_indices] = self.v_buffer[layer_id][host_indices]


class Alloc:  # PagedTokenToKVPoolAllocator: whole pages; a free drops the pages' tags, as the patched free does
    def __init__(self, st, first=False):  # first: freed pages go out first, as in SGLang 0.5.1
        self.st, self.free_pages, self.first = st, list(range(1, POOL_PAGES)), first

    def alloc(self, n):
        out, self.free_pages = self.free_pages[:n // PAGE], self.free_pages[n // PAGE:]
        return (torch.tensor(out).unsqueeze(1) * PAGE + torch.arange(PAGE)).flatten()

    def free(self, indices):
        pages = torch.unique(indices // PAGE)
        self.free_pages = pages.tolist() + self.free_pages if self.first else self.free_pages + pages.tolist()
        self.st.drop(pages)


class Controller:  # HiCacheController: write and load queue copies for the write and load threads
    def __init__(self, st, host, alloc):
        self.mem_pool_device, self.mem_pool_host, self.mem_pool_device_allocator = st.pool, host, alloc
        self.write_queue, self.load_queue = queue.Queue(), queue.Queue()

    def write(self, device_indices, priority=None, node_id=0):
        host_indices = self.mem_pool_host.alloc(len(device_indices))
        self.write_queue.put(Op(host_indices, device_indices, node_id, priority))
        return host_indices

    def load(self, host_indices, priority=None, node_id=0):  # SGLang's own (the patch replaces it): no tags carried
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        self.load_queue.put(Op(host_indices, device_indices, node_id, priority))
        return device_indices

    def run_writes(self):  # the write thread: every layer of each backup
        while not self.write_queue.empty():
            op = self.write_queue.get()
            for layer in range(LAYERS):
                self.mem_pool_host.k_buffer[layer][op.host_indices] = self.mem_pool_device.k_buffer[layer][op.device_indices]
                self.mem_pool_host.v_buffer[layer][op.host_indices] = self.mem_pool_device.v_buffer[layer][op.device_indices]

    def run_loads(self):  # the load thread: the queued loads merged into one, copied layer by layer
        def loop():
            ops = []
            while not self.load_queue.empty():
                ops.append(self.load_queue.get())
            hs, ds = torch.cat([o.host_indices for o in ops]), torch.cat([o.device_indices for o in ops])
            for layer in range(LAYERS):
                self.mem_pool_host.load_to_device_per_layer(self.mem_pool_device, hs, ds, layer, "kernel")

        th = threading.Thread(target=loop)
        th.start()
        th.join()


PLAIN_LOAD = Controller.load
_PATCHED = []


def _patch_once():
    if not _PATCHED:
        hook.patch_cache_controller(types.SimpleNamespace(HiCacheController=Controller, CacheOperation=Op))
        hook.patch_memory_pool_host(types.SimpleNamespace(MHATokenToKVPoolHost=Host, HostKVCache=HostBase))
        _PATCHED.append(True)


def _kv(pool, slots):
    return torch.stack([torch.stack([pool.k_buffer[l][slots], pool.v_buffer[l][slots]]) for l in range(LAYERS)])


def _setup(rank, first=False):
    """The fill request's prompt on device pages 1-6, written through the store path; an earlier owner's KV on pages
    7-12; a controller, a host pool and a scheduler with R2's token table still empty. first: the allocator hands
    freed pages out first, as SGLang's does, so the load gets pages 1-6 back."""
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    st.reset()
    pool._kvm = st
    alloc, host = Alloc(st, first), Host(pool)
    ctl = Controller(st, host, alloc)
    ids = list(range(1000, 1000 + PROMPT))
    slots = alloc.alloc(PROMPT)
    tk = T.token_keys(torch.tensor(ids), torch.arange(PROMPT))
    g = torch.Generator().manual_seed(100 + rank)  # each rank holds its own heads
    old = torch.arange(NEW[0] * PAGE, (NEW[-1] + 1) * PAGE)
    for layer in range(LAYERS):
        pool.k_buffer[layer][slots] = torch.randn(PROMPT, HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][slots] = torch.randn(PROMPT, HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, slots, tk)
        pool.k_buffer[layer][old] = torch.randn(len(old), HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][old] = torch.randn(len(old), HEADS, DIM, generator=g).to(torch.bfloat16)
    world = dist.group.WORLD
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE, tree_cache=types.SimpleNamespace(token_to_kv_pool_host=host),
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world))
    req = types.SimpleNamespace(rid="R2", req_pool_idx=0, origin_input_ids=ids, output_ids=[5000])
    return st, pool, host, alloc, ctl, sched, req, slots


def _round_trip(st, pool, host, alloc, ctl, sched, req, slots, load=None, arm=None, split=False, host_damage=False,
                free_host=False):
    """Back the prompt up to host memory, evict its device copy, then load it back for R2 and point R2's token table
    at the new pages. arm: a partial-load fault armed before the load (with split, before the second of two loads,
    which the load thread copies as one); host_damage: one bit of the host copy of page 12's first token flips (layer
    0) on rank 1 after the backup."""
    clean = _kv(pool, slots).clone()
    tags = st.tags[:, OLD].clone()
    host_slots = ctl.write(slots, node_id=1)
    ctl.run_writes()
    if host_damage and sched.tp_rank == 1:
        words = host.k_buffer[0].view(torch.int16).view(host.size, -1)
        words[int(host_slots[5 * PAGE]), 3] ^= 1 << 3
    if free_host:
        host.free(host_slots)
    alloc.free(slots)
    load = load or Controller.load
    if split:
        first = load(ctl, host_slots[:3 * PAGE], node_id=2)
        hook.act_fault(st, sched, arm, req)
        dev = torch.cat([first, load(ctl, host_slots[3 * PAGE:], node_id=3)])
    else:
        if arm:
            hook.act_fault(st, sched, arm, req)
        dev = load(ctl, host_slots, node_id=2)
    ctl.run_loads()
    sched.req_to_token_pool = types.SimpleNamespace(
        req_to_token=torch.cat([dev, torch.zeros(256 - PROMPT, dtype=torch.int64)]).view(1, 256))
    st.saved_span = (dev.clone(), clean)  # what a recompute must write back, and the yardstick of act_mend
    return dict(dev=dev, clean=clean, tags=tags)


def _fake_recompute(st, sched, req, s, e):
    """Stand-in for the extend forward: write R2's clean KV of [s, e) back through the store path, keyed by R2's
    tokens."""
    tgt, clean = st.saved_span
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, s:e].to(torch.int64)
    assert torch.equal(row, tgt[s:e]), "the recompute must write over R2's own slots"
    tk = T.token_keys(torch.tensor(req.origin_input_ids[s:e]), torch.arange(s, e))
    for layer in range(st.layers):
        st.pool.k_buffer[layer][row] = clean[layer, 0, s:e]
        st.pool.v_buffer[layer][row] = clean[layer, 1, s:e]
        st.update(layer, row, tk)


def _events(st, op):
    return [x for x in st.load_events if x["op"] == op]


def _rank_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    _patch_once()
    hook._recompute_range = _fake_recompute
    res = {}
    try:
        # the tags travel with the host copy: the loaded pages carry the fill request's tags and pass R2's check
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        rt = _round_trip(*setup)
        r = res["carry"] = dict(check=hook.act_check(st, sched, {}, req), backup=_events(st, "backup"),
                                issue=_events(st, "load_issue"), arrival=_events(st, "load_check"))
        r.update(dev_pages=sorted(set((rt["dev"] // PAGE).tolist())),
                 tags_carried=torch.equal(st.tags[:, NEW], rt["tags"]),
                 full=bool((st.mask[:, NEW] == hook.FULL).all()),
                 keys_carried=torch.equal(st.tkey[rt["dev"]],
                                          T.token_keys(torch.tensor(req.origin_input_ids), torch.arange(PROMPT))),
                 bytes_clean=torch.equal(_kv(pool, rt["dev"]), rt["clean"]),
                 src=st.host_copy(NEW[0]).tolist())
        hook.drain_load_events(st, types.SimpleNamespace(tp_rank=rank, cur_batch=types.SimpleNamespace(reqs=[req])))
        r["drained"] = [(x["op"], x["rids"]) for x in hook._BUF]
        del hook._BUF[:]
        # SGLang's own load carries no tags: every loaded page fails R2's check as uncovered, on every GPU
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        _round_trip(*setup, load=PLAIN_LOAD)
        res["no_carry"] = dict(check=hook.act_check(st, sched, {}, req), arrival=_events(st, "load_check"))
        # host slots freed after the backup lose their records: a load of them is uncovered too
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        _round_trip(*setup, free_host=True)
        res["host_freed"] = dict(check=hook.act_check(st, sched, {}, req), arrival=_events(st, "load_check"))
        # a partial load on rank 1 (two loads copied as one; the fault strikes the second): the check on arrival
        # finds the stale pieces there and copies them again before anything reads them
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        rt = _round_trip(*setup, arm=dict(kind="partial_load", rank=1, pages=2), split=True)
        res["arrival"] = dict(issue=_events(st, "load_issue"), arrival=_events(st, "load_check"),
                              check=hook.act_check(st, sched, {}, req),
                              bytes_clean=torch.equal(_kv(pool, rt["dev"]), rt["clean"]), errors=_events(st, "error"))
        # SGLang's allocator gives the load the pages the eviction just freed: their stale bytes are the very bytes
        # the load brings, so a transfer that does not land there changes nothing, and nothing is found
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank, first=True)
        rt = _round_trip(*setup, arm=dict(kind="partial_load", rank=1, pages=2))
        res["same_slots"] = dict(dev_pages=sorted(set((rt["dev"] // PAGE).tolist())), arrival=_events(st, "load_check"),
                                 check=hook.act_check(st, sched, {}, req),
                                 bytes_clean=torch.equal(_kv(pool, rt["dev"]), rt["clean"]))
        # the same fault, not checked on arrival: R2's reader check finds it on rank 1 by tag; mend copies it again
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        rt = _round_trip(*setup, arm=dict(kind="partial_load", rank=1, pages=2, check_on_arrival=False))
        r = res["reader"] = dict(arrival=_events(st, "load_check"), stale=not torch.equal(_kv(pool, rt["dev"]),
                                                                                        rt["clean"]))
        r["check1"] = hook.act_check(st, sched, {}, req)
        r["mend"] = hook.act_mend(st, sched, {}, req)
        r["check2"] = hook.act_check(st, sched, {}, req)
        r["bytes_clean"] = torch.equal(_kv(pool, rt["dev"]), rt["clean"])
        # zeros, not checked on arrival, and mend told not to use the host copy: recompute in place
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        rt = _round_trip(*setup, arm=dict(kind="partial_load", rank=1, pages=2, mode="zeros", check_on_arrival=False))
        r = res["recompute"] = dict(zeros=bool((_kv(pool, rt["dev"][4 * PAGE:]) == 0).all()))
        r["check1"] = hook.act_check(st, sched, {}, req)
        r["mend"] = hook.act_mend(st, sched, dict(recopy=False), req)
        r["check2"] = hook.act_check(st, sched, {}, req)
        r["bytes_clean"] = torch.equal(_kv(pool, rt["dev"]), rt["clean"])
        # the host copy on rank 1 is damaged too: the copy on arrival cannot fix page 12 in layer 0, R2's check
        # finds it, mend's copy fails again, and the page is recomputed
        st, pool, host, alloc, ctl, sched, req, slots = setup = _setup(rank)
        rt = _round_trip(*setup, arm=dict(kind="partial_load", rank=1, pages=2), host_damage=True)
        r = res["host_bad"] = dict(arrival=_events(st, "load_check"))
        r["check1"] = hook.act_check(st, sched, {}, req)
        r["mend"] = hook.act_mend(st, sched, {}, req)
        r["check2"] = hook.act_check(st, sched, {}, req)
        r["bytes_clean"] = torch.equal(_kv(pool, rt["dev"]), rt["clean"])
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_hicache_loads_on_two_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    every = list(range(RANKS))
    new = [str(p) for p in NEW]
    for rank in every:
        c = res[rank]["carry"]
        assert c["dev_pages"] == NEW and c["backup"] == [dict(op="backup", tokens=PROMPT, carried=True)]
        assert c["issue"] == [dict(op="load_issue", tokens=PROMPT, pages=PROMPT_PAGES, carried_pages=PROMPT_PAGES)]
        assert c["tags_carried"] and c["full"] and c["keys_carried"] and c["bytes_clean"]
        assert c["src"] == list(range((HOST_FIRST + 0) * PAGE, (HOST_FIRST + 1) * PAGE))  # host page 3, not page 7
        a = c["arrival"]
        assert len(a) == 1 and a[0]["checked"] and a[0]["pages"] == PROMPT_PAGES and a[0]["found_pieces"] == 0
        assert a[0]["uncovered_pieces"] == 0 and a[0]["recopied_pieces"] == 0 and "fault" not in a[0]
        assert c["check"]["found"] == {}, c["check"]  # no false alarm after a load
        assert c["drained"] == [("backup", ["R2"]), ("load_issue", ["R2"]), ("load_check", ["R2"])]
        for case in ("no_carry", "host_freed"):  # no record: uncovered on arrival (not copied again) and to R2
            n = res[rank][case]
            assert n["arrival"][0]["uncovered_pieces"] == PROMPT_PAGES * LAYERS, case
            assert n["arrival"][0]["recopied_pieces"] == 0, case
            assert n["check"]["found"] == {p: {"coverage": {"ranks": every, "layers": LAYERS}} for p in new}, case
        # on arrival: only rank 1's struck pieces fail; they are copied again, and R2 then reads clean KV
        a = res[rank]["arrival"]
        assert a["errors"] == []
        assert [x.get("fault", {}).get("pages") for x in a["issue"]] == [None, [11, 12]]
        assert a["issue"][1]["fault"]["load_positions"] == [16, 48]  # the second load's last two pages
        assert len(a["arrival"]) == 1 and a["arrival"][0]["pages"] == PROMPT_PAGES  # two loads, copied as one
        got = a["arrival"][0]
        assert got["fault"]["pages"] == [11, 12] and got["fault"]["struck_here"] == (rank == 1)
        if rank == 1:
            assert got["found_pages"] == [11, 12] and got["found_pieces"] == 2 * LAYERS
            assert got["recopied_pieces"] == 2 * LAYERS and got["still_failing_pieces"] == 0
            assert got["recopy_equals_host_copy"] is True and got["fault"]["stale_was_zero"] is False
            assert got["fault"]["changed_layers"] == LAYERS
        else:
            assert got["found_pieces"] == 0 and got["recopied_pieces"] == 0
        assert a["check"]["found"] == {} and a["bytes_clean"]
        # the load got the pages just freed: the struck pieces already hold the right bytes, nothing changed
        same = res[rank]["same_slots"]
        assert same["dev_pages"] == OLD and same["check"]["found"] == {} and same["bytes_clean"]
        assert same["arrival"][0]["found_pieces"] == 0
        assert same["arrival"][0]["fault"]["changed_layers"] == (0 if rank == 1 else None)
        # by the reader's check: rank 1 by tag, both layers; mend copies the pieces again, nothing is recomputed
        r = res[rank]["reader"]
        assert r["arrival"][0]["checked"] is False and r["stale"] == (rank == 1)
        assert r["check1"]["found"] == {p: {"tag": {"ranks": [1], "layers": LAYERS}} for p in STRUCK}
        m = r["mend"]
        assert m["recomputed"] == [] and m["rebuilt"] == [] and "recopy_failed_pages" not in m
        assert sorted((x["layer"], x["page"]) for x in m["recopied"]) == [(l, p) for l in range(LAYERS) for p in (11, 12)]
        assert all(x.get("recopied") and x.get("tag_matches") and x.get("equals_host_copy")
                   for x in m["recopied"]) == (rank == 1)
        assert m["max_abs_diff_vs_clean"] == 0.0 and r["check2"]["found"] == {} and r["bytes_clean"]
        # recompute in place instead of the copy
        r = res[rank]["recompute"]
        assert r["zeros"] == (rank == 1)
        assert r["check1"]["found"] == {p: {"tag": {"ranks": [1], "layers": LAYERS}} for p in STRUCK}
        assert r["mend"]["recopied"] == [] and [(x["start"], x["end"]) for x in r["mend"]["recomputed"]] == [(64, 96)]
        assert r["mend"]["still_failing_here"] == 0 and r["check2"]["found"] == {} and r["bytes_clean"]
        # a damaged host copy: the copies cannot fix page 12 in layer 0 on rank 1, so it is recomputed
        r = res[rank]["host_bad"]
        got = r["arrival"][0]
        if rank == 1:
            assert got["found_pieces"] == 2 * LAYERS and got["still_failing"] == [(0, 12)]
            assert got["recopy_equals_host_copy"] is True  # the copy is exact: it is the host copy that is wrong
        else:
            assert got["found_pieces"] == 0
        assert r["check1"]["found"] == {"12": {"tag": {"ranks": [1], "layers": 1}}}
        assert r["mend"]["recopy_failed_pages"] == [12]
        assert [(x["start"], x["end"]) for x in r["mend"]["recomputed"]] == [(80, 96)]
        assert r["check2"]["found"] == {} and r["bytes_clean"]


class _Acks:
    """HiCache's queue of finished backups (a queue.Queue: its deque and its lock): the backup of node 7 lands at
    the third look."""

    def __init__(self):
        self.mutex, self.looks = threading.Lock(), 0

    @property
    def queue(self):
        self.looks += 1
        return [7] if self.looks >= 3 else []


def test_offload_waits_for_the_trees_backups_then_evicts_the_device_copy():
    """It waits for the backup of node 7, which is in the tree, and not for node 5, a backup of an earlier tree whose
    finished backup SGLang's reset dropped: waiting for node 5 would wait for ever."""
    calls = []
    leaf = types.SimpleNamespace(id=7, value=torch.arange(32), host_value=torch.arange(64, 96), children={})
    root = types.SimpleNamespace(id=0, value=None, host_value=None, children={1: leaf})
    stale = types.SimpleNamespace(id=5, value=torch.arange(16), host_value=torch.arange(16), children={})

    class Tree:
        root_node, ongoing_write_through = root, {7: leaf, 5: stale}
        cache_controller = types.SimpleNamespace(ack_write_queue=_Acks())

        def writing_check(self):  # SGLang's releases the nodes whose backups landed
            calls.append(("writing_check", self.cache_controller.ack_write_queue.looks))
            del self.ongoing_write_through[7]

        def evictable_size(self):
            return 0 if leaf.value is None else len(leaf.value)

        def evict(self, n):
            calls.append(("evict", n))
            leaf.value = None  # a node with a host copy keeps it

    rec = hook.act_offload(None, types.SimpleNamespace(tree_cache=Tree()), dict(timeout=60), None)
    assert calls == [("writing_check", 3), ("evict", 32)] and rec["waited_s"] < 30
    assert rec["offloaded"] and rec["writes_waited_for"] == 1 and rec["pending_writes"] == 0
    assert rec["evictable_before"] == 32 and rec["evictable_after"] == 0 and rec["host_only_tokens"] == 32


def test_offload_without_hierarchical_cache_does_nothing():
    rec = hook.act_offload(None, types.SimpleNamespace(tree_cache=types.SimpleNamespace()), {}, None)
    assert rec == dict(offloaded=False, reason="no hierarchical cache")


class _Node:  # a node of SGLang's radix tree: its tokens, its device slots, its host slots (HiCache), its locks
    def __init__(self, nid, key, value, host_value, parent):
        self.id, self.key, self.value, self.host_value, self.parent = nid, key, value, host_value, parent
        self.children, self.lock_ref = {}, 0
        if parent is not None:
            parent.children[tuple(key[:PAGE])] = self


class _Tree:
    """The parts of SGLang 0.5.1's RadixCache, and with hicache of its HiRadixCache, that a rollback touches, with the
    same lock and size accounting."""

    def __init__(self, hicache, freed, host_freed):
        self.root_node, self.evictable_size_, self.ongoing_write_through = _Node(0, [], None, None, None), 0, {}
        self.token_to_kv_pool_allocator = types.SimpleNamespace(free=lambda idx: freed.append(idx.clone()))
        if hicache:
            self.cache_controller = types.SimpleNamespace(
                ack_write_queue=queue.Queue(),
                mem_pool_host=types.SimpleNamespace(free=lambda idx: host_freed.append(idx.clone())),
                evict_device=lambda dev, host: (freed.append(dev.clone()), len(dev))[1])

    def add(self, nid, key, value, host_value, parent):
        node = _Node(nid, key, value, host_value, parent)
        self.evictable_size_ += len(value) if value is not None else 0
        return node

    def inc_lock_ref(self, node):
        while node is not self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node):
        while node is not self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
            node.lock_ref -= 1
            node = node.parent

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v is node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _evict_backuped(self, node):
        n = self.cache_controller.evict_device(node.value, node.host_value)
        self.evictable_size_ -= n
        node.value = None
        return n

    def writing_check(self):  # releases the nodes whose backups landed
        q = self.cache_controller.ack_write_queue
        for _ in range(q.qsize()):
            self.dec_lock_ref(self.ongoing_write_through.pop(q.get()))


def _slots(*pages):
    return torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in pages])


def _rolled_back_prefill(case):
    """Request R loaded tokens 0-31 (pages 1-2) from host memory, and its prefill computed tokens 32-47 into page 3
    from them, after the load struck. SGLang cached that prefill right after it (cache_unfinished_req), before any
    check ran. case "adopted": a fill request had cached tokens 32-47 too, evicted since with a clean host copy, so
    SGLang's insert gave R's page to that node (job 326443's case); "no_host": a plain radix cache, which made a new
    node; "new_backup": HiCache made a new node and backed it up from R's page (the backup has landed, not yet
    released). R's reader check then found the damage, and R is rolled back from its prefill."""
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)])
    st = hook.PoolState(pool)
    st.reset()
    st.wstep[torch.tensor([1, 2])] = 3  # written by the fill request at step 3 (the load restored that step)
    st.wstep[3] = 7  # R's prefill, at step 7
    st.hrec = hook.HostRecord(st, HOST_PAGES * PAGE)
    st.hrec.valid[torch.tensor([3, 4, 5, 6])] = True
    st.hrec.wstep[torch.tensor([3, 4, 5])] = 3  # the fill's backups
    st.hrec.wstep[6] = 7  # a backup made from R's page 3
    freed, host_freed = [], []
    tree = _Tree(case != "no_host", freed, host_freed)
    a = tree.add(1, list(range(32)), _slots(1, 2), _slots(3, 4) if case != "no_host" else None, tree.root_node)
    b = tree.add(2, list(range(32, 48)), _slots(3), {"adopted": _slots(5), "new_backup": _slots(6)}.get(case), a)
    tree.inc_lock_ref(a)  # load_back's lock, released when the load is acknowledged
    tree.inc_lock_ref(b)  # R's own lock on its last node
    if case == "new_backup":
        tree.inc_lock_ref(b)  # the backup's lock
        tree.ongoing_write_through[b.id] = b
        tree.cache_controller.ack_write_queue.put(b.id)
    calls = {}
    req = types.SimpleNamespace(rid="R", req_pool_idx=0, fill_ids=list(range(48)), extend_input_len=16,
                                prefix_indices=list(range(48)), last_node=b, output_ids=[9], send_token_offset=0,
                                reset_for_retract=lambda: calls.setdefault("reset", True))
    prefill = types.SimpleNamespace(reqs=[req], seq_lens=torch.tensor([48]),
                                    forward_mode=types.SimpleNamespace(is_extend=lambda: True),
                                    filter_batch=lambda keep_indices: calls.setdefault("kept", keep_indices))
    row = torch.cat([_slots(1, 2, 3), torch.zeros(208, dtype=torch.int64)]).view(1, 256)
    sched = types.SimpleNamespace(
        page_size=PAGE, running_batch=types.SimpleNamespace(reqs=[]), cur_batch=prefill, tree_cache=tree,
        token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator,
        req_to_token_pool=types.SimpleNamespace(req_to_token=row, free=lambda i: calls.setdefault("slot", i)),
        _extend_requests_to_queue=lambda reqs, is_retracted: calls.setdefault("queued", (len(reqs), is_retracted)))
    rec = hook.act_retract(st, sched, dict(keep=0, from_prefill=True), req)
    return rec, tree, a, b, freed, host_freed, calls, req


def test_a_rollback_from_the_prefill_drops_what_the_prefill_cached():
    """Everything the prefill computed after the damage struck is suspect: R goes back to the prefix it loaded (32),
    and its page 3 leaves the prefix cache. Kept on the GPU, it would serve R's second prefill: SGLang's insert would
    free the page that prefill computes and point R at the cached one, computed from the damage (job 326443)."""
    for case in ("adopted", "no_host", "new_backup"):
        rec, tree, a, b, freed, host_freed, calls, req = _rolled_back_prefill(case)
        assert rec["retracted"] and (rec["kept_prefix"], rec["freed_tokens"]) == (32, 16), (case, rec)
        assert rec["regenerated"] == 1 and req.output_ids == [] and calls["queued"] == (1, True), case
        dropped = [x for x in freed if x.numel()]
        assert len(dropped) == 1 and torch.equal(dropped[0], _slots(3)), (case, freed)  # page 3, from the cache
        assert a.value is not None and torch.equal(a.value, _slots(1, 2)) and a.lock_ref == 1, case  # the load stays
        d = rec["dropped_from_cache"]
        if case == "adopted":  # the node keeps the fill's clean host copy; the next prefill's page replaces R's
            assert b.value is None and torch.equal(b.host_value, _slots(5)) and a.children == {tuple(range(32, 48)): b}
            assert host_freed == [] and d == dict(nodes=1, device_tokens=16, host_tokens=0, host_kept_tokens=16,
                                                  locked_tokens=0)
        else:  # no clean host copy: the node goes, and a host copy made from R's page goes with it
            assert a.children == {}, case
            assert [x.tolist() for x in host_freed] == ([_slots(6).tolist()] if case == "new_backup" else []), case
            assert d == dict(nodes=1, device_tokens=16, host_tokens=16 if case == "new_backup" else 0,
                             host_kept_tokens=0, locked_tokens=0), (case, d)
            assert tree.ongoing_write_through == {}, case
        assert tree.evictable_size_ == 0, (case, tree.evictable_size_)  # a is still locked by the load
