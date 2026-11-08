"""Damage during a chunked prefill, on 2 CPU processes with gloo (no SGLang, no GPU). SGLang computes a long prompt in
chunks: each later chunk reads the earlier chunks' K and V from the pool, and its own from the forward. A store skipped
during one chunk leaves an earlier owner's bytes in a page, and every later chunk reads them. The check right after the
prefill finds the page (coverage), and KVMend recomputes it and every position a later chunk computed, before any
decode step reads them: on the release rule's path (strict, which then generates the first token again) and on the path
of KVMEND_CHECK_AFTER_PREFILL=all (released_suspect, which reports the size of the recomputed suffix). The other
positions of the damaged chunk read the page from the forward itself, and are recomputed only where the rule says so.

The stand-ins: a pool that hook.patch_memory_pool patches as it patches SGLang's, so the skipped store goes through the
patched store path; a forward that writes a chunk's clean K and V when the prefix it reads from the pool is clean, and
wrong K and V otherwise (what reading damaged KV does); and an in-place recompute that does the same for the span it
recomputes. R's prompt is 8 pages (positions 0-127; block j on pool page j + 1), cut as SGLang 0.5.1 cuts it with a
64-token chunk and 16-token pages: [0, 49), [49, 113), [113, 128) (a chunk ends inside a page)."""
import os
import queue
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS, POOL_PAGES, HOST_PAGES = 16, 1, 64, 2, 2, 48, 24
PROMPT, CHUNK = 8 * PAGE, 64
CHUNKS = [(0, 49), (49, 113), (113, PROMPT)]
R_IDS = list(range(1000, 1000 + PROMPT))
ROW = list(range(PAGE, PAGE + 256))  # position p on slot 16 + p


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins

class Pool:
    """MHATokenToKVPool's fields that KVMend reads, and its store."""
    page_size, start_layer = PAGE, 0

    def __init__(self):
        self.k_buffer = [torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)]
        self.v_buffer = [torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)]

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
        lid = layer.layer_id if layer_id_override is None else layer_id_override
        self.k_buffer[lid][loc] = cache_k
        self.v_buffer[lid][loc] = cache_v


_PATCHED = []


def _pool():
    """A pool and its state, the store path patched as in SGLang (once per process)."""
    if not _PATCHED:
        hook.patch_memory_pool(types.SimpleNamespace(MHATokenToKVPool=Pool))
        _PATCHED.append(True)
    pool = Pool()
    st = pool._kvm
    st.reset()
    return pool, st


def _clean(rank, seed, n=PROMPT):
    """A request's clean K and V on this rank: [layers][K, V] of [n, heads, dim]."""
    g = torch.Generator().manual_seed(seed + rank)
    return [[torch.randn(n, HEADS, DIM, generator=g).to(torch.bfloat16) for _ in range(2)] for _ in range(LAYERS)]


def _owner(pool, rank):
    """An earlier owner's KV in every slot, written around the store path: what a page holds until a store writes it,
    with no tags (the reset that every flush makes)."""
    g = torch.Generator().manual_seed(900 + rank)
    for buf in pool.k_buffer + pool.v_buffer:
        buf.copy_(torch.randn(buf.shape, generator=g).to(torch.bfloat16))


def _row(sched, req):
    return sched.req_to_token_pool.req_to_token[req.req_pool_idx].to(torch.int64)


def _holds(pool, slots, clean, lo, hi):
    """Do the slots of positions [lo, hi) hold the request's clean K and V, in every layer?"""
    s = slots[lo:hi]
    return all(torch.equal(pool.k_buffer[l][s], clean[l][0][lo:hi])
               and torch.equal(pool.v_buffer[l][s], clean[l][1][lo:hi]) for l in range(LAYERS))


def _write(pool, req, slots, lo, hi, clean, ok):
    """Store the request's K and V of positions [lo, hi) through the (patched) store path, as a forward of its tokens
    does: the clean ones if `ok`, else wrong ones."""
    ids = list(req.origin_input_ids) + list(req.output_ids)
    loc = slots[lo:hi]
    hook._FB[0] = types.SimpleNamespace(out_cache_loc=loc, input_ids=torch.tensor(ids[lo:hi]),
                                        positions=torch.arange(lo, hi))
    try:
        for l in range(LAYERS):
            pool.set_kv_buffer(types.SimpleNamespace(layer_id=l), loc, clean[l][0][lo:hi] + (0 if ok else 1),
                               clean[l][1][lo:hi] + (0 if ok else 1))
    finally:
        hook._FB[0] = None


def _forward(pool, sched, req, lo, hi, clean):
    """One prefill chunk: positions [lo, hi) read [0, lo) from the pool and their own K and V from the forward."""
    slots = _row(sched, req)
    _write(pool, req, slots, lo, hi, clean, _holds(pool, slots, clean, 0, lo))


CALLS = []  # (rid, start, end) of every in-place recompute, in order


def _fake_recompute(st, sched, req, s, e):
    """KVMend's in-place recompute (an extend forward of R's tokens [s, e) on top of R's KV of [0, s) in the pool,
    written over R's own slots): it writes clean K and V if what it reads is clean."""
    CALLS.append((req.rid, s, e))
    clean, slots = st.clean[req.rid], _row(sched, req)
    _write(st.pool, req, slots, s, e, clean, _holds(st.pool, slots, clean, 0, s))


def _batch(reqs, seq_len=PROMPT + 1):
    b = types.SimpleNamespace(reqs=list(reqs), seq_lens=torch.full((len(reqs),), seq_len, dtype=torch.int64))

    def filter_batch(keep_indices):
        b.reqs = [b.reqs[j] for j in keep_indices]
        b.seq_lens = b.seq_lens[torch.tensor(keep_indices, dtype=torch.int64)]

    b.filter_batch = filter_batch
    return b


def _req(rid, row, prompt):
    """A request about to be prefilled (no output token yet)."""
    r = types.SimpleNamespace(rid=rid, req_pool_idx=row, origin_input_ids=list(prompt), output_ids=[],
                              fill_ids=[], extend_input_len=0, prefix_indices=list(range(len(prompt))),
                              last_node=rid, send_token_offset=0, to_abort=False, is_chunked=0, finished_reason=None)
    r.finished = lambda r=r: r.finished_reason is not None

    def reset_for_retract(r=r):  # as SGLang's: the request gives up its row of the token table
        r.prefix_indices, r.req_pool_idx = [], None

    r.reset_for_retract = reset_for_retract
    return r


def _setup(rank, strict, after_prefill, rows, tree=None):
    """The settings, a pool and its state, and a scheduler whose token table has `rows` (lists of slots)."""
    hook.STRICT, hook.ON_FAIL = hook._rules(strict)
    hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL = "mend", after_prefill
    hook.PLAN.items, hook.PLAN.done = {}, set()
    pool, st = _pool()
    st.step = 1
    st.clean = {}
    table = torch.zeros(len(rows), 256, dtype=torch.int32)
    for i, slots in enumerate(rows):
        table[i] = torch.tensor(slots[:256], dtype=torch.int32)
    world = dist.group.WORLD if dist.is_initialized() else None
    queued = []
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE, chunked_prefill_size=CHUNK, running_batch=_batch([]), cur_batch=None,
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=table, free=lambda i: None),
        token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: None),
        tree_cache=tree or types.SimpleNamespace(dec_lock_ref=lambda node: None),
        _extend_requests_to_queue=lambda reqs, is_retracted: queued.extend(r.rid for r in reqs))
    st.sched = sched
    del hook._BUF[:], CALLS[:]
    return st, pool, sched, queued


def _step(st, sched, batch=None):
    """The end of a scheduler iteration: the prefill batch's forwards are recorded (note_fills, as the patched
    process_batch_result does), then on_step moves to the next step and runs the step's checks."""
    if batch is not None:
        hook.note_fills(st, batch)
    st.step += 1
    st.step_t.fill_(st.step)
    hook.release_checks(st, sched)


def _prefill(st, pool, sched, reqs, chunks, out0=7000):
    """The requests' prefill, chunk by chunk, all in one batch per chunk, as the scheduler runs it: right before each
    forward the plan's skipped stores are armed (before_forward, as the patched run_batch does), after it they are
    disarmed; then the step ends. The last chunk chooses each request's first output token, and the current batch
    still holds the requests at the step's checks."""
    for k, (lo, hi) in enumerate(chunks):
        for req in reqs:
            req.fill_ids, req.extend_input_len = list(req.origin_input_ids)[:hi], hi - lo
        batch = _batch(reqs)
        sched.cur_batch = batch
        armed = hook.before_forward(st, sched, batch)
        st.step_t.fill_(st.step)
        for req in reqs:
            _forward(pool, sched, req, lo, hi, st.clean[req.rid])
        if armed or st.skip is not None:
            hook.after_forward(st, sched, armed)
        if k == len(chunks) - 1:
            for j, req in enumerate(reqs):
                req.output_ids.append(out0 + 1000 * j)
        _step(st, sched, batch)


def _decode(st, pool, sched, reqs, seed):
    """One decode step: each request stores its newest token's KV (random) through the store path and chooses its next
    token."""
    for k, req in enumerate(reqs):
        pos = len(req.origin_input_ids) + len(req.output_ids) - 1
        g = torch.Generator().manual_seed(seed + k)
        noise = [[torch.randn(pos + 1, HEADS, DIM, generator=g).to(torch.bfloat16) for _ in range(2)]
                 for _ in range(LAYERS)]
        st.step_t.fill_(st.step)
        _write(pool, req, _row(sched, req), pos, pos + 1, noise, True)
        req.output_ids.append(req.output_ids[-1] + 1)


def _records(op, **match):
    return [x for x in hook._BUF if x.get("op") == op and all(x.get(k) == v for k, v in match.items())]


def _spans(mend):
    return [(x["start"], x["end"]) for x in mend.get("recomputed", [])]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(main):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    return dict(out.get(timeout=300) for _ in range(RANKS))


def _init(rank, port):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute


# ---------------------------------------------------------------------------------------------------------------
# The rule, on the record of a request's prefill forwards (no processes)

def test_the_rule_takes_every_later_forward_and_only_those():
    fills = [(1, 0, 49), (2, 49, 113), (3, 113, 128)]
    st = types.SimpleNamespace(fills={"R": fills}, good_step={})
    R = types.SimpleNamespace(rid="R")

    def rule(blocks, n=PROMPT, **kw):
        s = hook.suspect_suffix(st, R, set(blocks), n, **kw)
        return None if s is None else (s["first"], s["start"], s["end"], s["tokens"], s["forwards"], s["damage_in"])

    # a page of chunk 1: chunks 2 and 3 read it; the suffix starts at the page that holds chunk 2's first position
    assert rule({1}) == (49, 48, 128, 80, 2, [0, 49])
    assert rule({1, 6}) == rule({1})  # the first damaged page decides
    # a page that chunks 1 and 2 share counts as chunk 1's (the safe side)
    assert rule({3}) == (49, 48, 128, 80, 2, [0, 49])
    # a page of chunk 2: only chunk 3 read it; the rest of chunk 2 read it from the forward
    assert rule({4}) == (113, 112, 128, 16, 1, [49, 113])
    assert rule({7}) == (113, 112, 128, 16, 1, [49, 113])  # shared by chunks 2 and 3
    # a page of the last chunk alone: no later chunk read it
    st.fills["R"] = [(1, 0, 49), (2, 49, 96), (3, 96, 128)]
    assert rule({6}) is None and rule({7}) is None
    st.fills["R"] = fills
    # a decode page (after the prefill) was read by no forward of the prefill
    assert rule({8}, n=PROMPT + 2) is None
    # the last good check bounds it: damage found later struck after it, and the prefill ran before it
    st.good_step["R"] = 4
    assert rule({1}) is None
    assert rule({1}, use_last_good=False) == (49, 48, 128, 80, 2, [0, 49])  # forced: as if R had none
    st.good_step["R"] = 3  # a check after chunk 2: only chunk 3 ran after it
    assert rule({1})[:5] == (113, 112, 128, 16, 1)
    st.good_step.clear()
    # a prefill on top of a cached prefix: a damaged prefix page was read by every forward of the prefill
    st.fills["R"] = [(5, 32, 80), (6, 80, 128)]
    assert rule({1}) == (32, 32, 128, 96, 2, "prefix")
    # nothing to go on
    assert rule(set()) is None
    st.fills.clear()
    assert rule({1}) is None


def test_each_prefill_forward_is_recorded_until_the_request_rolls_back():
    pool, st = _pool()
    R, Q, D = _req("R", 0, R_IDS), _req("Q", 1, range(50)), _req("D", 2, range(40))

    def fill(req, lo, hi):
        req.fill_ids, req.extend_input_len = list(req.origin_input_ids)[:hi], hi - lo

    st.step = 7
    fill(R, 0, 49)
    fill(D, 39, 40)  # a decode token of a mixed batch: not prefill
    batch = _batch([R, D])
    batch.decoding_reqs = [D]
    hook.note_fills(st, batch)
    st.step = 8
    fill(R, 49, 113)
    fill(Q, 0, 50)
    Q.finished_reason = "length"  # finished by its prefill: nothing left to protect
    hook.note_fills(st, _batch([R, Q]))
    assert st.fills == {"R": [(7, 0, 49), (8, 49, 113)]}
    st.step = 9
    fill(R, 96, 113)  # not where the record ends: a new prefill (after a rollback that kept 96 tokens cached)
    hook.note_fills(st, _batch([R]))
    assert st.fills == {"R": [(9, 96, 113)]}
    hook.forget(st, "R")
    assert st.fills == {}


def test_a_rollback_from_a_chunked_prefill_goes_back_to_its_first_chunk():
    """A rollback from the prefill undoes it whole: back to the prefix its first chunk started from, not only to its
    last chunk's start (all SGLang keeps of it)."""
    calls = {}
    R = _req("R", 0, R_IDS)
    R.output_ids, R.fill_ids, R.extend_input_len = [9], R_IDS, 15  # its last chunk computed tokens 113-127
    st = types.SimpleNamespace(fills={"R": [(4, 32, 49), (5, 49, 113), (6, 113, 128)]}, alarm_found=True)
    prefill = types.SimpleNamespace(reqs=[R], seq_lens=torch.tensor([PROMPT]), forward_mode=types.SimpleNamespace(
        is_extend=lambda: True), filter_batch=lambda keep_indices: calls.setdefault("kept", keep_indices))
    sched = types.SimpleNamespace(
        page_size=PAGE, running_batch=types.SimpleNamespace(reqs=[]), cur_batch=prefill,
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.tensor(ROW).view(1, 256),
                                                free=lambda i: calls.setdefault("slot", i)),
        token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: calls.setdefault("freed", idx.numel())),
        tree_cache=types.SimpleNamespace(dec_lock_ref=lambda node: calls.setdefault("unlocked", node)),
        _extend_requests_to_queue=lambda reqs, is_retracted: calls.setdefault("queued", (len(reqs), is_retracted)))
    rec = hook.act_retract(st, sched, dict(keep=0, from_prefill=True), R)
    assert rec["retracted"] and (rec["kept_prefix"], rec["freed_tokens"]) == (32, 96), rec
    assert rec["regenerated"] == 1 and calls["queued"] == (1, True) and calls["kept"] == []
    # without a record (or one that does not reach what the request fills now), the last chunk's start, as before
    R2 = _req("R2", 0, R_IDS)
    R2.output_ids, R2.fill_ids, R2.extend_input_len = [9], R_IDS, 15
    prefill.reqs = [R2]
    st.fills = {"R2": [(4, 32, 49)]}
    assert hook.act_retract(st, sched, dict(keep=0, from_prefill=True), R2)["kept_prefix"] == 112


def test_a_skipped_store_strikes_the_forward_that_computes_its_span():
    """The fault is armed right before a prefill forward, for that forward only. The span's stores are dropped: its
    pages keep the earlier owner's bytes and have no tags, while the rest of the chunk lands with its tags. With
    keep_bytes the bytes land and only the tags are skipped. A span that two forwards compute is struck in both."""
    cases = {}
    saved = (hook.STRICT, hook.ON_FAIL, hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL, hook.PLAN.items, hook.PLAN.done)
    try:
        for name, fault in (("skip", dict(pos=16, pages=1)), ("keep_bytes", dict(pos=16, pages=1, keep_bytes=True)),
                            ("two_forwards", dict(pos=48, pages=2)), ("in_chunk_2", dict(pos=64, pages=1))):
            cases[name] = _skip_case(fault)
    finally:
        (hook.STRICT, hook.ON_FAIL, hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL, hook.PLAN.items,
         hook.PLAN.done) = saved
        del hook._BUF[:]
    s = cases["skip"]
    assert s["due_before"] == [] and s["skip_after"] is None  # not an after-step action; disarmed after its forward
    assert s["recs"] == [dict(at=0, n_out=0, armed=True, positions=[16, 32], pages=[2], forward=[0, 49],
                              later_forwards=True, stores_dropped=LAYERS, stale_same_stores=0, keep_bytes=False,
                              reason=None)], s["recs"]
    assert s["done"] == [("R", 0)] and s["stale"] == [True] * LAYERS and not s["clean_page1"] and s["rest_clean"]
    assert s["mask"] == [0] * LAYERS and s["mask_next"] == [(1 << PAGE) - 1] * LAYERS  # no tags; the next page has
    assert s["fills"] == [(1, 0, 49), (2, 49, 113), (3, 113, 128)]
    k = cases["keep_bytes"]
    assert k["recs"][0]["armed"] and k["recs"][0]["stores_dropped"] == 0 and k["clean_page1"]  # the bytes landed
    assert k["mask"] == [0] * LAYERS  # and their tags did not
    t = cases["two_forwards"]
    assert [(x["positions"], x["forward"]) for x in t["recs"]] == [([48, 49], [0, 49]), ([49, 80], [49, 113])]
    assert t["done"] == [("R", 0)]
    c = cases["in_chunk_2"]  # chunk 1 does not compute the span: armed for chunk 2 only
    assert [(x["positions"], x["forward"]) for x in c["recs"]] == [([64, 80], [49, 113])], c["recs"]


def _skip_case(fault):
    """R's prefill with a skipped store (and an after-step check, to show the plan's own steps skip the fault)."""
    st, pool, sched, _ = _setup(0, "", "strict", [ROW])
    _owner(pool, 0)
    R = _req("R", 0, R_IDS)
    st.clean = {"R": _clean(0, 100)}
    stale = [pool.k_buffer[l].clone() for l in range(LAYERS)]
    hook.PLAN.items = {"R": [dict(op="fault", at=0, kind="skipped_store", **fault), dict(op="check", at=1)]}
    due_before = [a["op"] for _, _, a, _ in hook.PLAN.due(types.SimpleNamespace(running_batch=_batch([R]),
                                                                                cur_batch=None))]
    _prefill(st, pool, sched, [R], CHUNKS)
    slots = _row(sched, R)
    recs = [{k: x.get(k) for k in ("at", "n_out", "armed", "positions", "pages", "forward", "later_forwards",
                                   "stores_dropped", "stale_same_stores", "keep_bytes", "reason")}
            for x in _records("fault", rid="R")]
    return dict(
        due_before=due_before, recs=recs, done=sorted(hook.PLAN.done), skip_after=st.skip,
        stale=[torch.equal(pool.k_buffer[l][slots[16:32]], stale[l][slots[16:32]]) for l in range(LAYERS)],
        clean_page1=_holds(pool, slots, st.clean["R"], 16, 32), rest_clean=_holds(pool, slots, st.clean["R"], 0, 16),
        mask=[int(st.mask[l][2]) for l in range(LAYERS)], mask_next=[int(st.mask[l][3]) for l in range(LAYERS)],
        fills=list(st.fills["R"]))


def test_the_digest_tells_kv_apart_bit_for_bit():
    pool, st = _pool()
    g = torch.Generator().manual_seed(3)
    for buf in pool.k_buffer + pool.v_buffer:
        buf.copy_(torch.randn(buf.shape, generator=g).to(torch.bfloat16))
    slots = torch.arange(PAGE, PAGE + PROMPT)
    d0 = hook._kv_digest(st, slots)
    assert d0 == hook._kv_digest(st, slots.clone()) and hook._kv_digest(st, slots[:0]) == 0
    words = pool.v_buffer[1].view(torch.int16)
    words[PAGE + 70, 0, 9] ^= 1  # one bit of one V value, in layer 1
    assert hook._kv_digest(st, slots) != d0
    st.fills = {"R": [(1, 0, 49), (2, 49, 113), (3, 113, 128)]}
    R = _req("R", 0, R_IDS)
    sched = types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=torch.tensor(ROW).view(1, 256)))
    rec = hook.act_digest(st, sched, dict(pos=16, pages=1), R)
    # chunk 1 ends at 49, inside the page 48-63 that the rule recomputes with the later chunks
    assert rec["positions"] == dict(whole=[0, 128], before=[0, 16], span=[16, 32], same_chunk=[32, 48],
                                    later=[48, 128])
    assert rec["digest"]["whole"] == hook._kv_digest(st, slots) and rec["forwards"] == [[0, 49], [49, 113], [113, 128]]


# ---------------------------------------------------------------------------------------------------------------
# A store skipped during chunk 1 or 2, found right after the prefill: strict and all

def _chunks_main(rank, port, out):
    _init(rank, port)
    res = {}
    try:
        for path in ("strict", "all"):
            for pos in (16, 64):  # block 1 (chunk 1) or block 4 (chunk 2)
                st, pool, sched, queued = _setup(rank, "R" if path == "strict" else "",
                                                 "strict" if path == "strict" else "all", [ROW])
                _owner(pool, rank)
                R = _req("R", 0, R_IDS)
                st.clean = {"R": _clean(rank, 100)}
                hook.PLAN.items = {"R": [dict(op="fault", at=0, kind="skipped_store", pos=pos, pages=1)]}
                _prefill(st, pool, sched, [R], CHUNKS)
                slots = _row(sched, R)
                found = _records("release_check" if path == "strict" else "released_suspect", rid="R")
                held = hook.held_back(st, [R])
                for q, tail in held:  # as around SGLang's stream call: what was cut goes back
                    q.output_ids.extend(tail)
                r = res[(path, pos)] = dict(
                    records=len(found), found=found[0]["found"] if found else None,
                    spans=_spans(found[0]["mend"]) if found else None,
                    suffix={k: found[0]["mend"]["suffix"][k] for k in ("first", "start", "end", "tokens", "damage_in")}
                    if found and found[0]["mend"].get("suffix") else None,
                    suffix_tokens=found[0].get("suffix_tokens") if found else None,
                    reported=found[0].get("suffix") if found else None,
                    still=found[0]["mend"].get("still_failing_here") if found else None,
                    calls=[c[1:] for c in CALLS], kv_clean=_holds(pool, slots, st.clean["R"], 0, PROMPT),
                    pending=dict(st.pending_redo), outputs=list(R.output_ids), held=[(q.rid, t) for q, t in held])
                if path == "strict":  # the next step: R joins the running batch, and its rollback happens
                    sched.running_batch, sched.cur_batch = _batch([R], PROMPT + 2), None
                    _decode(st, pool, sched, [R], seed=50 + rank)
                    _step(st, sched)
                    r["redo"] = [(x["retracted"], x["regenerated"]) for x in _records("pending_redo", rid="R")]
                    r.update(queued=list(queued), record_gone="R" not in st.fills)
        # damage found later, after a passing check right after the prefill: the last good check bounds it, and only
        # the damaged page is recomputed (decode tokens since that check are generated again, as before)
        st, pool, sched, queued = _setup(rank, "R", "strict", [ROW])
        _owner(pool, rank)
        R = _req("R", 0, R_IDS)
        st.clean = {"R": _clean(rank, 100)}
        _prefill(st, pool, sched, [R], CHUNKS)
        sched.running_batch, sched.cur_batch = _batch([R], PROMPT + 2), None
        _decode(st, pool, sched, [R], seed=60 + rank)
        _step(st, sched)  # output 2, checked
        checked_before, slots = st.checked_upto.get("R"), _row(sched, R)  # R gives up its row when it rolls back
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=16, pages=1, donor=0), R)
        sched.running_batch.seq_lens[0] = PROMPT + 3
        _decode(st, pool, sched, [R], seed=70 + rank)
        _step(st, sched)  # output 3: the check finds the late write
        late = _records("release_check", rid="R")[-1]
        res["late"] = dict(checked_before=checked_before, records=len(_records("release_check", rid="R")),
                           spans=_spans(late["mend"]), suffix=late["mend"].get("suffix"),
                           suffix_tokens=late["suffix_tokens"], regenerated=late["regenerate"].get("regenerated"),
                           kv_clean=_holds(pool, slots, st.clean["R"], 0, PROMPT))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_later_chunks_are_recomputed_after_a_failed_check_after_the_prefill():
    res = _spawn(_chunks_main)
    every = {"ranks": list(range(RANKS)), "layers": LAYERS}
    for rank in range(RANKS):
        for path in ("strict", "all"):
            one, two = res[rank][(path, 16)], res[rank][(path, 64)]
            # the check right after the prefill finds the page whose store was skipped, by coverage, on every GPU
            assert one["records"] == 1 and one["found"] == {"2": {"coverage": every}}, (path, one)
            assert two["records"] == 1 and two["found"] == {"5": {"coverage": every}}, (path, two)
            # chunk 1's page: it is recomputed, then every position from the page holding chunk 2's first position
            # (49) to the end, in pieces of one prefill chunk. Positions 0-15 and 32-47, computed by chunk 1 beside
            # the damage, read it from the forward and are not recomputed; 48 shares a page with chunk 2 and is
            assert one["spans"] == one["calls"] == [(16, 32), (48, 112), (112, 128)], (path, one)
            assert one["suffix"] == dict(first=49, start=48, end=128, tokens=80, damage_in=[0, 49])
            # chunk 2's page: only chunk 3 read it; positions 80-111 of chunk 2 are not recomputed
            assert two["spans"] == two["calls"] == [(64, 80), (112, 128)], (path, two)
            assert two["suffix"] == dict(first=113, start=112, end=128, tokens=16, damage_in=[49, 113])
            for r in (one, two):
                assert r["kv_clean"] and r["still"] == 0, (path, r)  # every position holds R's clean KV again
        for pos, tokens, start in ((16, 80, 48), (64, 16, 112)):
            s, a = res[rank][("strict", pos)], res[rank][("all", pos)]
            # strict: no unchecked token leaves; the first token is generated again (the rollback waits a step)
            assert s["suffix_tokens"] == tokens and s["pending"] == {"R": 0} and s["held"] == [("R", [7000])]
            assert s["redo"] == [(True, 2)] and s["queued"] == ["R"] and s["record_gone"]
            # all: the first token has left; the report says so, and how much was recomputed after the damage
            assert a["suffix_tokens"] == tokens
            assert a["reported"] == dict(first=a["suffix"]["first"], start=start, end=PROMPT,
                                         damage_in=a["suffix"]["damage_in"])
            assert a["pending"] == {} and a["held"] == [] and a["outputs"] == [7000]
        late = res[rank]["late"]
        assert late["checked_before"] == 2 and late["records"] == 1  # passed right after the prefill and at output 2
        assert late["spans"] == [(16, 32)] and late["suffix"] is None and late["suffix_tokens"] == 0
        assert late["regenerated"] == 1 and late["kv_clean"]


# ---------------------------------------------------------------------------------------------------------------
# Damage in the prefix cache, read by every forward of a prefill; two readers of it in one batched check

SHARED = 2 * PAGE  # a cached prefix of 2 pages (pool pages 1-2), written before the prefills
S_IDS, N_IDS = R_IDS[:SHARED] + list(range(3032, 3096)), R_IDS[:SHARED] + list(range(4032, 4096))
S_ROW = ROW[:SHARED] + list(range(20 * PAGE, 20 * PAGE + 224))  # S's own part on pool pages 20-23
N_ROW = ROW[:SHARED] + list(range(30 * PAGE, 30 * PAGE + 224))  # N's on 30-33


def _prefix_main(rank, port, out):
    _init(rank, port)
    res = {}
    try:
        # R alone, outside the release rule: its prefill started on top of the cached prefix, after a late write
        # landed in the prefix's page 2 (block 1); both chunks of the prefill read it
        st, pool, sched, _ = _setup(rank, "", "all", [ROW])
        R = _req("R", 0, R_IDS)
        st.clean = {"R": _clean(rank, 100)}
        _write(pool, R, _row(sched, R), 0, SHARED, st.clean["R"], True)  # cached before, by an earlier request
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=16, pages=1, donor=0), R)
        _prefill(st, pool, sched, [R], [(SHARED, 80), (80, PROMPT)])
        rep = _records("released_suspect", rid="R")
        res["prefix"] = dict(found=rep[0]["found"], spans=_spans(rep[0]["mend"]), suffix_tokens=rep[0]["suffix_tokens"],
                             reported=rep[0]["suffix"], calls=list(CALLS),
                             kv_clean=_holds(pool, _row(sched, R), st.clean["R"], 0, PROMPT))
        # strict S and N, outside the rule, share that prefix; S's check comes first in the batched check and mends
        # the page, so N's own check passes, but N's prefill read the damage all the same
        st, pool, sched, _ = _setup(rank, "S", "all", [S_ROW, N_ROW])
        S, N = _req("S", 0, S_IDS), _req("N", 1, N_IDS)
        st.clean = {"S": _clean(rank, 300, len(S_IDS)), "N": _clean(rank, 400, len(N_IDS))}
        for l in range(LAYERS):
            for kv in range(2):
                st.clean["N"][l][kv][:SHARED] = st.clean["S"][l][kv][:SHARED]
        _write(pool, S, _row(sched, S), 0, SHARED, st.clean["S"], True)
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=16, pages=1, donor=0), S)
        _prefill(st, pool, sched, [S, N], [(SHARED, len(S_IDS))])
        n_rep, s_rep = _records("released_suspect", rid="N"), _records("release_check", rid="S")
        res["shared"] = dict(
            calls=list(CALLS), s_spans=_spans(s_rep[0]["mend"]), s_suffix=s_rep[0]["suffix_tokens"],
            n_found=n_rep[0]["found"], n_mended=n_rep[0]["mend"]["mended"], n_spans=_spans(n_rep[0]["mend"]),
            n_suffix=n_rep[0]["suffix_tokens"], n_where=(n_rep[0]["suffix"] or {}).get("damage_in"),
            clean={q.rid: _holds(pool, _row(sched, q), st.clean[q.rid], 0, len(q.origin_input_ids)) for q in (S, N)},
            pending=dict(st.pending_redo))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_damage_in_the_prefix_cache_taints_everything_the_prefill_computed():
    res = _spawn(_prefix_main)
    every = {"ranks": list(range(RANKS)), "layers": LAYERS}
    for rank in range(RANKS):
        p = res[rank]["prefix"]
        assert p["found"] == {"2": {"tag": every}}
        # the page is recomputed, and everything the prefill computed after it (32-127), in pieces of 64 tokens
        assert p["spans"] == [(16, 80), (80, 128)] and p["suffix_tokens"] == 96 and p["kv_clean"], p
        assert p["reported"] == dict(first=SHARED, start=SHARED, end=PROMPT, damage_in="prefix")
        s = res[rank]["shared"]
        # S's check mends the shared page and S's own part; N's check then passes, and N's own part, which its
        # prefill computed from the damage, is recomputed all the same
        assert s["calls"] == [("S", 16, 80), ("S", 80, 96), ("N", 32, 96)], s["calls"]
        assert s["s_suffix"] == 64 and s["pending"] == {"S": 0}
        assert s["n_found"] == {} and s["n_mended"] and s["n_spans"] == [(32, 96)] and s["n_suffix"] == 64
        assert s["n_where"] == "prefix" and s["clean"] == {"S": True, "N": True}


# ---------------------------------------------------------------------------------------------------------------
# HiCache: host copies made from the recomputed KV go

class _Node:
    def __init__(self, nid, value, host_value, parent):
        self.id, self.value, self.host_value, self.parent = nid, value, host_value, parent
        self.children, self.lock_ref = {}, 1  # the running request holds its path
        if parent is not None:
            parent.children[nid] = self


class _HiTree:
    """The parts of SGLang 0.5.1's HiRadixCache that dropping host copies touches."""

    def __init__(self, host_freed):
        self.root_node = _Node(0, None, None, None)
        self.ongoing_write_through = {}
        self.cache_controller = types.SimpleNamespace(
            ack_write_queue=queue.Queue(),
            mem_pool_host=types.SimpleNamespace(free=lambda idx: host_freed.append(idx.clone())))
        self.dec_lock_ref = lambda node: None

    def writing_check(self):  # releases the nodes whose backups landed
        q = self.cache_controller.ack_write_queue
        for _ in range(q.qsize()):
            self.ongoing_write_through.pop(q.get())


def _slots(lo, hi, first=PAGE):
    return torch.arange(first + lo, first + hi, dtype=torch.int64)


def _hicache_main(rank, port, out):
    _init(rank, port)
    try:
        host_freed = []
        tree = _HiTree(host_freed)
        st, pool, sched, _ = _setup(rank, "", "strict", [ROW], tree=tree)
        _owner(pool, rank)
        R = _req("R", 0, R_IDS)
        st.clean = {"R": _clean(rank, 100)}
        hook.PLAN.items = {"R": [dict(op="fault", at=0, kind="skipped_store", pos=16, pages=1)]}
        _prefill(st, pool, sched, [R], CHUNKS)
        # R's path: A holds positions 0-47, B 48-111, C 112-127. A and B were backed up after their chunks (write
        # through; B's backup has landed but is not released yet); C's host copy is older than its device page
        st.hrec = hook.HostRecord(st, HOST_PAGES * PAGE)
        A = _Node(1, _slots(0, 48), _slots(48, 96, 0), tree.root_node)
        B = _Node(2, _slots(48, 112), _slots(96, 160, 0), A)
        C = _Node(3, _slots(112, 128), _slots(160, 176, 0), B)
        R.last_node = C
        for node, made in ((A, None), (B, None), (C, 0)):
            hp = torch.div(node.host_value[::PAGE], PAGE, rounding_mode="floor")
            dp = torch.div(node.value[::PAGE], PAGE, rounding_mode="floor")
            st.hrec.valid[hp] = True
            st.hrec.wstep[hp] = st.wstep[dp] if made is None else made
        tree.ongoing_write_through[B.id] = B
        tree.cache_controller.ack_write_queue.put(B.id)
        check = hook.act_check(st, sched, {}, R)
        rec = hook.act_mend(st, sched, dict(suffix=True), R)
        out.put((rank, dict(found=check["found"], spans=_spans(rec), dropped=rec.get("host_copies_dropped"),
                            freed=[x.tolist() for x in host_freed],
                            host={n.id: n.host_value is not None for n in (A, B, C)},
                            ongoing=sorted(tree.ongoing_write_through),
                            kv_clean=_holds(pool, _row(sched, R), st.clean["R"], 0, PROMPT))))
    finally:
        dist.destroy_process_group()


def test_host_copies_made_from_the_recomputed_kv_are_dropped():
    res = _spawn(_hicache_main)
    for rank in range(RANKS):
        r = res[rank]
        assert r["spans"] == [(16, 32), (48, 112), (112, 128)] and r["kv_clean"], r
        # A's and B's host copies were made from the bytes now recomputed (B's once its backup was released); C's
        # is older than its device page, made before the damage, and stays
        assert r["host"] == {1: False, 2: False, 3: True} and r["dropped"] == 48 + 64
        assert r["freed"] == [_slots(96, 160, 0).tolist(), _slots(48, 96, 0).tolist()] and r["ongoing"] == []
