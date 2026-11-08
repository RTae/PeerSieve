"""Checks on read that repair (KVMEND_ATTN_ACT=repair), on 2 CPU processes with gloo (no SGLang, no GPU).

Every step goes through the scheduler's process_batch_result as hook.patch_scheduler patches it in SGLang, with
stand-ins for what that patch wraps and reads:
- SGLang's own processing: a decode batch passes over a request marked is_retracted (as SGLang 0.5.1's
  process_batch_result_decode does) and appends every other request's token; an extend batch appends each request's
  token; both stream what is new for streamed requests (fast release), the patch cutting strict requests' unchecked
  tokens;
- the check on read: the fake decode step records the slots each request's attention read, and the verdict compares
  every piece read with its stored tag and coverage, as attn_check's verdict does;
- the model: a forward that reads a byte that differs from the clean KV, on any GPU, writes wrong KV and samples WRONG
  (the GPUs agree, as after the model's tensor-parallel all-reduce), so a token computed from damage shows;
- the in-place recompute: it writes the clean KV if the prefix it reads is clean.

  repair    a bit flip on GPU 1 in a prompt page that strict S and fast F share: the next decode step's check on read
            finds it, the page is rebuilt from parity bit for bit, S and F drop that step's token (WRONG is never
            appended or streamed) and generate it again, S keeping its tokens before that step, still held until its
            next release check; N, which reads other pages, goes on; KV cached on top of the page since the readers'
            last passing check leaves the prefix cache
  log       the same with KVMEND_ATTN_ACT=log: the failure is logged, and everything else is as with no check on read
            at all (none): F streams WRONG
  gave_up   F was rolled back ATTN_MAX_REPAIRS times already: its token goes on, as with log; S is repaired
  every GPU a late write on every GPU into a cached page that A (decoding, checked) and two new requests read, and damage
            in A's own decode page: the cached page is recomputed in place, A's own page goes with A's rollback; the
            new requests' prefills read the damage before any check, so their prefill KV after the page is recomputed
            too, and their prefill token is dropped if it has not left (C), or reported (B, streamed)
"""
import functools
import os
import socket
import subprocess
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS, POOL_PAGES = 16, 1, 64, 2, 2, 48
PROMPT = 6 * PAGE
WRONG = 66666  # what a forward that read damaged KV samples
BASE = dict(F=7000, S=8000, N=9000, A=7000, B=8000, C=9000)  # the right token at output j is BASE[rid] + j


def _right(rid, n):
    return [BASE[rid] + j for j in range(n)]


# ---------------------------------------------------------------------------------------------------------------
# Stand-ins

class Sched:
    """SGLang's Scheduler: what KVMend's patch wraps (process_batch_result, stream_output_generation, ...) and reads."""

    def process_batch_result(self, batch, result):
        for req, tok in zip(batch.reqs, result.next_token_ids):
            if req.is_retracted:  # SGLang 0.5.1: a request retracted while its batch was in flight
                continue
            req.output_ids.append(tok)
        self.stream_output_generation(batch.reqs, False)

    def stream_output_generation(self, reqs, return_logprob, skip_req=None):
        for req in reqs:
            if req.stream:
                self.sent.setdefault(req.rid, []).extend(req.output_ids[req.send_token_offset:])
                req.send_token_offset = len(req.output_ids)

    def self_check_during_idle(self):
        pass

    def flush_cache(self):
        return True

    def run_batch(self, batch):
        return None


class Check:
    """attn_check.AttnCheck's interface on CPU: read() takes the slots a request's attention read in this step, and
    verdict() compares every piece read with its stored tag and coverage (coverage: the offsets read differ from those
    written; tag: the bytes, keyed by the slots' own token keys, do not match the tag), then starts over."""

    def __init__(self, st):
        self.st, self.reads, self.unsupported, self.reuse = st, {}, {}, True
        self.layer_of = {b.data_ptr(): l for l, b in enumerate(st.pool.k_buffer)}
        self.reset()

    def reset(self):
        self.stats = dict(steps=0, pieces=0, failures=0, layout=0, seconds=0.0, logged=0)
        self.reads = {}

    def read(self, slots):
        for s in slots.tolist():
            self.reads[s // PAGE] = self.reads.get(s // PAGE, 0) | (1 << (s % PAGE))

    def verdict(self):
        st, reads, self.reads = self.st, self.reads, {}
        pages = sorted(p for p in reads if p > 0)  # page 0 is SGLang's padding page
        found = []
        if pages:
            idx = torch.tensor(pages)
            want = torch.tensor([reads[p] for p in pages], dtype=torch.int32)
            for l in range(st.layers):
                cover, ok = (st.mask[l][idx] == want).tolist(), st.check(l, idx).tolist()
                found += [(l, p, "tag" if c else "coverage") for p, c, o in zip(pages, cover, ok) if not (c and o)]
        self.stats["steps"] += 1
        return found, len(pages) * st.layers, 0


class Node:
    """A node of the prefix cache: the slots of its pages."""

    def __init__(self, parent, pages, lock=0):
        self.parent, self.children, self.lock_ref = parent, {}, lock
        self.value = (torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in pages]) if pages
                      else torch.zeros(0, dtype=torch.int64))
        if parent is not None:
            parent.children[len(parent.children)] = self


def _tree(evicted):
    """The prefix cache: root, then A holding pages 1-4 (the shared prompt pages, locked by their running readers),
    with two cached children: B (page 40) and C (page 41), each a finished request's KV."""
    root = Node(None, [], lock=1)
    a = Node(root, [1, 2, 3, 4], lock=1)
    Node(a, [40])
    Node(a, [41])

    def delete_leaf(node):
        del node.parent.children[next(k for k, v in node.parent.children.items() if v is node)]

    return types.SimpleNamespace(root_node=root, dec_lock_ref=lambda node: None, _delete_leaf=delete_leaf,
                                 token_to_kv_pool_allocator=types.SimpleNamespace(
                                     free=lambda v: evicted.append(int(v[0]) // PAGE)))


def _batch(reqs, mode):
    b = types.SimpleNamespace(reqs=list(reqs), seq_lens=torch.zeros(len(reqs), dtype=torch.int64), decoding_reqs=None,
                              forward_mode=types.SimpleNamespace(is_decode=lambda: mode == "decode",
                                                                 is_extend=lambda: mode == "extend"))

    def filter_batch(keep_indices):
        b.reqs = [b.reqs[j] for j in keep_indices]
        b.seq_lens = b.seq_lens[torch.tensor(keep_indices, dtype=torch.int64)]

    b.filter_batch = filter_batch
    return b


def _req(rid, row, prompt, stream=True):
    r = types.SimpleNamespace(rid=rid, req_pool_idx=row, origin_input_ids=list(prompt), output_ids=[],
                              prefix_indices=[], last_node=rid, send_token_offset=0, stream=stream, to_abort=False,
                              is_chunked=0, finished_reason=None, is_retracted=False, fill_ids=[], extend_input_len=0)
    r.finished = lambda r=r: r.finished_reason is not None

    def reset_for_retract(r=r):  # as SGLang's: the request gives up its row of the token table
        r.prefix_indices, r.req_pool_idx, r.is_retracted = [], None, True

    r.reset_for_retract = reset_for_retract
    return r


def _setup(rank, rows, act, tree=None):
    """A pool, its state with the check on read (None for act "none"), and a scheduler whose token table has `rows`."""
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    st.reset()
    pool._kvm = st
    st.attn = Check(st) if act != "none" else None
    st.ctrl["attn_act"] = "log" if act == "none" else act  # what the KVMEND_CTRL file sets at a flush
    table = torch.zeros(len(rows), 256, dtype=torch.int32)
    for i, slots in enumerate(rows):
        table[i] = torch.tensor(slots[:256], dtype=torch.int32)
    world = dist.group.WORLD
    sched = Sched()
    sched.__dict__.update(
        tp_rank=rank, page_size=PAGE, running_batch=_batch([], "decode"), cur_batch=None, sent={}, queued=[],
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=table, free=lambda i: None),
        token_to_kv_pool_allocator=types.SimpleNamespace(  # as the patched allocator: freed pages lose their tags
            get_kvcache=lambda: pool, free=lambda idx: st.drop(torch.unique(idx.to(torch.int64) // PAGE))),
        tree_cache=tree if tree is not None else types.SimpleNamespace(dec_lock_ref=lambda node: None))
    sched._extend_requests_to_queue = lambda reqs, is_retracted: sched.queued.extend(r.rid for r in reqs)
    st.sched = sched
    hook._BUF.clear()
    return types.SimpleNamespace(rank=rank, pool=pool, st=st, sched=sched, table=table, calls=[])


# ---------------------------------------------------------------------------------------------------------------
# The fake model

@functools.lru_cache(maxsize=None)
def _kv(tok, pos, rank):
    """The clean K and V of token `tok` at position `pos` on this GPU: [layers, 2, heads, dim]."""
    g = torch.Generator().manual_seed((tok * 7919 + pos * 104729 + rank * 1299709) % (2 ** 31))
    return torch.randn(LAYERS, 2, HEADS, DIM, generator=g).to(torch.bfloat16)


def _ids(req):
    return list(req.origin_input_ids) + list(req.output_ids)


def _store(ctx, req, positions, ok):
    """Write the KV of the request's positions through the store path, keyed by its tokens: the clean KV if what the
    forward read was clean, else wrong KV."""
    if not positions:
        return
    ids = _ids(req)
    loc = ctx.table[req.req_pool_idx, torch.tensor(positions)].to(torch.int64)
    kv = torch.stack([_kv(ids[p], p, ctx.rank) for p in positions])
    kv = kv if ok else kv + 1
    tk = T.token_keys(torch.tensor([ids[p] for p in positions]), torch.tensor(positions))
    for l in range(LAYERS):
        ctx.pool.k_buffer[l][loc] = kv[:, l, 0]
        ctx.pool.v_buffer[l][loc] = kv[:, l, 1]
        ctx.st.update(l, loc, tk)


def _intact(ctx, req, n):
    """Does the request's KV of positions [0, n) hold the clean bytes on this GPU?"""
    if n <= 0:
        return True
    ids = _ids(req)
    row = ctx.table[req.req_pool_idx, :n].to(torch.int64)
    want = torch.stack([_kv(ids[p], p, ctx.rank) for p in range(n)])
    have = torch.stack([torch.stack([ctx.pool.k_buffer[l][row], ctx.pool.v_buffer[l][row]], 1)
                        for l in range(LAYERS)], 1)
    return torch.equal(have, want)


def _agree(flags):
    """The model's tensor-parallel all-reduce: damage read on any GPU makes every GPU's result wrong."""
    t = torch.tensor([int(f) for f in flags], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return [bool(x) for x in t.tolist()]


def _decode(ctx):
    """One decode step of the running batch: each request stores its newest token's KV (its attention reads its
    positions [0, n), which the check on read records), samples its next token, and the step's result goes through the
    patched process_batch_result."""
    batch = ctx.sched.running_batch
    ctx.sched.cur_batch = batch
    n = [len(_ids(r)) for r in batch.reqs]
    ok = _agree([_intact(ctx, r, m - 1) for r, m in zip(batch.reqs, n)])
    for i, (req, m, good) in enumerate(zip(batch.reqs, n, ok)):
        _store(ctx, req, [m - 1], good)
        batch.seq_lens[i] = m
        if ctx.st.attn is not None:
            ctx.st.attn.read(ctx.table[req.req_pool_idx, :m].to(torch.int64))
    toks = [BASE[r.rid] + len(r.output_ids) if good else WRONG for r, good in zip(batch.reqs, ok)]
    ctx.sched.process_batch_result(batch, types.SimpleNamespace(next_token_ids=toks))


def _prefill(ctx, reqs, start):
    """One extend batch: each request computes positions [start, n) on top of its cached prefix [0, start) (a chunk
    reads its prefix from the pool and its own positions from the forward, so they are clean iff the prefix is), then
    samples its next token; SGLang caches what it filled (prefix_indices) and merges the batch into the running batch
    at its next step."""
    batch = _batch(reqs, "extend")
    ctx.sched.cur_batch = batch
    for req in reqs:
        req.req_pool_idx = ctx.row_of[req.rid]  # a rolled-back request gets its row back (the same one, here)
        req.is_retracted = False
    ok = _agree([_intact(ctx, r, start[r.rid]) for r in reqs])
    for i, (req, good) in enumerate(zip(reqs, ok)):
        ids = _ids(req)
        req.fill_ids, req.extend_input_len = ids, len(ids) - start[req.rid]
        _store(ctx, req, list(range(start[req.rid], len(ids))), good)
        req.prefix_indices = list(range(len(ids)))
        batch.seq_lens[i] = len(ids)
    toks = [BASE[r.rid] + len(r.output_ids) if good else WRONG for r, good in zip(reqs, ok)]
    ctx.sched.process_batch_result(batch, types.SimpleNamespace(next_token_ids=toks))
    rb = ctx.sched.running_batch
    rb.reqs += list(reqs)
    rb.seq_lens = torch.cat([rb.seq_lens, batch.seq_lens])


def _fake_recompute(st, sched, req, s, e):
    """KVMend's in-place recompute: R's tokens [s, e) on top of R's KV of [0, s) in the pool, over R's own slots; it
    writes clean KV if what it reads is clean."""
    ctx = st.test_ctx
    ctx.calls.append((req.rid, s, e))
    _store(ctx, req, list(range(s, e)), _intact(ctx, req, s))


def _records(op):
    return [x for x in hook._BUF if x.get("op") == op]


def _page_clean(ctx, req, page, positions):
    """Does `page` hold the request's clean KV of `positions` on this GPU (its slots in the request's row)?"""
    ids = _ids(req)
    return all(torch.equal(torch.stack([ctx.pool.k_buffer[l][page * PAGE + o], ctx.pool.v_buffer[l][page * PAGE + o]]),
                           _kv(ids[p], p, ctx.rank)[l]) for l in range(LAYERS) for o, p in enumerate(positions))


def _start_rank(rank, port):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook.patch_scheduler(types.SimpleNamespace(Scheduler=Sched))
    hook._flush_log = lambda sched: None  # the records stay in hook._BUF; no file is written
    hook._recompute_range = _fake_recompute
    hook.STRICT, hook.ON_FAIL = hook._rules("S:4")
    hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL = "mend", "strict"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(main):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    return dict(out.get(timeout=300) for _ in range(RANKS))


# ---------------------------------------------------------------------------------------------------------------
# A bit flip on one GPU in a shared page: repair, log, no check, and a reader that gave up

F_IDS = list(range(1000, 1000 + PROMPT))
S_IDS = F_IDS[:4 * PAGE] + list(range(3064, 3064 + 2 * PAGE))  # S shares F's first four prompt pages
F_ROW = list(range(PAGE, PAGE + 256))                            # F: pages 1-6, then its decode pages from 7
S_ROW = F_ROW[:4 * PAGE] + list(range(20 * PAGE, 20 * PAGE + 192))  # S: F's pages 1-4, then pages 20-21, from 22
N_ROW = list(range(30 * PAGE, 30 * PAGE + 256))                  # N: pages 30-31, then from 32


def _one_gpu(rank, act, gave_up=False):
    evicted = []
    ctx = _setup(rank, [F_ROW, S_ROW, N_ROW], act, _tree(evicted))
    st, sched = ctx.st, ctx.sched
    st.test_ctx = ctx
    ctx.row_of = dict(F=0, S=1, N=2)
    if gave_up:  # F has been rolled back by checks on read as often as allowed
        st.attn_repairs = dict(F=getattr(hook, "ATTN_MAX_REPAIRS", 3))
    F, S, N = _req("F", 0, F_IDS), _req("S", 1, S_IDS), _req("N", 2, range(5000, 5000 + 2 * PAGE))
    sched.running_batch.batch_is_full = True  # SGLang's flag, set when the running batch was full
    _prefill(ctx, [F, N], dict(F=0, N=0))
    _prefill(ctx, [S], dict(S=4 * PAGE))  # on top of F's cached pages; its check right after the prefill passes
    hook.act_encode(st, sched, {}, None)  # parity of the sealed prompt pages
    for _ in range(5):  # outputs 2-6; S passes its release check at output 4
        _decode(ctx)
    # the cached children of the shared pages: written in the step right after the readers' last passing check
    # (a forward stamps its writes with the step before its own), and in the step of that check
    st.wstep[40], st.wstep[41] = st.step, st.step - 1
    clean3 = _page_clean(ctx, F, 3, range(32, 48))
    # one bit of one K value of position 40 (pool page 3, read by F and S), layer 1, on GPU 1
    hook.act_inject(st, sched, dict(pos=40, layer=1, rank=1, kv="k", elem=5, bit=14), F)
    damaged = not _page_clean(ctx, F, 3, range(32, 48))
    _decode(ctx)  # the step that reads it
    act_recs = _records("attn_act")
    res = dict(clean_before=clean3, damaged=damaged,
               after=dict(F=list(F.output_ids), S=list(S.output_ids), N=list(N.output_ids),
                          sent={k: list(v) for k, v in sched.sent.items()}, queued=list(sched.queued),
                          running=[r.rid for r in sched.running_batch.reqs], checked_S=st.checked_upto.get("S"),
                          page3_clean=_page_clean(ctx, F, 3, range(32, 48)), evicted=list(evicted),
                          batch_is_full=sched.running_batch.batch_is_full, good=sorted(getattr(st, "attn_good", ()))),
               found=[(x["step"], [tuple(f) for f in x["found"]], x["rids"]) for x in _records("attn_found")],
               act=[{k: x[k] for k in ("step", "pages", "summary", "mended_pages", "freed_pages", "rids", "gave_up",
                                       "evicted_tokens", "since", "repair")}
                    | dict(readers=[{k: r[k] for k in ("rid", "strict", "n_out", "keep", "regenerate", "checked_upto")}
                                    for r in x["readers"]],
                           mends=[(m["rid"], [(e.get("rank"), e.get("page"), e.get("rebuilt"), e.get("bit_exact"),
                                               e.get("tag_matches")) for e in m.get("rebuilt", [])],
                                   m.get("recomputed")) for m in x["mends"]])
                    for x in act_recs],
               errors=_records("error"))
    redo = [r for r in (F, S) if r.rid in sched.queued]
    if redo:  # SGLang re-prefills the rolled-back readers, on top of their cached prompt
        _prefill(ctx, redo, {r.rid: PROMPT for r in redo})
    for _ in range(2):  # two more decode steps; S passes its release check at output 8
        _decode(ctx)
    res["final"] = dict(F=list(F.output_ids), S=list(S.output_ids), N=list(N.output_ids),
                        sent={k: list(v) for k, v in sched.sent.items()}, queued=list(sched.queued),
                        page3_clean=_page_clean(ctx, F, 3, range(32, 48)))
    res["attn_found_after"] = [x["step"] for x in _records("attn_found")]
    return res


def _one_gpu_main(rank, port, out):
    _start_rank(rank, port)
    try:
        res = {act: _one_gpu(rank, act) for act in ("repair", "log", "none")}
        res["gave_up"] = _one_gpu(rank, "repair", gave_up=True)
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_a_failed_check_on_read_mends_the_page_and_regenerates_its_readers_tokens():
    res = _spawn(_one_gpu_main)
    for rank in range(RANKS):
        r = res[rank]["repair"]
        assert r["clean_before"] and r["damaged"] == (rank == 1) and not r["errors"], r["errors"]
        a = r["after"]
        # the readers' token of that step (WRONG) was never appended or streamed; N, which read other pages, went on
        assert a["F"] == _right("F", 6) and a["S"] == _right("S", 6) and a["N"] == _right("N", 7), a
        assert a["sent"] == dict(F=_right("F", 6), S=_right("S", 4), N=_right("N", 7)), a["sent"]
        # both readers rolled back to the last token before that step and queued to generate it again; S keeps
        # outputs 5 and 6, which only checks on read covered: still held for its release rule (checked_upto 4)
        assert a["queued"] == ["F", "S"] and a["running"] == ["N"] and a["checked_S"] == 4, a
        assert not a["batch_is_full"] and a["good"] == ["N"], a
        # the page rebuilt from parity, bit for bit, and the KV cached on top of it since the readers' last passing
        # check (page 40) evicted, not the KV cached before it (page 41)
        assert a["page3_clean"] and a["evicted"] == [40], a
        [act] = r["act"]
        assert act["pages"] == [3] and act["summary"] == {"3": {"tag": {"ranks": [1], "layers": 1}}}, act
        assert act["rids"] == ["F", "S"] and act["gave_up"] == [] and act["freed_pages"] == [], act
        assert act["mended_pages"] == [3] and act["evicted_tokens"] == PAGE and act["since"] > 0, act
        assert act["repair"] == dict(rebuilt=[3], recopied=[], recomputed=[], freed=[]), act["repair"]
        assert act["readers"] == [dict(rid="F", strict=False, n_out=6, keep=6, regenerate=1, checked_upto=0),
                                  dict(rid="S", strict=True, n_out=6, keep=6, regenerate=1, checked_upto=4)], act
        [(rid, rebuilt, recomputed)] = act["mends"]  # through F, its first reader; GPU 1 rebuilt its piece
        here = True if rank == 1 else None
        assert rid == "F" and recomputed == [] and rebuilt == [(1, 3, here, here, here)], act["mends"]
        # found at that very step: the verdict's record (GPU 1) and the repair's carry the same step
        if rank == 1:
            assert [(s, f) for s, f, _ in r["found"]] == [(act["step"], [(1, 1, 3, "tag")])], r["found"]
        else:
            assert r["found"] == []
        # regenerated: every text right, nothing wrong ever streamed, no check failed again
        f = r["final"]
        assert f["F"] == _right("F", 9) and f["S"] == _right("S", 9) and f["N"] == _right("N", 9), f
        assert f["sent"] == dict(F=_right("F", 9), S=_right("S", 8), N=_right("N", 9)), f["sent"]
        assert r["attn_found_after"] == [s for s, _, _ in r["found"]], r["attn_found_after"]
        # log (as before): the failure is logged and nothing acts: F streams WRONG, S holds it, the page stays
        # damaged; the run is the one with no check on read at all, token for token
        lg, no = res[rank]["log"], res[rank]["none"]
        assert lg["act"] == [] and lg["after"]["queued"] == [] and lg["after"]["good"] == [], lg
        assert lg["after"]["F"] == _right("F", 6) + [WRONG] and lg["after"]["sent"]["F"][-1] == WRONG, lg["after"]
        assert lg["after"]["S"] == _right("S", 6) + [WRONG] and lg["after"]["sent"]["S"] == _right("S", 4)
        assert lg["after"]["page3_clean"] == (rank == 0) and len(lg["found"]) == (rank == 1), lg
        assert lg["after"] == no["after"] and lg["final"] == no["final"], (lg, no)
        assert no["found"] == [] and not lg["errors"] and not no["errors"]
        # gave_up: F's token goes on, as with log; S, its other reader, is repaired, through its own mend
        g = res[rank]["gave_up"]
        [gact] = g["act"]
        assert gact["gave_up"] == ["F"] and gact["rids"] == ["S"] and g["after"]["queued"] == ["S"], gact
        assert g["after"]["F"][-1] == WRONG and g["after"]["S"] == _right("S", 6) and g["after"]["page3_clean"]
        assert [m[0] for m in gact["mends"]] == ["S"], gact["mends"]


# ---------------------------------------------------------------------------------------------------------------
# A late write on every GPU, read by a checked request and by two whose prefill read it before any check

A_IDS = list(range(1000, 1000 + PROMPT))
B_IDS = A_IDS[:3 * PAGE] + list(range(2048, 2048 + 3 * PAGE))  # B and C share A's first three prompt pages
C_IDS = A_IDS[:3 * PAGE] + list(range(4048, 4048 + 3 * PAGE))
A_ROW = list(range(PAGE, PAGE + 256))                           # A: pages 1-6, decode pages from 7
B_ROW = A_ROW[:3 * PAGE] + list(range(20 * PAGE, 20 * PAGE + 208))  # B: A's pages 1-3, then pages 20-22, from 23
C_ROW = A_ROW[:3 * PAGE] + list(range(26 * PAGE, 26 * PAGE + 208))  # C: A's pages 1-3, then pages 26-28, from 29


def _every_gpu_main(rank, port, out):
    _start_rank(rank, port)
    try:
        ctx = _setup(rank, [A_ROW, B_ROW, C_ROW], "repair")
        st, sched = ctx.st, ctx.sched
        st.test_ctx = ctx
        ctx.row_of = dict(A=0, B=1, C=2)
        A, B, C = _req("A", 0, A_IDS), _req("B", 1, B_IDS), _req("C", 2, C_IDS, stream=False)
        _prefill(ctx, [A], dict(A=0))
        for _ in range(3):  # outputs 2-4, each step's check on read passing
            _decode(ctx)
        # a late write around the store path into A's page 2 (positions 16-31), on every GPU, and damage in A's own
        # decode page 7 (position 96)
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=PAGE, pages=1, donor=4 * PAGE), A)
        for l in range(LAYERS):
            ctx.pool.k_buffer[l][7 * PAGE] += 1
        # B and C arrive: their prefills read the damaged page 2 from the prefix cache, with no check after them;
        # B's first token leaves at once (streamed), C's has not left
        _prefill(ctx, [B, C], dict(B=3 * PAGE, C=3 * PAGE))
        first = dict(B=list(B.output_ids), C=list(C.output_ids), sent={k: list(v) for k, v in sched.sent.items()})
        _decode(ctx)  # A, B and C read page 2; A also its own page 7
        x = _records("attn_act")
        res = dict(first=first, errors=_records("error"), calls=list(ctx.calls),
                   after=dict(A=list(A.output_ids), B=list(B.output_ids), C=list(C.output_ids),
                              queued=list(sched.queued), page2=_page_clean(ctx, A, 2, range(PAGE, 2 * PAGE)),
                              B_suffix=all(_page_clean(ctx, B, p, range(PAGE * (p - 17), PAGE * (p - 16)))
                                           for p in (20, 21, 22)),
                              C_suffix=all(_page_clean(ctx, C, p, range(PAGE * (p - 23), PAGE * (p - 22)))
                                           for p in (26, 27, 28))),
                   act=[{k: y[k] for k in ("pages", "mended_pages", "freed_pages", "rids", "since", "repair")}
                        | dict(readers=[{k: r.get(k) for k in ("rid", "n_out", "keep", "regenerate", "last_pass",
                                                               "released_suspect")} for r in y["readers"]],
                               mends=[(m["rid"], [(s["start"], s["end"]) for s in m.get("recomputed", [])],
                                       (m.get("suffix") or {}).get("damage_in")) for m in y["mends"]])
                        for y in x])
        # SGLang re-prefills them, then one decode step. C kept nothing: SGLang matches at most all but its last
        # prompt token, in whole pages, so it computes its last prompt page again (here over the same slots)
        _prefill(ctx, [A, B, C], dict(A=PROMPT, B=PROMPT, C=PROMPT - PAGE))
        _decode(ctx)
        res["final"] = dict(A=list(A.output_ids), B=list(B.output_ids), C=list(C.output_ids),
                            sent={k: list(v) for k, v in sched.sent.items()})
        res["attn_found"] = len(_records("attn_found"))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_damage_on_every_gpu_is_recomputed_and_prefills_that_read_it_are_covered():
    res = _spawn(_every_gpu_main)
    for rank in range(RANKS):
        r = res[rank]
        assert not r["errors"], r["errors"]
        # the new requests' prefills read the damage: their first token is WRONG; B's has left, C's has not
        assert r["first"] == dict(B=[WRONG], C=[WRONG], sent=dict(A=_right("A", 4), B=[WRONG])), r["first"]
        [act] = r["act"]
        assert act["pages"] == [2, 7] and act["rids"] == ["A", "B", "C"] and act["since"] == 0, act  # B and C: none
        # A's own page 7 goes with A's rollback; the cached page 2 is recomputed in place (no parity here), and so is
        # what B's and C's prefills computed after it (their pages 20-22 and 26-28)
        assert act["freed_pages"] == [7] and act["mended_pages"] == [2, 20, 21, 22, 26, 27, 28], act
        assert act["repair"] == dict(rebuilt=[], recopied=[], recomputed=[2, 20, 21, 22, 26, 27, 28], freed=[7])
        assert act["mends"] == [("A", [(16, 32)], None), ("B", [(48, 96)], "prefix"), ("C", [(48, 96)], "prefix")]
        assert r["calls"] == [("A", 16, 32), ("B", 48, 96), ("C", 48, 96)], r["calls"]
        # A (checked at every step) drops only that step's token; B and C had no check since their prefill: C drops
        # its prefill token too, B's has left and is reported
        ra, rb, rc = act["readers"]
        assert (ra["rid"], ra["n_out"], ra["keep"], ra["regenerate"], ra["released_suspect"]) == ("A", 4, 4, 1, None)
        assert ra["last_pass"] is not None and rb["last_pass"] is None and rc["last_pass"] is None, act["readers"]
        assert (rb["keep"], rb["regenerate"]) == (1, 1), rb
        assert rb["released_suspect"] == dict(released_tokens=1, suspect_tokens=1, reason=(
            "its latest prefill read the damage before any check covered it, and that prefill's token has left")), rb
        assert (rc["keep"], rc["regenerate"], rc["released_suspect"]) == (0, 2, None), rc
        a = r["after"]
        assert a["A"] == _right("A", 4) and a["B"] == [WRONG] and a["C"] == [] and a["queued"] == ["A", "B", "C"], a
        assert a["page2"] and a["B_suffix"] and a["C_suffix"], a
        # regenerated: A and C right; B's text after its released first token is right
        f = r["final"]
        assert f["A"] == _right("A", 6) and f["C"] == _right("C", 2) and f["B"] == [WRONG] + _right("B", 3)[1:], f
        assert f["sent"] == dict(A=_right("A", 6), B=[WRONG] + _right("B", 3)[1:]), f["sent"]
        assert r["attn_found"] == 1, r["attn_found"]  # found once, at that step, and never again


# ---------------------------------------------------------------------------------------------------------------
# The setting

def test_the_setting_is_log_unless_set_to_repair():
    """KVMEND_ATTN_ACT is "log" when unset, "log" or "repair" when set; anything else stops the import. The module is
    imported in a process of its own, so this process keeps its settings."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import importlib, os\n"
            "from kvmend import hook\n"
            "got = [(hook.ATTN_ACT, hook.ATTN_MAX_REPAIRS)]\n"
            "for v in ('repair', 'log', 'fix'):\n"
            "    os.environ['KVMEND_ATTN_ACT'] = v\n"
            "    try:\n"
            "        got.append(importlib.reload(hook).ATTN_ACT)\n"
            "    except ValueError:\n"
            "        got.append('refused')\n"
            "print(got)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("KVMEND_")}
    env.update(PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "[('log', 3), 'repair', 'log', 'refused']", r.stdout
