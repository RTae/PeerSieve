"""The release rule's eviction bound, on 2 CPU processes with gloo (no SGLang, no GPU). on_step sets the step before
the step's checks run, and the next forward stamps its writes with that same step (wstep), so the first forward after a
passing check at step g already writes KV stamped g. When a later check finds damage that struck after that check,
evict_suspect, which keeps what is stamped `since` or less, must get since = g - 1: KV cached in that first forward,
on top of the damaged page, may have read the damage and must leave the prefix cache; KV cached before the check
passed stays.

Strict S's prompt sits on pool pages 1-6 (a check every output token). P and Q share S's prompt and finish at their
prefill (one token each), so their own pages (30, 31) stay cached below S's prompt pages: P's prefill runs with S's,
before S's check passes at step g; then a late write strikes S's page 3; Q's prefill runs in the very next forward
(stamped g) and reads it. At S's next check the page is found and mended, S rolls back, and Q's page must go."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS, POOL_PAGES = 16, 1, 64, 2, 2, 40
PROMPT = 6 * PAGE


class Node:
    """A node of the prefix cache: the slots of its pages."""

    def __init__(self, parent, pages, lock=0):
        self.parent, self.children, self.lock_ref = parent, {}, lock
        self.value = (torch.cat([torch.arange(p * PAGE, (p + 1) * PAGE) for p in pages]) if pages
                      else torch.zeros(0, dtype=torch.int64))
        if parent is not None:
            parent.children[len(parent.children)] = self


def _write(st, pool, loc, ids, positions, seed):
    """Store KV at the slots `loc` through the store path, keyed by the tokens' ids and positions; the step's forward
    stamps them (wstep) with the step on_step set last."""
    tk = T.token_keys(torch.tensor(ids), torch.tensor(positions))
    g = torch.Generator().manual_seed(seed)
    for layer in range(LAYERS):
        pool.k_buffer[layer][loc] = torch.randn(len(loc), HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][loc] = torch.randn(len(loc), HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, loc, tk)


def _fake_recompute(st, sched, req, s, e):
    """The in-place recompute: S's clean KV of [s, e) (the copy act_fault saved), back through the store path."""
    tgt, clean = st.saved_span
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, s:e].to(torch.int64)
    ids = (list(req.origin_input_ids) + list(req.output_ids))[s:e]
    for layer in range(st.layers):
        st.pool.k_buffer[layer][row] = clean[layer, 0, :e - s]
        st.pool.v_buffer[layer][row] = clean[layer, 1, :e - s]
        st.update(layer, row, T.token_keys(torch.tensor(ids), torch.arange(s, e)))


def _rank_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    hook._flush_log = lambda sched: None  # the records stay in hook._BUF; no file is written
    hook.STRICT, hook.ON_FAIL = hook._rules("S")
    hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL = "mend", "strict"
    try:
        pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                     k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)],
                                     v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)])
        st = hook.PoolState(pool)
        st.reset()
        pool._kvm = st
        S_IDS = list(range(1000, 1000 + PROMPT))
        S = types.SimpleNamespace(rid="S", req_pool_idx=0, origin_input_ids=S_IDS, output_ids=[], last_node="S",
                                  prefix_indices=list(range(PROMPT)), send_token_offset=0, to_abort=False, is_chunked=0,
                                  finished_reason=None)
        S.finished = lambda: S.finished_reason is not None
        S.reset_for_retract = lambda: S.__dict__.update(prefix_indices=[], req_pool_idx=None)
        batch = types.SimpleNamespace(reqs=[S], seq_lens=torch.tensor([PROMPT]))

        def filter_batch(keep_indices):
            batch.reqs = [batch.reqs[j] for j in keep_indices]
            batch.seq_lens = batch.seq_lens[torch.tensor(keep_indices, dtype=torch.int64)]

        batch.filter_batch = filter_batch
        root = Node(None, [], lock=1)
        prompt_node = Node(root, list(range(1, 7)), lock=1)  # S's prompt, held by S
        evicted, queued = [], []

        def delete_leaf(node):
            del node.parent.children[next(k for k, v in node.parent.children.items() if v is node)]

        tree = types.SimpleNamespace(root_node=root, dec_lock_ref=lambda node: None, _delete_leaf=delete_leaf,
                                     token_to_kv_pool_allocator=types.SimpleNamespace(
                                         free=lambda v: evicted.append(int(v[0]) // PAGE)))
        world = dist.group.WORLD
        sched = types.SimpleNamespace(
            tp_rank=rank, page_size=PAGE, running_batch=batch, cur_batch=None, tree_cache=tree,
            tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                           cpu_group=world, device_group=world),
            req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256),
                                                    free=lambda i: None),
            token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: pool, free=lambda idx: None),
            _extend_requests_to_queue=lambda reqs, is_retracted: queued.extend(r.rid for r in reqs))
        hook._BUF.clear()
        for _ in range(3):  # idle steps: the step numbers below are well clear of 0
            hook.on_step(sched)
        # step g's forward: S's prefill, and P's (S's prompt, then its own page 30), which finishes at its prefill
        _write(st, pool, torch.arange(PAGE, PAGE + PROMPT), S_IDS, list(range(PROMPT)), seed=10 + rank)
        _write(st, pool, torch.arange(30 * PAGE, 31 * PAGE), list(range(5000, 5016)), list(range(PROMPT, PROMPT + 16)),
               seed=20 + rank)
        Node(prompt_node, [30])
        S.output_ids.append(7000)
        hook.on_step(sched)  # step g: S's check right after its prefill passes
        g = st.good_step.get("S")
        # a late write strikes S's page 3 after that check, on every GPU; then step g + 1's forward: Q's prefill reads
        # the cached prompt, page 3 included, and finishes at its prefill: its own page 31 is cached below S's prompt
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=2 * PAGE, pages=1, donor=0), S)
        _write(st, pool, torch.arange(31 * PAGE, 32 * PAGE), list(range(6000, 6016)), list(range(PROMPT, PROMPT + 16)),
               seed=30 + rank)
        Node(prompt_node, [31])
        hook.on_step(sched)  # step g + 1: S has nothing new to check
        # step g + 2: S decodes one token; its check finds page 3, which is mended, and S rolls back
        _write(st, pool, torch.tensor([PAGE + PROMPT]), [7000], [PROMPT], seed=40 + rank)
        S.output_ids.append(7001)
        batch.seq_lens[0] = PROMPT + 1
        hook.on_step(sched)
        rec = [x for x in hook._BUF if x.get("op") == "release_check" and x.get("found_pieces")]
        out.put((rank, dict(g=g, stamps={p: int(st.wstep[p]) for p in (30, 31)}, evicted=list(evicted),
                            left=sorted(int(c.value[0]) // PAGE for c in prompt_node.children.values()),
                            found=rec[0]["found"] if rec else None, evicted_tokens=rec[0].get("evicted_tokens") if rec
                            else None, mended=rec[0]["mend"].get("mended_pages") if rec else None,
                            queued=list(queued), outputs=list(S.output_ids))))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_kv_cached_right_after_the_last_good_check_is_evicted():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    for rank in range(RANKS):
        r = res[rank]
        g = r["g"]
        # P's page was written by the forward before S's check passed at step g, Q's by the first forward after it
        assert g is not None and r["stamps"] == {30: g - 1, 31: g}, r
        # S's check found the late write in page 3 on every GPU; the page was mended and S rolled back
        assert r["found"] == {"3": {"tag": {"ranks": list(range(RANKS)), "layers": LAYERS}}}, r["found"]
        assert r["mended"] == [3] and r["queued"] == ["S"] and r["outputs"] == [7000], r
        # Q's KV, computed on the damaged page in the step right after the check, leaves the prefix cache; P's stays
        assert r["evicted"] == [31] and r["evicted_tokens"] == PAGE and r["left"] == [30], r
