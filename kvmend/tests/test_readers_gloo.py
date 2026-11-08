"""The release rule with two strict requests that share prompt pages, on 2 CPU processes with gloo (no SGLang, no
GPU): damage in a shared page, found by one reader's check, is mended once, and both readers drop their unreleased
tokens, since both may have read it."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS = 16, 1, 64, 2, 2
PROMPT, SHARED = 6 * PAGE, 4 * PAGE  # R and S share their first four prompt pages


def _store(st, pool, row, ids, positions, seed):
    loc = row[torch.tensor(positions)].to(torch.int64)
    tk = T.token_keys(torch.tensor([ids[p] for p in positions]), torch.tensor(positions))
    g = torch.Generator().manual_seed(seed)
    for layer in range(LAYERS):
        pool.k_buffer[layer][loc] = torch.randn(len(positions), HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][loc] = torch.randn(len(positions), HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, loc, tk)


def _fake_recompute(st, sched, req, s, e):
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
    hook.STRICT = {"R": None, "S": None}
    try:
        pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                     k_buffer=[torch.zeros(32 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)],
                                     v_buffer=[torch.zeros(32 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)])
        st = hook.PoolState(pool)
        st.reset()
        table = torch.zeros(2, 256, dtype=torch.int32)
        table[0] = torch.arange(PAGE, PAGE + 256)                             # R: pages 1, 2, ...
        table[1, :SHARED] = table[0, :SHARED]                                # S: R's first four pages, then its own
        table[1, SHARED:] = torch.arange(10 * PAGE, 10 * PAGE + 256 - SHARED)
        calls = []

        def make(rid, row, ids, out0):
            r = types.SimpleNamespace(rid=rid, req_pool_idx=row, origin_input_ids=ids, output_ids=[out0],
                                      prefix_indices=list(range(PROMPT)), last_node=rid, send_token_offset=0,
                                      finished=lambda: False)
            r.reset_for_retract = lambda: None
            return r

        R = make("R", 0, list(range(1000, 1000 + PROMPT)), 7000)
        S = make("S", 1, list(range(1000, 1000 + SHARED)) + list(range(2000 + SHARED, 2000 + PROMPT)), 8000)
        world = dist.group.WORLD
        batch = types.SimpleNamespace(reqs=[R, S], seq_lens=torch.tensor([PROMPT + 3, PROMPT + 3]),
                                      filter_batch=lambda keep_indices: None)
        sched = types.SimpleNamespace(
            tp_rank=rank, page_size=PAGE, running_batch=batch, cur_batch=None,
            tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                           cpu_group=world, device_group=world),
            req_to_token_pool=types.SimpleNamespace(req_to_token=table, free=lambda i: None),
            token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: None),
            tree_cache=types.SimpleNamespace(dec_lock_ref=lambda node: None),
            _extend_requests_to_queue=lambda reqs, is_retracted: calls.append([r.rid for r in reqs]))
        _store(st, pool, table[0], R.origin_input_ids, list(range(PROMPT)), seed=10 + rank)
        _store(st, pool, table[1], S.origin_input_ids, list(range(SHARED, PROMPT)), seed=20 + rank)
        for step in (1, 2):  # two decode steps for both
            for req, base, seed in ((R, 7000, 30), (S, 8000, 40)):
                req.output_ids.append(base + step)
                ids = list(req.origin_input_ids) + req.output_ids
                _store(st, pool, table[req.req_pool_idx], ids, [PROMPT + step - 1], seed=seed + 10 * rank + step)
            hook.release_checks(st, sched)
        res = {"checked": dict(st.checked_upto)}
        for req, base, seed in ((R, 7000, 50), (S, 8000, 60)):  # one more step each: a token no check covers yet
            req.output_ids.append(base + 3)
            ids = list(req.origin_input_ids) + req.output_ids
            _store(st, pool, table[req.req_pool_idx], ids, [PROMPT + 2], seed=seed + rank)
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=2 * PAGE, pages=1, donor=0), R)  # a shared page
        hook.release_checks(st, sched)
        res.update(R=list(R.output_ids), S=list(S.output_ids), queued=calls, after=dict(st.checked_upto))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_readers_of_a_mended_page_regenerate_on_two_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    for rank in range(RANKS):
        r = res[rank]
        assert r["checked"] == {"R": 3, "S": 3}
        assert r["R"] == [7000, 7001, 7002] and r["S"] == [8000, 8001, 8002]  # each drops its unchecked token
        assert r["queued"] == [["R"], ["S"]] and r["after"] == {"R": 3, "S": 3}
