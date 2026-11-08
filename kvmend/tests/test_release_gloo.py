"""The release rule on 2 CPU processes with gloo (no SGLang, no GPU): a strict request's tokens leave only after a
check covers them; when a check finds damage, the pages are mended and the unreleased tokens are generated again;
the stream call sends only the checked tokens and keeps the rest for later."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS = 16, 1, 64, 2, 2
PROMPT = 6 * PAGE


def _store(st, pool, req, positions, seed):
    """Write KV for R's positions through the store path, keyed by R's tokens."""
    ids = list(req.origin_input_ids) + list(req.output_ids)
    loc = torch.tensor([PAGE + p for p in positions])
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
    torch.set_num_threads(1)  # four ranks on a few CPUs: one thread each, or they oversubscribe the cores
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    hook.STRICT = {"R": None}
    try:
        pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                     k_buffer=[torch.zeros(16 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)],
                                     v_buffer=[torch.zeros(16 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)])
        st = hook.PoolState(pool)
        st.reset()
        calls = {}
        req = types.SimpleNamespace(rid="R", req_pool_idx=0, origin_input_ids=list(range(1000, 1000 + PROMPT)),
                                    output_ids=[7000], prefix_indices=list(range(PROMPT)), last_node="node",
                                    send_token_offset=0, finished=lambda: False)
        req.reset_for_retract = lambda: calls.setdefault("reset", True)
        world = dist.group.WORLD
        batch = types.SimpleNamespace(reqs=[req], seq_lens=torch.tensor([PROMPT + 5]),
                                      filter_batch=lambda keep_indices: calls.setdefault("kept", keep_indices))
        sched = types.SimpleNamespace(
            tp_rank=rank, page_size=PAGE, running_batch=batch,
            tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                           cpu_group=world, device_group=world),
            req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256),
                                                    free=lambda i: calls.setdefault("slot", i)),
            token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: calls.setdefault("freed", idx.numel())),
            tree_cache=types.SimpleNamespace(dec_lock_ref=lambda node: calls.setdefault("unlocked", node)),
            _extend_requests_to_queue=lambda reqs, is_retracted: calls.setdefault("queued", (len(reqs), is_retracted)))
        _store(st, pool, req, list(range(PROMPT)), seed=10 + rank)  # the prefill
        res = {}
        for step in range(1, 4):  # three decode steps: each stores the previous token's KV and adds a token
            req.output_ids.append(7000 + step)
            _store(st, pool, req, [PROMPT + step - 1], seed=20 + 10 * rank + step)
            hook.release_checks(st, sched)
        res["after_clean_steps"] = dict(checked=st.checked_upto.get("R"), n_out=len(req.output_ids))
        # the stream call sends only checked tokens: one more token is chosen, but no check covers it yet
        req.output_ids.append(7004)
        sent = {}

        def fake_stream(reqs):
            sent["ids"] = list(reqs[0].output_ids)

        held = hook.held_back(st, [req])
        fake_stream([req])
        for r, tail in held:
            r.output_ids.extend(tail)
        res["stream"] = dict(sent=sent["ids"], restored=list(req.output_ids))
        req.send_token_offset = len(sent["ids"])
        # a late write lands around the store path in two prompt pages on every rank; the next check finds it
        res["fault"] = hook.act_fault(st, sched, dict(kind="late_bypass", pages=2, donor=0), req)
        _store(st, pool, req, [PROMPT + 3], seed=99 + rank)
        hook.release_checks(st, sched)
        res["after_fault"] = dict(output_ids=list(req.output_ids), checked=st.checked_upto.get("R"), calls=calls,
                                  record_cleared="R" not in st.req_gen)
        # R regenerates: its decode page was freed (a new generation) and is written again, then a new token comes.
        # The next check must pass: a rollback is not a read after free (job 326399 looped here for ever).
        st.drop(torch.tensor([7]))  # pool page 7 holds R's positions 96-111
        _store(st, pool, req, [PROMPT, PROMPT + 1, PROMPT + 2], seed=200 + rank)
        req.output_ids.append(7104)
        _store(st, pool, req, [PROMPT + 3], seed=300 + rank)
        hook.release_checks(st, sched)
        res["after_regenerate"] = dict(checked=st.checked_upto.get("R"), output_ids=list(req.output_ids))
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_release_rule_on_two_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    for rank in range(RANKS):
        r = res[rank]
        assert r["after_clean_steps"] == dict(checked=4, n_out=4)  # every chosen token is covered by a check
        assert r["stream"]["sent"] == [7000, 7001, 7002, 7003]  # the unchecked fifth token stays
        assert r["stream"]["restored"] == [7000, 7001, 7002, 7003, 7004]
        f = r["after_fault"]
        # the damage is found before 7004 leaves: the pages are mended, and 7004 is dropped to be generated again
        assert f["output_ids"] == [7000, 7001, 7002, 7003] and f["checked"] == 4
        assert f["calls"]["queued"] == (1, True) and f["calls"]["kept"] == [] and f["record_cleared"]
        assert r["after_regenerate"] == dict(checked=5, output_ids=[7000, 7001, 7002, 7003, 7104])


def _first_check_main(rank, port, out):
    """One rank, a check every 8 tokens: the first check still comes right after the prefill."""
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=1)
    hook.STRICT = {"R": 8}
    try:
        pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                     k_buffer=[torch.zeros(16 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)],
                                     v_buffer=[torch.zeros(16 * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                               for _ in range(LAYERS)])
        st = hook.PoolState(pool)
        st.reset()
        req = types.SimpleNamespace(rid="R", req_pool_idx=0, origin_input_ids=list(range(1000, 1000 + PROMPT)),
                                    output_ids=[7000], prefix_indices=list(range(PROMPT)), last_node="node",
                                    send_token_offset=0, finished=lambda: False)
        world = dist.group.WORLD
        batch = types.SimpleNamespace(reqs=[req], seq_lens=torch.tensor([PROMPT + 1]))
        sched = types.SimpleNamespace(
            tp_rank=rank, page_size=PAGE, running_batch=batch,
            tp_group=types.SimpleNamespace(world_size=1, rank_in_group=rank, ranks=[0], cpu_group=world,
                                           device_group=world),
            req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256)))
        _store(st, pool, req, list(range(PROMPT)), seed=10)  # the prefill; output 0 is chosen
        st.step += 1
        hook.release_checks(st, sched)
        seen = [(len(req.output_ids), st.checked_upto.get("R"), "R" in st.good_step)]
        for step in range(1, 9):  # outputs 2..9: the next check comes at output 8
            req.output_ids.append(7000 + step)
            _store(st, pool, req, [PROMPT + step - 1], seed=20 + step)
            st.step += 1
            hook.release_checks(st, sched)
            seen.append((len(req.output_ids), st.checked_upto.get("R"), "R" in st.good_step))
        out.put(seen)
    finally:
        dist.destroy_process_group()


def test_first_check_comes_right_after_the_prefill():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_first_check_main, args=(_free_port(), out), nprocs=1, start_method="spawn")
    seen = out.get(timeout=300)
    assert seen[0] == (1, 1, True)  # checked at output 1: a last good check before any decode step
    assert [c for n, c, _ in seen[1:]] == [1, 1, 1, 1, 1, 1, 8, 8]  # then only at output 8
