"""The hook's collective actions, end to end, on 4 CPU processes with gloo: encode, scrub, inject on rank 2, scrub,
repair, scrub. Rank 2 must get its piece back bit for bit. No SGLang and no GPU: the scheduler and the request are
stand-ins with the fields the actions read."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook

PAGE, HEADS, DIM, LAYERS, PAGES, RANKS = 16, 1, 64, 2, 8, 4


def _rank_main(rank, port, out):
    torch.set_num_threads(1)  # four ranks on a few CPUs: one thread each, or they oversubscribe the cores
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    try:
        pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                     k_buffer=[torch.zeros(PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)],
                                     v_buffer=[torch.zeros(PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)])
        st = hook.PoolState(pool)
        n_tok = 5 * PAGE + 5  # five sealed pages and one open page, starting at page 1
        loc = torch.arange(PAGE, PAGE + n_tok)
        g = torch.Generator().manual_seed(100 + rank)  # each rank holds its own heads
        for layer in range(LAYERS):
            pool.k_buffer[layer][loc] = torch.randn(n_tok, HEADS, DIM, generator=g).to(torch.bfloat16)
            pool.v_buffer[layer][loc] = torch.randn(n_tok, HEADS, DIM, generator=g).to(torch.bfloat16)
            st.update(layer, loc)
        world = dist.group.WORLD
        sched = types.SimpleNamespace(
            tp_rank=rank, page_size=PAGE,
            tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                           cpu_group=world, device_group=world),
            req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256)))
        req = types.SimpleNamespace(rid="R", req_pool_idx=0)
        res = {"encode": hook.act_encode(st, sched, {}, req), "scrub0": hook.act_scrub(st, sched, {}, req)}
        res["inject"] = hook.act_inject(st, sched, dict(pos=2 * PAGE + 6, layer=1, rank=2, kv="k", elem=5, bit=14), req)
        res["scrub1"] = hook.act_scrub(st, sched, {}, req)
        res["repair"] = hook.act_repair(st, sched, {}, req)
        res["scrub2"] = hook.act_scrub(st, sched, {}, req)
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_encode_scrub_inject_repair_on_four_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=120) for _ in range(RANKS))
    for rank in range(RANKS):
        assert res[rank]["encode"]["pages"] == 5 and not res[rank]["encode"]["unclean_pieces"]
        assert res[rank]["scrub0"]["damaged_all"] == []
        assert res[rank]["scrub1"]["damaged_all"] == [(2, 1, 3)]  # rank 2, layer 1, page 3
        assert res[rank]["scrub2"]["damaged_all"] == []
    fix = res[2]["repair"]["repairs"]
    assert len(fix) == 1 and fix[0]["rebuilt"] and fix[0]["tag_matches"] and fix[0]["bit_exact"]
    assert res[2]["inject"]["before"] != res[2]["inject"]["after"]
