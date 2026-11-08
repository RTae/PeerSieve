"""The comparison's collective actions on 4 CPU processes with gloo: GhostServe's share rebuild, told the GPU or the
GPU and the chunk, and the background encode and scrub of the overhead benchmark. No SGLang and no GPU: the
scheduler and the request are stand-ins with the fields the actions read."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook

PAGE, HEADS, DIM, LAYERS, PAGES, RANKS = 16, 1, 64, 2, 8, 4
N_TOK = 5 * PAGE + 5  # five sealed pages (pool pages 1-5) and one open page (6)
PIECE = 2 * PAGE * HEADS * DIM * 2


def _setup(rank):
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    loc = torch.arange(PAGE, PAGE + N_TOK)
    g = torch.Generator().manual_seed(100 + rank)  # each rank holds its own heads
    for layer in range(LAYERS):
        pool.k_buffer[layer][loc] = torch.randn(N_TOK, HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][loc] = torch.randn(N_TOK, HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, loc)
    world = dist.group.WORLD
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE,
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256)))
    req = types.SimpleNamespace(rid="R", req_pool_idx=0, origin_input_ids=list(range(N_TOK)))
    return st, sched, req


def _rank_main(rank, port, out):
    torch.set_num_threads(1)  # four ranks on a few CPUs: one thread each, or they oversubscribe the cores
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    try:
        st, sched, req = _setup(rank)
        sealed = torch.arange(1, 6)
        res = {"encode": hook.act_encode(st, sched, {"check": False}, req)}
        clean = st.pieces(sealed)
        # told the GPU: rank 2's piece of page 3, layer 1 is damaged; GhostServe rebuilds rank 2's whole share
        hook.act_inject(st, sched, dict(pos=2 * PAGE + 6, layer=1, rank=2, kv="k", elem=5, bit=14), req)
        res["gpu"] = hook.act_repair_share(st, sched, dict(rank=2, scope="gpu"), req)
        res["gpu_restored"] = torch.equal(st.pieces(sealed), clean)
        st.saved.clear()
        # told the GPU and the chunk: position 67 sits on page 5; the 32-token chunk [64, 96) holds pages 5 and 6,
        # and only page 5 has parity
        hook.act_inject(st, sched, dict(pos=4 * PAGE + 3, layer=0, rank=2, kv="v", elem=7, bit=3), req)
        res["chunk"] = hook.act_repair_share(st, sched, dict(rank=2, scope="chunk", pos=4 * PAGE + 3, chunk_tokens=32),
                                             req)
        res["chunk_restored"] = torch.equal(st.pieces(sealed), clean)
        # background work: encode every batch step, scrub 4 pieces per step, sum the alarms every step
        hook.SYNC_EVERY = 1
        st.encoded.zero_()
        st.holder.fill_(-1)
        st.ctrl.update(encode_every=1, scrub_pieces=4, label="t")
        for _ in range(4):  # 7 pages x 2 layers = 14 pieces: 4 steps cover them all
            hook.periodic(st, sched)
        res["clean_stats"] = dict(st.stats)
        hook.act_inject(st, sched, dict(pos=PAGE + 2, layer=0, rank=1, kv="k", elem=1, bit=9), req)
        for _ in range(4):
            hook.periodic(st, sched)
        res["fault_stats"] = dict(st.stats)
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_share_rebuild_and_background_work_on_four_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=120) for _ in range(RANKS))
    for rank in range(RANKS):
        assert res[rank]["encode"]["pages"] == 5 and res[rank]["encode"]["checked"] is False
        gpu, chunk = res[rank]["gpu"], res[rank]["chunk"]
        assert (gpu["pages"], gpu["pieces"], gpu["bytes_read"]) == (5, 10, RANKS * 10 * PIECE)
        assert (chunk["pages"], chunk["pieces"], chunk["bytes_read"]) == (1, 2, RANKS * 2 * PIECE)
        assert res[rank]["gpu_restored"] and res[rank]["chunk_restored"]
        c, f = res[rank]["clean_stats"], res[rank]["fault_stats"]
        assert (c["encodes"], c["pages_encoded"], c["unclean"], c["scrub_pieces"], c["alarms"]) == (4, 5, 0, 14, 0)
        assert (f["encodes"], f["scrub_pieces"], f["alarms"]) == (8, 28, 1)  # every rank hears rank 1's alarm
    assert res[2]["gpu"]["damaged_pieces"] == [dict(layer=1, page=3, rebuilt=True, bit_exact=True)]
    assert res[2]["gpu"]["tag_mismatches_after"] == 0
    assert res[2]["chunk"]["damaged_pieces"] == [dict(layer=0, page=5, rebuilt=True, bit_exact=True)]
    assert res[2]["chunk"]["tag_mismatches_after"] == 0
