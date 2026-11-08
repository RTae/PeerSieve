"""The incident's faults and the reader's check on 4 CPU processes with gloo. Each fault is well formed but wrong KV on
every rank: which check finds it, and how mend repairs it. No SGLang and no GPU: the scheduler and the request are
stand-ins with the fields the actions read, and the in-place recompute (an extend forward in SGLang) is replaced by a
stand-in that writes the clean KV back through the store path, keyed by R's own tokens."""
import os
import socket
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM, LAYERS, RANKS = 16, 1, 64, 2, 4
PROMPT_PAGES, POOL_PAGES = 6, 12
PROMPT = PROMPT_PAGES * PAGE   # R's prompt sits on pool pages 1-6; R has one output token, so it stores the prompt
SPAN = dict(pages=2, donor=0)  # the fault replaces R's last two prompt pages (positions 64-95) with positions 0-31


def _setup(rank):
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    st.reset()  # as after a flush: every page in a new generation
    ids = list(range(1000, 1000 + PROMPT))
    loc = torch.arange(PAGE, PAGE + PROMPT)
    tk = T.token_keys(torch.tensor(ids), torch.arange(PROMPT))
    g = torch.Generator().manual_seed(100 + rank)  # each rank holds its own heads
    for layer in range(LAYERS):
        pool.k_buffer[layer][loc] = torch.randn(PROMPT, HEADS, DIM, generator=g).to(torch.bfloat16)
        pool.v_buffer[layer][loc] = torch.randn(PROMPT, HEADS, DIM, generator=g).to(torch.bfloat16)
        st.update(layer, loc, tk)
    world = dist.group.WORLD
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE,
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(PAGE, PAGE + 256).view(1, 256)))
    req = types.SimpleNamespace(rid="R", req_pool_idx=0, origin_input_ids=ids, output_ids=[5000])
    return st, sched, req


def _fake_recompute(st, sched, req, s, e):
    """Stand-in for the extend forward: write R's clean KV of [s, e) back through the store path, keyed by R's
    tokens (the clean copy is the one act_fault saved)."""
    tgt, clean = st.saved_span
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, s:e].to(torch.int64)
    assert torch.equal(row, tgt[:e - s]), "the recompute must write over R's own slots"
    ids = (list(req.origin_input_ids) + list(req.output_ids))[s:e]
    tk = T.token_keys(torch.tensor(ids), torch.arange(s, e))
    for layer in range(st.layers):
        st.pool.k_buffer[layer][row] = clean[layer, 0, :e - s]
        st.pool.v_buffer[layer][row] = clean[layer, 1, :e - s]
        st.update(layer, row, tk)


def _span_bytes(st):
    tgt = torch.arange(PAGE + PROMPT - 2 * PAGE, PAGE + PROMPT)
    return torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]]) for l in range(LAYERS)])


def _rank_main(rank, port, out):
    torch.set_num_threads(1)  # four ranks on a few CPUs: one thread each, or they oversubscribe the cores
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    res = {}
    try:
        for kind in ("unwritten", "late_bypass", "late_store", "late_store_open", "foreign"):
            st, sched, req = _setup(rank)
            r = res[kind] = {"check0": hook.act_check(st, sched, {}, req)}
            clean = _span_bytes(st)
            fault = dict(SPAN, kind="late_store", fence=False) if kind == "late_store_open" else dict(SPAN, kind=kind)
            r["fault"] = hook.act_fault(st, sched, fault, req)
            r["bytes_changed"] = not torch.equal(_span_bytes(st), clean)
            r["scrub"] = hook.act_scrub(st, sched, {}, req)
            r["check1"] = hook.act_check(st, sched, {}, req)
            r["mend"] = hook.act_mend(st, sched, {}, req)
            r["bytes_restored"] = torch.equal(_span_bytes(st), clean)
            r["check2"] = hook.act_check(st, sched, {}, req)
        # one GPU damaged: parity rebuilds the piece, nothing is recomputed
        st, sched, req = _setup(rank)
        r = res["one_gpu"] = {"encode": hook.act_encode(st, sched, {}, req), "check0": hook.act_check(st, sched, {}, req)}
        r["inject"] = hook.act_inject(st, sched, dict(pos=2 * PAGE + 6, layer=1, rank=2, kv="k", elem=5, bit=14), req)
        r["check1"] = hook.act_check(st, sched, {}, req)
        r["mend"] = hook.act_mend(st, sched, {}, req)
        r["check2"] = hook.act_check(st, sched, {}, req)
        # a read after free: one of R's pages is freed and reused, so it moves to a new generation
        st, sched, req = _setup(rank)
        r = res["after_free"] = {"check0": hook.act_check(st, sched, {}, req)}
        st.drop(torch.tensor([3]))
        st.update(0, torch.arange(3 * PAGE, 4 * PAGE))  # the new owner writes page 3 again (layer 0)
        st.update(1, torch.arange(3 * PAGE, 4 * PAGE))
        r["check1"] = hook.act_check(st, sched, {}, req)
        # the fault-free control of the recompute path: force a recompute of the span
        st, sched, req = _setup(rank)
        st.saved_span = (torch.arange(PAGE + PROMPT - 2 * PAGE, PAGE + PROMPT), _span_bytes(st).clone())
        res["forced"] = {"mend": hook.act_mend(st, sched, dict(SPAN, force=True), req),
                         "check": hook.act_check(st, sched, {}, req)}
        # a burst on rank 2 (pages 2-3, both layers): rebuilt from parity, or, without parity, recomputed
        for how, parity in (("burst_rebuild", True), ("burst_recompute", False)):
            st, sched, req = _setup(rank)
            r = res[how] = {"encode": hook.act_encode(st, sched, {}, req), "check0": hook.act_check(st, sched, {}, req)}
            clean = torch.stack([torch.stack([st.pool.k_buffer[l][PAGE * 2:PAGE * 4], st.pool.v_buffer[l][PAGE * 2:PAGE * 4]])
                                 for l in range(LAYERS)]).clone()
            r["fault"] = hook.act_fault(st, sched, dict(kind="burst", pos=PAGE, pages=2, layer=0, layers=2, rank=2), req)
            r["check1"] = hook.act_check(st, sched, {}, req)
            r["mend"] = hook.act_mend(st, sched, dict(parity=parity), req)
            now = torch.stack([torch.stack([st.pool.k_buffer[l][PAGE * 2:PAGE * 4], st.pool.v_buffer[l][PAGE * 2:PAGE * 4]])
                               for l in range(LAYERS)])
            r["restored"] = torch.equal(now, clean)
            r["check2"] = hook.act_check(st, sched, {}, req)
        # stripes within each GPU (2 pages each): a late write into one page on every GPU is rebuilt by every GPU
        # from its own stripe; two damaged pages in one stripe are recomputed instead
        for how, pages in (("local_one_page", 1), ("local_crowded", 2)):
            st, sched, req = _setup(rank)
            r = res[how] = {"encode": hook.act_encode(st, sched, dict(stripe="local", k=2), req),
                            "check0": hook.act_check(st, sched, {}, req)}
            r["fault"] = hook.act_fault(st, sched, dict(kind="late_bypass", pages=pages, donor=0), req)
            r["check1"] = hook.act_check(st, sched, {}, req)
            r["mend"] = hook.act_mend(st, sched, {}, req)
            r["check2"] = hook.act_check(st, sched, {}, req)
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _reasons(check):
    return {p: sorted(v) for p, v in check["found"].items()}


def test_incident_faults_on_four_ranks():
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(_rank_main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    res = dict(out.get(timeout=300) for _ in range(RANKS))
    every = list(range(RANKS))
    span_pages = ["5", "6"]  # pool pages of R's positions 64-95
    for rank in every:
        r = res[rank]
        for kind in ("unwritten", "late_bypass", "late_store", "late_store_open", "foreign"):
            assert r[kind]["check0"]["found"] == {}, (kind, r[kind]["check0"])
        # a skipped write: the scrub skips pages with no tag; the reader finds them uncovered on every rank
        u = r["unwritten"]
        assert u["bytes_changed"] and u["scrub"]["damaged_all"] == []
        assert _reasons(u["check1"]) == {p: ["coverage"] for p in span_pages}
        assert all(u["check1"]["found"][p]["coverage"]["ranks"] == every for p in span_pages)
        # a late write around the store path: bytes change and tags do not, so even the scrub finds it
        b = r["late_bypass"]
        assert b["bytes_changed"] and {x[2] for x in b["scrub"]["damaged_all"]} == {5, 6}
        assert _reasons(b["check1"]) == {p: ["tag"] for p in span_pages}
        # a late write through the store path with the old generation: the fence drops it
        s = r["late_store"]
        assert s["fault"]["fenced"] and s["fault"]["dropped_tokens"] == 2 * PAGE and not s["bytes_changed"]
        assert s["check1"]["found"] == {} and s["mend"]["mended"] is False
        # the same write with no fence, or another sequence's KV with its own tags: only the token keys catch it
        for kind in ("late_store_open", "foreign"):
            f = r[kind]
            assert f["bytes_changed"] and f["scrub"]["damaged_all"] == [], kind
            assert _reasons(f["check1"]) == {p: ["tag"] for p in span_pages}, kind
        # damage on every rank: mend recomputes the span in place, and R's pages pass again
        for kind in ("unwritten", "late_bypass", "late_store_open", "foreign"):
            m = r[kind]["mend"]
            assert m["rebuilt"] == [] and [(x["start"], x["end"]) for x in m["recomputed"]] == [(64, 96)], kind
            assert m["still_failing_here"] == 0 and r[kind]["bytes_restored"], kind
            assert r[kind]["check2"]["found"] == {}, kind
        # damage on one rank: parity rebuilds it bit for bit
        o = r["one_gpu"]
        assert o["check1"]["found"] == {"3": {"tag": {"ranks": [2], "layers": 1}}}
        assert o["mend"]["recomputed"] == [] and o["check2"]["found"] == {}
        # a read after free
        assert _reasons(r["after_free"]["check1"]) == {"3": ["generation"]}
        assert r["forced"]["mend"]["recomputed"][0]["tokens"] == 2 * PAGE and r["forced"]["check"]["found"] == {}
        assert r["forced"]["mend"]["max_abs_diff_vs_clean"] == 0.0
    fix = res[2]["one_gpu"]["mend"]["rebuilt"]
    assert len(fix) == 1 and fix[0]["rebuilt"] and fix[0]["tag_matches"] and fix[0]["bit_exact"]
    for rank in every:
        for how in ("burst_rebuild", "burst_recompute"):
            b = res[rank][how]
            assert b["fault"]["pieces"] == 4 and b["check0"]["found"] == {}, how
            assert b["check1"]["found"] == {p: {"tag": {"ranks": [2], "layers": 2}} for p in ("2", "3")}, how
            assert b["restored"] and b["check2"]["found"] == {}, how
        assert res[rank]["burst_rebuild"]["mend"]["recomputed"] == []
        assert [(x["start"], x["end"]) for x in res[rank]["burst_recompute"]["mend"]["recomputed"]] == [(16, 48)]
    burst = [r for r in res[2]["burst_rebuild"]["mend"]["rebuilt"] if r.get("rebuilt")]
    assert len(burst) == 4 and all(r["bit_exact"] and r["tag_matches"] for r in burst)
    for rank in every:
        one, crowded = res[rank]["local_one_page"], res[rank]["local_crowded"]
        assert one["encode"]["stripes"] == 3 and one["encode"]["stripe"] == "local"
        assert one["check1"]["found"] == {"6": {"tag": {"ranks": every, "layers": LAYERS}}}
        mine = [x for x in one["mend"]["rebuilt"] if x.get("rebuilt")]  # this GPU's own pieces, from its own stripe
        assert len(mine) == LAYERS and all(x["tag_matches"] and x["holder"] == "local" for x in mine)
        assert one["mend"]["recomputed"] == [] and one["mend"]["max_abs_diff_vs_clean"] == 0.0
        assert one["check2"]["found"] == {}
        assert crowded["mend"]["rebuilt"] == []
        assert [(x["start"], x["end"]) for x in crowded["mend"]["recomputed"]] == [(64, 96)]
        assert crowded["check2"]["found"] == {}
