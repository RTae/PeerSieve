"""The check right after the prefill for requests outside the release rule (KVMEND_CHECK_AFTER_PREFILL), on 2 CPU
processes with gloo (no SGLang, no GPU). N is outside the release rule; S is strict.

  strict  the default: only strict requests are checked right after their prefill, as before
  all     every request is: N joins S's batched check (one pass over their pages, one collective). A pass is N's last
          good check (checked_upto, good_step, its pages' generations), so a later finding keeps the tokens it covered
          (retract keep "last_good"). A failure mends the pages and logs a report that N's first token, already
          released, may be wrong (op released_suspect); nothing is held back, generated again or aborted, and a strict
          reader of a mended page regenerates its unreleased tokens, as before."""
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

PAGE, HEADS, DIM, LAYERS, RANKS, POOL_PAGES = 16, 1, 64, 2, 2, 40
PROMPT = 6 * PAGE
N_IDS, N_ROW = list(range(1000, 1000 + PROMPT)), list(range(PAGE, PAGE + 256))  # N's prompt on pool pages 1-6
TAG3 = {"3": {"tag": {"ranks": list(range(RANKS)), "layers": LAYERS}}}  # a late write into page 3, on every rank


def _batch(reqs):
    b = types.SimpleNamespace(reqs=list(reqs), seq_lens=torch.full((len(reqs),), PROMPT + 1, dtype=torch.int64))

    def filter_batch(keep_indices):
        b.reqs = [b.reqs[j] for j in keep_indices]
        b.seq_lens = b.seq_lens[torch.tensor(keep_indices, dtype=torch.int64)]

    b.filter_batch = filter_batch
    return b


def _req(rid, row, prompt, out0):
    """A request that has just finished its prefill: one output token."""
    r = types.SimpleNamespace(rid=rid, req_pool_idx=row, origin_input_ids=list(prompt), output_ids=[out0],
                              prefix_indices=list(range(len(prompt))), last_node=rid, send_token_offset=0,
                              to_abort=False, is_chunked=0, finished_reason=None)
    r.finished = lambda r=r: r.finished_reason is not None

    def reset_for_retract(r=r):  # as SGLang's: the request gives up its row of the token table
        r.prefix_indices, r.req_pool_idx = [], None

    r.reset_for_retract = reset_for_retract
    return r


def _setup(rank, strict, after_prefill, rows):
    """The settings, a pool and its state, and a scheduler whose token table has `rows` (lists of slots)."""
    hook.STRICT, hook.ON_FAIL = hook._rules(strict)
    hook.ON_FAIL_DEFAULT, hook.CHECK_AFTER_PREFILL = "mend", after_prefill
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)],
                                 v_buffer=[torch.zeros(POOL_PAGES * PAGE, HEADS, DIM, dtype=torch.bfloat16)
                                           for _ in range(LAYERS)])
    st = hook.PoolState(pool)
    st.reset()
    table = torch.zeros(len(rows), 256, dtype=torch.int32)
    for i, slots in enumerate(rows):
        table[i] = torch.tensor(slots[:256], dtype=torch.int32)
    world = dist.group.WORLD
    queued = []
    sched = types.SimpleNamespace(
        tp_rank=rank, page_size=PAGE, running_batch=_batch([]), cur_batch=None,
        tp_group=types.SimpleNamespace(world_size=RANKS, rank_in_group=rank, ranks=list(range(RANKS)),
                                       cpu_group=world, device_group=world),
        req_to_token_pool=types.SimpleNamespace(req_to_token=table, free=lambda i: None),
        token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: None),
        tree_cache=types.SimpleNamespace(dec_lock_ref=lambda node: None),
        _extend_requests_to_queue=lambda reqs, is_retracted: queued.extend(r.rid for r in reqs))
    st.sched = sched
    del hook._BUF[:]
    return st, pool, sched, queued


def _store(st, pool, sched, req, positions, seed):
    """Write the KV of the request's positions through the store path, keyed by its tokens."""
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
        _store(st, pool, sched, req, [len(req.origin_input_ids) + len(req.output_ids) - 1], seed=seed + k)
        req.output_ids.append(req.output_ids[-1] + 1)


def _fake_recompute(st, sched, req, s, e):
    """Stand-in for the extend forward: write the clean KV that act_fault saved back through the store path, keyed
    by the request's tokens."""
    tgt, clean = st.saved_span
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, s:e].to(torch.int64)
    assert torch.equal(row, tgt[:e - s]), "the recompute must write over the damaged slots"
    ids = (list(req.origin_input_ids) + list(req.output_ids))[s:e]
    for layer in range(st.layers):
        st.pool.k_buffer[layer][row] = clean[layer, 0, :e - s]
        st.pool.v_buffer[layer][row] = clean[layer, 1, :e - s]
        st.update(layer, row, T.token_keys(torch.tensor(ids), torch.arange(s, e)))


def _kv(pool, page):
    s = slice(page * PAGE, (page + 1) * PAGE)
    return torch.stack([torch.stack([pool.k_buffer[l][s], pool.v_buffer[l][s]]) for l in range(LAYERS)]).clone()


def _records(op, **match):
    return [x for x in hook._BUF if x.get("op") == op and all(x.get(k) == v for k, v in match.items())]


def _spans(mend):
    """(start, end) of each span a repair recomputed, or False when it mended nothing."""
    return [(x["start"], x["end"]) for x in mend["recomputed"]] if mend.get("mended") else False


def _no_last_good(st, rid):
    return (rid in st.good_step, rid in st.checked_upto, rid in st.req_gen)


# ---------------------------------------------------------------------------------------------------------------
# strict and all; a pass

def _pass_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    gathers, gather = [0], hook._gather_found

    def counting_gather(*args):  # the batched check's collective
        gathers[0] += 1
        return gather(*args)

    hook._gather_found = counting_gather
    res = {}
    try:
        for setting in ("strict", "all"):
            # N on pages 1-6, S on pages 20-25; both have just finished their prefill (the current batch only)
            st, pool, sched, queued = _setup(rank, "S", setting, [N_ROW, list(range(20 * PAGE, 20 * PAGE + 256))])
            N, S = _req("N", 0, N_IDS, 7000), _req("S", 1, range(3000, 3000 + PROMPT), 8000)
            _store(st, pool, sched, N, list(range(PROMPT)), seed=10 + rank)
            _store(st, pool, sched, S, list(range(PROMPT)), seed=20 + rank)
            sched.cur_batch = _batch([N, S])
            passes, check_codes = [], st.check_codes  # the batched check's passes over pages
            st.check_codes = lambda p, e, t: (passes.append(int(p.numel())), check_codes(p, e, t))[1]
            gathers[0] = 0
            st.step = 1
            hook.release_checks(st, sched)
            r = res[setting] = dict(
                first=[{k: x.get(k) for k in ("requests", "passed", "not_strict", "pages")}
                       for x in _records("release_checks")],
                passes=list(passes), gathers=gathers[0], checked=dict(st.checked_upto), good=dict(st.good_step),
                gens=sorted(st.req_gen), prefill_checked=sorted(st.prefill_checked), cached_keys=sorted(st.view_cache))
            # two decode steps in the running batch: S is checked at each, N never again
            sched.running_batch, sched.cur_batch = _batch([N, S]), None
            for step in (2, 3):
                _decode(st, pool, sched, [N, S], seed=30 + 10 * rank + step)
                st.step = step
                hook.release_checks(st, sched)
            r["later"] = [(x["step"], x["passed"], x.get("not_strict")) for x in _records("release_checks")][1:]
            # the guard stays strict-only: N's newest token is not held back, S's unchecked one is
            _decode(st, pool, sched, [N, S], seed=60 + rank)  # output 4
            held = hook.held_back(st, [N, S])
            r["held"] = {req.rid: list(tail) for req, tail in held}
            for req, tail in held:
                req.output_ids.extend(tail)
            # a late write into N's prompt page 3 on every rank, found at output 4 by a check of N alone (as the
            # incident test schedules it); the page is mended and N rolls back to its last good check
            hook.act_fault(st, sched, dict(kind="late_bypass", pos=32, pages=1, donor=0), N)
            r["check"] = hook.act_check(st, sched, {}, N)["found"]
            hook.act_mend(st, sched, {}, N)
            r["retract"] = {k: v for k, v in hook.act_retract(st, sched, dict(keep="last_good"), N).items()
                            if k in ("retracted", "regenerated", "kept_outputs")}
            r["N_after"], r["queued"] = list(N.output_ids), list(queued)
        out.put((rank, res))
    finally:
        hook._gather_found = gather
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn(main):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(main, args=(_free_port(), out), nprocs=RANKS, start_method="spawn")
    return dict(out.get(timeout=300) for _ in range(RANKS))


def test_only_all_checks_a_request_outside_the_release_rule_after_its_prefill():
    res = _spawn(_pass_main)
    for rank in range(RANKS):
        s, a = res[rank]["strict"], res[rank]["all"]
        # strict (the default): only S is checked after the prefill, as before
        assert s["first"] == [dict(requests=1, passed=["S"], not_strict=[], pages=6)], s["first"]
        assert s["passes"] == [6] and s["gathers"] == 1
        assert s["checked"] == {"S": 1} and s["good"] == {"S": 1} and s["gens"] == ["S"]
        assert s["prefill_checked"] == []
        # all: N joins S's check, one pass over both requests' pages and one collective
        assert a["first"] == [dict(requests=2, passed=["N", "S"], not_strict=["N"], pages=12)], a["first"]
        assert a["passes"] == [12] and a["gathers"] == 1
        # the pass is N's last good check, in the release rule's records; N's token keys do not stay on the GPU
        assert a["checked"] == {"N": 1, "S": 1} and a["good"] == {"N": 1, "S": 1} and a["gens"] == ["N", "S"]
        assert a["prefill_checked"] == ["N"] and a["cached_keys"] == ["S"]
        for r in (s, a):
            assert r["later"] == [(2, ["S"], []), (3, ["S"], [])], r["later"]  # N is checked once at most
            assert r["held"] == {"S": [8003]}
            assert r["check"] == TAG3 and r["queued"] == ["N"]
        # the later finding: with all, the rollback keeps the token N's last good check covered; without it, none
        assert s["retract"] == dict(retracted=True, regenerated=4, kept_outputs=0) and s["N_after"] == []
        assert a["retract"] == dict(retracted=True, regenerated=3, kept_outputs=1) and a["N_after"] == [7000]


# ---------------------------------------------------------------------------------------------------------------
# a failure

def _fail_main(rank, port, out):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=RANKS)
    hook._recompute_range = _fake_recompute
    res = {}
    try:
        # N alone, with KVMEND_ON_FAIL=abort, a rule for strict requests only; a late write into its page 3 lands
        # between its prefill and the check right after it
        st, pool, sched, queued = _setup(rank, "", "all", [N_ROW])
        hook.ON_FAIL_DEFAULT = "abort"
        N = _req("N", 0, N_IDS, 7000)
        _store(st, pool, sched, N, list(range(PROMPT)), seed=10 + rank)
        clean = _kv(pool, 3)
        hook.act_fault(st, sched, dict(kind="late_bypass", pos=32, pages=1, donor=0), N)
        sched.cur_batch = _batch([N])
        st.step = 1
        hook.release_checks(st, sched)
        rep = _records("released_suspect", rid="N")
        r = res["alone"] = dict(
            report=[{k: x.get(k) for k in ("n_out", "released_tokens", "found", "other_readers")} for x in rep],
            mend=[(_spans(x["mend"]), x["mend"].get("still_failing_here")) for x in rep],
            batch=[(x["requests"], x["passed"], x.get("not_strict"), x["found_pieces"])
                   for x in _records("release_checks")],
            restored=torch.equal(_kv(pool, 3), clean), outputs=list(N.output_ids), queued=list(queued),
            aborted=N.to_abort, pending=dict(st.pending_redo), held=[(q.rid, t) for q, t in hook.held_back(st, [N])],
            good=_no_last_good(st, "N"), prefill_checked=sorted(st.prefill_checked))
        sched.running_batch, sched.cur_batch = _batch([N]), None  # the next step: N is not checked again
        _decode(st, pool, sched, [N], seed=40 + rank)
        n_records = len(hook._BUF)
        st.step = 2
        hook.release_checks(st, sched)
        r["checked_again"] = len(hook._BUF) != n_records
        # N and strict S share the damaged page (N's pages 1-4 are S's prefix), and both have just finished their
        # prefill. NS: N's check comes first and mends the page; SN: S's does, then N's own check passes.
        S_ROW = N_ROW[:4 * PAGE] + list(range(20 * PAGE, 20 * PAGE + 192))
        S_IDS = N_IDS[:4 * PAGE] + list(range(3064, 3096))
        for order in ("NS", "SN"):
            st, pool, sched, queued = _setup(rank, "S", "all", [N_ROW, S_ROW])
            N, S = _req("N", 0, N_IDS, 7000), _req("S", 1, S_IDS, 8000)
            _store(st, pool, sched, N, list(range(PROMPT)), seed=10 + rank)  # N writes the shared prefix
            _store(st, pool, sched, S, list(range(4 * PAGE, PROMPT)), seed=20 + rank)
            clean = _kv(pool, 3)
            hook.act_fault(st, sched, dict(kind="late_bypass", pos=32, pages=1, donor=0), N)
            sched.cur_batch = _batch([N, S] if order == "NS" else [S, N])
            st.step = 1
            hook.release_checks(st, sched)
            r = res[order] = dict(
                suspect=[dict(found=x["found"], mended=_spans(x["mend"]),
                              other_readers=[(o["rid"], o.get("pending", False)) for o in x.get("other_readers", [])])
                         for x in _records("released_suspect", rid="N")],
                strict=[dict(mended=_spans(x["mend"]), pending=x["regenerate"].get("pending", False))
                        for x in _records("release_check", rid="S")],
                batch=[(x["requests"], x["passed"], x.get("not_strict"), x["found_pieces"])
                       for x in _records("release_checks")],
                restored=torch.equal(_kv(pool, 3), clean), pending=dict(st.pending_redo), good=_no_last_good(st, "N"))
            sched.running_batch, sched.cur_batch = _batch([N, S]), None  # S's rollback, which waited, happens now
            _decode(st, pool, sched, [N, S], seed=50 + 10 * rank)
            st.step = 2
            hook.release_checks(st, sched)
            r.update(redo=[(x["retracted"], x["regenerated"]) for x in _records("pending_redo", rid="S")],
                     queued=list(queued), after={"N": list(N.output_ids), "S": list(S.output_ids)})
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_a_failed_check_after_the_prefill_mends_and_reports_without_regenerating():
    res = _spawn(_fail_main)
    for rank in range(RANKS):
        a = res[rank]["alone"]
        # found right after the prefill: the page is recomputed and N's report logged, though on_fail says abort
        assert a["report"] == [dict(n_out=1, released_tokens=1, found=TAG3, other_readers=[])], a["report"]
        assert a["mend"] == [([(32, 48)], 0)] and a["restored"]
        assert a["batch"] == [(1, [], ["N"], RANKS * LAYERS)], a["batch"]
        # nothing is held back, generated again or aborted, and the failed check is not N's last good check
        assert a["outputs"] == [7000] and a["queued"] == [] and not a["aborted"] and a["pending"] == {}
        assert a["held"] == [] and a["good"] == (False, False, False) and a["prefill_checked"] == ["N"]
        assert not a["checked_again"]
        # NS: N's check mends the shared page, and S, which read it, regenerates as any strict reader of a mended
        # page does (its rollback waits a step: S has just finished its prefill)
        ns = res[rank]["NS"]
        assert ns["suspect"] == [dict(found=TAG3, mended=[(32, 48)], other_readers=[("S", True)])], ns["suspect"]
        assert ns["strict"] == [] and ns["batch"] == [(2, [], ["N"], RANKS * LAYERS)] and ns["restored"]
        # SN: S's own check mends the page; N's check then passes, but N's prefill read the damage all the same
        sn = res[rank]["SN"]
        assert sn["strict"] == [dict(mended=[(32, 48)], pending=True)], sn["strict"]
        assert sn["suspect"] == [dict(found={}, mended=False, other_readers=[])], sn["suspect"]
        assert sn["batch"] == [(2, [], ["N"], RANKS * LAYERS)] and sn["restored"]
        for r in (ns, sn):
            assert r["pending"] == {"S": 0} and r["good"] == (False, False, False)
            assert r["redo"] == [(True, 2)] and r["queued"] == ["S"]
            assert r["after"] == {"N": [7000, 7001], "S": []}  # N keeps its tokens; S generates its own again


# ---------------------------------------------------------------------------------------------------------------
# the setting

def test_the_setting_is_strict_unless_set_to_all():
    """KVMEND_CHECK_AFTER_PREFILL is "strict" when unset, "strict" or "all" when set; anything else stops the import.
    The module is imported in a process of its own, so this process keeps its settings."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = ("import importlib, os\n"
            "from kvmend import hook\n"
            "got = [hook.CHECK_AFTER_PREFILL]\n"
            "for v in ('all', 'strict', 'every'):\n"
            "    os.environ['KVMEND_CHECK_AFTER_PREFILL'] = v\n"
            "    try:\n"
            "        got.append(importlib.reload(hook).CHECK_AFTER_PREFILL)\n"
            "    except ValueError:\n"
            "        got.append('refused')\n"
            "print(got)\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("KVMEND_")}
    env.update(PYTHONPATH=root, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "['strict', 'all', 'strict', 'refused']", r.stdout
