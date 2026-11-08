"""The step path that runs inside SGLang's scheduler: Plan picks due actions, on_step runs them once and in order,
and _log writes one line per action under the rank that wrote it (CPU only, no SGLang)."""
import json
import types

import torch

from kvmend import hook

PAGE = 16


def _sched(n_out, tmp_path):
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(4 * PAGE, 1, 64, dtype=torch.bfloat16)],
                                 v_buffer=[torch.zeros(4 * PAGE, 1, 64, dtype=torch.bfloat16)])
    pool._kvm = hook.PoolState(pool)
    req = types.SimpleNamespace(rid="R", req_pool_idx=0, output_ids=list(range(n_out)))
    return types.SimpleNamespace(
        tp_rank=1, running_batch=types.SimpleNamespace(reqs=[req]), cur_batch=None,
        token_to_kv_pool_allocator=types.SimpleNamespace(get_kvcache=lambda: pool)), req


def test_log_keeps_the_rank_that_wrote_the_line(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "OUT", str(tmp_path))
    sched = types.SimpleNamespace(tp_rank=1)
    hook._log(sched, {"op": "inject", "target_rank": 2, "rank": 7})
    assert not (tmp_path / "rank1.jsonl").exists()  # kept in memory until the scheduler is idle or flushes
    hook._flush_log(sched)
    rec = json.loads((tmp_path / "rank1.jsonl").read_text())
    assert rec["rank"] == 1 and rec["target_rank"] == 2


def test_on_step_runs_due_actions_once_and_in_order(tmp_path, monkeypatch):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"R": [{"op": "b", "at": 3}, {"op": "a", "at": 1}, {"op": "c", "at": 3}]}))
    monkeypatch.setattr(hook, "OUT", str(tmp_path))
    monkeypatch.setattr(hook, "PLAN", hook.Plan(str(plan)))
    calls = []
    fake = {op: (lambda op: lambda st, sched, a, req: calls.append(op) or {"rank": 99, "op_result": op})(op)
            for op in "abc"}
    monkeypatch.setattr(hook, "ACTIONS", fake)
    sched, req = _sched(1, tmp_path)
    hook.on_step(sched)
    hook.on_step(sched)  # nothing new is due
    req.output_ids = list(range(3))
    hook.on_step(sched)
    hook._flush_log(sched)
    assert calls == ["a", "b", "c"]
    lines = [json.loads(x) for x in (tmp_path / "rank1.jsonl").read_text().splitlines()]
    assert [(x["op"], x["at"], x["n_out"], x["rank"]) for x in lines] == [("a", 1, 1, 1), ("b", 3, 3, 1), ("c", 3, 3, 1)]
    assert all(x["t0"] <= x["time"] for x in lines)


def test_evict_drops_every_unlocked_page():
    calls = []
    tree = types.SimpleNamespace(evictable_size=lambda: 0 if calls else 48, evict=calls.append)
    rec = hook.act_evict(None, types.SimpleNamespace(tree_cache=tree), {}, None)
    assert calls == [48] and (rec["evictable_before"], rec["evictable_after"]) == (48, 0)
