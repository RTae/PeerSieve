"""The fair test's two scheduler actions with stand-ins (CPU only, no SGLang): a strict rollback drops the output
tokens after the last good check unless they were already sent, and an abort marks the request."""
import types

import torch

from kvmend import hook


def _sched_with(req, seq_len=20):
    calls = {}
    batch = types.SimpleNamespace(reqs=[req], seq_lens=torch.tensor([seq_len]),
                                  filter_batch=lambda keep_indices: calls.setdefault("kept", keep_indices))
    sched = types.SimpleNamespace(
        running_batch=batch, page_size=16,
        token_to_kv_pool_allocator=types.SimpleNamespace(free=lambda idx: calls.setdefault("freed", idx.numel())),
        req_to_token_pool=types.SimpleNamespace(req_to_token=torch.arange(64).view(1, 64),
                                                free=lambda i: calls.setdefault("slot", i)),
        tree_cache=types.SimpleNamespace(dec_lock_ref=lambda node: calls.setdefault("unlocked", node)),
        _extend_requests_to_queue=lambda reqs, is_retracted: calls.setdefault("queued", (len(reqs), is_retracted)))
    return sched, calls


def _req(n_out, sent=0):
    r = types.SimpleNamespace(rid="R", req_pool_idx=0, prefix_indices=list(range(16)), last_node="node",
                              output_ids=list(range(100, 100 + n_out)), send_token_offset=sent)
    r.reset_for_retract = lambda: None
    return r


def test_strict_rollback_drops_the_unreleased_suspect_tokens():
    req = _req(5)
    sched, calls = _sched_with(req)
    rec = hook.act_retract(None, sched, {"keep": 1}, req)
    assert req.output_ids == [100] and (rec["regenerated"], rec["kept_outputs"]) == (4, 1)
    assert calls["freed"] == 4 and calls["queued"] == (1, True) and calls["kept"] == []


def test_strict_rollback_keeps_tokens_already_sent():
    req = _req(5, sent=3)
    sched, _ = _sched_with(req)
    rec = hook.act_retract(None, sched, {"keep": 1}, req)
    assert req.output_ids == [100, 101, 102, 103, 104] and rec["refused"] == "tokens already sent"


def test_abort_marks_the_request():
    req = types.SimpleNamespace(to_abort=False, to_abort_message=None)
    assert hook.act_abort(None, None, {}, req)["aborted"] and req.to_abort and req.to_abort_message


def test_only_if_found_waits_for_a_finding():
    """A rollback or an abort marked only_if_found runs only when the last check found damage."""
    quiet = types.SimpleNamespace(alarm_found=False)
    req = _req(5)
    sched, calls = _sched_with(req)
    assert hook.act_retract(quiet, sched, {"keep": 1, "only_if_found": True}, req)["retracted"] is False
    assert req.output_ids == [100, 101, 102, 103, 104] and not calls
    target = types.SimpleNamespace(to_abort=False, to_abort_message=None)
    assert hook.act_abort(quiet, None, {"only_if_found": True}, target)["aborted"] is False and not target.to_abort
    alarm = types.SimpleNamespace(alarm_found=True)
    assert hook.act_retract(alarm, sched, {"keep": 1, "only_if_found": True}, req)["retracted"] is True
    assert hook.act_abort(alarm, None, {"only_if_found": True}, target)["aborted"] and target.to_abort
