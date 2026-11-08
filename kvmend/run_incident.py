#!/usr/bin/env python3
"""The incident's faults (decision 20): KV that is well formed but wrong, on every GPU, as in the Cloudflare/SGLang
incident of 2026, where responses were computed from KV pages that were never written for them, and a dead request's
late write landed in another request's KV.

Request R (a fixed prompt, greedy, --max-new tokens) runs on an empty cache. The fault replaces R's last --pages
pages of the prompt (the question at its end) with KV of R's tokens at --donor, in every layer on every GPU:

  unwritten    at output 1, right after the prefill: the pages were never written for R (an earlier owner's bytes,
               no tags), as when part of a transferred prompt never arrives
  late_bypass  at output --late-at: a dead request's late write lands around the store path (bytes change, tags not)
  late_store   at output --late-at: the same write arrives through the store path with the pages' old generation
  foreign      at output --late-at: the pages hold another sequence's KV with matching tags (a misplaced page)

Responses:
  silent_*     no checks (plain SGLang, or GhostServe as published): does the text go wrong, with no error?
  kvmend_*     KVMend: the reader's check (coverage and token-keyed tags), then mend (recompute the pages in place,
               since every GPU is damaged and parity cannot rebuild them), then strict release regenerates the
               suspect tokens; the scrub runs first, to show which faults it alone would miss
  retry_*      abort and retry: KVMend's check detects the damage, R is aborted, the cache flushed, R resubmitted
  retry_keep_* a fair retry: the same abort, but the prompt before the damaged page stays cached (the tags locate
               the damage), so the resubmitted R recomputes from the damaged page on; its time counts the aborted
               attempt and the resubmission only
  control_keep no damage, the fair retry's path: the prompt before that page cached, the rest computed on top of
               it. Two passes are not bit for bit with one, so the fair retry is right when it equals this text or
               the clean text (a near-tie can flip, as on TinyLlama at output 73)
  kvmend_bad_tag, kvmend_bad_parity   damage to what KVMend keeps, on one GPU in one layer (with parity across the
               GPUs): a stored tag, or the KV with the parity that would rebuild it; the rebuild fails its tag, so the
               pages are recomputed
  parity_late  GhostServe told the GPU: parity rebuilds GPU --rank's share from the other GPUs' damaged pieces
  control_*    no fault, the same recompute (and rollback) as KVMend: the text a fault-free server gives on that path

R's last good check comes right after the prefill, at output 1. With --last-good plan (the default) the plan schedules
it (op check). With --last-good hook it does not: KVMend makes it itself, for every request
(KVMEND_CHECK_AFTER_PREFILL=all), and each rollback keeps the tokens KVMend's last passing check covered (retract keep
"last_good"). Every run then gets that check, the no-check rows too: silent_unwritten's damage, which strikes at output
1, is found by it and mended, and its report says that R's first token may be wrong (released_suspect).
A check right after the prefill bounds nothing, so KVMend's mend after it (kvmend_unwritten, and released_suspect)
also recomputes every position the prefill computed in a later chunk than the damaged page (only the damaged pages when
the prompt fits in one --chunk, or the damage is in its last chunk); control_mend1 forces the same repair.

--set prefill runs a store skipped during the prefill instead (choose a --chunk small enough for several chunks):
  owner_*           first, another prompt of the same length, so that R gets its very pages after the flush and the
                    skipped pages hold that earlier owner's KV
  kvmend_prefill    strict release: the chunk that computes --pages pages from --pos stores none of them, and every
                    later chunk reads the stale pages; the check right after the prefill finds them (coverage), and
                    KVMend recomputes them and every position of the later chunks, then generates the first token again
  control_prefill   no damage, the same path: that chunk stores the K and V but not their tags, so the same check
                    fails and the same repair runs (the text and every KV bit must equal kvmend_prefill's)
  pagesonly_prefill the skip, then the repair of the damaged pages only (as before this rule): its later chunks keep
                    KV computed from the stale pages
  suspect_prefill   the skip, outside the release rule: the check after the prefill (KVMend's with --last-good hook,
                    the plan's otherwise) mends the pages and the later chunks; the first token has left already
  clean             no fault: the fault-free copy of the text and of the KV
Every run takes fingerprints of R's prompt KV at output 2 (op digest): the whole prompt, before the damaged pages, the
rest of their chunk, and the later chunks.

  uv run python run_incident.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 4 --out runs/incident-check
  uv run python run_incident.py --set prefill --chunk 256 --pos 1024 --pages 1 --model ... --tp 2 --out runs/prefill
"""
import argparse
import faulthandler
import glob
import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAULTS = ("unwritten", "late_bypass", "late_store", "foreign")
PREFILL_RUNS = ("clean", "control_prefill", "kvmend_prefill", "pagesonly_prefill", "suspect_prefill")
PREFILL_STRICT = ("kvmend_prefill", "control_prefill")  # the release rule's path, a check every PREFILL_EVERY tokens
PREFILL_EVERY = 8
OWNED = ("kvmend_prefill", "pagesonly_prefill", "suspect_prefill")  # an earlier owner's KV in R's pages first


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--set", default="incident", choices=("incident", "prefill"),
                   help="the incident's faults, or a store skipped during the prefill (see above)")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--late-at", type=int, default=10)
    p.add_argument("--detect-at", type=int, default=41)
    p.add_argument("--keep", type=int, default=1, help="output tokens computed before the last good check")
    p.add_argument("--last-good", default="plan", choices=("plan", "hook"),
                   help="who makes R's last good check, right after the prefill: the plan (op check at output 1), or "
                        "KVMend for every request (KVMEND_CHECK_AFTER_PREFILL=all); with hook, rollbacks keep what "
                        "KVMend's last passing check covered")
    p.add_argument("--pages", type=int, default=4, help="pages the fault replaces")
    p.add_argument("--pos", type=int, default=-1,
                   help="first position of the damaged pages (a page boundary); default: the end of the prompt")
    p.add_argument("--donor", type=int, default=512, help="position of the tokens whose KV the fault writes")
    p.add_argument("--rank", type=int, default=2, help="the GPU an oracle names to GhostServe")
    p.add_argument("--stripe", default="gpus", choices=["gpus", "local"],
                   help="parity stripes across the GPUs, or within each GPU (always within at TP=1)")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.last_good == "hook" and a.keep != 1:
        p.error("--keep: KVMend's check after the prefill covers output 1, so --last-good hook keeps 1")
    if a.set == "prefill":
        a.pos = 1024 if a.pos < 0 else a.pos  # page 64, the running example
        if a.pos % 16 or a.pos + a.pages * 16 > a.prompt_len:
            p.error("--pos must be a page boundary with --pages pages inside the prompt")
    return a


def make_plan(a):
    span = dict(pages=a.pages, donor=a.donor, **(dict(pos=a.pos) if a.pos >= 0 else {}))
    enc = dict(op="encode", at=1, stripe=a.stripe)
    hook = a.last_good == "hook"
    # R's last good check, right after the prefill: the plan's own, or KVMend's (hook), whose record each rollback reads
    last_good = [enc] if hook else [enc, dict(op="check", at=1)]
    keep = "last_good" if hook else a.keep
    ghost = [dict(enc, check=False)]  # parity, and nothing is checked (with hook, KVMend checks after the prefill)
    unwritten = dict(op="fault", at=1, kind="unwritten", **span)

    def late(kind, **kw):
        return dict(op="fault", at=a.late_at, kind=kind, **span, **kw)

    detect = [dict(op="scrub", at=a.detect_at), dict(op="check", at=a.detect_at)]
    fix = [dict(op="mend", at=a.detect_at), dict(op="retract", at=a.detect_at, keep=keep, only_if_found=True)]
    # found right after the prefill, which no check bounds: KVMend's rule also recomputes what later chunks computed
    # (suffix); its fault-free control forces the same repair, as if that check had failed
    forced1 = dict(op="mend", at=1, force=True, suffix=True, **span)
    plans = {
        "warmup": last_good + [forced1],
        "clean": list(last_good),
        "control_mend1": last_good + [forced1],
        "control_mend41": last_good + [dict(op="mend", at=a.detect_at, force=True, **span),
                                       dict(op="retract", at=a.detect_at, keep=keep)],
        "control_retract41": last_good + [dict(op="retract", at=a.detect_at, keep=keep)],
        "silent_unwritten": ghost + [unwritten],
        "silent_late": ghost + [late("late_store", fence=False)],  # plain SGLang has no fence: the write lands
        "kvmend_unwritten": [enc, unwritten, dict(op="scrub", at=1), dict(op="check", at=1),
                             dict(op="mend", at=1, suffix=True)],
        "kvmend_late_bypass": last_good + [late("late_bypass")] + detect + fix,
        "kvmend_late_store": last_good + [late("late_store")] + detect + fix,
        "kvmend_foreign": last_good + [late("foreign")] + detect + fix,
        "parity_late": ghost + [late("late_bypass"), dict(op="repair_share", at=a.detect_at, rank=a.rank, scope="gpu")],
        "retry_unwritten": [enc, unwritten, dict(op="check", at=1),
                            dict(op="abort", at=1, only_if_found=True)],
        "retry_late": last_good + [late("late_bypass"), dict(op="check", at=a.detect_at),
                                   dict(op="abort", at=a.detect_at, only_if_found=True)],
        "retry2_unwritten": [],
        "retry2_late": [],
    }
    plans["control_keep"] = []  # no damage: only the cached prefix differs from the clean run
    plans["retry_keep_unwritten"] = list(plans["retry_unwritten"])  # the fair retry's first attempt: the same abort
    plans["retry_keep_late"] = list(plans["retry_late"])
    for kind in ("bad_tag", "bad_parity"):
        plans[f"kvmend_{kind}"] = last_good + [late(kind, rank=a.rank, layer=1)] + detect + fix
    if a.tp == 1 or a.stripe == "local":  # parity across the GPUs: no such rows here
        for rid in ("parity_late", "kvmend_bad_tag", "kvmend_bad_parity"):
            del plans[rid]
    return plans


def make_prefill_plan(a):
    """--set prefill: the plans of the skipped store during the prefill (see the top of this file). The skip is armed
    right before the chunk that computes the span (at 0: during the prefill); the strict runs are checked right after
    the prefill by the release rule (KVMEND_STRICT), the others by the plan (op check) or, with --last-good hook,
    suspect_prefill by KVMend's own check after every prefill."""
    span = dict(pos=a.pos, pages=a.pages)
    skip = dict(op="fault", at=0, kind="skipped_store", **span)
    digest = dict(op="digest", at=2, **span)
    check = [dict(op="check", at=1)]
    return {
        "warmup": [],
        "clean": [digest],
        "control_prefill": [dict(skip, keep_bytes=True), digest],
        "kvmend_prefill": [skip, digest],
        "pagesonly_prefill": [skip] + check + [dict(op="mend", at=1, suffix=False), digest],
        "suspect_prefill": [skip] + ([] if a.last_good == "hook" else check + [dict(op="mend", at=1, suffix=True)])
        + [digest],
    }


def rule_blocks(forwards, start, n, prompt_len):
    """The 16-token blocks KVMend's rule recomputes for damage in positions [start, start + n) found right after a
    prefill made of `forwards` ([first, end) of each, in order): the damaged blocks, and every block from the one
    that holds the first position of a later forward than the one that computed the first damaged block (of every
    forward, if that block lies before the prefill) to the end of the prompt. As hook.suspect_suffix."""
    blocks = set(range(start // 16, -(-(start + n) // 16)))
    lo = start // 16 * 16
    if forwards and lo < forwards[0][0]:
        nxt = forwards[0][0]
    else:
        c = next((i for i, (s, e) in enumerate(forwards) if s < lo + 16 and e > lo), None)
        nxt = forwards[c + 1][0] if c is not None and c + 1 < len(forwards) else None
    if nxt is not None:
        blocks |= set(range(nxt // 16, -(-prompt_len // 16)))
    return blocks


def block_tokens(blocks, prompt_len):
    return sum(min(16, prompt_len - 16 * j) for j in blocks)


def first_forwards(extends):
    """The forwards of a request's first prefill, [first, end) each, from its extend records (rank 0, in order).
    Records older than cached_prefix have prefix_len, which is the chunk's start for every chunk but the last."""
    out = []
    for x in extends:
        s = x.get("cached_prefix", x.get("prefix_len", 0))
        if out and out[-1][1] != s:
            break
        out.append((s, s + x["extend_input_len"]))
    return out


def recomputed_blocks(mend):
    return {j for x in (mend or {}).get("recomputed") or [] for j in range(x["start"] // 16, -(-x["end"] // 16))}


CONTROL = {"unwritten": "control_mend1", "late_bypass": "control_mend41", "late_store": "clean",
           "foreign": "control_mend41", "bad_tag": "control_mend41", "bad_parity": "control_mend41"}
META = ("bad_tag", "bad_parity")
CONTROL_REBUILT = {"late_bypass": "control_retract41", "foreign": "control_retract41"}  # pages rebuilt, not recomputed


def summarize(out, a, res, meta):
    from run_fair import classify

    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]

    def pick(rid, op, rank=0, at=None):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op and x["rank"] == rank
                and (at is None or x.get("at") == at)]

    def every_rank(rid, op, at=None):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op and (at is None or x.get("at") == at)]

    clean = res["clean"]["ids"]
    hook = a.last_good == "hook"
    ref = statistics.median(res[r]["seconds"] for r in ("clean", "silent_unwritten", "silent_late"))
    span_pages = None
    rows = {}
    for f in FAULTS + tuple(m for m in META if f"kvmend_{m}" in res):
        at = 1 if f == "unwritten" else a.detect_at
        kv = f"kvmend_{f}"
        fault = pick(kv, "fault")
        span_pages = span_pages or (fault[0]["pages"] if fault else None)
        scrub = pick(kv, "scrub", at=at)
        check = pick(kv, "check", at=at)
        mends = every_rank(kv, "mend", at=at)
        mend0 = [m for m in mends if m["rank"] == 0]
        retract = pick(kv, "retract", at=at)
        found = check[0]["found"] if check else None
        rebuilt_only = bool(mend0) and not mend0[0].get("recomputed") and bool(mend0[0].get("rebuilt"))
        twin = CONTROL_REBUILT.get(f, CONTROL[f]) if rebuilt_only else CONTROL[f]
        control = res[twin]["ids"]
        rows[f] = dict(
            fault=fault[0] if fault else None,
            rebuild_failed_pages=mend0[0].get("rebuild_failed_pages") if mend0 else None,
            rebuild_tried=sum(1 for m in mends for x in m.get("rebuilt", []) if x.get("rebuilt")),
            scrub_found_pages=sorted({d[2] for d in scrub[0]["damaged_all"]}) if scrub else None,
            check_found=found,
            check_seconds=check[0]["seconds"] if check else None,
            recomputed=mend0[0].get("recomputed") if mend0 else None,
            rebuilt=mend0[0].get("rebuilt") if mend0 else None,
            still_failing=sum(m.get("still_failing_here", 0) for m in mends) if mends else None,
            max_abs_diff_vs_clean=max((m.get("max_abs_diff_vs_clean", 0.0) for m in mends), default=None),
            bit_equal_share_vs_clean=min((m.get("bit_equal_share_vs_clean", 1.0) for m in mends), default=None),
            regenerated=retract[0].get("regenerated") if retract else None,
            retracted=retract[0].get("retracted") if retract else None,
            kvmend_text_right=res[kv]["ids"] == clean,
            kvmend_equals_control=res[kv]["ids"] == control, control_run=twin,
            kvmend_first_diff=classify(res[kv]["ids"], clean)[1],
            kvmend_seconds=res[kv]["seconds"], kvmend_extra_s=res[kv]["seconds"] - ref)
        silent = "silent_unwritten" if f == "unwritten" else "silent_late"
        rows[f]["silent_outcome"], rows[f]["silent_first_diff"] = (
            classify(res[silent]["ids"], clean) if f not in META else ("not run", None))
        if hook and f == "unwritten":  # KVMend's check after the prefill found this run's damage
            rows[f]["silent_outcome"] = "mended after the prefill: " + rows[f]["silent_outcome"]
        retry = res.get({"unwritten": "retry_unwritten", "late_bypass": "retry_late"}.get(f, ""))
        if retry:
            rows[f].update(retry_first_finish=retry["first"]["finish"], retry_text_right=retry["ids"] == clean,
                           retry_seconds=retry["seconds"], retry_extra_s=retry["seconds"] - ref)
        keep = res.get({"unwritten": "retry_keep_unwritten", "late_bypass": "retry_keep_late"}.get(f, ""))
        if keep:
            rows[f].update(retry_keep_equals_control=("control_keep" in res and keep["ids"] == res["control_keep"]["ids"]))
            rows[f].update(retry_keep_first_finish=keep["first"]["finish"], retry_keep_text_right=keep["ids"] == clean,
                           retry_keep_seconds=keep["seconds"], retry_keep_extra_s=keep["seconds"] - ref,
                           retry_keep_cached_prefix=keep["cached_prefix"])
    share = pick("parity_late", "repair_share", rank=a.rank) if "parity_late" in res else []
    every = list(range(a.tp))

    def caught(found, reason):
        return bool(found) and span_pages is not None and all(
            found.get(str(p), {}).get(reason, {}).get("ranks") == every for p in span_pages)

    def passed_at_1(rid):
        """The GPUs where R's last good check at output 1 passed: the plan's check op, or KVMend's check after the
        prefill (its batched record lists R among the requests outside the release rule, and among those that
        passed)."""
        if hook:
            return sum(1 for x in logs if x["op"] == "release_checks" and rid in (x.get("not_strict") or [])
                       and rid in x.get("passed", []))
        return sum(1 for x in every_rank(rid, "check", at=1) if x.get("found") == {})

    def check_s(rid):  # how long that check took on GPU 0
        xs = ([x["seconds"] for x in logs if x["op"] == "release_checks" and x["rank"] == 0
               and rid in x.get("passed", [])] if hook else [x["seconds"] for x in pick(rid, "check", at=1)])
        return xs[0] if xs else None

    # the runs that start from R's last good check at output 1 (their damage, if any, strikes after it)
    relying = [r for r in ("control_mend41", "control_retract41", "kvmend_late_bypass", "kvmend_late_store",
                           "kvmend_foreign", "kvmend_bad_tag", "kvmend_bad_parity", "retry_late", "retry_keep_late")
               if r in res]
    check_times = [t for t in map(check_s, relying) if t is not None]
    suspect = [dict(rid=x["rid"], rank=x["rank"], n_out=x["n_out"], found=x.get("found"),
                    mended_pages=x["mend"].get("mended_pages"),
                    recomputed_tokens=sum(t["tokens"] for t in x["mend"].get("recomputed", [])),
                    suffix_tokens=x.get("suffix_tokens"), still_failing_here=x["mend"].get("still_failing_here"))
               for x in logs if x["op"] == "released_suspect"]
    unwritten_start = (rows["unwritten"]["fault"] or {}).get("start")

    def rule_tokens(rid):
        """What KVMend's rule recomputes for the unwritten pages, found right after rid's prefill: the pages, and
        the later chunks of that prefill (from its extend records)."""
        if unwritten_start is None:
            return None
        forwards = first_forwards(pick(rid, "extend"))
        return block_tokens(rule_blocks(forwards, unwritten_start, a.pages * 16, a.prompt_len), a.prompt_len)
    summary = dict(
        meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new, late_at=a.late_at,
        detect_at=a.detect_at, keep=a.keep, pages=a.pages, pos=a.pos, donor=a.donor, chunk=a.chunk,
        prefill_forwards=first_forwards(pick("clean", "extend")),
        last_good=a.last_good, last_good_checks={r: passed_at_1(r) for r in relying},
        last_good_check_seconds=statistics.median(check_times) if check_times else None, released_suspect=suspect,
        reference_seconds=ref,
        reference_runs={r: res[r]["seconds"] for r in ("clean", "silent_unwritten", "silent_late")},
        controls={c: dict(equals_clean=res[c]["ids"] == clean, first_diff=classify(res[c]["ids"], clean)[1],
                          seconds=res[c]["seconds"])
                  for c in ("control_mend1", "control_mend41", "control_retract41", "control_keep") if c in res},
        control_recompute=[dict(rank=m["rank"], max_abs_diff=m.get("max_abs_diff_vs_clean"),
                                bit_equal_share=m.get("bit_equal_share_vs_clean"))
                           for m in every_rank("control_mend1", "mend", at=1)],
        faults=rows,
        parity_late=dict(outcome=classify(res["parity_late"]["ids"], clean)[0],
                         tag_mismatches_after=share[0].get("tag_mismatches_after") if share else None)
        if "parity_late" in res else None, stripe=a.stripe,
        texts={k: v["text"][:200] for k, v in res.items()},
        errors=[x for x in logs if x["op"] == "error"])
    fr = rows
    summary["verdict"] = {
        "no_errors": not summary["errors"],
        "unwritten_missed_by_the_scrub": fr["unwritten"]["scrub_found_pages"] == [],
        "unwritten_found_uncovered_on_every_gpu": caught(fr["unwritten"]["check_found"], "coverage"),
        "late_bypass_found_by_the_scrub": fr["late_bypass"]["scrub_found_pages"] == sorted(span_pages or []),
        "late_bypass_found_by_the_check": caught(fr["late_bypass"]["check_found"], "tag"),
        "late_store_dropped_by_the_fence": bool(fr["late_store"]["fault"]) and fr["late_store"]["fault"].get(
            "written_tokens") == 0 and fr["late_store"]["check_found"] == {} and not fr["late_store"]["retracted"],
        "foreign_missed_by_the_scrub": fr["foreign"]["scrub_found_pages"] == [],
        "foreign_found_by_the_token_keys": caught(fr["foreign"]["check_found"], "tag"),
        "mended_pages_pass_the_check": all(fr[f]["still_failing"] == 0 for f in ("unwritten", "late_bypass", "foreign")),
        # found right after the prefill: the unwritten pages and every later chunk of the prefill are recomputed
        "unwritten_recompute_follows_the_rule": sum(x["tokens"] for x in fr["unwritten"]["recomputed"] or [])
        == rule_tokens("kvmend_unwritten"),
        "suspect_tokens_regenerated": all(fr[f]["regenerated"] == a.detect_at - a.keep
                                          for f in ("late_bypass", "foreign")),
        "kvmend_equals_its_control": all(fr[f]["kvmend_equals_control"] for f in FAULTS),
        "retry_first_attempt_aborted": all(fr[f]["retry_first_finish"] == "abort" for f in ("unwritten", "late_bypass")),
        "retry_final_text_is_clean": all(fr[f]["retry_text_right"] for f in ("unwritten", "late_bypass")),
    }
    if all("retry_keep_seconds" in fr[f] for f in ("unwritten", "late_bypass")):
        summary["verdict"].update(
            retry_keep_first_attempt_aborted=all(fr[f]["retry_keep_first_finish"] == "abort"
                                                 for f in ("unwritten", "late_bypass")),
            retry_keep_final_text_right=all(fr[f]["retry_keep_text_right"] or fr[f]["retry_keep_equals_control"]
                                            for f in ("unwritten", "late_bypass")))
    if hook:  # KVMend made R's last good check itself, and it found silent_unwritten's damage at output 1
        mine = [x for x in suspect if x["rid"] == "silent_unwritten"]
        first = [x for x in mine if x["rank"] == 0]
        summary["verdict"].update(
            last_good_by_the_check_after_the_prefill=all(n == a.tp for n in summary["last_good_checks"].values()),
            unwritten_after_the_prefill_found_mended_and_reported=(
                len(mine) == a.tp and len(first) == 1 and caught(first[0]["found"], "coverage")
                and first[0]["recomputed_tokens"] == rule_tokens("silent_unwritten")
                and all(x["still_failing_here"] == 0 for x in mine)
                and res["silent_unwritten"]["ids"] in (clean, res["control_mend1"]["ids"])))
    for m in META:
        if m in fr:  # the rebuild from parity fails its tag on the damaged pages, which are then recomputed
            summary["verdict"][f"{m}_rebuild_failed_then_recomputed"] = (
                fr[m]["rebuild_failed_pages"] == sorted(span_pages or [])
                and sum(x["tokens"] for x in fr[m]["recomputed"] or []) == a.pages * 16 and fr[m]["still_failing"] == 0)
            summary["verdict"][f"{m}_text_right"] = fr[m]["kvmend_text_right"] or fr[m]["kvmend_equals_control"]
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "incident.md"), "w") as fh:
        fh.write(incident_md(summary))
    return summary


def incident_md(s):
    def yes(x):
        return "yes" if x else "no"

    where = (f"{s['pages']} pages from position {s['pos']}" if s.get("pos", -1) >= 0
             else f"the last {s['pages']} pages of the prompt")
    hook = s.get("last_good") == "hook"
    checks, check_s = s.get("last_good_checks", {}), s.get("last_good_check_seconds")
    lines = [f"Reference: {s['reference_seconds']:.2f} s, the median of {', '.join(s['reference_runs'])} "
             f"(no recovery{'; but silent_unwritten mends its page after the prefill' if hook else ''}). "
             f"Fault: {where}, every layer, every GPU (bad_tag, bad_parity: one layer, one GPU). "
             f"R's last good check, at output 1: "
             + ("KVMend's check after every prefill (KVMEND_CHECK_AFTER_PREFILL=all; every run gets it, the reference "
                "runs and the retries' resubmissions too)" if hook else "the test's own check")
             + f", passed on every GPU in {sum(n == s['tp'] for n in checks.values())} of the {len(checks)} runs "
             f"that start from it"
             + (f"; it took {check_s:.3f} s (median)" if check_s is not None else "") + ".", "",
             "| Fault | No checks | Scrub finds it | Reader check finds it | KVMend repair | KVMend text right "
             "| KVMend extra | Abort and retry text right | Retry extra | Retry from the damaged page extra |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for f, r in s["faults"].items():
        reasons = sorted({w for v in (r["check_found"] or {}).values() for w in v})
        repair = ("dropped by the fence" if f == "late_store" else
                  (f"rebuild of {r['rebuild_tried']} pieces failed its tag; " if r.get("rebuild_failed_pages") else "")
                  + f"recompute {sum(x['tokens'] for x in r['recomputed'] or [])} tokens"
                  + (f", regenerate {r['regenerated']}" if r["regenerated"] else ""))
        retry = (f"{yes(r.get('retry_text_right'))} | {r['retry_extra_s']:+.2f} s" if "retry_seconds" in r
                 else "not run | ")
        retry += (f" | {r['retry_keep_extra_s']:+.2f} s (text right: {yes(r['retry_keep_text_right'])}, "
                  f"control: {yes(r['retry_keep_equals_control'])})" if "retry_keep_seconds" in r else " | ")
        lines.append(f"| {f} | {r['silent_outcome']} | {yes(r['scrub_found_pages'])} | "
                     f"{', '.join(reasons) or 'nothing'} | {repair} | {yes(r['kvmend_text_right'])} "
                     f"(control: {yes(r['kvmend_equals_control'])}) | {r['kvmend_extra_s']:+.2f} s | {retry} |")
    ghost = s["parity_late"]
    lines += ["", f"Parity stripes: {s['stripe']}. Controls (no fault, the same repair path) equal the clean text: "
              f"{ {k: v['equals_clean'] for k, v in s['controls'].items()} }."
              + (f" GhostServe told the GPU, after a late write on every GPU: {ghost['outcome']}, "
                 f"{ghost['tag_mismatches_after']} tag mismatches after its rebuild." if ghost else "")]
    return "\n".join(lines) + "\n"


def summarize_prefill(out, a, res, meta):
    """--set prefill: what the skip struck, what the check after the prefill found, what each run recomputed, and
    the fingerprints of R's prompt KV against the fault-free runs, rank by rank."""
    from run_fair import classify

    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]

    def every_rank(rid, op):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op]

    def rank0(rid, op):
        return [x for x in every_rank(rid, op) if x["rank"] == 0]

    P, start, n, ranks, hook = a.prompt_len, a.pos, a.pages * 16, list(range(a.tp)), a.last_good == "hook"
    clean = res["clean"]["ids"]
    forwards = first_forwards(rank0("clean", "extend"))
    want = rule_blocks(forwards, start, n, P)
    span_blocks = set(range(start // 16, -(-(start + n) // 16)))
    digests = {rid: {x["rank"]: x["digest"] for x in every_rank(rid, "digest")} for rid in PREFILL_RUNS}

    def kv(rid1, rid2, part):  # per rank: the two runs' fingerprints of `part` agree (None: one is missing)
        d1, d2 = digests.get(rid1, {}), digests.get(rid2, {})
        return [None if r not in d1 or r not in d2 or d1[r].get(part) is None else d1[r][part] == d2[r][part]
                for r in ranks]

    def skips(rid):
        return [x for x in every_rank(rid, "fault") if x.get("kind") == "skipped_store" and x.get("armed")]

    def checks(rid):  # the check right after the prefill that found the skip: its record and its repair, per rank
        if rid in PREFILL_STRICT:
            return [(x, x.get("mend")) for x in every_rank(rid, "release_check") if x.get("n_out") == 1]
        if rid == "suspect_prefill" and hook:
            return [(x, x.get("mend")) for x in every_rank(rid, "released_suspect")]
        found = {x["rank"]: x for x in every_rank(rid, "check") if x.get("at") == 1}
        return [(found.get(m["rank"]), m) for m in every_rank(rid, "mend") if m.get("at") == 1]

    rows = {}
    for rid in PREFILL_RUNS:
        sk, ch = skips(rid), checks(rid)
        sk0 = [x for x in sk if x["rank"] == 0]
        c0 = [(c, m) for c, m in ch if (c or m or {}).get("rank") == 0]
        mend = c0[0][1] if c0 else None
        redo = rank0(rid, "pending_redo")
        rows[rid] = dict(
            struck_forwards=[x["forward"] for x in sk0], struck_pages=sk0[0]["pages"] if sk0 else None,
            skips=[{k: x.get(k) for k in ("rank", "n_out", "positions", "forward", "later_forwards", "stores_dropped",
                                          "stale_same_stores", "keep_bytes", "layers")} for x in sk],
            found=(c0[0][0] or {}).get("found") if c0 else None,
            recomputed=(mend or {}).get("recomputed"), recomputed_tokens=block_tokens(recomputed_blocks(mend), P),
            recomputed_blocks=sorted(recomputed_blocks(mend)), suffix=(mend or {}).get("suffix"),
            suffix_tokens=(c0[0][0] or {}).get("suffix_tokens") if c0 and c0[0][0] else None,
            reported=(c0[0][0] or {}).get("suffix") if c0 and rid == "suspect_prefill" and hook else None,
            released_tokens=(c0[0][0] or {}).get("released_tokens") if c0 and c0[0][0] else None,
            still_failing=sum((m or {}).get("still_failing_here", 0) for _, m in ch) if ch else None,
            mend_seconds=(mend or {}).get("mend_seconds"),
            regenerated=[(x.get("retracted"), x.get("regenerated")) for x in redo],
            text_equals_clean=res[rid]["ids"] == clean, first_diff_vs_clean=classify(res[rid]["ids"], clean)[1],
            kv_vs_clean={p: kv(rid, "clean", p) for p in ("whole", "before", "span", "same_chunk", "later")},
            seconds=res[rid]["seconds"], extra_s=res[rid]["seconds"] - res["clean"]["seconds"])
    kvm, ctl, pages_only, sus = (rows[r] for r in ("kvmend_prefill", "control_prefill", "pagesonly_prefill",
                                                   "suspect_prefill"))
    later = sorted(want - span_blocks)
    later_tokens = P - later[0] * 16 if later else 0
    every = dict(ranks=ranks)

    def struck_right(row):  # the skip struck, during the prefill, the forward that computes the span, on every GPU
        return (len(row["skips"]) >= a.tp and {x["rank"] for x in row["skips"]} == set(ranks)
                and all(x["n_out"] == 0 and x["forward"][0] <= start and x["stores_dropped"] == x["layers"]
                        and x["stale_same_stores"] == 0 for x in row["skips"] if not x["keep_bytes"])
                and any(x["later_forwards"] for x in row["skips"]))

    def found_right(row):  # the check right after the prefill found the span's pages by coverage on every GPU
        return bool(row["found"]) and bool(row["struck_pages"]) and all(
            row["found"].get(str(p), {}).get("coverage", {}).get("ranks") == every["ranks"]
            for p in row["struck_pages"])

    errors = [x for x in logs if x["op"] == "error"]
    verdict = {
        "no_errors": not errors,
        "the_skip_struck_a_chunk_that_later_chunks_read": all(struck_right(rows[r]) for r in OWNED)
        and bool(later),
        "found_by_the_check_right_after_the_prefill": all(found_right(rows[r]) for r in OWNED + ("control_prefill",)),
        # KVMend's repair: the damaged pages and every later chunk, from the page holding its first position; the
        # rest of the damaged chunk is not recomputed
        "later_chunks_recomputed": all(set(rows[r]["recomputed_blocks"]) == want and rows[r]["still_failing"] == 0
                                       for r in ("kvmend_prefill", "control_prefill", "suspect_prefill")),
        "suffix_size_reported": kvm["suffix_tokens"] == later_tokens and ctl["suffix_tokens"] == later_tokens
        and (not hook or (sus["suffix_tokens"] == later_tokens and sus["released_tokens"] == 1
                          and (sus["reported"] or {}).get("start") == (later[0] * 16 if later else None))),
        "first_token_generated_again": all(rows[r]["regenerated"][:1] and rows[r]["regenerated"][0][0]
                                           and rows[r]["regenerated"][0][1] >= 1 for r in PREFILL_STRICT),
        "kvmend_kv_equals_its_control": all(x is True for x in kv("kvmend_prefill", "control_prefill", "whole")),
        "kvmend_text_equals_its_control": res["kvmend_prefill"]["ids"] == res["control_prefill"]["ids"],
        # the rule's premise: a chunk reads its own K and V from the forward, so the rest of the damaged chunk did not
        # read the skipped pages
        "the_damaged_chunk_did_not_read_the_skip": all(
            x is True for r in ("kvmend_prefill", "pagesonly_prefill") for x in kv(r, "clean", "same_chunk")),
        "without_the_fix_later_chunks_differ_or_the_text_is_wrong": all(
            x is False for x in kv("pagesonly_prefill", "clean", "later")) or not pages_only["text_equals_clean"],
        "outside_the_rule_the_later_chunks_are_recomputed_too": all(
            x is True for x in kv("suspect_prefill", "control_prefill", "whole")),
    }
    summary = dict(meta=meta, set="prefill", model=a.model, tp=a.tp, prompt_len=P, chunk=a.chunk, pos=start,
                   pages=a.pages, max_new=a.max_new, last_good=a.last_good, prefill_forwards=forwards,
                   rule_blocks=sorted(want), later_tokens=later_tokens, rule_tokens=block_tokens(want, P), runs=rows,
                   texts={k: v["text"][:200] for k, v in res.items()}, errors=errors, verdict=verdict)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "incident.md"), "w") as fh:
        fh.write(prefill_md(summary))
    return summary


def prefill_md(s):
    def yes(x):
        return "yes" if x else "no"

    def kv(xs):
        return "/".join("=" if x else ("missing" if x is None else "differs") for x in xs)

    fw = ", ".join(f"[{x}, {y})" for x, y in s["prefill_forwards"])
    lines = [f"A store skipped during the prefill: {s['pages']} page(s) from position {s['pos']} of a "
             f"{s['prompt_len']}-token prompt, computed in chunks of {s['chunk']} tokens: {fw}. KVMend's rule "
             f"recomputes {s['rule_tokens']} tokens ({s['later_tokens']} of them in later chunks). KV fingerprints at "
             f"output 2, per GPU, against clean (= equal).", "",
             "| Run | Skip struck forward | Check found | Recomputed tokens | Text equals clean | Before | Damaged "
             "chunk, after the pages | Later chunks | Extra |", "|---|---|---|---|---|---|---|---|---|"]
    for rid, r in s["runs"].items():
        found = ", ".join(f"page {p}: {', '.join(w)}" for p, w in (r["found"] or {}).items()) or "-"
        lines.append(f"| {rid} | {r['struck_forwards'] or '-'} | {found} | {r['recomputed_tokens']} | "
                     f"{yes(r['text_equals_clean'])} | {kv(r['kv_vs_clean']['before'])} | "
                     f"{kv(r['kv_vs_clean']['same_chunk'])} | {kv(r['kv_vs_clean']['later'])} | "
                     f"{r['extra_s']:+.2f} s |")
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)  # the GPU the oracle names must exist (a TP=2 check has GPUs 0 and 1)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    prefill = a.set == "prefill"
    plan = make_prefill_plan(a) if prefill else make_plan(a)
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_FENCE",
              "KVMEND_CHECK_AFTER_PREFILL", "KVMEND_STRICT", "KVMEND_STRICT_EVERY", "KVMEND_ON_FAIL"):
        os.environ.pop(k, None)
    if a.last_good == "hook":
        os.environ["KVMEND_CHECK_AFTER_PREFILL"] = "all"
    if prefill:  # the release rule's path for kvmend_prefill and its control
        os.environ["KVMEND_STRICT"] = ",".join(f"{rid}:{PREFILL_EVERY}" for rid in PREFILL_STRICT)

    import sglang as sgl
    from transformers import AutoTokenizer

    from kvmend.meta import run_meta
    from run_fair import run
    from run_poc import build_prompt, engine_args, flush, start_loop

    meta = run_meta()
    prompt = build_prompt(AutoTokenizer.from_pretrained(a.model), a.prompt_len)
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    res = {}

    def go(rid, n=None):
        flush(engine)
        res[rid] = run(engine, prompt, rid, n or a.max_new, a.variant_timeout)
        print(f"{rid:20s} {len(res[rid]['ids'])} tokens in {res[rid]['seconds']:.2f} s ({res[rid]['finish']})",
              flush=True)
        return res[rid]

    start = a.pos if a.pos >= 0 else (len(prompt) // 16) * 16 - a.pages * 16  # the first damaged position

    def retry_keep(name):
        """The fair retry: abort as retry() does, then drop only what is damaged: the cache keeps the prompt before
        the damaged page (warmed here, untimed), and the resubmitted R recomputes from that page on."""
        flush(engine)
        first = run(engine, prompt, f"retry_keep_{name}", a.max_new, a.variant_timeout)
        flush(engine)
        run(engine, prompt[:start], f"retry_keep_warm_{name}", 1, a.variant_timeout)
        second = run(engine, prompt, f"retry_keep2_{name}", a.max_new, a.variant_timeout)
        res[f"retry_keep_{name}"] = dict(first=first, second=second, ids=second["ids"], text=second["text"],
                                         seconds=first["seconds"] + second["seconds"], cached_prefix=start)
        print(f"retry_keep_{name:9s} first {len(first['ids'])} tokens ({first['finish']}); retry from position "
              f"{start}: {res[f'retry_keep_{name}']['seconds']:.2f} s", flush=True)

    def retry(name):
        flush(engine)
        t0 = time.perf_counter()
        first = run(engine, prompt, f"retry_{name}", a.max_new, a.variant_timeout)
        flush(engine)  # the damaged cached KV must go, or the retry would read it again
        second = run(engine, prompt, f"retry2_{name}", a.max_new, a.variant_timeout)
        res[f"retry_{name}"] = dict(first=first, second=second, ids=second["ids"], text=second["text"],
                                    seconds=time.perf_counter() - t0)
        print(f"retry_{name:14s} first {len(first['ids'])} tokens ({first['finish']}); retry {len(second['ids'])} "
              f"tokens; {res[f'retry_{name}']['seconds']:.2f} s in all", flush=True)

    def owned(rid):
        """An earlier owner of R's pages: another prompt of the same length (the prompt's halves swapped), so that
        after the flush R gets its very pages, chunk by chunk, and a page whose store is skipped keeps its KV."""
        flush(engine)
        owner = run(engine, prompt[len(prompt) // 2:] + prompt[:len(prompt) // 2], f"owner_{rid}", 1,
                    a.variant_timeout)
        print(f"owner_{rid:14s} {len(owner['ids'])} token in {owner['seconds']:.2f} s", flush=True)
        go(rid)

    def incident():
        for rid in ("clean", "control_mend1", "control_mend41", "control_retract41", "silent_unwritten",
                    "silent_late", "kvmend_unwritten", "kvmend_late_bypass", "kvmend_late_store", "kvmend_foreign",
                    "parity_late", "kvmend_bad_tag", "kvmend_bad_parity"):
            if rid in plan:
                go(rid)
        retry("unwritten")
        retry("late")
        retry_keep("unwritten")
        retry_keep("late")
        flush(engine)  # the fair retry's fault-free control: the same cached prefix, then R on top of it
        run(engine, prompt[:start], "control_keep_warm", 1, a.variant_timeout)
        res["control_keep"] = run(engine, prompt, "control_keep", a.max_new, a.variant_timeout)
        print(f"control_keep         {len(res['control_keep']['ids'])} tokens in {res['control_keep']['seconds']:.2f} s",
              flush=True)

    try:
        go("warmup", n=8)
        if prefill:
            for rid in PREFILL_RUNS:
                (owned if rid in OWNED else go)(rid)
        else:
            incident()
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            flush(engine)
        except Exception as e:
            print("final flush failed:", repr(e), flush=True)
        faulthandler.cancel_dump_traceback_later()
        engine.shutdown()
    with open(os.path.join(out, "outputs.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    s = summarize_prefill(out, a, res, meta) if prefill else summarize(out, a, res, meta)
    print(json.dumps(s["verdict"], indent=1))
    print(prefill_md(s) if prefill else incident_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
