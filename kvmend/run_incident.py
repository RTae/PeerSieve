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
  parity_late  GhostServe told the GPU: parity rebuilds GPU --rank's share from the other GPUs' damaged pieces
  control_*    no fault, the same recompute (and rollback) as KVMend: the text a fault-free server gives on that path

  uv run python run_incident.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 4 --out runs/incident-check
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


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--late-at", type=int, default=10)
    p.add_argument("--detect-at", type=int, default=41)
    p.add_argument("--keep", type=int, default=1, help="output tokens computed before the last good check")
    p.add_argument("--pages", type=int, default=4, help="pages the fault replaces, at the end of the prompt")
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
    return p.parse_args()


def make_plan(a):
    span = dict(pages=a.pages, donor=a.donor)
    enc = dict(op="encode", at=1, stripe=a.stripe)
    last_good = [enc, dict(op="check", at=1)]  # R's last good check, right after the prefill
    ghost = [dict(enc, check=False)]  # parity, and nothing is checked
    unwritten = dict(op="fault", at=1, kind="unwritten", **span)

    def late(kind, **kw):
        return dict(op="fault", at=a.late_at, kind=kind, **span, **kw)

    detect = [dict(op="scrub", at=a.detect_at), dict(op="check", at=a.detect_at)]
    fix = [dict(op="mend", at=a.detect_at), dict(op="retract", at=a.detect_at, keep=a.keep, only_if_found=True)]
    forced1 = dict(op="mend", at=1, force=True, **span)
    plans = {
        "warmup": last_good + [forced1],
        "clean": list(last_good),
        "control_mend1": last_good + [forced1],
        "control_mend41": last_good + [dict(op="mend", at=a.detect_at, force=True, **span),
                                       dict(op="retract", at=a.detect_at, keep=a.keep)],
        "control_retract41": last_good + [dict(op="retract", at=a.detect_at, keep=a.keep)],
        "silent_unwritten": ghost + [unwritten],
        "silent_late": ghost + [late("late_store", fence=False)],  # plain SGLang has no fence: the write lands
        "kvmend_unwritten": [enc, unwritten, dict(op="scrub", at=1), dict(op="check", at=1),
                             dict(op="mend", at=1)],
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
    if a.tp == 1 or a.stripe == "local":  # GhostServe's parity spans the GPUs: no such row here
        del plans["parity_late"]
    return plans


CONTROL = {"unwritten": "control_mend1", "late_bypass": "control_mend41", "late_store": "clean",
           "foreign": "control_mend41"}
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
    ref = statistics.median(res[r]["seconds"] for r in ("clean", "silent_unwritten", "silent_late"))
    span_pages = None
    rows = {}
    for f in FAULTS:
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
        rows[f]["silent_outcome"], rows[f]["silent_first_diff"] = classify(res[silent]["ids"], clean)
        retry = res.get({"unwritten": "retry_unwritten", "late_bypass": "retry_late"}.get(f, ""))
        if retry:
            rows[f].update(retry_first_finish=retry["first"]["finish"], retry_text_right=retry["ids"] == clean,
                           retry_seconds=retry["seconds"], retry_extra_s=retry["seconds"] - ref)
    share = pick("parity_late", "repair_share", rank=a.rank) if "parity_late" in res else []
    every = list(range(a.tp))

    def caught(found, reason):
        return bool(found) and span_pages is not None and all(
            found.get(str(p), {}).get(reason, {}).get("ranks") == every for p in span_pages)

    summary = dict(
        meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new, late_at=a.late_at,
        detect_at=a.detect_at, keep=a.keep, pages=a.pages, donor=a.donor, reference_seconds=ref,
        reference_runs={r: res[r]["seconds"] for r in ("clean", "silent_unwritten", "silent_late")},
        controls={c: dict(equals_clean=res[c]["ids"] == clean, first_diff=classify(res[c]["ids"], clean)[1],
                          seconds=res[c]["seconds"])
                  for c in ("control_mend1", "control_mend41", "control_retract41")},
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
        "suspect_tokens_regenerated": all(fr[f]["regenerated"] == a.detect_at - a.keep
                                          for f in ("late_bypass", "foreign")),
        "kvmend_equals_its_control": all(fr[f]["kvmend_equals_control"] for f in FAULTS),
        "retry_first_attempt_aborted": all(fr[f]["retry_first_finish"] == "abort" for f in ("unwritten", "late_bypass")),
        "retry_final_text_is_clean": all(fr[f]["retry_text_right"] for f in ("unwritten", "late_bypass")),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "incident.md"), "w") as fh:
        fh.write(incident_md(summary))
    return summary


def incident_md(s):
    def yes(x):
        return "yes" if x else "no"

    lines = [f"Reference: {s['reference_seconds']:.2f} s, the median of {', '.join(s['reference_runs'])} "
             f"(no recovery). Fault: the last {s['pages']} pages of the prompt, every layer, every GPU.", "",
             "| Fault | No checks | Scrub finds it | Reader check finds it | KVMend repair | KVMend text right "
             "| KVMend extra | Abort and retry text right | Retry extra |", "|---|---|---|---|---|---|---|---|---|"]
    for f, r in s["faults"].items():
        reasons = sorted({w for v in (r["check_found"] or {}).values() for w in v})
        repair = ("dropped by the fence" if f == "late_store" else
                  f"recompute {sum(x['tokens'] for x in r['recomputed'] or [])} tokens"
                  + (f", regenerate {r['regenerated']}" if r["regenerated"] else ""))
        retry = (f"{yes(r.get('retry_text_right'))} | {r['retry_extra_s']:+.2f} s" if "retry_seconds" in r
                 else "same as late_bypass | ")
        lines.append(f"| {f} | {r['silent_outcome']} | {yes(r['scrub_found_pages'])} | "
                     f"{', '.join(reasons) or 'nothing'} | {repair} | {yes(r['kvmend_text_right'])} "
                     f"(control: {yes(r['kvmend_equals_control'])}) | {r['kvmend_extra_s']:+.2f} s | {retry} |")
    ghost = s["parity_late"]
    lines += ["", f"Parity stripes: {s['stripe']}. Controls (no fault, the same repair path) equal the clean text: "
              f"{ {k: v['equals_clean'] for k, v in s['controls'].items()} }."
              + (f" GhostServe told the GPU, after a late write on every GPU: {ghost['outcome']}, "
                 f"{ghost['tag_mismatches_after']} tag mismatches after its rebuild." if ghost else "")]
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)  # the GPU the oracle names must exist (a TP=2 check has GPUs 0 and 1)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    plan = make_plan(a)
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_FENCE"):
        os.environ.pop(k, None)

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

    try:
        go("warmup", n=8)
        for rid in ("clean", "control_mend1", "control_mend41", "control_retract41", "silent_unwritten",
                    "silent_late", "kvmend_unwritten", "kvmend_late_bypass", "kvmend_late_store", "kvmend_foreign",
                    "parity_late"):
            if rid in plan:
                go(rid)
        retry("unwritten")
        retry("late")
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
    s = summarize(out, a, res, meta)
    print(json.dumps(s["verdict"], indent=1))
    print(incident_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
