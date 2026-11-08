#!/usr/bin/env python3
"""Checks on read that repair (KVMEND_CHECK_IN_ATTN=1, KVMEND_ATTN_ACT=repair) on SGLang 0.5.1: damage strikes R's KV
between two decode steps, the check inside the next step's decode attention finds it, and KVMend mends the pages and
generates that step's token again before it leaves. Request R (a fixed prompt, greedy decoding, --max-new tokens) runs
alone on an empty cache, in one engine with fused tags and CUDA graphs, as bench_overhead.py's attn mode:

  clean          no fault
  probe_repair   parity of the prompt's pages at output 1; at output --probe-at, one bit of one K value of R's prompt
                 flips on the last GPU (bench_overhead.py's probe): the next decode step must find it, parity rebuild
                 the piece bit for bit, and R drop that step's token and generate it again
  probe_twin     the same parity, no fault, R rolled back at output --probe-at: the fault-free run on the repair's path
                 (the rebuild restores the very bytes, so nothing is recomputed)
  late_silent    KVMEND_ATTN_ACT=log, as before this change: at output --late-at a late write lands around the store
                 path in --pages pages of R's prompt, on every GPU (run_incident.py's late_bypass); the check on read
                 logs it at the next step, and nothing acts
  late_repair    the same late write: found at the next step on every GPU, the pages recomputed in place, that step's
                 token dropped and generated again
  late_twin      no fault; the same pages recomputed in place at output --late-at, and R rolled back there
  strict_repair, strict_twin   late_repair and late_twin under strict release, a check every --strict-every tokens and
                 none due at --late-at, so that the check on read finds the damage before the release rule does

Pass: each damage found at the next decode step, every GPU agreeing (the probe: exactly the flipped piece); each repair's
text equals its twin's, token for token; late_silent logged the damage at the next step and nothing acted; no check on
read failed in a fault-free run; no errors. With --require-wrong, late_silent's text must also differ from late_twin's:
the wrong tokens that the repair avoided (true on TinyLlama; on a model whose text this damage leaves unchanged, it is
only reported).

  uv run python run_attn_repair.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --require-wrong \
      --out runs/attn-repair-check
"""
import argparse
import faulthandler
import glob
import json
import os

from bench_overhead import PROBE_AT, PROBE_FAULT, PROBE_LAYER

HERE = os.path.dirname(os.path.abspath(__file__))
TWIN = dict(probe_repair="probe_twin", late_repair="late_twin", strict_repair="strict_twin")
FAULT_FREE = ("warmup", "clean", "probe_twin", "late_twin", "strict_twin")
RUNS = ("clean", "probe_twin", "probe_repair", "late_twin", "late_silent", "late_repair", "strict_twin",
        "strict_repair")
PROBE_POS = 40  # bench_overhead.py's probe position: a full page of every prompt of 48 tokens or more


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--probe-at", type=int, default=PROBE_AT, help="output count at which the probe's bit flips")
    p.add_argument("--late-at", type=int, default=10, help="output count at which the late write lands")
    p.add_argument("--pages", type=int, default=4, help="pages the late write replaces")
    p.add_argument("--pos", type=int, default=-1, help="first position of those pages (default: the prompt's end)")
    p.add_argument("--donor", type=int, default=512, help="position of the tokens whose KV the late write lands")
    p.add_argument("--strict-every", type=int, default=8, help="output tokens between the strict runs' checks")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-wrong", action="store_true", help="late_silent's text must differ from late_twin's")
    p.add_argument("--require-pass", action="store_true", help="exit 3 unless every pass check holds")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    a.start = a.pos if a.pos >= 0 else (a.prompt_len // 16) * 16 - a.pages * 16  # the late write's first position
    if a.prompt_len < PROBE_POS + 16 - PROBE_POS % 16:
        p.error(f"--prompt-len: the probe's page (position {PROBE_POS}) must be a full prompt page")
    if a.start % 16 or a.start < 0 or a.start + a.pages * 16 > a.prompt_len or a.donor + a.pages * 16 > a.start:
        p.error("the late write's pages must start at a page boundary inside the prompt, after the donor's tokens")
    if a.late_at < 2 or a.late_at % a.strict_every == 0:
        p.error("--late-at: no strict release check may be due there (output 1 or a multiple of --strict-every)")
    if a.probe_at < 2 or a.max_new <= max(a.late_at, a.probe_at) + 1:
        p.error("--probe-at must be 2 or more, and --max-new must reach past it and --late-at")
    return a


def make_plan(a):
    span = dict(pos=a.start, pages=a.pages, donor=a.donor)
    enc = dict(op="encode", at=1)
    probe = dict(op="inject", at=a.probe_at, pos=PROBE_POS, layer=PROBE_LAYER, rank=a.tp - 1, **PROBE_FAULT)
    late = dict(op="fault", at=a.late_at, kind="late_bypass", **span)
    twin = [dict(op="mend", at=a.late_at, force=True, **span), dict(op="retract", at=a.late_at, keep=a.late_at)]
    return dict(warmup=[], clean=[], probe_repair=[enc, probe],
                probe_twin=[enc, dict(op="retract", at=a.probe_at, keep=a.probe_at)],
                late_silent=[late], late_repair=[late], late_twin=twin, strict_repair=[late], strict_twin=list(twin))


def read_logs(out):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    return logs


def summarize(out, a, res, meta):
    from run_fair import classify

    logs = read_logs(out)
    ranks = list(range(a.tp))
    clean = res["clean"]["ids"]

    def of(op, rid):  # the records of `op` that name rid: its own (plan actions) or among the readers (checks on read)
        return [x for x in logs if x["op"] == op and (x.get("rid") == rid or rid in (x.get("rids") or []))]

    rows = {}
    for rid, twin in TWIN.items():
        cause = [x for x in of("inject" if rid == "probe_repair" else "fault", rid) if x["rank"] == 0]
        acts = of("attn_act", rid)
        a0 = next((x for x in acts if x["rank"] == 0), None)
        step = cause[0]["step"] if cause else None
        reader = next((r for r in (a0 or {}).get("readers", []) if r["rid"] == rid), {})
        mends = [m for x in acts for m in x.get("mends", [])]  # every GPU's; GPU 0's alone for what all GPUs share
        rows[rid] = dict(
            twin=twin, fault=cause[0] if cause else None, found_step=(a0 or {}).get("step"),
            found_next_step=a0 is not None and step is not None and a0["step"] == step + 1
            and sorted(x["rank"] for x in acts) == ranks and len({x["step"] for x in acts}) == 1,
            every_gpu_agrees=len(acts) == a.tp and len({json.dumps(x["summary"], sort_keys=True) for x in acts}) == 1,
            found=(a0 or {}).get("found"), summary=(a0 or {}).get("summary"), pages=(a0 or {}).get("pages"),
            mended_pages=(a0 or {}).get("mended_pages"), freed_pages=(a0 or {}).get("freed_pages"),
            repair=(a0 or {}).get("repair"),
            reader={k: reader.get(k) for k in ("strict", "n_out", "keep", "regenerate", "checked_upto", "last_pass")},
            rebuilt=[e for m in mends for e in m.get("rebuilt", []) if e.get("rebuilt")],
            recomputed_tokens=sum(s["tokens"] for m in (a0 or {}).get("mends", []) for s in m.get("recomputed", [])),
            still_failing=sum(m.get("still_failing_here", 0) for m in mends),
            repair_seconds=(a0 or {}).get("seconds"),
            found_again_after=[x["step"] for x in of("attn_found", rid) if a0 and x["step"] > a0["step"]],
            text_equals_twin=res[rid]["ids"] == res[twin]["ids"], tokens=len(res[rid]["ids"]),
            first_diff_vs_twin=classify(res[rid]["ids"], res[twin]["ids"])[1],
            first_diff_vs_clean=classify(res[rid]["ids"], clean)[1],
            seconds=res[rid]["seconds"], extra_s=res[rid]["seconds"] - res[twin]["seconds"])
    silent_cause = [x for x in of("fault", "late_silent") if x["rank"] == 0]
    silent_found = of("attn_found", "late_silent")
    first_found = {x["rank"]: min(y["step"] for y in silent_found if y["rank"] == x["rank"]) for x in silent_found}
    outcome, first_wrong = classify(res["late_silent"]["ids"], res["late_twin"]["ids"])
    silent = dict(fault=silent_cause[0] if silent_cause else None, found_on_ranks=sorted(first_found),
                  first_found_step=first_found, found_records=len(silent_found),
                  acted=len(of("attn_act", "late_silent")), outcome_vs_twin=outcome, first_wrong_vs_twin=first_wrong,
                  first_diff_vs_clean=classify(res["late_silent"]["ids"], clean)[1])
    errors = [x for x in logs if x["op"] == "error"]
    alarms = [x for x in logs if x["op"] in ("attn_found", "attn_act") and set(x.get("rids") or []) & set(FAULT_FREE)]
    pr, lr, sr = rows["probe_repair"], rows["late_repair"], rows["strict_repair"]
    span_pages = (lr["fault"] or {}).get("pages") or []
    probe_piece = [[a.tp - 1, PROBE_LAYER, (pr["fault"] or {}).get("page"), "tag"]]
    probe_rank = [x for x in of("inject", "probe_repair") if x["rank"] == a.tp - 1]
    if probe_rank:
        probe_piece = [[a.tp - 1, probe_rank[0]["layer"], probe_rank[0]["page"], "tag"]]
    verdict = {
        "no_errors": not errors,
        "no_false_alarm": not alarms,
        "probe_found_at_the_next_step": pr["found_next_step"] and pr["every_gpu_agrees"],
        "probe_names_exactly_the_piece": pr["found"] == probe_piece,
        "probe_rebuilt_bit_exact": len(pr["rebuilt"]) == 1 and all(e.get("tag_matches") and e.get("bit_exact")
                                                                   for e in pr["rebuilt"]),
        "late_found_on_every_gpu_at_the_next_step": lr["found_next_step"] and lr["every_gpu_agrees"]
        and lr["pages"] == sorted(span_pages) and all(
            (lr["summary"] or {}).get(str(p), {}).get("tag", {}).get("ranks") == ranks for p in span_pages),
        "late_pages_recomputed": lr["recomputed_tokens"] == a.pages * 16 and lr["still_failing"] == 0
        and lr["mended_pages"] == sorted(span_pages),
        "strict_found_at_the_next_step": sr["found_next_step"] and sr["every_gpu_agrees"],
        "strict_kept_its_tokens_and_held_them": sr["reader"].get("strict") is True
        and sr["reader"].get("keep") == a.late_at
        and sr["reader"].get("checked_upto") == a.late_at // a.strict_every * a.strict_every,
        "each_step_token_dropped_and_generated_again": all(
            r["reader"].get("regenerate") == 1 and r["reader"].get("keep") == r["reader"].get("n_out")
            and r["tokens"] == a.max_new for r in rows.values()),
        "found_once": not any(r["found_again_after"] for r in rows.values()),
        "repairs_equal_their_twins": all(r["text_equals_twin"] for r in rows.values()),
        "log_mode_only_logged": silent["acted"] == 0 and silent["found_on_ranks"] == ranks and bool(silent["fault"])
        and all(s == silent["fault"]["step"] + 1 for s in first_found.values()),
    }
    if a.require_wrong:
        verdict["wrong_tokens_avoided"] = first_wrong is not None
    summary = dict(meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new,
                   probe_at=a.probe_at, late_at=a.late_at, pages=a.pages, start=a.start, donor=a.donor,
                   strict_every=a.strict_every, runs=rows, late_silent=silent,
                   seconds={k: v["seconds"] for k, v in res.items()},
                   texts={k: v["text"][:200] for k, v in res.items()}, errors=errors[:20], false_alarms=alarms[:20],
                   verdict=verdict)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "attn_repair.md"), "w") as fh:
        fh.write(report_md(summary))
    return summary


def report_md(s):
    def yes(x):
        return "yes" if x else "no"

    lines = [f"Checks on read that repair: {os.path.basename(str(s['model']).rstrip('/'))} at TP={s['tp']}, a "
             f"{s['prompt_len']}-token prompt, {s['max_new']} output tokens, greedy. Probe: one bit at output "
             f"{s['probe_at']} on the last GPU. Late write: {s['pages']} pages from position {s['start']} at output "
             f"{s['late_at']}, on every GPU.", "",
             "| Run | Found at step (damage at) | Repair | Kept, generated again | Text equals its twin | First "
             "difference from clean | Repair time |", "|---|---|---|---|---|---|---|"]
    for rid, r in s["runs"].items():
        kind = (f"rebuilt {len(r['rebuilt'])} piece(s) from parity" if r["rebuilt"] else
                f"recomputed {r['recomputed_tokens']} tokens")
        t = r["repair_seconds"]
        lines.append(f"| {rid} | {r['found_step']} ({(r['fault'] or {}).get('step')}) | {kind} | "
                     f"{r['reader'].get('keep')}, {r['reader'].get('regenerate')} | {yes(r['text_equals_twin'])} | "
                     f"{r['first_diff_vs_clean']} | {'' if t is None else f'{1e3 * t:.1f} ms'} |")
    v = s["late_silent"]
    lines += ["", f"late_silent (KVMEND_ATTN_ACT=log): found at step {v['first_found_step']} on GPUs "
                  f"{v['found_on_ranks']} after the damage at step {(v['fault'] or {}).get('step')}; acted "
                  f"{v['acted']} times; text vs late_twin: {v['outcome_vs_twin']} (first wrong token at output "
                  f"{v['first_wrong_vs_twin']})."]
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path, ctrl_path = os.path.join(out, "plan.json"), os.path.join(out, "ctrl.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a), fh, indent=1)
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_TAGS", "KVMEND_FENCE", "KVMEND_CHECK_AFTER_PREFILL",
              "KVMEND_STRICT_EVERY", "KVMEND_ON_FAIL", "KVMEND_ATTN_RELOAD", "KVMEND_ATTN_LOG_MAX",
              "KVMEND_ATTN_MAX_REPAIRS", "KVMEND_LOG_READS", "KVMEND_LOAD_CHECK", "KVMEND_STRIPE"):
        os.environ.pop(k, None)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out, KVMEND_CTRL=ctrl_path,
                      KVMEND_FUSED="1", KVMEND_CHECK_IN_ATTN="1", KVMEND_ATTN_ACT="repair",
                      KVMEND_STRICT=",".join(f"{rid}:{a.strict_every}" for rid in ("strict_repair", "strict_twin")),
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))

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

    def go(rid, act="repair", n=None):
        with open(ctrl_path, "w") as fh:  # what a failed check on read does in this run, read at the flush
            json.dump(dict(attn_act=act, label=rid), fh)
        flush(engine)
        res[rid] = run(engine, prompt, rid, n or a.max_new, a.variant_timeout)
        print(f"{rid:14s} ({act}) {len(res[rid]['ids'])} tokens in {res[rid]['seconds']:.2f} s "
              f"({res[rid]['finish']})", flush=True)

    try:
        go("warmup", n=8)
        for rid in RUNS:
            go(rid, "log" if rid == "late_silent" else "repair")
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            flush(engine)  # every rank writes its buffered log lines
        except Exception as e:
            print("final flush failed:", repr(e), flush=True)
        faulthandler.cancel_dump_traceback_later()
        engine.shutdown()
    with open(os.path.join(out, "outputs.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    s = summarize(out, a, res, meta)
    print(json.dumps(s["verdict"], indent=1))
    print(report_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
