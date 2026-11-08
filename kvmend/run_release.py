#!/usr/bin/env python3
"""The release rule (step 2 of decision 20): what a streaming client receives when the KV goes bad mid-request.

Request R (a fixed prompt, greedy, streamed token by token) runs on an empty cache. At output --late-at, damage of
kind --kind strikes the last --pages pages of the prompt, on every GPU (the kinds of run_incident.py): by default a late
write around the store path at output 10. A model whose text that does not change (Llama-3.1-8B) needs earlier damage,
such as --kind unwritten --late-at 2: the pages hold bytes that were never written for R. Damage at output 2 strikes
after the check that follows the prefill, so each strict run keeps that check's token and its control replays it.

  fast_clean     fast release, no fault: the reference token stream and its timing
  strict_clean   strict release, a check before every token leaves, no fault: what holding tokens back costs
  fast_fault     fast release, the damage, no checks: tokens computed from the damaged KV reach the client
  strict_fault   strict release with a check every --every tokens: the damage is found at the next check, the pages
                 are recomputed in place, and the tokens chosen since the last pass, never sent, are generated again
  strict_fault1  strict release with a check before every token: the damage is found before any token reads it
  control_fault, control_fault1   no damage, but the repair path of strict_fault and strict_fault1: the same pages
                 recomputed at the step its check finds the damage, and the same rollback. A recompute is not bit for
                 bit, so a strict run's text is right when it equals the clean text or its control's (a near-tie can
                 flip, as on TinyLlama at output 73).

Each streamed run records the arrival time and tokens of every chunk. A run is clean for the client when every token
it received is in its final text and the final text is the fault-free text.

  uv run python run_release.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --out runs/release-check
"""
import argparse
import asyncio
import faulthandler
import glob
import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--late-at", type=int, default=10, help="output count at which the damage strikes")
    p.add_argument("--kind", default="late_bypass", choices=("late_bypass", "unwritten", "foreign"),
                   help="the kind of damage (run_incident.py)")
    p.add_argument("--every", type=int, default=8, help="tokens between checks for strict_fault")
    p.add_argument("--pages", type=int, default=4)
    p.add_argument("--donor", type=int, default=512)
    p.add_argument("--fused", type=int, default=1, help="1: the fused Triton tag kernels")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def found_at(a, every):
    """The output count of the first check at or after the damage, and the tokens kept: those of the last pass. The
    release rule checks right after the prefill (output 1) and every `every` outputs; at an output where the damage
    strikes, the damage comes first, so that output's check finds it."""
    def checked(n):
        return n == 1 or n % every == 0

    at = next(n for n in range(a.late_at, a.late_at + every + 1) if checked(n))
    keep = max((n for n in range(1, a.late_at) if checked(n)), default=0)
    return at, keep


def make_plan(a):
    fault = dict(op="fault", at=a.late_at, kind=a.kind, pages=a.pages, donor=a.donor)
    base = [dict(op="encode", at=1)]
    span = dict(pages=a.pages, donor=a.donor)

    def control(every):
        # The check that finds the damage mends at once. Its rollback waits one step when the request has only just
        # finished its prefill (output 1), as the release rule's pending redo does.
        at, keep = found_at(a, every)
        return base + [dict(op="mend", at=at, force=True, **span),
                       dict(op="retract", at=at if at > 1 else 2, keep=keep)]

    return {"warmup": base, "fast_clean": base, "strict_clean": base, "fast_fault": base + [fault],
            "strict_fault": base + [fault], "strict_fault1": base + [fault],
            "control_fault": control(a.every), "control_fault1": control(1)}


def stream(engine, ids, rid, max_new):
    """Stream one request; return its chunks as (seconds since submission, new token ids, finish reason)."""
    from sglang.srt.managers.io_struct import GenerateReqInput

    obj = GenerateReqInput(input_ids=ids, rid=rid, stream=True,
                           sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))

    async def collect():
        out, t0 = [], time.perf_counter()
        async for chunk in engine.tokenizer_manager.generate_request(obj, None):
            fin = chunk["meta_info"].get("finish_reason")
            out.append((time.perf_counter() - t0, list(chunk["output_ids"]),
                        fin.get("type") if isinstance(fin, dict) else fin, chunk["meta_info"].get("completion_tokens")))
        return out

    return asyncio.get_event_loop().run_until_complete(collect())


def received(chunks, prompt):
    """Every token the client received, in order, without SGLang's leading prompt tokens."""
    from run_poc import output_tokens

    ids = [t for _, new, _, _ in chunks for t in new]
    n = chunks[-1][3] if chunks and chunks[-1][3] is not None else len(ids)
    return output_tokens(dict(output_ids=ids), prompt, n)[0]


def summarize(out, a, res, meta):
    from run_poc import first_diff

    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    clean = res["fast_clean"]["ids"]
    control_of = {"strict_fault": "control_fault", "strict_fault1": "control_fault1"}
    rows = {}
    for rid, r in res.items():
        if rid == "warmup":
            continue
        refs = [clean] + ([res[control_of[rid]]["ids"]] if control_of.get(rid) in res else [])
        full = [c for c in r["chunks"] if c[1]]  # chunks that carry tokens (a held-back step sends an empty one)
        gaps = [b[0] - a_[0] for a_, b in zip(full, full[1:])]
        single = [x for x in logs if x.get("rid") == rid and x["op"] == "release_check" and x["rank"] == 0]
        batched = [x for x in logs if x["op"] == "release_checks" and x["rank"] == 0 and rid in x.get("passed", [])]
        checks = single + batched  # passing checks are logged per step, for every request that was due
        found = [x for x in single if x.get("found_pieces")]
        wrong = [0 if d is None else len(r["ids"]) - d for d in (first_diff(r["ids"], ref) for ref in refs)]
        rows[rid] = dict(
            final_text_right=r["ids"] == clean, received_tokens=len(r["ids"]),
            equals_control=(r["ids"] == refs[1]) if len(refs) > 1 else None,
            wrong_tokens_received=min(wrong),  # against the clean text, or the run's same-path control
            first_token_s=full[0][0] if full else None,
            median_gap_s=statistics.median(gaps) if gaps else None, total_s=r["seconds"],
            checks=len(checks), check_median_s=statistics.median(x["seconds"] for x in checks) if checks else None,
            found_at=[x["n_out"] for x in found],
            regenerated=[x.get("regenerate", {}).get("regenerated") for x in found],
            recomputed=[sum(s["tokens"] for s in x.get("mend", {}).get("recomputed", [])) for x in found],
            finish=r["chunks"][-1][2] if r["chunks"] else None)
    s = dict(meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new, late_at=a.late_at,
             kind=a.kind, every=a.every, pages=a.pages, fused=a.fused, rows=rows,
             errors=[x for x in logs if x["op"] == "error"])
    s["verdict"] = {
        "no_errors": not s["errors"],
        "strict_clean_is_clean": rows["strict_clean"]["final_text_right"],
        "fast_fault_sends_wrong_tokens": rows["fast_fault"]["wrong_tokens_received"] > 0,
        "strict_fault_sends_no_wrong_token": rows["strict_fault"]["wrong_tokens_received"] == 0,
        "strict_fault_final_text_right": rows["strict_fault"]["final_text_right"] or rows["strict_fault"]["equals_control"],
        "strict_fault_found_at_its_next_check": rows["strict_fault"]["found_at"][:1] == [found_at(a, a.every)[0]],
        "strict_fault1_sends_no_wrong_token": rows["strict_fault1"]["wrong_tokens_received"] == 0,
        "strict_fault1_final_text_right": rows["strict_fault1"]["final_text_right"] or rows["strict_fault1"]["equals_control"],
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    return s


def release_md(s):
    lines = ["| Run | Final text right | Equals its control | Tokens received | Wrong tokens received | First token "
             "| Median gap | Checks (median time) | Found at output | Regenerated |", "|---|---|---|---|---|---|---|---|---|---|"]
    for rid, r in s["rows"].items():
        chk = f"{r['checks']} ({1e3 * r['check_median_s']:.1f} ms)" if r["checks"] else "none"
        gap = f"{1e3 * r['median_gap_s']:.1f} ms" if r["median_gap_s"] is not None else ""
        ctl = "" if r.get("equals_control") is None else ("yes" if r["equals_control"] else "no")
        lines.append(f"| {rid} | {'yes' if r['final_text_right'] else 'no'} | {ctl} | {r['received_tokens']} | "
                     f"{r['wrong_tokens_received']} | {r['first_token_s']:.2f} s | {gap} | {chk} | "
                     f"{r['found_at'] or ''} | {r['regenerated'] or ''} |")
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a), fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out, KVMEND_FUSED=str(a.fused),
                      KVMEND_STRICT=(f"warmup:1,strict_clean:1,strict_fault:{a.every},strict_fault1:1,"
                                     f"control_fault:{a.every},control_fault1:1"),
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_FENCE",
              "KVMEND_STRICT_EVERY"):
        os.environ.pop(k, None)

    import sglang as sgl
    from transformers import AutoTokenizer

    from kvmend.meta import run_meta
    from run_poc import build_prompt, engine_args, flush, start_loop

    meta = run_meta()
    prompt = build_prompt(AutoTokenizer.from_pretrained(a.model), a.prompt_len)
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    res = {}
    try:
        for rid in ("warmup", "fast_clean", "strict_clean", "fast_fault", "strict_fault", "strict_fault1",
                    "control_fault", "control_fault1"):
            flush(engine)
            faulthandler.dump_traceback_later(a.variant_timeout, exit=True)
            t0 = time.perf_counter()
            chunks = stream(engine, prompt, rid, 8 if rid == "warmup" else a.max_new)
            faulthandler.cancel_dump_traceback_later()
            res[rid] = dict(chunks=chunks, ids=received(chunks, prompt), seconds=time.perf_counter() - t0)
            print(f"{rid:14s} {len(res[rid]['ids'])} tokens in {len(chunks)} chunks, {res[rid]['seconds']:.2f} s",
                  flush=True)
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
    print(release_md(s), flush=True)
    with open(os.path.join(out, "release.md"), "w") as fh:
        fh.write(release_md(s))
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
