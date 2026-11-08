#!/usr/bin/env python3
"""The fair test of KVMend (decision 18): every response must return the right text.

Request R (a fixed prompt, greedy, --max-new tokens) runs on an empty cache. At output 1 KVMend encodes parity and
scrubs once: R's last good check. At output 10 one bit of one stored K value of token --pos flips. At output 41 a
scrub finds it. Variants:

  clean            no fault: the right text
  sweep_e*_b*      the fault at element e, bit b, and nothing checks: is the text unchanged, fluent but different,
                   or garbage?
  strict_e*_b*     KVMend with strict release: rebuild the piece from parity, drop the suspect tokens, which were never
                   released, and generate them again
  retry_e*_b*      abort and retry, today's practice: abort R at detection, drop the cached KV, resubmit R
  control_strict   no fault, the same rollback as strict: the regenerate path on its own
  page_recompute   what KVMend without parity does instead of the rebuild: a 16-token extend of the damaged page

strict and retry run for the NaN fault of the earlier runs and for the first sweep fault that leaves the text fluent.
Every response is judged by one bar: is its final text the clean text, and at what cost in tokens and time?

  uv run python run_fair.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 4 --out runs/fair-smoke
"""
import argparse
import faulthandler
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--inject-at", type=int, default=10)
    p.add_argument("--detect-at", type=int, default=41)
    p.add_argument("--keep", type=int, default=1, help="output tokens computed before the last good check")
    p.add_argument("--pos", type=int, default=1030)
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--kv", default="k", choices=["k", "v"])
    p.add_argument("--elems", default="37,5,60", help="elements of the token's K on that rank to try")
    p.add_argument("--bits", default="14,10,6", help="bit positions to try")
    p.add_argument("--nan-site", default="37:14", help="elem:bit of the NaN fault of the earlier runs")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def sites(a):
    return [(int(e), int(b)) for e in a.elems.split(",") for b in a.bits.split(",")]


def tag(site):
    return f"e{site[0]}_b{site[1]}"


def make_plan(a):
    base = [dict(op="encode", at=1), dict(op="scrub", at=1)]
    detect = dict(op="scrub", at=a.detect_at)
    plans = {"warmup": [], "clean": list(base), "page_recompute": [],
             "control_strict": base + [dict(op="retract", at=a.detect_at, keep=a.keep)]}
    for e, b in sites(a):
        fault = dict(op="inject", at=a.inject_at, pos=a.pos, layer=a.layer, rank=a.rank, kv=a.kv, elem=e, bit=b)
        t = tag((e, b))
        plans[f"sweep_{t}"] = base + [fault]
        plans[f"strict_{t}"] = base + [fault, detect, dict(op="repair", at=a.detect_at),
                                       dict(op="retract", at=a.detect_at, keep=a.keep)]
        plans[f"retry_{t}"] = base + [fault, detect, dict(op="abort", at=a.detect_at)]
        plans[f"retry2_{t}"] = []
    return plans


def longest_run(ids):
    best = cur = 0
    prev = object()
    for t in ids:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


def classify(ids, clean):
    """unchanged, fluent but different, or garbage (token 0, which NaN logits select, or a long repeat)."""
    from run_poc import first_diff

    d = first_diff(ids, clean)
    if d is None:
        return "unchanged", None
    tail = ids[d:]
    if tail.count(0) >= 5 or longest_run(tail) >= 8:
        return "garbage", d
    return "fluent but different", d


def run(engine, ids, rid, max_new, timeout):
    from run_poc import generate, output_tokens

    faulthandler.dump_traceback_later(timeout, exit=True)
    t0 = time.perf_counter()
    r = generate(engine, ids, rid, max_new)
    dt = time.perf_counter() - t0
    faulthandler.cancel_dump_traceback_later()
    n = r["meta_info"].get("completion_tokens", max_new)
    out, _ = output_tokens(r, ids, n)
    finish = r["meta_info"].get("finish_reason") or {}
    return dict(ids=out, text=r["text"], seconds=dt, finish=finish.get("type") if isinstance(finish, dict) else finish)


def summarize(out, a, res, sweep, tested, meta):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]

    def pick(rid, op, rank=None):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op and (rank is None or x["rank"] == rank)]

    clean = res["clean"]
    sweep_rows = []
    for site in sites(a):
        t = tag(site)
        inj = pick(f"sweep_{t}", "inject", rank=a.rank)
        outcome, d = sweep[site]
        sweep_rows.append(dict(site=t, before=inj[0].get("value_before") if inj else None,
                               after=inj[0].get("value_after") if inj else None, outcome=outcome, first_diff=d,
                               text=res[f"sweep_{t}"]["text"][:160]))
    fair = []
    for site in tested:
        t = tag(site)
        st, rt = res[f"strict_{t}"], res[f"retry_{t}"]
        fixes = [r for x in pick(f"strict_{t}", "repair", rank=a.rank) for r in x["repairs"]]
        retract = pick(f"strict_{t}", "retract", rank=0)
        pre = [x for x in pick(f"strict_{t}", "extend", rank=0) if x.get("n_out", 0) > 1]
        fair.append(dict(
            site=t, silent_outcome=sweep[site][0],
            strict=dict(final_text_right=st["ids"] == clean["ids"], first_diff=classify(st["ids"], clean["ids"])[1],
                        equals_control_strict=st["ids"] == res["control_strict"]["ids"],
                        rebuild_bit_exact=bool(fixes) and all(r.get("bit_exact") and r.get("tag_matches") for r in fixes),
                        bytes_read=sum(r.get("bytes_read", 0) for r in fixes if r.get("rebuilt")),
                        reprefilled=sum(x["extend_input_len"] for x in pre),
                        regenerated=retract[0].get("regenerated") if retract else None,
                        seconds=st["seconds"], extra_s=st["seconds"] - clean["seconds"]),
            retry=dict(first_finish=rt["first"]["finish"], first_tokens=len(rt["first"]["ids"]),
                       final_text_right=rt["ids"] == clean["ids"], reprefilled=a.prompt_len, regenerated=a.max_new,
                       seconds=rt["seconds"], extra_s=rt["seconds"] - clean["seconds"])))
    page = pick("page_recompute", "extend", rank=0)
    summary = dict(
        meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new, inject_at=a.inject_at,
        detect_at=a.detect_at, keep=a.keep, clean_seconds=clean["seconds"],
        control_strict_equals_clean=res["control_strict"]["ids"] == clean["ids"],
        sweep=sweep_rows, fluent_site=next((tag(s) for s in tested[1:]), None), fair=fair,
        page_recompute=dict(extend_tokens=page[0]["extend_input_len"] if page else None,
                            seconds=res["page_recompute"]["seconds"]),
        errors=[x for x in logs if x["op"] == "error"])
    summary["verdict"] = {
        "no_errors": not summary["errors"],
        "sweep_ran": len(sweep_rows) == len(sites(a)),
        "retry_first_attempt_aborted": all(f["retry"]["first_finish"] == "abort" for f in fair),
        "retry_final_text_is_clean": all(f["retry"]["final_text_right"] for f in fair),
        "strict_rebuild_bit_exact": all(f["strict"]["rebuild_bit_exact"] for f in fair),
        "strict_regenerated_the_suspect_tokens": all(f["strict"]["regenerated"] == a.detect_at - a.keep for f in fair),
        "strict_equals_control_strict": all(f["strict"]["equals_control_strict"] for f in fair),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "fair.md"), "w") as fh:
        fh.write(fair_md(summary))
    return summary


def fair_md(s):
    lines = ["| Fault (element, bit) | No checks | Response | Final text right | Recomputed | Time | Extra over fault-free |",
             "|---|---|---|---|---|---|---|"]
    for f in s["fair"]:
        st, rt = f["strict"], f["retry"]
        lines.append(f"| {f['site']} | {f['silent_outcome']} | abort and retry | {'yes' if rt['final_text_right'] else 'no'} | "
                     f"{rt['reprefilled']:,} prefilled + {rt['regenerated']} generated | {rt['seconds']:.2f} s | "
                     f"{rt['extra_s']:+.2f} s |")
        lines.append(f"| {f['site']} | {f['silent_outcome']} | KVMend, strict release | "
                     f"{'yes' if st['final_text_right'] else 'no (first differs at ' + str(st['first_diff']) + ')'} | "
                     f"{st['reprefilled']} re-prefilled + {st['regenerated']} regenerated; reads {st['bytes_read'] // 1024} KiB | "
                     f"{st['seconds']:.2f} s | {st['extra_s']:+.2f} s |")
    lines += ["", "| Sweep site | Value before | after | Outcome with no checks | First differing token |", "|---|---|---|---|---|"]
    for r in s["sweep"]:
        lines.append(f"| {r['site']} | {r['before']} | {r['after']} | {r['outcome']} | {r['first_diff']} |")
    lines.append(f"\nFault-free run: {s['clean_seconds']:.2f} s. Regenerate path alone equals the fault-free text: "
                 f"{s['control_strict_equals_clean']}. Page recompute (KVMend without parity): "
                 f"{s['page_recompute']['extend_tokens']} tokens in {s['page_recompute']['seconds']:.3f} s.")
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a), fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS"):
        os.environ.pop(k, None)

    import sglang as sgl
    from transformers import AutoTokenizer

    from kvmend.meta import run_meta
    from run_poc import build_prompt, engine_args, flush, start_loop

    meta = run_meta()
    prompt = build_prompt(AutoTokenizer.from_pretrained(a.model), a.prompt_len)
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    res, sweep, tested = {}, {}, []

    def go(rid, ids=None, n=None):
        flush(engine)
        res[rid] = run(engine, prompt if ids is None else ids, rid, n or a.max_new, a.variant_timeout)
        print(f"{rid:16s} {len(res[rid]['ids'])} tokens in {res[rid]['seconds']:.2f} s ({res[rid]['finish']})", flush=True)
        return res[rid]

    try:
        go("warmup", n=8)
        clean = go("clean")
        for site in sites(a):
            sweep[site] = classify(go(f"sweep_{tag(site)}")["ids"], clean["ids"])
            print(f"{'':16s} outcome: {sweep[site][0]} (first differs at {sweep[site][1]})", flush=True)
        nan_site = tuple(int(x) for x in a.nan_site.split(":"))
        assert nan_site in sweep, "--nan-site must be one of the swept sites"
        fluent = next((s for s in sites(a) if sweep[s][0] == "fluent but different"), None)
        tested = [nan_site] + ([fluent] if fluent and fluent != nan_site else [])
        for site in tested:
            t = tag(site)
            go(f"strict_{t}")
            flush(engine)
            t0 = time.perf_counter()
            first = run(engine, prompt, f"retry_{t}", a.max_new, a.variant_timeout)
            flush(engine)  # the damaged cached KV must go, or the retry would read it again
            second = run(engine, prompt, f"retry2_{t}", a.max_new, a.variant_timeout)
            res[f"retry_{t}"] = dict(first=first, second=second, ids=second["ids"], text=second["text"],
                                     seconds=time.perf_counter() - t0)
            print(f"retry_{t:10s} first attempt {len(first['ids'])} tokens ({first['finish']}); "
                  f"retry {len(second['ids'])} tokens; {res[f'retry_{t}']['seconds']:.2f} s in all", flush=True)
        go("control_strict")
        # KVMend without parity recomputes the damaged page: a 16-token extend after the pages before it
        page0 = (a.pos // 16) * 16
        go("page_prefix", ids=prompt[:page0], n=1)
        res["page_recompute"] = run(engine, prompt[:page0 + 16], "page_recompute", 1, a.variant_timeout)
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
        json.dump({k: v for k, v in res.items()}, fh, indent=1, default=str)
    s = summarize(out, a, res, sweep, tested, meta)
    print(json.dumps(s["verdict"], indent=1))
    print(fair_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
