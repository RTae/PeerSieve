#!/usr/bin/env python3
"""When parity pays (step 3 of decision 20): a burst of damage on one GPU, rebuilt from parity or recomputed.

Request R (a fixed prompt, greedy) runs on an empty cache. At output 3, a failing memory region on GPU --rank
changes one bit in every token of a run of pages, in one or more layers (pieces = pages x layers). The reader's
check finds it at once, and mend repairs it either way:
  rebuild_PxL     parity: each damaged piece from the other GPUs' pieces and the parity (bit for bit)
  recompute_PxL   no parity (tags only): the damaged pages are recomputed in place, every layer, from R's tokens
Each pair gives the work (bytes read, or tokens recomputed) and time of the repair, and whether the final text is the
fault-free text. The crossover tells at what burst size parity starts to pay for its cost.

  uv run python run_burst.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --sizes 1x1,8x1,64x1 --out runs/b
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
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=32)
    p.add_argument("--sizes", default="1x1,8x1,64x1,64x8", help="pages x layers of each burst")
    p.add_argument("--pos", type=int, default=16, help="first token of the burst (a page boundary)")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def sizes(a):
    out = []
    for s in a.sizes.split(","):
        pages, layers = (int(x) for x in s.split("x"))
        if a.pos + pages * 16 > a.prompt_len:
            raise SystemExit(f"a burst of {pages} pages from position {a.pos} does not fit a {a.prompt_len}-token prompt")
        out.append((pages, layers))
    return out


def make_plan(a):
    plans = {"warmup": [dict(op="encode", at=1)], "clean": [dict(op="encode", at=1), dict(op="check", at=1)]}
    for pages, layers in sizes(a):
        fault = dict(op="fault", at=3, kind="burst", pos=a.pos, pages=pages, layers=layers, layer=a.layer, rank=a.rank)
        for how, parity in (("rebuild", True), ("recompute", False)):
            plans[f"{how}_{pages}x{layers}"] = [dict(op="encode", at=1), dict(op="check", at=1), fault,
                                                 dict(op="check", at=3), dict(op="mend", at=3, parity=parity)]
    return plans


def summarize(out, a, res, meta):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    clean = res["clean"]["ids"]
    rows = []
    for pages, layers in sizes(a):
        row = dict(pages=pages, layers=layers, pieces=pages * layers, tokens_in_pages=16 * pages)
        for how in ("rebuild", "recompute"):
            rid = f"{how}_{pages}x{layers}"
            mend = [x for x in logs if x.get("rid") == rid and x["op"] == "mend" and x["rank"] == a.rank]
            fault = [x for x in logs if x.get("rid") == rid and x["op"] == "fault" and x["rank"] == a.rank]
            m = mend[0] if mend else {}
            fixes = [r for r in m.get("rebuilt", []) if r.get("rebuilt")]
            row[how] = dict(
                pieces_damaged=fault[0].get("pieces") if fault else None,
                pieces_rebuilt=len(fixes), bytes_read=sum(r["bytes_read"] for r in fixes),
                rebuild_s=sum(r["seconds"] for r in fixes), bit_exact=bool(fixes) and all(r["bit_exact"] for r in fixes),
                tokens_recomputed=sum(s["tokens"] for s in m.get("recomputed", [])),
                recompute_s=sum(s["seconds"] for s in m.get("recomputed", [])),
                mend_s=m.get("mend_seconds"), still_failing=m.get("still_failing_here"),
                final_text_right=res[rid]["ids"] == clean, seconds=res[rid]["seconds"])
        rows.append(row)
    s = dict(meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, rank=a.rank, layer=a.layer, rows=rows,
             errors=[x for x in logs if x["op"] == "error"])
    s["verdict"] = {
        "no_errors": not s["errors"],
        "every_burst_rebuilt_bit_for_bit": all(r["rebuild"]["bit_exact"] and r["rebuild"]["pieces_rebuilt"] == r["pieces"]
                                               for r in rows),
        "every_burst_recomputed": all(r["recompute"]["tokens_recomputed"] == r["tokens_in_pages"] for r in rows),
        "rebuild_text_right": all(r["rebuild"]["final_text_right"] for r in rows),
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    return s


def burst_md(s):
    lines = ["| Burst (pages x layers) | Pieces | Rebuild: data read | Rebuild time | Recompute: tokens | Recompute time "
             "| Text right (rebuild, recompute) |", "|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        b, c = r["rebuild"], r["recompute"]
        lines.append(f"| {r['pages']} x {r['layers']} | {r['pieces']:,} | {b['bytes_read'] / 2**20:.2f} MiB | "
                     f"{1e3 * b['rebuild_s']:.1f} ms | {c['tokens_recomputed']:,} | {1e3 * c['recompute_s']:.1f} ms | "
                     f"{'yes' if b['final_text_right'] else 'no'}, {'yes' if c['final_text_right'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a), fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_STRICT"):
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
    try:
        for rid in make_plan(a):
            flush(engine)
            res[rid] = run(engine, prompt, rid, 8 if rid == "warmup" else a.max_new, a.variant_timeout)
            print(f"{rid:18s} {len(res[rid]['ids'])} tokens in {res[rid]['seconds']:.2f} s", flush=True)
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
    print(burst_md(s), flush=True)
    with open(os.path.join(out, "burst.md"), "w") as fh:
        fh.write(burst_md(s))
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
