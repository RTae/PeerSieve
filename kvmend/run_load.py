#!/usr/bin/env python3
"""Heavier tests (step 5 of decision 20): many requests at once, shared prefixes, several faults per run, and
detection by the release rule's own checks rather than at a time the test chooses.

The workload: --groups groups of --per-group requests, sent at once. The requests of a group share a --shared-token
prefix (page aligned, so the radix cache shares its pages) and each adds its own question. Every request is greedy,
with --max-new output tokens. Parity is encoded every 16 batch steps in the background.

Four faults strike during the run, each at an output count of one request:
  shared   g0_r0 at output 10: a late write around the store path into the last 4 pages of group 0's shared prefix,
           on every GPU; all of group 0 read those pages
  foreign  g1_r2 at output 20: the first page of its own question holds another sequence's KV with matching tags
  burst    g2_r1 at output 15: one GPU's memory region fails: 8 prefix pages, 2 layers, on GPU --rank (parity rebuilds)
  fenced   g3_r0 at output 12: a late write through the store path with an old generation (the fence drops it)

Runs, all in one engine, the cache flushed between them; before each, one request per group with a single output
token puts the group's prefix in the radix cache, so the run's requests share its pages:
  a_*, b_*   no fault, strict release: two fault-free replays, to measure how much outputs vary under load
  c_*        no fault, fast release: the throughput that strict release is compared with
  d_*, e_*   with --clean-runs 5: two more fault-free replays (d strict, e fast), run last
  f_*        the four faults, strict release (a check every --every tokens): every fault is found by a reader's
             check, mended, and every reader that may have read it regenerates its unreleased tokens
  x_*        the same faults with no checks (fast release): which outputs go wrong silently
Batching makes some requests' outputs vary between fault-free runs (many, on Llama-3.1-8B at TP=8). The verdict judges
strict release on the stable requests, whose fault-free runs all agree: each must give that text. A varying request
is right when it equals one of its fault-free runs, which is reported but is no verdict: it cannot tell damage from
batching.

  uv run python run_load.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --out runs/load-check
"""
import argparse
import asyncio
import faulthandler
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CLEAN = ("a", "b", "c", "d", "e")  # the fault-free runs (f and x carry the faults)


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--per-group", type=int, default=6)
    p.add_argument("--shared", type=int, default=1024, help="shared prefix tokens (a multiple of 16)")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--every", type=int, default=4, help="output tokens between a strict request's checks")
    p.add_argument("--rank", type=int, default=1, help="the GPU of the burst")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--clean-runs", type=int, default=3, choices=(3, 5), help="fault-free runs: a, b, c (and d, e)")
    p.add_argument("--run-timeout", type=int, default=900)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def names(a):
    return [f"g{g}_r{r}" for g in range(a.groups) for r in range(a.per_group)]


def prompts(tok, a):
    """Each group's shared prefix is a different stretch of the server log; each request asks its own question."""
    assert a.shared % 16 == 0
    lines = [f"Event {i}: request {(i * 37) % 1009} on GPU {i % 8} finished in {10 + (i * 53) % 990} ms."
             for i in range(3000 + a.groups * a.shared // 10)]
    ids = tok.encode("\n".join(lines), add_special_tokens=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    out = {}
    for g in range(a.groups):
        head = bos + ids[g * a.shared:(g + 1) * a.shared - len(bos)]
        for r in range(a.per_group):
            k = g * a.per_group + r
            q = tok.encode(f"\n\nQuestion {k}: which events on GPU {k % 8} took longer than {300 + 61 * k} ms? "
                           f"Answer briefly.\nAnswer:", add_special_tokens=False)
            out[f"g{g}_r{r}"] = head + q
    return out


def make_plan(a, prompts_):
    span_end = a.shared
    faults = {
        "g0_r0": dict(op="fault", at=10, kind="late_bypass", pos=span_end - 64, pages=4, donor=0),
        "g1_r2": dict(op="fault", at=20, kind="foreign", pos=span_end, pages=1, donor=0),
        "g2_r1": dict(op="fault", at=15, kind="burst", pos=16, pages=8, layers=2, layer=1, rank=a.rank),
        "g3_r0": dict(op="fault", at=12, kind="late_store", pos=span_end - 32, pages=2, donor=0),
    }
    plan = {}
    for run in ("f", "x"):
        for name, f in faults.items():
            f = dict(f)
            if run == "x" and f["kind"] == "late_store":
                f["fence"] = False  # plain SGLang has no fence: the late write lands
            plan[f"{run}_{name}"] = [f]
    return plan


async def _many(engine, items, max_new):
    from sglang.srt.managers.io_struct import GenerateReqInput

    async def one(rid, ids):
        obj = GenerateReqInput(input_ids=ids, rid=rid,
                               sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))
        out = await engine.tokenizer_manager.generate_request(obj, None).__anext__()
        return rid, out

    return await asyncio.gather(*[one(rid, ids) for rid, ids in items])


def warm_prefixes(engine, run, prompts_, a):
    """One request per group, one output token: the group's shared prefix stays in the radix cache."""
    leaders = [(f"{run}w_g{g}", prompts_[f"g{g}_r0"][:a.shared]) for g in range(a.groups)]
    asyncio.get_event_loop().run_until_complete(_many(engine, leaders, 1))


def run_batch(engine, run, prompts_, a):
    from run_poc import output_tokens

    items = [(f"{run}_{name}", ids) for name, ids in prompts_.items()]
    t0 = time.perf_counter()
    outs = asyncio.get_event_loop().run_until_complete(_many(engine, items, a.max_new))
    dt = time.perf_counter() - t0
    res = {}
    for rid, out in outs:
        name = rid.split("_", 1)[1]
        n = out["meta_info"].get("completion_tokens", a.max_new)
        fin = out["meta_info"].get("finish_reason") or {}
        res[name] = dict(ids=output_tokens(out, prompts_[name], n)[0], text=out["text"][:160],
                         finish=fin.get("type") if isinstance(fin, dict) else fin)
    return dict(outputs=res, seconds=dt, tokens=sum(len(r["ids"]) for r in res.values()))


def summarize(out, a, runs, meta):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    every = names(a)
    clean = [k for k in CLEAN if k in runs]
    F, X = runs["f"]["outputs"], runs["x"]["outputs"]
    ref = {n: [runs[k]["outputs"][n]["ids"] for k in clean] for n in every}
    vary = [n for n in every if any(r != ref[n][0] for r in ref[n])]
    stable = [n for n in every if n not in vary]
    right_f = [n for n in every if F[n]["ids"] in ref[n]]
    wrong_x = [n for n in every if X[n]["ids"] not in ref[n]]
    stable_wrong_f = [n for n in stable if F[n]["ids"] != ref[n][0]]
    stable_wrong_x = [n for n in stable if X[n]["ids"] != ref[n][0]]
    checks = [x for x in logs if x["op"] == "release_check" and x["rank"] == 0 and str(x.get("rid", "")).startswith("f_")]
    found = [x for x in checks if x.get("found_pieces")]
    faults = [x for x in logs if x["op"] == "fault" and x["rank"] == 0 and x["rid"].startswith("f_")]
    found_pages = set()
    for x in found:
        found_pages |= {int(p) for p in x["found"]}
    fault_rows = []
    for x in faults:
        hit = sorted(set(x["pages"]) & found_pages)
        fault_rows.append(dict(rid=x["rid"], kind=x["kind"], pages=x["pages"], fenced=x.get("fenced", False),
                               written=x.get("written_tokens"), found_pages=hit))
    s = dict(meta=meta, model=a.model, tp=a.tp, groups=a.groups, per_group=a.per_group, shared=a.shared,
             max_new=a.max_new, every=a.every,
             runs={k: dict(seconds=v["seconds"], tokens=v["tokens"], tok_per_s=v["tokens"] / v["seconds"])
                   for k, v in runs.items()},
             clean_runs=clean, fault_free_variation=vary, stable=len(stable), stable_strict_wrong=stable_wrong_f,
             stable_fast_wrong=stable_wrong_x, strict_right=len(right_f),
             strict_wrong=sorted(set(every) - set(right_f)), fast_wrong=wrong_x, faults=fault_rows,
             detections=[dict(rid=x["rid"], n_out=x["n_out"], found=x["found"],
                              regenerated=x.get("regenerate", {}).get("regenerated"),
                              other_readers=[o["rid"] for o in x.get("other_readers", [])],
                              evicted=x.get("evicted_tokens"),
                              recomputed=sum(r["tokens"] for r in x.get("mend", {}).get("recomputed", [])),
                              rebuilt=sum(1 for r in x.get("mend", {}).get("rebuilt", []) if r.get("rebuilt")))
                         for x in found],
             checks=len(checks) + sum(len(x.get("passed", [])) for x in logs if x["op"] == "release_checks"
                                     and x["rank"] == 0),
             check_seconds_per_step=sorted(x["seconds"] for x in logs if x["op"] == "release_checks"
                                           and x["rank"] == 0)[len([1 for x in logs if x["op"] == "release_checks"
                                                                    and x["rank"] == 0]) // 2:][:1],
             errors=[x for x in logs if x["op"] == "error"])
    s["verdict"] = {
        "no_errors": not s["errors"],
        "every_unfenced_fault_found": all(r["found_pages"] for r in fault_rows if not r["fenced"]),
        "fenced_write_dropped": all(r["written"] == 0 for r in fault_rows if r["fenced"]),
        "strict_right_on_stable_requests": not stable_wrong_f,
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    return s


def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    P = prompts(tok, a)
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a, P), fh, indent=1)
    runs_order = ["a", "b", "c", "f", "x"] + (["d", "e"] if a.clean_runs == 5 else [])
    strict = ",".join(f"{run}_{n}:{a.every}" for run in ("a", "b", "d", "f") if run in runs_order for n in names(a))
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out, KVMEND_FUSED="1",
                      KVMEND_STRICT=strict, KVMEND_ENCODE_EVERY="16",
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_FENCE", "KVMEND_STRICT_EVERY"):
        os.environ.pop(k, None)

    import sglang as sgl

    from kvmend.meta import run_meta
    from run_poc import engine_args, flush, start_loop

    meta = run_meta()
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    runs = {}
    try:
        flush(engine)
        run_batch(engine, "w", dict(list(P.items())[:4]), a)  # warm-up, not counted
        for run in runs_order:
            flush(engine)
            faulthandler.dump_traceback_later(a.run_timeout, exit=True)
            warm_prefixes(engine, run, P, a)
            runs[run] = run_batch(engine, run, P, a)
            faulthandler.cancel_dump_traceback_later()
            print(f"run {run}: {runs[run]['tokens']} tokens in {runs[run]['seconds']:.2f} s", flush=True)
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
        json.dump(runs, fh, indent=1, default=str)
    s = summarize(out, a, runs, meta)
    print(json.dumps(s["verdict"], indent=1))
    print(json.dumps({k: s[k] for k in ("clean_runs", "fault_free_variation", "stable", "stable_strict_wrong",
                                        "stable_fast_wrong", "strict_right", "strict_wrong", "fast_wrong", "faults",
                                        "detections", "runs")}, indent=1, default=str), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
