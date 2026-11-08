#!/usr/bin/env python3
"""Fault-free cost of KVMend on SGLang 0.5.1, against GhostServe-style parity alone and plain SGLang.

One engine per mode, each in its own process, because the tag update is captured into SGLang's decode CUDA graphs
when the engine starts:
  plain   SGLang without the patch
  parity  GhostServe-style parity alone: sealed pages are encoded every --encode-every batch steps; no tags
  kvmend  running tags in every KV write, parity with a tag check before each encode, and a background scrub of
          --scrub pieces per batch step (one segment per setting; 0 means no scrub)
  fused   kvmend with the fused Triton tag kernels (kvmend/tag_kernels.py)
  tags    running tags and nothing else: no parity, no scrub (the watch alone, as KVMend without parity)
  tags_fused  tags with the fused kernels
  attn    tags_fused plus checks on read (KVMEND_CHECK_IN_ATTN=1): the decode attention kernel adds up the tag of
          every piece it reads, and every decode step ends with a comparison against the stored tags
  attn_reload  attn, with the kernel reading the words again instead of reusing the attention's tiles
Each run sends --requests requests of --prompt-len distinct tokens at once on an empty cache, greedy, with exactly
--max-new output tokens each. A smaller first run warms up and is not counted. Every mode uses run_poc.py's engine
settings. Writes <out>/<mode>.json; --report merges the modes into overhead.json and overhead.md.

The attn modes check themselves: no check may fail in the fault-free runs, the check must have read pieces, and a
probe after the timed runs (one request; at output 4, one bit of one K value of its prompt flips on the last rank)
must be found at the next decode step, naming exactly that rank, layer and page. Otherwise the mode exits 3.

  uv run python bench_overhead.py --model M --tp 8 --mode plain --out runs/x/overhead
  uv run python bench_overhead.py --report --out runs/x/overhead
"""
import argparse
import faulthandler
import glob
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODES = ("plain", "parity", "kvmend", "fused", "tags", "tags_fused", "attn", "attn_reload")
TAGGED = ("kvmend", "fused", "tags", "tags_fused", "attn", "attn_reload")  # modes with running tags
FUSED = ("fused", "tags_fused", "attn", "attn_reload")  # modes with the Triton tag kernels
ATTN = ("attn", "attn_reload")  # modes with checks on read inside the decode attention
H7 = 0.03  # the proposal's target: fault-free throughput within 3% of parity alone
# The attn modes' probe: request PROBE_RID gets one flipped bit, which the check on read must find at the next step
PROBE_RID, PROBE_AT, PROBE_NEW, PROBE_LAYER = "attn-probe", 4, 12, 1
PROBE_FAULT = dict(kv="k", elem=5, bit=3)  # a low mantissa bit: any change in a word changes its piece's tag


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--encode-every", type=int, default=16, help="batch steps between parity encodes")
    p.add_argument("--scrub", default="0,1024,4096", help="kvmend: pieces checked per batch step, one segment each")
    p.add_argument("--max-total-tokens", type=int, default=98304)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192, help="chunked prefill size")
    p.add_argument("--run-timeout", type=int, default=600, help="seconds before a stuck run ends the job")
    p.add_argument("--report", action="store_true", help="merge <out>/<mode>.json into overhead.json and .md")
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_prompts(tok, n, length):
    """n prompts of exactly `length` tokens from a plain log, each from its own offset, so no two share a page."""
    lines = [f"Event {i}: request {(i * 37) % 1009} on GPU {i % 8} finished in {10 + (i * 53) % 990} ms."
             for i in range(4000 + n * length // 20)]
    ids = tok.encode("\n".join(lines), add_special_tokens=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body = length - len(bos)
    step = (len(ids) - body) // n
    assert step >= 16, "prompt source too short"
    return [bos + ids[i * step:i * step + body] for i in range(n)]


def run(engine, prompts, max_new):
    """Seconds, output tokens, and a digest of every request's output token ids (greedy decoding: modes that leave
    the model's arithmetic alone give the same digest)."""
    t0 = time.perf_counter()
    res = engine.generate(input_ids=prompts,
                          sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))
    dt = time.perf_counter() - t0
    digest = hashlib.sha256(json.dumps([list(r.get("output_ids", [])) for r in res]).encode()).hexdigest()[:16]
    return dt, sum(r["meta_info"].get("completion_tokens", max_new) for r in res), digest


def segments(a):
    if a.mode in ("kvmend", "fused"):
        return [dict(label=f"scrub{s}", encode_every=a.encode_every, scrub_pieces=int(s)) for s in a.scrub.split(",")]
    if a.mode in ("tags", "tags_fused"):
        return [dict(label="tags", encode_every=0, scrub_pieces=0)]
    if a.mode in ATTN:
        return [dict(label="attn", encode_every=0, scrub_pieces=0)]
    if a.mode == "parity":
        return [dict(label="parity", encode_every=a.encode_every, scrub_pieces=0)]
    return [dict(label="plain", encode_every=0, scrub_pieces=0)]


def periodic_stats(log_dir):
    """Per segment, the background work of each rank, averaged over the segment's runs."""
    recs = []
    for path in glob.glob(os.path.join(log_dir, "rank*.jsonl")):
        with open(path) as fh:
            recs += [x for x in map(json.loads, fh) if x["op"] == "periodic" and x["label"] != "warmup"]
    out = {}
    for label in sorted({x["label"] for x in recs}):
        rs = [x for x in recs if x["label"] == label]
        pairs = [(x, x["t_last"] - x["t_first"]) for x in rs if x["t_first"] is not None and x["t_last"] is not None]
        busy = [b for _, b in pairs]
        rates = [x["scrub_pieces"] * x["piece_bytes"] / b for x, b in pairs if b > 0]
        out[label] = dict(
            records=len(rs), runs_per_rank=len(rs) / max(1, len({x["rank"] for x in rs})),
            batch_steps=statistics.mean(x["batch_steps"] for x in rs),
            mean_step_s=statistics.mean(b / max(1, x["batch_steps"] - 1) for x, b in pairs) if pairs else None,
            encodes=statistics.mean(x["encodes"] for x in rs), pages_encoded=statistics.mean(x["pages_encoded"] for x in rs),
            encode_s=statistics.mean(x["encode_s"] for x in rs), unclean=sum(x["unclean"] for x in rs),
            scrub_pieces=statistics.mean(x["scrub_pieces"] for x in rs), alarms=sum(x["alarms"] for x in rs),
            scrub_bytes_per_s=statistics.mean(rates) if rates else 0.0,
            pool_bytes=rs[0]["pool_pages"] * rs[0]["layers"] * rs[0]["piece_bytes"])
    return out


def read_logs(log_dir):
    recs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "rank*.jsonl"))):
        with open(path) as fh:
            recs += [json.loads(line) for line in fh]
    return recs


def attn_summary(log_dir, probe_rank):
    """The attn modes' self-check, from the rank logs. Per segment: the check's decode steps, pieces read and
    failures (each flush logs one attn_check record per rank). The probe: the inject record on the probe's rank, and
    the first attn_found record after it, which must come at the next step and name exactly (rank, layer, page,
    "tag"). Any other attn_found record is a false alarm."""
    recs = read_logs(log_dir)
    by_label = {}
    for x in (r for r in recs if r["op"] == "attn_check" and r["label"] != "warmup"):
        d = by_label.setdefault(x["label"], dict(records=0, steps=0, pieces=0, failures=0, layout=0, seconds=0.0,
                                                 unsupported={}))
        for k in ("steps", "pieces", "failures", "layout", "seconds"):
            d[k] += x[k]
        d["records"] += 1
        for why, n in x.get("unsupported", {}).items():
            d["unsupported"][why] = d["unsupported"].get(why, 0) + n
    for d in by_label.values():
        d["pieces_per_step"] = d["pieces"] / d["steps"] if d["steps"] else 0.0
        d["verdict_ms_per_step"] = 1e3 * d["seconds"] / d["steps"] if d["steps"] else None
    found = [r for r in recs if r["op"] == "attn_found"]
    inj = [r for r in recs if r["op"] == "inject" and r.get("rid") == PROBE_RID and r["rank"] == probe_rank]
    probe = dict(injected=None, first_found=None, found_next_step=False, names_exactly_the_piece=False)
    window = []
    if inj:
        s = inj[0]
        probe["injected"] = dict(step=s["step"], rank=probe_rank, layer=s["layer"], page=s["page"], loc=s["loc"])
        window = [r for r in found if r["rank"] == probe_rank and r["step"] > s["step"]]
        if window:
            first = min(window, key=lambda r: r["step"])
            probe["first_found"] = dict(step=first["step"], found=first["found"], rids=first.get("rids"))
            probe["found_next_step"] = first["step"] == s["step"] + 1
            probe["names_exactly_the_piece"] = ([list(f) for f in first["found"]]
                                                == [[probe_rank, s["layer"], s["page"], "tag"]])
    false_alarms = [r for r in found if r not in window]
    timed = {k: v for k, v in by_label.items() if k != "probe"}
    checks = dict(
        no_failure_in_fault_free_runs=all(v["failures"] == 0 and v["layout"] == 0 for v in timed.values()),
        the_check_read_pieces=bool(timed) and all(v["pieces"] > 0 for v in timed.values()),
        no_false_alarm=not false_alarms,
        probe_found_at_the_next_step=probe["found_next_step"],
        probe_names_exactly_the_piece=probe["names_exactly_the_piece"],
        no_errors=not [r for r in recs if r["op"] == "error"])
    return dict(by_label=by_label, probe=probe, false_alarms=false_alarms[:20], checks=checks, ok=all(checks.values()))


def bench(a):
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log_dir = os.path.join(out, f"{a.mode}_logs")
    ctrl = os.path.join(out, f"{a.mode}_ctrl.json")
    for k in ("KVMEND_PLAN", "KVMEND_ENABLE", "KVMEND_TAGS", "KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES",
              "KVMEND_FUSED", "KVMEND_STRICT", "KVMEND_CHECK_IN_ATTN", "KVMEND_ATTN_RELOAD", "KVMEND_ATTN_ACT"):
        os.environ.pop(k, None)
    if a.mode != "plain":  # plain SGLang does not even load the hook
        os.environ.update(KVMEND_ENABLE="1", KVMEND_TAGS="1" if a.mode in TAGGED else "0", KVMEND_OUT=log_dir,
                          KVMEND_FUSED="1" if a.mode in FUSED else "0",
                          KVMEND_CTRL=ctrl, PYTHONPATH=os.pathsep.join(
                              p for p in (os.path.join(HERE, "hook_site"), HERE, os.environ.get("PYTHONPATH", "")) if p))
    probe_rank = a.tp - 1
    if a.mode in ATTN:
        # The probe's plan: the timed runs' requests have other ids, so they meet no action.
        plan_path = os.path.join(out, f"{a.mode}_plan.json")
        with open(plan_path, "w") as fh:
            json.dump({PROBE_RID: [dict(op="inject", at=PROBE_AT, pos=min(40, a.prompt_len - 1), layer=PROBE_LAYER,
                                        rank=probe_rank, **PROBE_FAULT)]}, fh, indent=1)
        os.environ.update(KVMEND_CHECK_IN_ATTN="1", KVMEND_ATTN_RELOAD="1" if a.mode == "attn_reload" else "0",
                          KVMEND_PLAN=plan_path)

    import sglang as sgl
    from transformers import AutoTokenizer

    from kvmend.meta import run_meta
    from run_poc import engine_args, flush, generate, start_loop

    meta = run_meta()
    prompts = make_prompts(AutoTokenizer.from_pretrained(a.model), a.requests, a.prompt_len)
    segs = segments(a)

    def set_ctrl(seg, label):
        with open(ctrl, "w") as fh:
            json.dump(dict(seg, label=label), fh)

    t_start = time.perf_counter()
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    print(f"{a.mode}: engine ready in {time.perf_counter() - t_start:.0f} s", flush=True)
    runs = []
    try:
        # Warm-up: compiles kernels and, with the patch, allocates the host parity at the first encode.
        faulthandler.dump_traceback_later(a.run_timeout, exit=True)
        set_ctrl(segs[0], "warmup")
        flush(engine)  # a flush applies the control file on every rank and empties the cache
        dt, _, _ = run(engine, prompts[:max(8, a.requests // 4)], 32)
        print(f"{a.mode}: warm-up {dt:.2f} s", flush=True)
        for seg in segs:
            set_ctrl(seg, seg["label"])
            for i in range(a.repeats):
                faulthandler.dump_traceback_later(a.run_timeout, exit=True)
                flush(engine)
                dt, n_out, digest = run(engine, prompts, a.max_new)
                n_in = len(prompts) * a.prompt_len
                runs.append(dict(segment=seg["label"], scrub_pieces=seg["scrub_pieces"], repeat=i, seconds=dt,
                                 output_tokens=n_out, out_tok_s=n_out / dt, total_tok_s=(n_in + n_out) / dt,
                                 output_digest=digest))
                print(f"{a.mode}: {seg['label']:9s} run {i}: {dt:.2f} s, {n_out / dt:,.0f} output tok/s, "
                      f"output digest {digest}", flush=True)
        if a.mode in ATTN:  # the probe, untimed, on an empty cache
            faulthandler.dump_traceback_later(a.run_timeout, exit=True)
            set_ctrl(segs[0], "probe")
            flush(engine)  # logs the last timed run's check statistics
            generate(engine, prompts[0], PROBE_RID, PROBE_NEW)
        faulthandler.dump_traceback_later(120, exit=True)
        flush(engine)  # every rank logs the last segment's background work
    finally:
        faulthandler.cancel_dump_traceback_later()
        engine.shutdown()
    res = dict(meta=meta, mode=a.mode, args=vars(a), runs=runs, periodic=periodic_stats(log_dir))
    if a.mode in ATTN:
        try:
            res["attn"] = attn_summary(log_dir, probe_rank)
        except Exception as e:  # keep the timed runs even if the logs cannot be read
            res["attn"] = dict(by_label={}, probe=dict(found_next_step=False), checks=dict(summary=repr(e)), ok=False)
    with open(os.path.join(out, f"{a.mode}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if a.mode in ATTN:
        s = res["attn"]
        print(f"{a.mode}: self-check {'passed' if s['ok'] else 'FAILED'}: {json.dumps(s['checks'])}", flush=True)
        print(f"{a.mode}: probe {json.dumps(s['probe'])}", flush=True)
        for label, d in s["by_label"].items():
            print(f"{a.mode}: {label}: {d['steps']} rank-steps checked, {d['pieces_per_step']:.0f} pieces read per "
                  f"step, {d['failures']} failures, verdict {d['verdict_ms_per_step'] or 0:.3f} ms per step, "
                  f"fallbacks {d['unsupported']}", flush=True)
        if not s["ok"]:
            raise SystemExit(3)


def report(a):
    out = os.path.abspath(a.out)
    res = {}
    for m in MODES:
        path = os.path.join(out, f"{m}.json")
        if os.path.exists(path):
            with open(path) as fh:
                res[m] = json.load(fh)
    rows = []
    for m, r in res.items():
        for label in dict.fromkeys(x["segment"] for x in r["runs"]):
            xs = [x["out_tok_s"] for x in r["runs"] if x["segment"] == label]
            p = r["periodic"].get(label, {})
            per_step = next(x["scrub_pieces"] for x in r["runs"] if x["segment"] == label)
            rows.append(dict(mode=m, segment=label, scrub_per_step=per_step, runs=len(xs),
                             out_tok_s=statistics.median(xs), min=min(xs),
                             max=max(xs), seconds=statistics.median(x["seconds"] for x in r["runs"] if x["segment"] == label),
                             **{k: p.get(k) for k in ("encodes", "pages_encoded", "encode_s", "scrub_pieces",
                                                      "scrub_bytes_per_s", "alarms", "unclean", "mean_step_s",
                                                      "pool_bytes")}))
            rows[-1]["output_digests"] = sorted({x.get("output_digest") for x in r["runs"] if x["segment"] == label
                                                 and x.get("output_digest")})
            if "attn" in r:  # checks on read: the check's own numbers, and its self-check
                c = r["attn"]["by_label"].get(label, {})
                rows[-1].update(attn_pieces_per_step=c.get("pieces_per_step"), attn_failures=c.get("failures"),
                                attn_verdict_ms_per_step=c.get("verdict_ms_per_step"), attn_self_check=r["attn"]["ok"],
                                attn_probe_found_next_step=r["attn"]["probe"]["found_next_step"])
    base = {m: next((x["out_tok_s"] for x in rows if x["mode"] == m), None) for m in ("plain", "parity")}
    # tags alone: the fused store kernels, as the attn modes use them (or the PyTorch ones if only those ran)
    base["tags"] = next((x["out_tok_s"] for m in ("tags_fused", "tags") for x in rows if x["mode"] == m), None)
    plain_digests = next((x["output_digests"] for x in rows if x["mode"] == "plain"), [])
    for x in rows:
        x["vs_plain"] = x["out_tok_s"] / base["plain"] - 1 if base["plain"] else None
        x["vs_parity"] = x["out_tok_s"] / base["parity"] - 1 if base["parity"] else None
        x["vs_tags"] = x["out_tok_s"] / base["tags"] - 1 if base["tags"] else None
        # greedy decoding: every run's tokens equal plain SGLang's (each mode's runs give one digest)
        x["same_tokens_as_plain"] = (x["output_digests"] == plain_digests
                                     if len(plain_digests) == 1 and x["output_digests"] else None)
        x["within_3pct_of_parity"] = (x["vs_parity"] >= -H7) if x["mode"] in ("kvmend", "fused") and x["vs_parity"] is not None else None
        rate = x.get("scrub_bytes_per_s") or 0
        x["pass_over_pool_s"] = x["pool_bytes"] / rate if rate and x.get("pool_bytes") else None
        x["pass_over_20GB_s"] = 20e9 / rate if rate else None
    first = next(iter(res.values()))
    max_new = first["args"]["max_new"]
    for x in rows:
        x["step_ms"] = 1e3 * x["seconds"] / max_new  # wall time per output step: the time per output token when decode dominates
    summary = dict(meta={m: r["meta"] for m, r in res.items()}, workload={k: first["args"][k] for k in (
        "model", "tp", "requests", "prompt_len", "max_new", "repeats", "encode_every", "max_total_tokens", "chunk")},
        rows=rows)
    with open(os.path.join(out, "overhead.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    w = summary["workload"]
    lines = [f"Workload: {w['requests']} requests x {w['prompt_len']} prompt tokens, {w['max_new']} output tokens "
             f"each, sent at once; {os.path.basename(str(w['model']).rstrip('/'))} at TP={w['tp']}; median of "
             f"{w['repeats']} runs; parity encoded every {w['encode_every']} batch steps.", "",
             "| Mode | Background scrub | Output tok/s (min-max) | vs plain | vs parity alone | vs tags alone | "
             "Time per output step | Scrub rate per GPU | Pass over 20 GB | Encode time per run | Alarms | "
             "Same tokens as plain |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def pct(v):
        return "" if v is None else f"{100 * v:+.1f}%"

    for x in rows:
        scrub = f"{x['scrub_per_step']:,} pieces per step" if x.get("scrub_per_step") else "none"
        rate = f"{x['scrub_bytes_per_s'] / 1e9:.2f} GB/s" if x.get("scrub_bytes_per_s") else ""
        pass20 = f"{x['pass_over_20GB_s']:.0f} s" if x.get("pass_over_20GB_s") else ""
        enc = f"{x['encode_s']:.2f} s" if x.get("encode_s") is not None else ""
        alarms = "" if x.get("alarms") is None else str(x["alarms"])
        same = {True: "yes", False: "no"}.get(x["same_tokens_as_plain"], "")
        lines.append(f"| {x['mode']} | {scrub} | {x['out_tok_s']:,.0f} ({x['min']:,.0f}-{x['max']:,.0f}) | "
                     f"{pct(x['vs_plain'])} | {pct(x['vs_parity'])} | {pct(x['vs_tags'])} | {x['step_ms']:.1f} ms | "
                     f"{rate} | {pass20} | {enc} | {alarms} | {same} |")
    checked = [x for x in rows if x.get("attn_self_check") is not None]
    if checked:
        lines += ["", "Checks on read (per rank and decode step; a piece is one page of one layer on one GPU):", "",
                  "| Mode | Pieces read per step | Failures in fault-free runs | Verdict time per step | "
                  "Probe found at the next step | Self-check |", "|---|---|---|---|---|---|"]
        for x in checked:
            ms = x.get("attn_verdict_ms_per_step")
            lines.append(f"| {x['mode']} | {x.get('attn_pieces_per_step') or 0:,.0f} | {x.get('attn_failures')} | "
                         f"{'' if ms is None else f'{ms:.3f} ms'} | {'yes' if x['attn_probe_found_next_step'] else 'no'}"
                         f" | {'passed' if x['attn_self_check'] else 'FAILED'} |")
    with open(os.path.join(out, "overhead.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    a = parse()
    if a.report:
        report(a)
    else:
        bench(a)


if __name__ == "__main__":
    main()
