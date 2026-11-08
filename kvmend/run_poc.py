#!/usr/bin/env python3
"""KVMend proof of concept on SGLang 0.5.1: request R under several responses to one silent fault, one engine.

Every variant runs R (a fixed prompt, greedy decoding) on an empty cache. At output 1, right after the prefill, each
variant encodes parity of the prompt's pages; KVMend's variants also scrub once, which is R's last good check.

  clean          no fault
  control        no fault; R is rolled back at output 41 and re-prefilled (the re-prefill path on its own)
  silent         GhostServe as published: parity, no check. At output 10, one bit of R's page 64 (tokens
                 1,024-1,039), layer 12, rank 2 flips; nothing notices
  kvmend         the same fault; at output 41 the scrub finds it, parity rebuilds the piece, R is rolled back and
                 re-prefilled
  repaironly     the same fault, found and rebuilt, but R is not rolled back: its suspect tokens keep their KV
  gs_gpu         GhostServe told the GPU: at output 41 an oracle names rank 2, and GhostServe rebuilds that GPU's
                 share of every encoded page from parity; no rollback
  gs_chunk       GhostServe told the GPU and the 2,048-token chunk that holds the damage; rebuilds that chunk only
  reprefill_all  detect, then re-prefill the whole request: the scrub finds the fault, and R is rolled back with its
                 cached prompt dropped, as when a checksum mismatch aborts the request and the retry recomputes it

After the variants, two fault-free runs produce the same text as the recovered ones, on the same compute path:
  forced         caches the prompt, extends it by kvmend's first 41 output tokens, and decodes the rest (KVMend's path)
  forced_full    computes the prompt and the same 41 tokens from an empty cache, then decodes (reprefill_all's path)

A recovery is correct if its output equals its fault-free twin token for token. The responses that do not roll back
(silent, repaironly, gs_gpu, gs_chunk) are compared with forced; control equal to clean shows that the re-prefill path
by itself does not change the output.

  uv run python run_poc.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 4 --out runs/smoke
"""
import argparse
import asyncio
import faulthandler
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ROWS = [  # the comparison with GhostServe: approach, run, how it learns of the fault
    ("GhostServe as published", "silent", "no"),
    ("GhostServe, told the GPU", "gs_gpu", "an oracle names the GPU"),
    ("GhostServe, told the GPU and the chunk", "gs_chunk", "an oracle names the GPU and the chunk"),
    ("Detect, then re-prefill the whole request", "reprefill_all", "the scrub names the piece"),
    ("KVMend", "kvmend", "the scrub names the piece"),
]
TWIN = {"reprefill_all": "forced_full"}  # every other run is compared with forced


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--inject-at", type=int, default=10, help="output count at which the bit flips")
    p.add_argument("--detect-at", type=int, default=41, help="output count of the scrub that finds it")
    p.add_argument("--pos", type=int, default=1030, help="token position of the damaged value (page 64)")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--rank", type=int, default=2)
    p.add_argument("--kv", default="k", choices=["k", "v"])
    p.add_argument("--elem", type=int, default=37, help="element within the token's K or V on that rank")
    p.add_argument("--bit", type=int, default=14)
    p.add_argument("--gs-chunk", type=int, default=2048, help="GhostServe's chunk, in tokens")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192, help="chunked prefill size")
    p.add_argument("--variants", default="clean,control,silent,kvmend,repaironly")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--variant-timeout", type=int, default=300, help="seconds before a stuck variant ends the job")
    p.add_argument("--require-pass", action="store_true", help="exit 3 unless every pass check holds")
    p.add_argument("--out", required=True)
    return p.parse_args()


def engine_args(model, tp, mem_frac, max_total, chunk, no_cuda_graph=False):
    """The engine settings of every run, the proof of concept and the overhead benchmark alike."""
    return dict(model_path=model, tp_size=tp, page_size=16, attention_backend="triton", sampling_backend="pytorch",
                disable_overlap_schedule=True, mem_fraction_static=mem_frac, max_total_tokens=max_total,
                chunked_prefill_size=chunk, random_seed=0, log_level="warning", disable_cuda_graph=no_cuda_graph,
                disable_custom_all_reduce=True)


def start_loop(engine):
    # SGLang switches asyncio to uvloop while it starts; on Python 3.12 that policy no longer makes a loop on demand,
    # and Engine's helpers call asyncio.get_event_loop(). Set one now, after the policy change.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    # The tokenizer manager reads the scheduler's replies in a loop that only generate_request() starts;
    # flush_cache() does not start it, so a flush before the first request would wait forever. Start it now.
    engine.tokenizer_manager.auto_create_handle_loop()


def make_plan(a):
    base = [dict(op="encode", at=1), dict(op="scrub", at=1)]
    gs_base = [dict(op="encode", at=1, check=False)]  # GhostServe: parity, and nothing is checked
    fault = dict(op="inject", at=a.inject_at, pos=a.pos, layer=a.layer, rank=a.rank, kv=a.kv, elem=a.elem, bit=a.bit)
    detect = [dict(op="scrub", at=a.detect_at), dict(op="repair", at=a.detect_at)]
    retract = [dict(op="retract", at=a.detect_at)]
    oracle = dict(op="repair_share", at=a.detect_at, rank=a.rank, pos=a.pos, chunk_tokens=a.gs_chunk)
    dump = dict(op="dump", at=a.detect_at, name="at_detect", **{"from": a.prompt_len, "to": a.prompt_len + a.detect_at})
    plans = dict(
        clean=base + [dump],
        control=base + retract,
        silent=gs_base + [fault, dump],
        kvmend=base + [fault] + detect + retract,
        repaironly=base + [fault] + detect,
        gs_gpu=gs_base + [fault, dict(oracle, scope="gpu")],
        gs_chunk=gs_base + [fault, dict(oracle, scope="chunk")],
        reprefill_all=base + [fault, dict(op="scrub", at=a.detect_at)] + retract + [dict(op="evict", at=a.detect_at)],
    )
    out = {name: plans[name] for name in a.variants.split(",")}
    if "kvmend" in out:
        out["forced"] = []  # no actions; listed so that the hook logs its re-prefill
    if "reprefill_all" in out:
        out["forced_full"] = []
    return out


def build_prompt(tokenizer, n):
    """A long, plain log with a question at the end, cut to exactly n tokens."""
    head = "Below is a long server log. Read it, then answer the question at the end.\n\n"
    lines = [f"Event {i}: request {(i * 37) % 1009} on GPU {i % 8} finished in {10 + (i * 53) % 990} ms."
             for i in range(6000)]
    body = tokenizer.encode(head + "\n".join(lines))
    question = tokenizer.encode("\n\nQuestion: which events took longer than 990 ms, and on which GPUs? "
                                "Answer in a few sentences.\nAnswer:", add_special_tokens=False)
    assert len(body) + len(question) >= n, f"prompt source too short for {n} tokens"
    return body[:n - len(question)] + question


def generate(engine, ids, rid, max_new):
    from sglang.srt.managers.io_struct import GenerateReqInput

    obj = GenerateReqInput(input_ids=ids, rid=rid,
                           sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(engine.tokenizer_manager.generate_request(obj, None).__anext__())


def flush(engine):
    """Empty the cache before each variant. A failed flush would let the next variant reuse the cached prompt,
    possibly with another variant's damaged page, so retry and stop if it keeps failing. The flush also makes every
    rank write its buffered log lines."""
    for _ in range(40):
        res = engine.flush_cache()
        if getattr(res, "success", res):
            return
        time.sleep(0.5)
    raise RuntimeError("flush_cache kept failing: a request is still in the scheduler")


def output_tokens(r, inputs, n):
    """SGLang's output_ids start with the input's last few tokens, kept for incremental detokenization."""
    ids = list(r["output_ids"])
    extra = len(ids) - n
    head = ids[:extra] if extra > 0 and ids[:extra] == list(inputs[-extra:]) else []
    return ids[len(head):], head


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def spread(out, a, ranks):
    """Per layer, the largest |K| and |V| difference between silent and clean at the first tainted position."""
    import torch

    rows = []
    for r in range(ranks):
        paths = [os.path.join(out, f"dump_{v}_at_detect_rank{r}.pt") for v in ("clean", "silent")]
        if not all(os.path.exists(p) for p in paths):
            return None
        clean, silent = (torch.load(p) for p in paths)
        tainted = a.inject_at - 1  # the first token whose KV was written after the flip, relative to the prompt end
        same_input = clean["output_ids"][:a.inject_at] == silent["output_ids"][:a.inject_at]
        for layer in range(clean["k"].shape[0]):
            dk = (clean["k"][layer, tainted].float() - silent["k"][layer, tainted].float()).abs().max().item()
            dv = (clean["v"][layer, tainted].float() - silent["v"][layer, tainted].float()).abs().max().item()
            rows.append(dict(rank=r, layer=layer, max_abs_diff=max(dk, dv), same_input=same_input))
    layers = sorted({x["layer"] for x in rows})
    worst = {l: max(x["max_abs_diff"] for x in rows if x["layer"] == l) for l in layers}
    return dict(position=a.prompt_len + a.inject_at - 1, same_input_tokens=rows[0]["same_input"] if rows else None,
                layers_unchanged=[l for l in layers if worst[l] == 0.0],
                layers_changed=[l for l in layers if not worst[l] == 0.0], per_layer_max_abs_diff=worst)


def comparison(logs, ids, a):
    """One row per response to the same fault: does it notice, what does it read and recompute, how long does the
    recovery take, and is the output after the recovery that of a healthy server for the same text."""
    rows = []
    for name, v, notices in ROWS:
        if v not in ids:
            continue
        mine = [x for x in logs if x.get("rid") == v]
        det = [x for x in mine if x["op"] == "scrub" and x.get("at") == a.detect_at and x["rank"] == 0]
        acts = [x for x in mine if x.get("at") == a.detect_at and x["op"] in ("repair", "repair_share", "retract", "evict")]
        pre = [x for x in mine if x["op"] == "extend" and x["rank"] == 0 and x.get("n_out", 0) > 1]
        share = [x for x in acts if x["op"] == "repair_share" and x["rank"] == a.rank]
        fixes = [r for x in acts if x["op"] == "repair" and x["rank"] == a.rank for r in x["repairs"]]
        row = dict(approach=name, run=v, notices=notices, found=det[0]["damaged_all"] if det else None,
                   detect_s=det[0]["seconds"] if det else None,
                   bytes_read=share[0]["bytes_read"] if share else sum(r["bytes_read"] for r in fixes if r.get("rebuilt")),
                   pieces_rebuilt=share[0]["pieces"] if share else sum(1 for r in fixes if r.get("rebuilt")),
                   repair_s=share[0]["seconds"] if share else (sum(r["seconds"] for r in fixes if r.get("rebuilt"))
                                                               if fixes else None),
                   tokens_recomputed=sum(x["extend_input_len"] for x in pre), recover_s=None)
        if share:
            row["damaged_piece_bit_exact"] = all(p["rebuilt"] and p["bit_exact"] for p in share[0]["damaged_pieces"])
            row["tag_mismatches_after"] = share[0]["tag_mismatches_after"]
        elif fixes:
            row["damaged_piece_bit_exact"] = all(r.get("bit_exact") is True for r in fixes)
        if acts:  # from the start of the first recovery action to the first token after it, or the end of the rebuild
            start = min(x["t0"] for x in acts)
            if pre:
                end = max(x["time"] for x in pre)
            elif share:  # the end of the rebuild, before the checks that only this proof of concept makes
                end = max(x["t_end"] for x in acts if x["op"] == "repair_share")
            else:
                end = max(x["time"] for x in acts)
            row["recover_s"] = end - start
        ref = TWIN.get(v, "forced")
        row.update(reference=ref, same_text_before_detection=ids[v][:a.detect_at] == ids["kvmend"][:a.detect_at]
                   if "kvmend" in ids else None,
                   equals_reference=ids[v] == ids[ref] if ref in ids else None,
                   first_diff_vs_reference=first_diff(ids[v], ids[ref]) if ref in ids else None)
        rows.append(row)
    return rows


def size(n):
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.4g} {unit}"
    return f"{n} B"


def comparison_md(rows, a):
    lines = ["| Approach | Notices the fault | Data read to repair | Tokens recomputed | Recovery time | "
             "Output after recovery equals the fault-free run of the same text |", "|---|---|---|---|---|---|"]
    for r in rows:
        t = "none" if r["recover_s"] is None else f"{r['recover_s']:.3f} s"
        ok = "yes" if r["equals_reference"] else f"no (first differs at token {r['first_diff_vs_reference']})"
        lines.append(f"| {r['approach']} | {r['notices']} | {size(r['bytes_read']) if r['bytes_read'] else 'none'} | "
                     f"{r['tokens_recomputed']:,} | {t} | {ok} |")
    page_start = (a.pos // 16) * 16
    lines.append(f"\nRe-prefill from the damaged page (computed, not run): {a.prompt_len + a.detect_at - page_start:,} "
                 f"tokens, against {a.prompt_len + a.detect_at:,} for the whole request.")
    return "\n".join(lines) + "\n"


def verdict(s, a):
    """The pass checks of the proof of concept, and of the comparison runs when they ran."""
    inj = s["inject"][0] if s["inject"] else {}
    det = s["detection"][0]["damaged_all"] if s["detection"] else []
    rep, pre = s["repair"], s["reprefill"]
    checks = {
        "kvmend_equals_fault_free_run_of_same_text": s["comparisons"]["kvmend_equals_forced"] is True,
        "scrub_names_the_injected_piece": bool(inj) and [list(x) for x in det] == [[a.rank, a.layer, inj.get("page")]],
        "repair_is_bit_exact": len(rep) == 1 and rep[0].get("bit_exact") is True and rep[0].get("tag_matches") is True,
        "reprefill_is_small": len(pre) == 1 and 0 < pre[0]["extend_input_len"] <= a.detect_at + 16,
        "no_errors": not s["errors"],
    }
    rows = {r["run"]: r for r in s["ghostserve_comparison"]}
    gs = [rows[v] for v in ("gs_gpu", "gs_chunk") if v in rows]
    if gs:
        checks["ghostserve_rebuild_is_bit_exact"] = all(r.get("damaged_piece_bit_exact") is True
                                                        and r.get("tag_mismatches_after") == 0 for r in gs)
    if "reprefill_all" in rows:
        r = rows["reprefill_all"]
        checks["reprefill_all_recomputes_the_whole_request"] = r["tokens_recomputed"] == a.prompt_len + a.detect_at
    return checks


def summarize(out, a, results, meta):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]

    def pick(rid, op, at=None, rank=None):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op and (at is None or x.get("at") == at)
                and (rank is None or x["rank"] == rank)]

    ids = {v: results[v]["output_ids"] for v in results}

    def diff(x, y):
        return first_diff(ids[x], ids[y]) if {x, y} <= ids.keys() else None

    summary = dict(
        meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new,
        fault=dict(pos=a.pos, layer=a.layer, rank=a.rank, kv=a.kv, elem=a.elem, bit=a.bit, at=a.inject_at),
        detect_at=a.detect_at,
        outputs={v: dict(tokens=len(ids[v]), text=results[v]["text"][:400], seconds=results[v]["seconds"])
                 for v in results},
        comparisons={
            "kvmend_equals_forced": ids.get("kvmend") == ids.get("forced") if {"kvmend", "forced"} <= ids.keys() else None,
            "kvmend_equals_control": ids.get("kvmend") == ids.get("control") if {"kvmend", "control"} <= ids.keys() else None,
            "kvmend_first_tokens_equal_repaironly": ids["kvmend"][:a.detect_at] == ids["repaironly"][:a.detect_at]
            if {"kvmend", "repaironly"} <= ids.keys() else None,
            "repaironly_vs_forced_first_diff": diff("repaironly", "forced"),
            "silent_vs_clean_first_diff": diff("silent", "clean"),
            "repaironly_vs_control_first_diff": diff("repaironly", "control"),
            "control_vs_clean_first_diff": diff("control", "clean"),
            "gs_gpu_vs_repaironly_first_diff": diff("gs_gpu", "repaironly"),
            "reprefill_all_vs_forced_full_first_diff": diff("reprefill_all", "forced_full"),
            "forced_full_vs_forced_first_diff": diff("forced_full", "forced"),
        },
        inject=pick("kvmend", "inject", rank=a.rank),
        last_good_check=pick("kvmend", "scrub", at=1, rank=0),
        detection=pick("kvmend", "scrub", at=a.detect_at, rank=0),
        repair=[r for x in pick("kvmend", "repair", rank=a.rank) for r in x["repairs"]],
        rollback=pick("kvmend", "retract", rank=0),
        # the re-prefill is the extend after the first output token (SGLang clears is_retracted when it reschedules)
        reprefill=[x for x in pick("kvmend", "extend", rank=0) if x.get("n_out", 0) > 1],
        forced_extend=pick("forced", "extend", rank=0),
        encode=pick("kvmend", "encode", at=1),
        scrub_rate=[dict(rank=x["rank"], gib_per_s=x["gib_per_s"], pieces=x["pieces_checked"]) for x in pick("kvmend", "scrub", at=1)],
        ghostserve_share_repairs=[x for v in ("gs_gpu", "gs_chunk") for x in pick(v, "repair_share", rank=a.rank)],
        evict=pick("reprefill_all", "evict", rank=0),
        errors=[x for x in logs if x["op"] == "error"],
        spread=spread(out, a, a.tp),
    )
    summary["ghostserve_comparison"] = comparison(logs, ids, a)
    summary["verdict"] = verdict(summary, a)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    if summary["ghostserve_comparison"]:
        with open(os.path.join(out, "comparison.md"), "w") as fh:
            fh.write(comparison_md(summary["ghostserve_comparison"], a))
    return summary


def fault_free_twin(engine, a, name, prompt, window, warm):
    """A healthy server producing `window` after the prompt: with warm, the prompt is cached first, so the window is
    an extend on top of it (KVMend's re-prefill path); without, everything is computed from an empty cache."""
    faulthandler.dump_traceback_later(a.variant_timeout, exit=True)
    print(f"{name:13s} flush and generate", flush=True)
    flush(engine)
    if warm:
        generate(engine, prompt, f"{name}_warm", 1)
    t0 = time.perf_counter()
    r = generate(engine, prompt + window, name, a.max_new - a.detect_at)
    faulthandler.cancel_dump_traceback_later()
    tail, head = output_tokens(r, prompt + window, a.max_new - a.detect_at)
    res = dict(output_ids=window + tail, prompt_tail=head, text=r["text"], seconds=time.perf_counter() - t0)
    print(f"{name:13s} {len(window)} kept + {len(tail)} new tokens in {res['seconds']:.2f} s", flush=True)
    return res


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

    meta = run_meta()
    prompt = build_prompt(AutoTokenizer.from_pretrained(a.model), a.prompt_len)
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk, a.no_cuda_graph))
    start_loop(engine)
    results = {}
    try:
        for name in a.variants.split(","):
            # A hang anywhere in a variant dumps every thread's stack and ends the job, instead of using the
            # whole time limit.
            faulthandler.dump_traceback_later(a.variant_timeout, exit=True)
            print(f"{name:13s} flush and generate", flush=True)
            flush(engine)
            t0 = time.perf_counter()
            r = generate(engine, prompt, name, a.max_new)
            faulthandler.cancel_dump_traceback_later()
            ids, head = output_tokens(r, prompt, a.max_new)
            results[name] = dict(output_ids=ids, prompt_tail=head, text=r["text"], seconds=time.perf_counter() - t0)
            print(f"{name:13s} {len(ids)} tokens in {results[name]['seconds']:.2f} s", flush=True)
        if "kvmend" in results:
            results["forced"] = fault_free_twin(engine, a, "forced", prompt,
                                                results["kvmend"]["output_ids"][:a.detect_at], warm=True)
        if "reprefill_all" in results:
            results["forced_full"] = fault_free_twin(engine, a, "forced_full", prompt,
                                                     results["reprefill_all"]["output_ids"][:a.detect_at], warm=False)
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            flush(engine)  # every rank writes its last buffered log lines
        except Exception as e:  # report and still shut down
            print("final flush failed:", repr(e), flush=True)
        faulthandler.cancel_dump_traceback_later()
        engine.shutdown()
    with open(os.path.join(out, "outputs.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    s = summarize(out, a, results, meta)
    print(json.dumps({k: s[k] for k in ("verdict", "comparisons", "inject", "detection", "repair", "rollback",
                                        "reprefill", "forced_extend", "spread", "errors")}, indent=1, default=str))
    if s["ghostserve_comparison"]:
        print(comparison_md(s["ghostserve_comparison"], a), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
