#!/usr/bin/env python3
"""Loads from host memory that land on only some GPUs (decision 39). SGLang's hierarchical cache (HiCache,
--enable-hierarchical-cache) backs KV pages up to a host pool and loads them back into new device pages on a prefix
hit. A failed per-GPU transfer leaves one GPU's copy of some loaded pages stale, and nothing reports an error.

Every variant starts from an empty cache. A fill request (rid fill_<variant>: the prompt, 1 output token) computes the
prompt's KV, and write_through backs it up to host memory; op offload then evicts the device copy. Request R (rid
<variant>: the same prompt, greedy, --max-new tokens) loads its prefix (the prompt but its last page) from host
memory and computes its last page and its output on top. The fault: the load lands on every GPU but --rank, where
--pages pages (the load's last pages, or from --pos) read zeros (--mode zeros, the default) or keep the bytes their
slots held (--mode stale). SGLang hands the pages an eviction has just freed out first, so here R's load lands in the
fill request's own pages, whose stale bytes are the very KV the load brings: a stale failure would change nothing.

  clean             R alone from an empty cache: the prompt in one pass, no load
  control_device    the fill request, then R reuses its KV on the GPU: no load
  control_load      the fill request, offload, then R loads the prefix from host memory: the faults' path, no fault
  silent            the fault, and nothing checks: does the text go wrong, with no error?
  kvmend            the fault; the check on arrival (each loaded layer against the tags its pages left with, before
                    the forward reads it) finds the stale pieces on that GPU and copies them again from the host
                    copy; R's reader check after its prefill then passes
  kvmend_reader     the fault, not checked on arrival: R's reader check after its prefill finds the stale pages on
                    that GPU by tag, mend copies them again from the host copy, and the tokens computed from them (R's
                    last page and its first output token) are computed again (retract, keep 0)
  kvmend_recompute  the same, but mend recomputes the stale pages in place, using no host copy
  control_recompute no fault, the same recompute and rollback: kvmend_recompute's fault-free twin
  retry             abort and retry: R's reader check finds the damage, R is aborted, the cache flushed (device and host
                    copies), and R resubmitted, computed from its first token; its time leaves out the flush

kvmend and kvmend_reader must return control_load's text; kvmend_recompute must return control_recompute's; the retry
must return clean's (one pass, like its resubmission).

  uv run python run_hicache.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --out runs/hicache-check
"""
import argparse
import faulthandler
import glob
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = 16
VARIANTS = ("clean", "control_device", "control_load", "silent", "kvmend", "kvmend_reader", "kvmend_recompute",
            "control_recompute", "retry")
LOADS = ("control_load", "silent", "kvmend", "kvmend_reader", "kvmend_recompute", "control_recompute", "retry")
ARRIVAL_KEYS = ("checked", "found_pieces", "found_pages", "found_layers", "uncovered_pieces", "recopied_pieces",
                "still_failing_pieces", "recopy_equals_host_copy", "check_seconds", "recopy_seconds", "fault")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--prompt-len", type=int, default=1536)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--pages", type=int, default=4, help="pages the partial load leaves stale")
    p.add_argument("--pos", type=int, default=-1,
                   help="first stale position of the load (a page boundary); default: the load's last --pages pages")
    p.add_argument("--rank", type=int, default=1, help="the GPU the load does not reach")
    p.add_argument("--mode", default="zeros", choices=["stale", "zeros"],
                   help="what the struck pages hold: zeros, or their slots' earlier bytes (here, the same KV)")
    p.add_argument("--io-backend", default="kernel", choices=["kernel", "direct"], help="HiCache's copy kernels")
    p.add_argument("--mem-layout", default="layer_first", choices=["layer_first", "page_first"])
    p.add_argument("--hicache-ratio", type=float, default=2.0, help="host pool size over device pool size")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--variant-timeout", type=int, default=600)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.pos >= 0 and (a.pos % PAGE or a.pos + a.pages * PAGE > loaded_tokens(a)):
        p.error(f"--pos must be a page boundary with --pages pages inside the load of {loaded_tokens(a)} tokens")
    return a


def loaded_tokens(a):
    """R's prefix from host memory: its prompt but the last token (SGLang computes at least one), in whole pages."""
    return ((a.prompt_len - 1) // PAGE) * PAGE


def struck_start(a):
    """The first stale position; the load starts at position 0, since the whole prompt was offloaded."""
    return a.pos if a.pos >= 0 else loaded_tokens(a) - a.pages * PAGE


def make_plan(a):
    fault = dict(op="fault", at=1, kind="partial_load", rank=a.rank, pages=a.pages, mode=a.mode,
                 **(dict(pos=a.pos) if a.pos >= 0 else {}))
    unchecked = dict(fault, check_on_arrival=False)
    offload = dict(op="offload", at=1)
    # R's reader check right after its prefill; on a finding, mend, then roll R back from its prefill batch: its last
    # page and its first output token were computed from the stale pages
    back = dict(op="retract", at=1, keep=0, from_prefill=True)
    reader = [dict(op="check", at=1), dict(op="mend", at=1), dict(back, only_if_found=True)]
    return {
        "fill_warmup": [offload], "warmup": [],
        "clean": list(reader),
        "fill_control_device": [], "control_device": list(reader),
        "fill_control_load": [offload], "control_load": list(reader),
        "fill_silent": [offload, unchecked], "silent": [],
        "fill_kvmend": [offload, fault], "kvmend": list(reader),
        "fill_kvmend_reader": [offload, unchecked], "kvmend_reader": list(reader),
        "fill_kvmend_recompute": [offload, unchecked],
        "kvmend_recompute": [dict(op="check", at=1), dict(op="mend", at=1, recopy=False),
                             dict(back, only_if_found=True)],
        "fill_control_recompute": [offload],
        "control_recompute": [dict(op="check", at=1),
                              dict(op="mend", at=1, force=True, pos=struck_start(a), pages=a.pages), back],
        "fill_retry": [offload, unchecked],
        "retry": [dict(op="check", at=1), dict(op="abort", at=1, only_if_found=True)],
        "retry2": [],
    }


def hicache_args(a):
    from run_poc import engine_args

    args = engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk)
    # SGLang 0.5.1 switches decode to FlashInfer when HiCache copies with its "kernel" I/O and no decode backend is
    # named (ModelRunner.model_specific_adjustment); name Triton for both, as in every other KVMend run
    args.update(enable_hierarchical_cache=True, hicache_write_policy="write_through", hicache_io_backend=a.io_backend,
                hicache_mem_layout=a.mem_layout, hicache_ratio=a.hicache_ratio,
                decode_attention_backend="triton", prefill_attention_backend="triton")
    return args


def summarize(out, a, res, meta, engine):
    from run_fair import classify
    from run_poc import first_diff

    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]

    def act(rid, op, rank=0):  # a plan action of request rid, as rank `rank` logged it
        return [x for x in logs if x.get("rid") == rid and x["op"] == op and x["rank"] == rank]

    def every_rank(rid, op):
        return [x for x in logs if x.get("rid") == rid and x["op"] == op]

    def event(rid, op, rank=None):  # a record of the HiCache hooks, made during a batch that held request rid
        return [x for x in logs if x["op"] == op and rid in x.get("rids", []) and (rank is None or x["rank"] == rank)]

    loaded, ref = loaded_tokens(a), res["control_load"]
    rows, layers = {}, None
    for v in VARIANTS:
        if v not in res:
            continue
        issue = event(v, "load_issue", rank=0)
        arrival = {x["rank"]: x for x in event(v, "load_check")}
        layers = layers or next((x["layers"] for x in arrival.values()), None)
        extends = act(v, "extend")
        offload = act(f"fill_{v}", "offload")
        check = act(v, "check")
        mends = every_rank(v, "mend")
        m0 = [m for m in mends if m["rank"] == 0]
        retract = act(v, "retract")
        copies = [x for m in mends for x in m.get("recopied", []) if x.get("recopied")]
        rows[v] = dict(
            offload={k: offload[0].get(k) for k in ("host_only_tokens", "waited_s", "pending_writes", "seconds")}
            if offload else None,
            loaded_tokens=issue[0]["tokens"] if issue else 0,
            carried_pages=issue[0].get("carried_pages") if issue else None,
            fault=issue[0].get("fault") if issue else None,
            prefix_len=extends[0]["prefix_len"] if extends else None,
            prefills=[dict(prefix_len=x["prefix_len"], tokens=x["extend_input_len"]) for x in extends],
            arrival={str(r): {k: x.get(k) for k in ARRIVAL_KEYS} for r, x in sorted(arrival.items())},
            reader_check=check[0]["found"] if check else None,
            reader_check_seconds=check[0]["seconds"] if check else None,
            mend=dict(recopied_pieces=len(copies), recopy_tag_matches=all(x.get("tag_matches") for x in copies),
                      recopy_equals_host_copy=all(x.get("equals_host_copy") for x in copies),
                      recomputed=m0[0].get("recomputed"),
                      recomputed_tokens=sum(x["tokens"] for x in m0[0].get("recomputed") or []),
                      recopy_failed_pages=m0[0].get("recopy_failed_pages"),
                      still_failing=sum(m.get("still_failing_here", 0) for m in mends),
                      seconds=m0[0].get("mend_seconds")) if m0 and m0[0].get("mended") else None,
            retract={k: retract[0].get(k) for k in ("retracted", "regenerated", "freed_tokens", "reason")}
            if retract else None,
            equals_control_load=res[v]["ids"] == ref["ids"],
            first_diff_vs_control_load=first_diff(res[v]["ids"], ref["ids"]),
            seconds=res[v]["seconds"], extra_s=res[v]["seconds"] - ref["seconds"])
    rows["silent"]["outcome"] = classify(res["silent"]["ids"], ref["ids"])[0]
    retry = res["retry"]
    rows["retry"].update(first_finish=retry["first"]["finish"], first_tokens=len(retry["first"]["ids"]),
                         equals_clean=retry["ids"] == res["clean"]["ids"], flush_seconds=retry["flush_seconds"],
                         resubmit_prefill=[x["extend_input_len"] for x in act("retry2", "extend")])
    errors = [x for x in logs if x["op"] == "error"]
    r, kv, rd, rc = str(a.rank), rows["kvmend"], rows["kvmend_reader"], rows["kvmend_recompute"]
    others = [str(x) for x in range(a.tp) if x != a.rank]
    pieces = a.pages * (layers or 0)

    def struck(row):  # the device pages the fault struck in this variant's load
        return (row["fault"] or {}).get("pages", [])

    hit = kv["arrival"].get(r, {})
    verdict = {
        "no_errors": not errors,
        "offload_left_the_prompt_in_host_memory_only": all(
            (rows[v]["offload"] or {}).get("host_only_tokens", 0) >= loaded for v in LOADS),
        "every_load_variant_loaded_its_prefix_from_host": all(
            rows[v]["loaded_tokens"] == loaded and rows[v]["prefix_len"] == loaded for v in LOADS),
        "control_device_reused_the_gpu_copy": rows["control_device"]["loaded_tokens"] == 0
        and rows["control_device"]["prefix_len"] == loaded,
        "tags_travel_no_false_alarm_after_a_load": rows["control_load"]["carried_pages"] == loaded // PAGE
        and len(rows["control_load"]["arrival"]) == a.tp
        and all(x["checked"] and x["found_pieces"] == 0 and x["uncovered_pieces"] == 0
                for x in rows["control_load"]["arrival"].values())
        and rows["control_load"]["reader_check"] == {},
        "the_fault_struck_one_gpu": all(len(struck(rows[v])) == a.pages and rows[v]["fault"]["rank"] == a.rank
                                        for v in ("silent", "kvmend", "kvmend_reader", "kvmend_recompute", "retry")),
        "the_fault_changed_the_bytes_in_every_layer": (hit.get("fault") or {}).get("changed_layers") == layers,
        "found_on_arrival_on_that_gpu_only": hit.get("found_pages") == struck(kv) and hit.get("found_layers") == layers
        and hit.get("found_pieces") == pieces and all(kv["arrival"].get(o, {}).get("found_pieces") == 0 for o in others),
        "copied_again_from_host_and_passes_its_tags": hit.get("recopied_pieces") == pieces
        and hit.get("still_failing_pieces") == 0 and hit.get("recopy_equals_host_copy") is True,
        "kvmend_reader_check_then_passes": kv["reader_check"] == {},
        "kvmend_text_equals_control_load": kv["equals_control_load"],
        "reader_check_finds_it_on_that_gpu_by_tag": rd["reader_check"] == {
            str(p): {"tag": {"ranks": [a.rank], "layers": layers}} for p in struck(rd)},
        "reader_path_copies_again_and_regenerates": bool(rd["mend"]) and rd["mend"]["recopied_pieces"] == pieces
        and rd["mend"]["recopy_tag_matches"] and rd["mend"]["recomputed"] == []
        and not rd["mend"]["recopy_failed_pages"] and (rd["retract"] or {}).get("regenerated") == 1,
        "kvmend_reader_text_equals_control_load": rd["equals_control_load"],
        "recompute_path_recomputes_the_struck_pages": bool(rc["mend"]) and rc["mend"]["recomputed_tokens"] == a.pages * PAGE
        and rc["mend"]["recopied_pieces"] == 0 and rc["mend"]["still_failing"] == 0
        and (rc["retract"] or {}).get("regenerated") == 1,
        "kvmend_recompute_text_equals_its_control": res["kvmend_recompute"]["ids"] == res["control_recompute"]["ids"],
        "retry_first_attempt_aborted": retry["first"]["finish"] == "abort",
        "retry_text_is_clean": rows["retry"]["equals_clean"],
    }

    def most(row, key):
        vals = [x.get(key) for x in row["arrival"].values() if x.get(key) is not None]
        return max(vals) if vals else None

    times = dict(reference_seconds=ref["seconds"], **{f"{v}_extra_s": rows[v]["extra_s"] for v in
                                                      ("kvmend", "kvmend_reader", "kvmend_recompute", "silent")},
                 retry_extra_s=retry["seconds"] - ref["seconds"],
                 retry_with_flush_extra_s=retry["seconds"] + retry["flush_seconds"] - ref["seconds"],
                 arrival_check_seconds={v: most(rows[v], "check_seconds") for v in ("control_load", "kvmend")},
                 arrival_copy_seconds=hit.get("recopy_seconds"),
                 reader_check_seconds={v: rows[v]["reader_check_seconds"] for v in ("control_load", "kvmend_reader")},
                 reader_mend_seconds=(rd["mend"] or {}).get("seconds"),
                 recompute_mend_seconds=(rc["mend"] or {}).get("seconds"))
    summary = dict(meta=meta, model=a.model, tp=a.tp, prompt_len=a.prompt_len, max_new=a.max_new,
                   loaded_tokens=loaded, pages=a.pages, pos=a.pos, struck_positions=[struck_start(a),
                                                                                    struck_start(a) + a.pages * PAGE],
                   rank=a.rank, mode=a.mode, layers=layers,
                   engine={k: v for k, v in engine.items() if k.startswith(("hicache", "enable_hier", "decode_att",
                                                                             "prefill_att", "attention", "tp_size",
                                                                             "page_size", "max_total"))},
                   rows=rows, times=times, controls=dict(
                       control_device_equals_control_load=res["control_device"]["ids"] == ref["ids"],
                       control_load_equals_clean=ref["ids"] == res["clean"]["ids"],
                       control_load_first_diff_vs_clean=first_diff(ref["ids"], res["clean"]["ids"]),
                       control_recompute_equals_control_load=res["control_recompute"]["ids"] == ref["ids"]),
                   texts={k: v["text"][:200] for k, v in res.items()}, errors=errors, verdict=verdict)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(os.path.join(out, "hicache.md"), "w") as fh:
        fh.write(hicache_md(summary))
    return summary


def hicache_md(s):
    def yes(x):
        return "yes" if x else "no"

    r = str(s["rank"])
    lines = [f"Prompt {s['prompt_len']} tokens; R loads its first {s['loaded_tokens']} from host memory. The fault: "
             f"{s['pages']} pages (positions {s['struck_positions'][0]}-{s['struck_positions'][1] - 1}) do not land on "
             f"GPU {s['rank']} ({s['mode']}), in all {s['layers']} layers. Reference: control_load, "
             f"{s['times']['reference_seconds']:.2f} s.", "",
             "| Variant | Loaded from host | Found by | Repair | Text equals the fault-free load | R's time | Extra |",
             "|---|---|---|---|---|---|---|"]
    for v, row in s["rows"].items():
        arrival = row["arrival"].get(r, {})
        found = []
        if arrival.get("found_pieces"):
            found.append(f"check on arrival, GPU {r}: {arrival['found_pieces']} pieces")
        if row["reader_check"]:
            ranks = sorted({rk for x in row["reader_check"].values() for w in x.values() for rk in w["ranks"]})
            found.append(f"reader check, GPUs {ranks}: {', '.join(sorted({w for x in row['reader_check'].values() for w in x}))}")
        repair = []
        if arrival.get("recopied_pieces"):
            repair.append(f"copy {arrival['recopied_pieces']} pieces from host")
        m = row["mend"] or {}
        if m.get("recopied_pieces"):
            repair.append(f"copy {m['recopied_pieces']} pieces from host")
        if m.get("recomputed_tokens"):
            repair.append(f"recompute {m['recomputed_tokens']} tokens")
        if (row["retract"] or {}).get("retracted"):
            repair.append(f"regenerate {row['retract']['regenerated']}")
        if v == "retry":
            repair.append(f"abort, flush, recompute {sum(row['resubmit_prefill'])} tokens")
        text = yes(row["equals_control_load"]) + (f" ({row['outcome']})" if "outcome" in row else "")
        if v == "retry":
            text += f"; equals clean: {yes(row['equals_clean'])}"
        lines.append(f"| {v} | {row['loaded_tokens']} tokens | {'; '.join(found) or 'nothing'} | "
                     f"{'; '.join(repair) or '-'} | {text} | {row['seconds']:.2f} s | {row['extra_s']:+.2f} s |")
    t = s["times"]
    lines += ["", f"Check on arrival: {t['arrival_check_seconds']} s per load (the slowest GPU); copy from host on "
              f"GPU {s['rank']}: {t['arrival_copy_seconds']} s. Abort and retry: {t['retry_extra_s']:+.2f} s "
              f"({t['retry_with_flush_extra_s']:+.2f} s with the flush). Controls: {s['controls']}."]
    return "\n".join(lines) + "\n"


def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)  # the GPU the fault strikes must exist
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    plan_path = os.path.join(out, "plan.json")
    with open(plan_path, "w") as fh:
        json.dump(make_plan(a), fh, indent=1)
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_ENCODE_EVERY", "KVMEND_SCRUB_PIECES", "KVMEND_CTRL", "KVMEND_TAGS", "KVMEND_FENCE",
              "KVMEND_STRICT", "KVMEND_LOAD_CHECK"):
        os.environ.pop(k, None)

    import sglang as sgl
    from transformers import AutoTokenizer

    from kvmend.meta import run_meta
    from run_fair import run
    from run_poc import build_prompt, flush, start_loop

    meta = run_meta()
    prompt = build_prompt(AutoTokenizer.from_pretrained(a.model), a.prompt_len)
    engine_cfg = hicache_args(a)
    engine = sgl.Engine(**engine_cfg)
    start_loop(engine)
    res = {}

    def go(name, fill, n=None):
        flush(engine)
        if fill:
            res[f"fill_{name}"] = run(engine, prompt, f"fill_{name}", 1, a.variant_timeout)
        res[name] = run(engine, prompt, name, n or a.max_new, a.variant_timeout)
        print(f"{name:18s} {len(res[name]['ids'])} tokens in {res[name]['seconds']:.2f} s ({res[name]['finish']})",
              flush=True)

    try:
        go("warmup", True, n=8)  # the load path warm too
        go("clean", False)
        for v in ("control_device", "control_load", "silent", "kvmend", "kvmend_reader", "kvmend_recompute",
                  "control_recompute"):
            go(v, True)
        flush(engine)
        res["fill_retry"] = run(engine, prompt, "fill_retry", 1, a.variant_timeout)
        first = run(engine, prompt, "retry", a.max_new, a.variant_timeout)
        t0 = time.perf_counter()
        flush(engine)  # the damaged cached KV must go, device and host copies, or the retry would read it again
        flush_s = time.perf_counter() - t0
        second = run(engine, prompt, "retry2", a.max_new, a.variant_timeout)
        res["retry"] = dict(first=first, second=second, ids=second["ids"], text=second["text"], finish=first["finish"],
                            seconds=first["seconds"] + second["seconds"], flush_seconds=flush_s)
        print(f"retry              first {len(first['ids'])} tokens ({first['finish']}); retry {len(second['ids'])} "
              f"tokens; {res['retry']['seconds']:.2f} s, and the flush {flush_s:.2f} s", flush=True)
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
    s = summarize(out, a, res, meta, engine_cfg)
    print(json.dumps(s["verdict"], indent=1))
    print(hicache_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
