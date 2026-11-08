#!/usr/bin/env python3
"""Requests that arrive over time, under three responses to damaged KV (the proposal's fairness rule: costs are
measured on requests that arrive over time, not on a batch sent at once).

The workload is run_load.py's: --groups groups, each with a --shared-token prefix (page aligned) that its requests
share, and one question per request. --requests requests arrive by a Poisson process of --rate requests per second
(seed --seed); request i (q000, q001, ...) asks the next question of group i % --groups. Every request is greedy, with
--max-new output tokens, and streamed. Parity is encoded every 16 batch steps in the background, in every mode. Before
each run, one request per group with one output token puts the group's prefix in the radix cache.

Four faults strike, run_load.py's four (run_load.fault_specs), each when its request has produced the fault's output
count, each in a group of its own. The seed picks the groups, and in each group the request, among the first two
thirds of the group's requests, so that later ones arrive while the damage may still be there:
  shared   output 10: a late write around the store path into the last 4 pages of the group's prefix, every GPU
  foreign  output 20: the first page of the request's question holds another sequence's KV, with matching tags
  burst    output 15: one GPU's memory region fails: 8 prefix pages, 2 layers, on GPU --rank (parity rebuilds it)
  fenced   output 12: a late write through the store path with an old generation (the fence drops it; fast mode has
           no fence, as plain SGLang has none)

Three modes in one engine, the cache flushed between runs; each mode runs once without the faults and once with them:
  fast    fc, ff   no checks: tokens leave as they are chosen
  strict  sc, sf   KVMend: a request is checked right after its prefill and every --every output tokens, and its
                   tokens leave only after a check covers them; damage is mended, and every reader that may have read
                   it generates its unreleased tokens again
  retry   rc, rf   the same checks, but a failed check mends nothing: the request is aborted, and so is every other
                   reader of the damaged pages; those pages leave the prefix cache with everything cached on top of
                   them (hook: KVMEND_STRICT entries rid:every:abort), and the client resubmits the request, which
                   recomputes from the first damaged page on
A request that ends with an abort is resubmitted in any mode, up to --max-attempts attempts (in strict mode, only a
check at finish aborts). --replays more fault-free runs follow, fast and strict in turn (fc2, sc2, fc3, ...).

Outputs are judged as in run_load.py. A request is stable when every fault-free run gives it the same text; a stable
request's final text is right when it is that text, and any request's is right when it equals some fault-free run's.
Per run: time to first token and end-to-end latency (median and 99th percentile, from the request's arrival to the
first token and to the end of its last attempt), throughput, goodput (requests whose final text is right, per second),
wrong outputs on stable requests, aborts and resubmissions, and checks per batch step. summary.json holds them,
arrivals.md the tables.

  uv run python run_arrivals.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --out runs/arrivals-check
"""
import argparse
import asyncio
import faulthandler
import glob
import json
import os
import random
import statistics
import time

from run_load import fault_specs

HERE = os.path.dirname(os.path.abspath(__file__))
MODES = {"f": "fast", "s": "strict", "r": "retry"}
FAULT_RUNS = ("ff", "sf", "rf")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--rate", type=float, default=8.0, help="arrivals per second (Poisson)")
    p.add_argument("--seed", type=int, default=0, help="picks the arrival times and the faulted requests")
    p.add_argument("--groups", type=int, default=4, help="shared prefixes, one fault in each of four of them")
    p.add_argument("--shared", type=int, default=1024, help="shared prefix tokens (a multiple of 16)")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--every", type=int, default=4, help="output tokens between a checked request's checks")
    p.add_argument("--rank", type=int, default=1, help="the GPU of the burst")
    p.add_argument("--max-attempts", type=int, default=4, help="attempts per request, resubmissions included")
    p.add_argument("--replays", type=int, default=2, help="more fault-free runs, fast and strict in turn")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--run-timeout", type=int, default=900, help="seconds before a stuck run ends the job")
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    specs = fault_specs(argparse.Namespace(shared=a.shared, rank=0))
    if a.groups < len(specs):
        p.error(f"--groups must be at least {len(specs)}: each fault strikes a group of its own")
    if a.requests < 2 * a.groups:
        p.error("--requests must give every group at least two requests")
    if a.shared % 16 or a.shared < 144:
        p.error("--shared must be a multiple of 16 and hold the burst's pages (positions 16-143)")
    if a.max_new <= max(f["at"] for f in specs.values()):
        p.error(f"--max-new must exceed the last fault's output count ({max(f['at'] for f in specs.values())})")
    a.per_group = -(-a.requests // a.groups)  # run_load.prompts makes this many questions per group
    return a


def run_codes(a):
    return ["fc", "ff", "sc", "sf", "rc", "rf"] + [("fc", "sc")[k % 2] + str(2 + k // 2) for k in range(a.replays)]


def strict_rules(a, codes):
    """KVMEND_STRICT: the engine warm-up (w_*) and every strict and retry run, by rid prefix."""
    rules = [f"w_*:{a.every}"]
    for c in codes:
        if c[0] in "sr":
            rules.append(f"{c}_*:{a.every}" + (":abort" if c[0] == "r" else ""))
    return ",".join(rules)


def request_order(a):
    """(qid, prompt name) in arrival order: request i asks the next question of group i % groups."""
    return [(f"q{i:03d}", f"g{i % a.groups}_r{i // a.groups}") for i in range(a.requests)]


def schedule(a):
    """The arrival offsets in seconds (the first at 0) and the faulted requests {index: fault name}, from the seed."""
    rng = random.Random(a.seed)
    offsets, t = [], 0.0
    for _ in range(a.requests):
        offsets.append(t)
        t += rng.expovariate(a.rate)
    names = list(fault_specs(a))
    faults = {}
    for name, g in zip(names, rng.sample(range(a.groups), len(names))):
        members = [i for i in range(a.requests) if i % a.groups == g]
        faults[rng.choice(members[:max(1, 2 * len(members) // 3)])] = name
    return offsets, dict(sorted(faults.items()))


def make_plan(a, order, faults):
    specs = fault_specs(a)
    plan = {}
    for run in FAULT_RUNS:
        for i, name in faults.items():
            f = dict(specs[name])
            if run == "ff" and f["kind"] == "late_store":
                f["fence"] = False  # plain SGLang has no fence: the late write lands
            plan[f"{run}_{order[i][0]}"] = [f]
    return plan


# ---------------------------------------------------------------------------------------------------------------
# The clients

async def _attempt(engine, rid, ids, max_new, t0):
    """One streamed attempt: when its first token and its end arrive (seconds from the run's start), how it finished,
    and its tokens. The tokens are read from the tokenizer manager's state of the request, which keeps every chunk's:
    when chunks arrive faster than the client reads them, the stream yields only the newest."""
    from sglang.srt.managers.io_struct import GenerateReqInput

    from run_poc import output_tokens

    tm = engine.tokenizer_manager
    obj = GenerateReqInput(input_ids=ids, rid=rid, stream=True,
                           sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))
    rec = dict(rid=rid, submit=time.perf_counter() - t0, first=None, end=None, finish=None, chunks=0)
    state, got, last = None, [], None
    try:
        async for chunk in tm.generate_request(obj, None):
            rec["end"] = time.perf_counter() - t0
            rec["chunks"] += 1
            if state is None:  # gone if the request ended before its first chunk was read; then `lost` tells
                state = tm.rid_to_state.get(rid)  # whether the chunks read hold all its tokens
            got += list(chunk["output_ids"])
            if rec["first"] is None and (chunk["meta_info"].get("completion_tokens") or 0) > 0:
                rec["first"] = rec["end"]  # a held-back step streams a chunk with no output token
            last = chunk
    except Exception as e:  # recorded and counted as an error; the request is not resubmitted
        rec.update(end=time.perf_counter() - t0, finish="error", error=repr(e)[:500], ids=[], tokens=0, lost=False)
        return rec
    meta = last["meta_info"]
    fin = meta.get("finish_reason") or {}
    n = meta.get("completion_tokens") or 0
    out, _ = output_tokens(dict(output_ids=list(state.output_ids) if state is not None else got), ids, n)
    rec.update(finish=fin.get("type") if isinstance(fin, dict) else fin,
               message=fin.get("message") if isinstance(fin, dict) else None, tokens=n,
               cached=meta.get("cached_tokens"), ids=out, text=last.get("text", "")[:160], lost=len(out) != n)
    return rec


async def _client(engine, run, qid, ids, offset, a, t0):
    """A client: it sends its request at its arrival time and resubmits it after an abort."""
    await asyncio.sleep(max(0.0, t0 + offset - time.perf_counter()))
    arrive = time.perf_counter() - t0
    attempts = []
    while not attempts or (attempts[-1]["finish"] == "abort" and len(attempts) < a.max_attempts):
        rid = f"{run}_{qid}" + (f".{len(attempts)}" if attempts else "")
        attempts.append(await _attempt(engine, rid, ids, a.max_new, t0))
    final = attempts[-1]
    firsts = [x["first"] for x in attempts if x["first"] is not None]
    return dict(qid=qid, arrive=arrive, attempts=attempts, ids=final["ids"], finish=final["finish"],
                ttft=final["first"] - arrive if final["first"] is not None else None, e2e=final["end"] - arrive,
                first_token_any_attempt=min(firsts) - arrive if firsts else None)


def drive(engine, run, order, P, offsets, a):
    async def go():
        t0, wall0 = time.perf_counter(), time.time()
        reqs = await asyncio.gather(*[_client(engine, run, qid, P[name], offsets[i], a, t0)
                                      for i, (qid, name) in enumerate(order)])
        return dict(wall=[wall0, time.time()], seconds=time.perf_counter() - t0, requests={r["qid"]: r for r in reqs})

    return asyncio.get_event_loop().run_until_complete(go())


# ---------------------------------------------------------------------------------------------------------------
# The summary

def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _pages(found):
    return {int(p) for p in (found or {})}


def run_row(code, run, ctx):
    """One run's metrics, its faults and what found them."""
    qids, ref, stable, log0, plan = ctx["qids"], ctx["ref"], ctx["stable"], ctx["log0"], ctx["plan"]
    reqs = run["requests"]
    lo, hi = run["wall"]
    window = [x for x in log0 if lo <= x.get("time", 0.0) <= hi]
    mine = [x for x in log0 if str(x.get("rid", "")).startswith(code + "_")]
    right = [q for q in qids if reqs[q]["ids"] in ref[q]]
    attempts = [x for q in qids for x in reqs[q]["attempts"]]
    start = min(r["arrive"] for r in reqs.values())
    makespan = max(r["arrive"] + r["e2e"] for r in reqs.values()) - start
    batched = [x for x in window if x["op"] == "release_checks"]
    single = [x for x in mine if x["op"] == "release_check"]
    steps = next((x["batch_steps"] for x in log0 if x["op"] == "periodic" and x.get("label") == code), None)
    checks = sum(x.get("requests", 0) for x in batched) + len(single)
    found = [x for x in single if x.get("found_pieces")]
    faults = []
    for rid in (r for r in plan if r.startswith(code + "_")):
        rec = next((x for x in mine if x["op"] == "fault" and x["rid"] == rid), None)
        by = [f"{x['rid']}@{x['n_out']}" for x in found
              if rec is not None and x["time"] >= rec["time"] and set(rec["pages"]) & _pages(x.get("found"))]
        faults.append(dict(rid=rid, name=ctx["fault_of"][rid.split("_", 1)[1]], kind=plan[rid][0]["kind"],
                           at=plan[rid][0]["at"], struck=rec is not None, pages=rec["pages"] if rec else None,
                           fenced=bool(rec and rec.get("fenced")), written=rec.get("written_tokens") if rec else None,
                           found_by=by, request_right=rid.split("_", 1)[1] in right))
    detections = [dict(rid=x["rid"], n_out=x["n_out"], finishing=x.get("finishing"),
                       found_pages=sorted(_pages(x.get("found"))),
                       reasons=sorted({w for v in (x.get("found") or {}).values() for w in v}),
                       aborted=bool(x.get("aborted")), other_readers=[o["rid"] for o in x.get("other_readers", [])],
                       regenerated=x.get("regenerate", {}).get("regenerated"), evicted=x.get("evicted_tokens"),
                       recomputed=sum(s["tokens"] for s in x.get("mend", {}).get("recomputed", [])),
                       rebuilt=sum(1 for s in x.get("mend", {}).get("rebuilt", []) if s.get("rebuilt")),
                       poisoned=x.get("poisoned_pages"), dropped=x.get("dropped_tokens"))
                  for x in found]
    seconds = sorted(x["seconds"] for x in batched)
    return dict(
        mode=MODES[code[0]], with_faults=code in FAULT_RUNS, requests=len(qids), makespan_s=makespan,
        ttft_p50_s=pct([reqs[q]["ttft"] for q in qids], 0.5), ttft_p99_s=pct([reqs[q]["ttft"] for q in qids], 0.99),
        e2e_p50_s=pct([reqs[q]["e2e"] for q in qids], 0.5), e2e_p99_s=pct([reqs[q]["e2e"] for q in qids], 0.99),
        output_tokens=sum(len(reqs[q]["ids"]) for q in qids),
        out_tok_per_s=sum(len(reqs[q]["ids"]) for q in qids) / makespan,
        req_per_s=sum(1 for q in qids if reqs[q]["finish"] not in ("abort", "error")) / makespan,
        goodput_req_per_s=len(right) / makespan, right=len(right), wrong=[q for q in qids if q not in right],
        stable_wrong=[q for q in stable if reqs[q]["ids"] != ref[q][0]],
        aborts=sum(1 for x in attempts if x["finish"] == "abort"), resubmissions=len(attempts) - len(qids),
        aborted_again=sorted(x["rid"] for x in attempts if x["finish"] == "abort" and "." in x["rid"]),
        unfinished=[q for q in qids if reqs[q]["finish"] == "abort"],
        resubmission_cached_tokens={x["rid"]: x.get("cached") for x in attempts if "." in x["rid"]},
        errors=[dict(rid=x["rid"], finish=x["finish"], error=x.get("error"), lost=x.get("lost")) for x in attempts
                if x["finish"] == "error" or x.get("lost")],
        checks=checks, check_steps=len(batched), batch_steps=steps,
        checks_per_step=checks / steps if steps else None,
        check_ms_median=1e3 * statistics.median(seconds) if seconds else None,
        dropped_tokens=sum(x.get("dropped_tokens", 0) for x in window if x["op"] == "drop_poisoned")
        + sum(d["dropped"] or 0 for d in detections),
        faults=faults, detections=detections)


def summarize(out, a, runs, meta, order, offsets, faults, plan):
    logs = []
    for path in sorted(glob.glob(os.path.join(out, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    log0 = [x for x in logs if x.get("rank") == 0]
    qids = [q for q, _ in order]
    clean = [c for c in runs if c not in FAULT_RUNS]
    ref = {q: [runs[c]["requests"][q]["ids"] for c in clean] for q in qids}
    vary = [q for q in qids if any(x != ref[q][0] for x in ref[q])]
    stable = [q for q in qids if q not in vary]
    ctx = dict(qids=qids, ref=ref, stable=stable, log0=log0, plan=plan,
               fault_of={order[i][0]: name for i, name in faults.items()})
    rows = {c: run_row(c, runs[c], ctx) for c in runs}
    hook_errors = [x for x in logs if x["op"] == "error"]
    driver_errors = [e for r in rows.values() for e in r["errors"]]

    def all_found(code):
        fr = rows.get(code, {}).get("faults", [])
        return len(fr) == len(faults) and all(f["struck"] and (f["fenced"] or f["found_by"]) for f in fr)

    s = dict(meta=meta, model=a.model, tp=a.tp, requests=a.requests, rate=a.rate, seed=a.seed, groups=a.groups,
             shared=a.shared, max_new=a.max_new, every=a.every, max_attempts=a.max_attempts,
             arrival_span_s=offsets[-1], faults={order[i][0]: name for i, name in faults.items()},
             clean_runs=clean, fault_free_variation=vary, stable=len(stable), runs=rows,
             errors=dict(hook=hook_errors, driver=driver_errors))
    s["verdict"] = {
        "no_errors": not hook_errors and not driver_errors,
        "strict_sends_no_wrong_output_on_stable_requests": "sf" in rows and not rows["sf"]["stable_wrong"],
        "every_unfenced_fault_found_strict": all_found("sf"),
        "every_unfenced_fault_found_retry": all_found("rf"),
    }
    s["observations"] = {  # reported, not pass checks
        "fast_wrong_on_stable": len(rows["ff"]["stable_wrong"]) if "ff" in rows else None,
        "retry_wrong_on_stable": len(rows["rf"]["stable_wrong"]) if "rf" in rows else None,
        "fenced_write_dropped": all(f["written"] == 0 for c in ("sf", "rf") if c in rows
                                    for f in rows[c]["faults"] if f["fenced"]),
        "unfinished": {c: r["unfinished"] for c, r in rows.items() if r["unfinished"]},
        "retry_resubmissions_aborted_again": rows["rf"]["aborted_again"] if "rf" in rows else None,
    }
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(s, fh, indent=1, default=str)
    with open(os.path.join(out, "arrivals.md"), "w") as fh:
        fh.write(arrivals_md(s))
    return s


def arrivals_md(s):
    def ms(x):
        return "" if x is None else f"{1e3 * x:.0f}"

    def sec(x):
        return "" if x is None else f"{x:.2f}"

    lines = [f"{s['requests']} requests at {s['rate']:g} per second (seed {s['seed']}, the last at "
             f"{s['arrival_span_s']:.1f} s), {s['groups']} groups with {s['shared']:,}-token shared prefixes, "
             f"{s['max_new']} output tokens each; checks every {s['every']} tokens. {s['model']} at TP={s['tp']}. "
             f"Stable requests: {s['stable']} of {s['requests']} (fault-free runs {', '.join(s['clean_runs'])}).", "",
             "| Run | Mode | Faults | TTFT p50 / p99 (ms) | E2E p50 / p99 (s) | Output tok/s | Goodput (right req/s) "
             "| Wrong on stable | Right | Aborts / resubmissions | Checks per step |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in s["runs"].items():
        cps = "" if r["checks_per_step"] is None else f"{r['checks_per_step']:.2f}"
        lines.append(f"| {c} | {r['mode']} | {'yes' if r['with_faults'] else 'no'} | {ms(r['ttft_p50_s'])} / "
                     f"{ms(r['ttft_p99_s'])} | {sec(r['e2e_p50_s'])} / {sec(r['e2e_p99_s'])} | "
                     f"{r['out_tok_per_s']:.0f} | {r['goodput_req_per_s']:.2f} | {len(r['stable_wrong'])} | "
                     f"{r['right']} of {r['requests']} | {r['aborts']} / {r['resubmissions']} | {cps} |")
    lines += ["", "| Fault | Request | At output | Pages | Fast: request right | Strict: found by | Retry: found by |",
              "|---|---|---|---|---|---|---|"]
    by = {c: {f["name"]: f for f in s["runs"][c]["faults"]} for c in FAULT_RUNS if c in s["runs"]}
    for qid, name in s["faults"].items():
        row = [by.get(c, {}).get(name) for c in FAULT_RUNS]
        pages = next((f["pages"] for f in row if f and f["pages"]), None)

        def found(f):
            if f is None:
                return ""
            if f["fenced"]:
                return f"dropped by the fence ({f['written']} tokens written)"
            return ", ".join(f["found_by"]) or ("not found" if f["struck"] else "did not strike")

        lines.append(f"| {name} | {qid} | {row[0]['at'] if row[0] else ''} | {pages} | "
                     f"{'yes' if row[0] and row[0]['request_right'] else 'no'} | {found(row[1])} | {found(row[2])} |")
    lines += ["", "Verdict: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in s["verdict"].items()) + "."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------

def main():
    a = parse()
    a.rank = min(a.rank, a.tp - 1)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    from transformers import AutoTokenizer

    from run_load import prompts

    P = prompts(AutoTokenizer.from_pretrained(a.model), a)
    order = request_order(a)
    offsets, faults = schedule(a)
    plan = make_plan(a, order, faults)
    codes = run_codes(a)
    plan_path, ctrl_path = os.path.join(out, "plan.json"), os.path.join(out, "ctrl.json")
    with open(plan_path, "w") as fh:
        json.dump(plan, fh, indent=1)
    with open(os.path.join(out, "arrivals.json"), "w") as fh:
        json.dump(dict(order=order, offsets=offsets, faults={order[i][0]: n for i, n in faults.items()}), fh, indent=1)

    def label(name):  # read by the hook at the next flush: names the background-work record of the next window
        with open(ctrl_path, "w") as fh:
            json.dump({"label": name}, fh)

    label("warmup")
    os.environ.update(KVMEND_ENABLE="1", KVMEND_PLAN=plan_path, KVMEND_OUT=out, KVMEND_FUSED="1",
                      KVMEND_STRICT=strict_rules(a, codes), KVMEND_ENCODE_EVERY="16", KVMEND_CTRL=ctrl_path,
                      PYTHONPATH=os.pathsep.join(p for p in (os.path.join(HERE, "hook_site"), HERE,
                                                             os.environ.get("PYTHONPATH", "")) if p))
    for k in ("KVMEND_SCRUB_PIECES", "KVMEND_TAGS", "KVMEND_FENCE", "KVMEND_STRICT_EVERY", "KVMEND_ON_FAIL",
              "KVMEND_STRIPE"):
        os.environ.pop(k, None)

    import sglang as sgl

    from kvmend.meta import run_meta
    from run_load import run_batch, warm_prefixes
    from run_poc import engine_args, flush, start_loop

    meta = run_meta()
    engine = sgl.Engine(**engine_args(a.model, a.tp, a.mem_frac, a.max_total_tokens, a.chunk))
    start_loop(engine)
    runs = {}
    try:
        flush(engine)
        # warm-up, not counted: one request per group, checked, so the check kernels compile before the timed runs
        run_batch(engine, "w", {f"g{g}_r0": P[f"g{g}_r0"] for g in range(a.groups)}, a)
        for code in codes:
            label(code)
            flush(engine)
            faulthandler.dump_traceback_later(a.run_timeout, exit=True)
            warm_prefixes(engine, code, P, a)
            runs[code] = drive(engine, code, order, P, offsets, a)
            faulthandler.cancel_dump_traceback_later()
            reqs = runs[code]["requests"].values()
            print(f"run {code}: {sum(len(r['ids']) for r in reqs)} output tokens in {runs[code]['seconds']:.2f} s, "
                  f"{sum(len(r['attempts']) - 1 for r in reqs)} resubmissions", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(120, exit=True)
        try:
            label("end")
            flush(engine)  # every rank writes its last log lines, with the last run's background-work record
        except Exception as e:
            print("final flush failed:", repr(e), flush=True)
        faulthandler.cancel_dump_traceback_later()
        engine.shutdown()
    with open(os.path.join(out, "outputs.json"), "w") as fh:
        json.dump(runs, fh, indent=1, default=str)
    s = summarize(out, a, runs, meta, order, offsets, faults, plan)
    print(json.dumps(s["verdict"], indent=1))
    print(json.dumps(s["observations"], indent=1, default=str))
    print(arrivals_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
