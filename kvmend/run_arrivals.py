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
check at finish aborts). --replays more fault-free runs follow, fast and strict in turn (fc2, sc2, fc3, ...). Before
fc, one untimed warm-up run of the same workload (wu, fast, no faults) pays for the kernels compiled at their first
use. A run in which the server stalled for over --stall-s seconds (some request in flight, and none got its first token
or ended) is cold, and its timing is not reported.

Outputs are judged as in run_load.py. A request is stable when every fault-free run gives it the same text; a stable
request's final text is right when it is that text, and any request's is right when it equals some fault-free run's.
SGLang's output depends on the batch, so a stable request's text can also change in a run with faults without reading
any damage, when other requests' repairs change its batches, and a repair changes the compute path of the request it
repairs, which flips a near-tie later in its text. The hook logs the pages each request read and since when
(KVMEND_LOG_READS). A request read damage if a page it read was damaged while it read it (from the fault until the last
check that found the damage; to the end of the run when nothing checks), or if a check of it failed. Records without
the pages (older runs) are judged by group: a fault in a shared prefix reaches its group's requests that ran while it
was there; a fault in a request's own pages reaches only that request. A stable request with a wrong text is then
  wrong before the repair    it read damage, and its first wrong token among those it released before the repair
                             (before the detection that covered the damage: what the release rule had cleared by then
                             or the stream had delivered; an aborted attempt: all it released), or with no repair at
                             all, left after the damage struck: a miss
  varied before the damage   it read damage, but its first wrong released token left before the damage struck (the
                             stream's release timeline against the fault's strike time), so intact KV computed it:
                             batch variation, which SGLang shows at TP=8 between identical runs; reported
  diverged after the repair  it read damage, but everything it released before the repair is right; reported with the
                             repair (rebuilt bit-exact, recopied, recomputed, regenerated, or aborted and resubmitted)
  varied without damage      it read no damaged page and passed every check: reported
Records without the stream's release timeline (older runs) take a token's release time from the passing check that
first covered it, no earlier than its computation either; for a request never checked, a straight line between its
first and last token gives an estimate, which is reported but never excuses a wrong token.
Per run: time to first token and end-to-end latency (median and 99th percentile, from the request's arrival to the
first token and to the end of its last attempt), throughput, goodput (requests whose final text is right, per second),
wrong outputs on stable requests, aborts and resubmissions, and checks per batch step. summary.json holds them,
arrivals.md the tables.

  uv run python run_arrivals.py --model /home/tae/models/TinyLlama-1.1B-Chat-v1.0 --tp 2 --out runs/arrivals-check
  python run_arrivals.py --resummarize runs/arrivals-check --out /tmp/rejudged    # judge old records again
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
PAGE = 16
MODES = {"f": "fast", "s": "strict", "r": "retry"}
FAULT_RUNS = ("ff", "sf", "rf")
TIMING = ("makespan_s", "ttft_p50_s", "ttft_p99_s", "e2e_p50_s", "e2e_p99_s", "out_tok_per_s", "req_per_s",
          "goodput_req_per_s", "check_ms_median")


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model")
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
    p.add_argument("--stall-s", type=float, default=2.0, help="a run whose server stalled this long is cold")
    p.add_argument("--max-total-tokens", type=int, default=65536)
    p.add_argument("--mem-frac", type=float, default=0.6)
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--run-timeout", type=int, default=900, help="seconds before a stuck run ends the job")
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--resummarize", metavar="DIR", help="judge the records in DIR again; write the summary to --out")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.resummarize:
        return a
    if not a.model:
        p.error("--model is required")
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
    its tokens, and `released`: [seconds, tokens received so far] at each chunk that brought tokens. The tokens are
    read from the tokenizer manager's state of the request, which keeps every chunk's: when chunks arrive faster than
    the client reads them, the stream yields only the newest."""
    from sglang.srt.managers.io_struct import GenerateReqInput

    from run_poc import output_tokens

    tm = engine.tokenizer_manager
    obj = GenerateReqInput(input_ids=ids, rid=rid, stream=True,
                           sampling_params=dict(temperature=0.0, max_new_tokens=max_new, ignore_eos=True))
    rec = dict(rid=rid, submit=time.perf_counter() - t0, first=None, end=None, finish=None, chunks=0, released=[])
    state, got, last = None, [], None
    try:
        async for chunk in tm.generate_request(obj, None):
            rec["end"] = time.perf_counter() - t0
            rec["chunks"] += 1
            if state is None:  # gone if the request ended before its first chunk was read; then `lost` tells
                state = tm.rid_to_state.get(rid)  # whether the chunks read hold all its tokens
            got += list(chunk["output_ids"])
            n_now = chunk["meta_info"].get("completion_tokens") or 0
            if n_now > (rec["released"][-1][1] if rec["released"] else 0):  # a held-back step brings no token
                rec["released"].append([rec["end"], n_now])
                rec["first"] = rec["end"] if rec["first"] is None else rec["first"]
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


def _ranges(pages):
    out = []
    for p in sorted(pages):
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return out


def _show(ranges):
    return ", ".join(f"{a_}-{b}" if b > a_ else f"{a_}" for a_, b in ranges)


def longest_stall(run):
    """The longest time in the run when some request was in flight and none got its first token or ended: the whole
    server stood still, as when a kernel compiles at its first use."""
    atts = [x for r in run["requests"].values() for x in r["attempts"] if x.get("end") is not None]
    marks = sorted(t for x in atts for t in (x.get("first"), x.get("end")) if t is not None)
    worst = 0.0
    for t0, t1 in zip(marks, marks[1:]):
        if t1 - t0 > worst and any(x["submit"] <= t0 and x["end"] >= t1 for x in atts):
            worst = t1 - t0
    return worst


def damage_windows(code, run, ctx):
    """The faults of run `code` that changed KV: their pages, when they struck, and the last time the damage could be
    read: the last check that found it (strict: then mended; retry: its readers aborted), or the end of the run when
    nothing found it. A late write the fence dropped changed nothing."""
    hi = run["wall"][1]
    mine = [x for x in ctx["log0"] if str(x.get("rid", "")).startswith(code + "_")]
    found = [x for x in mine if x["op"] == "release_check" and x.get("found_pieces")]
    out = []
    for x in mine:
        if x["op"] != "fault" or (x.get("fenced") and not x.get("written_tokens")):
            continue
        pages = set(x["pages"])
        seen = [y["time"] for y in found if y["time"] >= x["time"] and pages & _pages(y.get("found"))]
        qid = x["rid"].split("_", 1)[1]
        out.append(dict(rid=x["rid"], qid=qid, kind=x["kind"], pages=pages, start=x["time"],
                        end=max(seen) if seen else hi, found=bool(seen), group=ctx["group"][qid],
                        own=x.get("start", 0) >= ctx["shared"]))  # past the shared prefix: the request's own pages
    return out


def judge(q, run, ctx, windows, failing):
    """Did request q read damage in this run: a page it read was damaged while it read it, or one of its checks
    failed? The pages, and the time from its first check to its end, come from the hook's read_pages records; without
    them, by group (see the module's docstring), from its submission to its end."""
    w0 = run["wall"][0]
    hits, logged = [], True
    for att in run["requests"][q]["attempts"]:
        since, until = ctx["read_span"].get(att["rid"], (None, None))
        lo = since if since is not None else w0 + att["submit"]
        hi = until if until is not None else w0 + (att["end"] if att["end"] is not None else att["submit"])
        got = ctx["reads"].get(att["rid"])
        logged = logged and got is not None
        for w in windows:
            if lo > w["end"] or hi < w["start"]:
                continue
            if got is not None:
                pages = got & w["pages"]
            else:
                pages = w["pages"] if ((w["qid"] == q) if w["own"] else ctx["group"][q] == w["group"]) else set()
            if pages:
                hits.append(dict(rid=att["rid"], fault=w["rid"], kind=w["kind"], pages=sorted(pages), struck=w["start"]))
    failed = sorted(x["rid"] for x in run["requests"][q]["attempts"] if x["rid"] in failing)
    return dict(read_damage=bool(hits or failed), damage_read=hits, failed_checks=failed, pages_logged=logged)


def repair_of(x, ctx):
    """What the repair in release_check record x did. A rebuilt or recopied piece is counted on the rank that did it:
    only that rank's record marks it (the others list the piece and what they sent)."""
    recs = ctx["by_key"].get((x.get("rid"), x.get("step"), x.get("n_out")), {x.get("rank", 0): x})
    mends = {r: (y.get("mend") or {}) for r, y in recs.items()}
    rebuilt = [s for r, m in mends.items() for s in m.get("rebuilt", []) if s.get("rebuilt") and s.get("rank") == r]
    recopied = [s for r, m in mends.items() for s in m.get("recopied", []) if s.get("recopied") and s.get("rank") == r]
    diffs = [m["max_abs_diff_vs_clean"] for m in mends.values() if m.get("max_abs_diff_vs_clean") is not None]
    return dict(rebuilt=len(rebuilt), rebuilt_bit_exact=sum(1 for s in rebuilt if s.get("bit_exact")),
                recopied=len(recopied),
                recomputed=sum(s["tokens"] for s in (x.get("mend") or {}).get("recomputed", [])),
                max_abs_diff_vs_clean=max(diffs) if diffs else None)


def _repair_words(rep, regenerated, note):
    words = []
    if rep["rebuilt"]:
        words.append(f"rebuilt bit-exact ({rep['rebuilt']} pieces)" if rep["rebuilt_bit_exact"] == rep["rebuilt"]
                     else f"rebuilt ({rep['rebuilt']} pieces, {rep['rebuilt_bit_exact']} bit for bit)")
    if rep["recopied"]:
        words.append(f"copied again from host memory ({rep['recopied']} pieces)")
    if rep["recomputed"]:
        d = rep["max_abs_diff_vs_clean"]
        words.append(f"recomputed {rep['recomputed']} tokens" + (f" (largest difference {d:g})" if d else ""))
    if regenerated:
        words.append(f"regenerated {regenerated} tokens")
    return words + ([note] if note else [])


def repairs_of(code, ctx):
    """Every repair that reached an attempt of run `code`, by attempt rid: the time of the detection that covered the
    damage, the output tokens the release rule had cleared for release by then (kept; None for an abort: its stream
    holds what it released), and what the repair did."""
    pend = {x["rid"]: x for x in ctx["log0"] if x["op"] == "pending_redo"}
    out = {}
    for x in ctx["log0"]:
        if x["op"] != "release_check" or not x.get("found_pieces") or not str(x.get("rid", "")).startswith(code + "_"):
            continue
        rep = repair_of(x, ctx)

        def add(rid, redo, note=None):  # redo: that request's rollback (regenerate, an other reader's entry, ...)
            if redo.get("pending") and rid in pend:  # rolled back a step later (it had only just been prefilled)
                redo = pend[rid]
            aborted = bool(redo.get("aborted"))
            out.setdefault(rid, []).append(dict(
                time=x["time"], by=x["rid"], kept=None if aborted else redo.get("kept_outputs"), aborted=aborted,
                how=_repair_words(rep, None if aborted else redo.get("regenerated"), note), repair=rep))

        if x.get("aborted"):  # on_fail "abort": nothing is mended
            add(x["rid"], dict(aborted=True), "aborted, not mended; resubmitted")
        elif x.get("aborted_at_finish"):
            add(x["rid"], dict(aborted=True), "aborted at finish; resubmitted")
        else:
            add(x["rid"], x.get("regenerate") or {})
        for o in x.get("other_readers", []):
            add(o["rid"], o, "aborted, not mended; resubmitted" if o.get("aborted") else None)
    return out


def released_by(att, w0, t):
    """The output tokens the client of attempt `att` had received by wall time t, from the stream's release
    timeline (None in records without one)."""
    rel = att.get("released")
    return None if rel is None else max([n for s, n in rel if w0 + s <= t], default=0)


def release_time(att, d, run, ctx):
    """When output token d of attempt `att` was released (wall time), and from which record: the stream's timeline
    (when the client got it); without one (older records), the passing check that first covered it (a strict request is
    checked right after its prefill and every --every tokens, so its j-th pass covers its first 1, every, 2 * every,
    ... tokens, until a rollback); without checks, a straight line from its first to its last token (an estimate)."""
    w0 = run["wall"][0]
    rel = att.get("released")
    if rel is not None:
        t = next((s for s, n in rel if n >= d + 1), None)
        return (None if t is None else w0 + t), "stream"
    passes = ctx["passes"].get(att["rid"], [])
    cover = [1] + [ctx["every"] * j for j in range(1, len(passes))]
    t = next((p for p, c in zip(passes, cover) if c >= d + 1), ctx["finish_checks"].get(att["rid"]))
    if t is not None:
        return t, "checks"
    n = len(att.get("ids") or [])
    if att.get("first") is None or att.get("end") is None or not n:
        return None, "unknown"
    return w0 + att["first"] + (att["end"] - att["first"]) * d / max(n - 1, 1), "estimated"


def before_repair(q, run, ctx, judged, repairs):
    """For a request that read damage, each attempt that read it: the tokens it released before the repair (before
    the detection that covered the damage: what the release rule had cleared by then or the stream had delivered,
    whichever is more), against the fault-free text; and for the first wrong one, when it was released and when the
    damage it read struck. An attempt that ended with an abort counts all it released, and one that read damage and
    was never repaired counts every token."""
    ref, w0 = ctx["ref"][q][0], run["wall"][0]
    hit = {h["rid"] for h in judged["damage_read"]} | set(judged["failed_checks"])
    out = []
    for att in run["requests"][q]["attempts"]:
        reps = repairs.get(att["rid"], [])
        if att["rid"] not in hit and not reps:
            continue
        ids = att.get("ids") or []
        if att["finish"] == "abort":
            k, point = len(ids), "aborted"
        elif reps:
            k, point = max(max(r["kept"] or 0, released_by(att, w0, r["time"]) or 0) for r in reps), "repaired"
        else:
            k, point = len(ids), "never repaired"
        k = min(k, len(ids))
        wrong = next((i for i in range(k) if i >= len(ref) or ids[i] != ref[i]), None)
        at, source = release_time(att, wrong, run, ctx) if wrong is not None else (None, None)
        struck = [h["struck"] for h in judged["damage_read"] if h["rid"] == att["rid"] and h.get("struck") is not None]
        out.append(dict(rid=att["rid"], released_before_repair=k, point=point, first_wrong_released=wrong,
                        first_wrong_released_s=None if at is None else at - w0, released_from=source,
                        damage_struck_s=min(struck) - w0 if struck else None,
                        repair=[w for r in reps for w in r["how"]]))
    return out


def verdict_of(judged, released):
    """A stable request with a wrong text: "varied without damage" if it read no damaged page and passed every check;
    if it read damage: "diverged after the repair" if everything it released before the repair is right; "varied
    before the damage" if its first wrong released token left before the damage it read struck, so it was computed
    from intact KV (SGLang's output varies with the batch); else "wrong before the repair", a miss: the first wrong
    token left after the damage struck and before the repair (or with none), or its times are unknown."""
    if not judged["read_damage"]:
        return "varied without damage"
    wrong = [c for c in released if c["first_wrong_released"] is not None]
    if released and not wrong:
        return "diverged after the repair"
    # only a time no earlier than the token's computation can excuse it: the client's receipt (stream) or the check
    # that cleared it (checks); a straight-line estimate is reported but excuses nothing
    if released and all(c["released_from"] in ("stream", "checks") and c["first_wrong_released_s"] is not None
                        and c["damage_struck_s"] is not None and c["first_wrong_released_s"] < c["damage_struck_s"]
                        for c in wrong):
        return "varied before the damage"
    return "wrong before the repair"


def reads_of(q, run, ctx, prefix):
    """The pages request q read, as logged, or what the records tell when they were not logged."""
    atts = run["requests"][q]["attempts"]
    if all(x["rid"] in ctx["reads"] for x in atts):
        return {x["rid"]: _show(_ranges(ctx["reads"][x["rid"]])) for x in atts}
    g = ctx["group"][q]
    known = f"pages {_show([prefix[g]])}, from the fault records" if g in prefix else "pages not in the records"
    return f"group {g}'s prefix ({known}) and its own pages (not logged)"


def run_row(code, run, ctx):
    """One run's metrics, its faults, what found them, and who read the damage."""
    from run_poc import first_diff

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
                       regenerated=x.get("regenerate", {}).get("regenerated"),
                       kept=x.get("regenerate", {}).get("kept_outputs"), evicted=x.get("evicted_tokens"),
                       **repair_of(x, ctx), poisoned=x.get("poisoned_pages"), dropped=x.get("dropped_tokens"))
                  for x in found]
    # who read the damage, and what each stable request with a wrong text read and released before its repair
    windows = damage_windows(code, run, ctx)
    failing = {x["rid"] for x in found}
    judged = {q: judge(q, run, ctx, windows, failing) for q in qids}
    repairs = repairs_of(code, ctx)
    prefix = {}  # group -> [first, last] page of its prefix, where a fault in the prefix tells it
    for x in mine:
        if x["op"] == "fault" and "start" in x and x["start"] < ctx["shared"]:
            base = x["pages"][0] - x["start"] // PAGE
            prefix[ctx["group"][x["rid"].split("_", 1)[1]]] = [base, base + ctx["shared"] // PAGE - 1]
    stable_wrong = [q for q in stable if reqs[q]["ids"] != ref[q][0]]
    detail = {}
    for q in stable_wrong:
        released = before_repair(q, run, ctx, judged[q], repairs) if judged[q]["read_damage"] else []
        detail[q] = dict(group=ctx["group"][q], first_diff=first_diff(reqs[q]["ids"], ref[q][0]), **judged[q],
                         released=released, verdict=verdict_of(judged[q], released),
                         reads=reads_of(q, run, ctx, prefix))
    damage = [dict(fault=w["rid"], kind=w["kind"], pages=_show(_ranges(w["pages"])),
                   struck_s=w["start"] - lo, last_found_s=w["end"] - lo if w["found"] else None,
                   readers=[q for q in qids if any(h["fault"] == w["rid"] for h in judged[q]["damage_read"])])
              for w in windows]
    seconds = sorted(x["seconds"] for x in batched)
    stall = longest_stall(run)
    timing = dict(
        makespan_s=makespan, ttft_p50_s=pct([reqs[q]["ttft"] for q in qids], 0.5),
        ttft_p99_s=pct([reqs[q]["ttft"] for q in qids], 0.99), e2e_p50_s=pct([reqs[q]["e2e"] for q in qids], 0.5),
        e2e_p99_s=pct([reqs[q]["e2e"] for q in qids], 0.99),
        out_tok_per_s=sum(len(reqs[q]["ids"]) for q in qids) / makespan,
        req_per_s=sum(1 for q in qids if reqs[q]["finish"] not in ("abort", "error")) / makespan,
        goodput_req_per_s=len(right) / makespan,
        check_ms_median=1e3 * statistics.median(seconds) if seconds else None)
    warm = stall < ctx["stall_s"]
    return dict(
        mode=MODES[code[0]], with_faults=code in FAULT_RUNS, requests=len(qids), warm=warm, longest_stall_s=stall,
        **(timing if warm else {k: None for k in TIMING}), cold_timing=None if warm else timing,
        output_tokens=sum(len(reqs[q]["ids"]) for q in qids), right=len(right),
        wrong=[q for q in qids if q not in right], stable_wrong=stable_wrong,
        stable_wrong_read_damage=[q for q in stable_wrong if judged[q]["read_damage"]],
        stable_wrong_before_repair=[q for q in stable_wrong if detail[q]["verdict"] == "wrong before the repair"],
        stable_diverged_after_repair=[q for q in stable_wrong if detail[q]["verdict"] == "diverged after the repair"],
        stable_varied_before_damage=[q for q in stable_wrong if detail[q]["verdict"] == "varied before the damage"],
        stable_varied_without_damage=[q for q in stable_wrong if detail[q]["verdict"] == "varied without damage"],
        stable_wrong_detail=detail, damage=damage,
        aborts=sum(1 for x in attempts if x["finish"] == "abort"), resubmissions=len(attempts) - len(qids),
        aborted_again=sorted(x["rid"] for x in attempts if x["finish"] == "abort" and "." in x["rid"]),
        unfinished=[q for q in qids if reqs[q]["finish"] == "abort"],
        resubmission_cached_tokens={x["rid"]: x.get("cached") for x in attempts if "." in x["rid"]},
        errors=[dict(rid=x["rid"], finish=x["finish"], error=x.get("error"), lost=x.get("lost")) for x in attempts
                if x["finish"] == "error" or x.get("lost")],
        checks=checks, check_steps=len(batched), batch_steps=steps, checks_per_step=checks / steps if steps else None,
        dropped_tokens=sum(x.get("dropped_tokens", 0) for x in window if x["op"] == "drop_poisoned")
        + sum(d["dropped"] or 0 for d in detections),
        faults=faults, detections=detections)


def summarize(src, out, a, runs, meta, order, offsets, faults, plan):
    """Judge the runs, from the rank logs in `src`; write summary.json and arrivals.md to `out`."""
    logs = []
    for path in sorted(glob.glob(os.path.join(src, "rank*.jsonl"))):
        with open(path) as fh:
            logs += [json.loads(line) for line in fh]
    log0 = [x for x in logs if x.get("rank") == 0]
    qids = [q for q, _ in order]
    clean = [c for c in runs if c not in FAULT_RUNS]
    ref = {q: [runs[c]["requests"][q]["ids"] for c in clean] for q in qids}
    vary = [q for q in qids if any(x != ref[q][0] for x in ref[q])]
    stable = [q for q in qids if q not in vary]
    reads = {x["rid"]: {p for a_, b in x["pages"] for p in range(a_, b + 1)} for x in log0 if x["op"] == "read_pages"}
    read_span = {x["rid"]: (x.get("since"), x.get("time")) for x in log0 if x["op"] == "read_pages"}
    by_key = {}  # the same release_check record on every rank
    for x in logs:
        if x["op"] == "release_check":
            by_key.setdefault((x.get("rid"), x.get("step"), x.get("n_out")), {})[x.get("rank", 0)] = x
    passes, finish_checks = {}, {}  # when each request's checks passed (for records without a stream timeline)
    for x in log0:
        if x["op"] == "release_checks":
            for rid in x.get("passed", []):
                passes.setdefault(rid, []).append(x["time"])
        elif x["op"] == "release_check" and "releasable" in x:
            if x.get("finishing"):
                finish_checks[x["rid"]] = x["time"]
            else:
                passes.setdefault(x["rid"], []).append(x["time"])
    for times in passes.values():
        times.sort()
    ctx = dict(qids=qids, ref=ref, stable=stable, log0=log0, plan=plan, reads=reads, read_span=read_span,
               by_key=by_key, passes=passes, finish_checks=finish_checks, every=a.every,
               shared=a.shared, stall_s=getattr(a, "stall_s", 2.0),
               group={q: int(n[1:].split("_")[0]) for q, n in order},
               fault_of={order[i][0]: name for i, name in faults.items()})
    rows = {c: run_row(c, runs[c], ctx) for c in runs}
    hook_errors = [x for x in logs if x["op"] == "error"]
    driver_errors = [e for r in rows.values() for e in r["errors"]]

    def all_found(code):
        fr = rows.get(code, {}).get("faults", [])
        return len(fr) == len(faults) and all(f["struck"] and (f["fenced"] or f["found_by"]) for f in fr)

    s = dict(meta=meta, model=a.model, tp=a.tp, requests=a.requests, rate=a.rate, seed=a.seed, groups=a.groups,
             shared=a.shared, max_new=a.max_new, every=a.every, max_attempts=a.max_attempts,
             stall_s=ctx["stall_s"], arrival_span_s=offsets[-1],
             faults={order[i][0]: name for i, name in faults.items()}, clean_runs=clean, fault_free_variation=vary,
             stable=len(stable), pages_read="logged by the hook" if reads else
             "not in the records: judged by group (a prefix fault reaches its group; an own-page fault its request)",
             runs=rows, errors=dict(hook=hook_errors, driver=driver_errors))
    s["verdict"] = {
        "no_errors": not hook_errors and not driver_errors,
        # a stable request with a wrong text fails only if its first wrong released token left after damage it read
        # struck and before the repair (verdict_of); one that varied before the damage, diverged after the repair, or
        # varied without damage is reported
        "strict_sends_no_wrong_output_on_stable_requests": "sf" in rows and not rows["sf"]["stable_wrong_before_repair"],
        "retry_sends_no_wrong_output_on_stable_requests": "rf" in rows and not rows["rf"]["stable_wrong_before_repair"],
        "every_unfenced_fault_found_strict": all_found("sf"),
        "every_unfenced_fault_found_retry": all_found("rf"),
    }
    s["observations"] = {  # reported, not pass checks
        "fast_wrong_on_stable": len(rows["ff"]["stable_wrong"]) if "ff" in rows else None,
        "retry_wrong_on_stable": len(rows["rf"]["stable_wrong"]) if "rf" in rows else None,
        "wrong_before_the_repair": {c: r["stable_wrong_before_repair"] for c, r in rows.items()
                                    if r["stable_wrong_before_repair"]},
        "diverged_after_the_repair": {c: {q: "; ".join(w for x in r["stable_wrong_detail"][q]["released"]
                                                       for w in x["repair"])
                                          for q in r["stable_diverged_after_repair"]}
                                      for c, r in rows.items() if r["stable_diverged_after_repair"]},
        "varied_before_the_damage": {c: r["stable_varied_before_damage"] for c, r in rows.items()
                                     if r["stable_varied_before_damage"]},
        "varied_without_damage": {c: r["stable_varied_without_damage"] for c, r in rows.items()
                                  if r["stable_varied_without_damage"]},
        "cold_runs_longest_stall_s": {c: round(r["longest_stall_s"], 2) for c, r in rows.items() if not r["warm"]},
        "fenced_write_dropped": all(f["written"] == 0 for c in ("sf", "rf") if c in rows
                                    for f in rows[c]["faults"] if f["fenced"]),
        "unfinished": {c: r["unfinished"] for c, r in rows.items() if r["unfinished"]},
        "retry_resubmissions_aborted_again": rows["rf"]["aborted_again"] if "rf" in rows else None,
    }
    os.makedirs(out, exist_ok=True)
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
             "| Wrong on stable: before the repair / diverged after it / varied before the damage / without damage "
             "| Right | Aborts / resubmissions | Checks per step |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in s["runs"].items():
        cps = "" if r["checks_per_step"] is None else f"{r['checks_per_step']:.2f}"
        if r["warm"]:
            timing = (f"{ms(r['ttft_p50_s'])} / {ms(r['ttft_p99_s'])} | {sec(r['e2e_p50_s'])} / {sec(r['e2e_p99_s'])} "
                      f"| {r['out_tok_per_s']:.0f} | {r['goodput_req_per_s']:.2f}")
        else:
            timing = " | ".join([f"cold ({r['longest_stall_s']:.1f} s stall)"] + ["cold"] * 3)
        lines.append(f"| {c} | {r['mode']} | {'yes' if r['with_faults'] else 'no'} | {timing} | "
                     f"{len(r['stable_wrong_before_repair'])} / {len(r['stable_diverged_after_repair'])} / "
                     f"{len(r['stable_varied_before_damage'])} / {len(r['stable_varied_without_damage'])} | "
                     f"{r['right']} of {r['requests']} | {r['aborts']} / {r['resubmissions']} | {cps} |")
    cold = {c: r["longest_stall_s"] for c, r in s["runs"].items() if not r["warm"]}
    if cold:
        lines += ["", f"Cold runs (the server stood still for over {s['stall_s']:g} s): "
                  + ", ".join(f"{c} {v:.1f} s" for c, v in cold.items()) + "; their timing is not reported."]
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
    lines += ["", f"Damage, and the requests that read it (pages read: {s['pages_read']}):", "",
              "| Run | Fault | Pages | Struck at (s) | Last found at (s) | Requests that read it |", "|---|---|---|---|---|---|"]
    for c in FAULT_RUNS:
        for d in s["runs"].get(c, {}).get("damage", []):
            last = "never found" if d["last_found_s"] is None else f"{d['last_found_s']:.2f}"
            lines.append(f"| {c} | {d['kind']} ({d['fault']}) | {d['pages']} | {d['struck_s']:.2f} | {last} | "
                         f"{', '.join(d['readers']) or 'none'} |")
    lines += ["", "Stable requests with a wrong text (released before the repair: the tokens the release rule had cleared, "
              "or the stream had delivered, when the detection that covered the damage ran; an aborted attempt: all it "
              "released):", "",
              "| Run | Request | Group | First differing token | Read damage | Released before the repair | Repair "
              "| Judged | Checks | Pages read |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in s["runs"].items():
        for q, d in r["stable_wrong_detail"].items():
            dmg = ("; ".join(f"{h['kind']} pages {_show(_ranges(h['pages']))}" for h in d["damage_read"])
                   or ("a check failed" if d["failed_checks"] else "no"))
            def at(x):  # when the first wrong token left, and when the damage it read struck (s from the run's start)
                t, s_ = x["first_wrong_released_s"], x["damage_struck_s"]
                return ((f", left at {t:.2f} s" if t is not None else ", left at an unknown time")
                        + ("" if x["released_from"] == "stream" else f" ({x['released_from']})")
                        + (f"; damage struck at {s_:.2f} s" if s_ is not None else ""))

            rel = "; ".join(f"{x['rid']}: {x['released_before_repair']}"
                            + (" (all right)" if x["first_wrong_released"] is None
                               else f" (token {x['first_wrong_released']} wrong{at(x)})")
                            + ("" if x["point"] == "repaired" else f", {x['point']}") for x in d["released"])
            how = "; ".join(w for x in d["released"] for w in x["repair"])
            chk = ("none (fast)" if r["mode"] == "fast" else
                   f"failed: {', '.join(d['failed_checks'])}" if d["failed_checks"] else "all passed")
            reads = d["reads"] if isinstance(d["reads"], str) else "; ".join(f"{k}: {v}" for k, v in d["reads"].items())
            lines.append(f"| {c} | {q} | {d['group']} | {d['first_diff']} | {dmg} | {rel} | {how} | {d['verdict']} | "
                         f"{chk} | {reads} |")
    lines += ["", "Verdict: " + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in s["verdict"].items()) + "."]
    return "\n".join(lines) + "\n"


def report(s, a):
    print(json.dumps(s["verdict"], indent=1))
    print(json.dumps(s["observations"], indent=1, default=str))
    print(arrivals_md(s), flush=True)
    if a.require_pass and not all(s["verdict"].values()):
        print("PASS CHECKS FAILED:", [k for k, ok in s["verdict"].items() if not ok], flush=True)
        raise SystemExit(3)


def resummarize(a, out):
    """Judge the records of an earlier run (its folder `a.resummarize`) with this code."""
    src = os.path.abspath(a.resummarize)

    def load(name):
        with open(os.path.join(src, name)) as fh:
            return json.load(fh)

    old, runs, plan, arr = load("summary.json"), load("outputs.json"), load("plan.json"), load("arrivals.json")
    for k in ("model", "tp", "requests", "rate", "seed", "groups", "shared", "max_new", "every", "max_attempts"):
        setattr(a, k, old[k])
    order = [tuple(x) for x in arr["order"]]
    qids = [q for q, _ in order]
    faults = {qids.index(q): name for q, name in arr["faults"].items()}
    report(summarize(src, out, a, runs, old.get("meta"), order, arr["offsets"], faults, plan), a)


# ---------------------------------------------------------------------------------------------------------------

def main():
    a = parse()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    if a.resummarize:
        return resummarize(a, out)
    a.rank = min(a.rank, a.tp - 1)
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
                      KVMEND_LOG_READS="1",
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
    runs, warmup = {}, None
    try:
        flush(engine)
        # warm-up, not counted: one request per group, checked, so the check kernels compile before the timed runs;
        # then the whole workload once (wu, fast, no faults), for the kernels the arriving batches compile
        run_batch(engine, "w", {f"g{g}_r0": P[f"g{g}_r0"] for g in range(a.groups)}, a)
        for code in ["wu"] + codes:
            label(code)
            flush(engine)
            faulthandler.dump_traceback_later(a.run_timeout, exit=True)
            warm_prefixes(engine, code, P, a)
            res = drive(engine, code, order, P, offsets, a)
            faulthandler.cancel_dump_traceback_later()
            if code == "wu":
                warmup = res
            else:
                runs[code] = res
            reqs = res["requests"].values()
            print(f"run {code}: {sum(len(r['ids']) for r in reqs)} output tokens in {res['seconds']:.2f} s, "
                  f"{sum(len(r['attempts']) - 1 for r in reqs)} resubmissions, longest stall "
                  f"{longest_stall(res):.2f} s" + (" (warm-up, not timed)" if code == "wu" else ""), flush=True)
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
    with open(os.path.join(out, "warmup.json"), "w") as fh:
        json.dump(warmup, fh, indent=1, default=str)
    report(summarize(out, out, a, runs, meta, order, offsets, faults, plan), a)


if __name__ == "__main__":
    main()
