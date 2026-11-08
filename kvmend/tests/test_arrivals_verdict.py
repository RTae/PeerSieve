"""run_arrivals.py's verdict tells a stable request whose text changed after it read damage from one whose text varied
without damage: from the pages the hook logged, or, in records without them, by group. And a run in which the server
stood still is cold (CPU only, no SGLang)."""
import run_arrivals as RA

W0 = 1000.0  # the run's start, wall clock


def _req(qid, submit, end, ids=(1, 2, 3)):
    return dict(qid=qid, arrive=submit, e2e=end - submit, ttft=0.01, ids=list(ids), finish="length",
                attempts=[dict(rid=f"sf_{qid}", submit=submit, first=submit + 0.01, end=end, finish="length")])


def _ctx(reads, read_span=None):
    log0 = [  # a late write into group 0's prefix pages 10-11 at 2.0 s, found and mended at 2.5 s; a fenced write in
        # group 1's prefix that the fence dropped; another sequence's KV in q003's own page 40, found at 3.25 s
        dict(op="fault", rid="sf_q000", kind="late_bypass", pages=[10, 11], start=960, time=W0 + 2.0),
        dict(op="release_check", rid="sf_q000", found_pieces=4, found={"10": {}, "11": {}}, time=W0 + 2.5),
        dict(op="fault", rid="sf_q001", kind="late_store", pages=[20, 21], start=992, fenced=True, written_tokens=0,
             time=W0 + 2.125),
        dict(op="fault", rid="sf_q003", kind="foreign", pages=[40], start=1024, time=W0 + 3.0),
        dict(op="release_check", rid="sf_q003", found_pieces=2, found={"40": {}}, time=W0 + 3.25),
        dict(op="release_check", rid="sf_q005", found_pieces=1, found={"77": {}}, time=W0 + 4.0)]
    group = {"q000": 0, "q001": 1, "q002": 0, "q003": 1, "q004": 0, "q005": 1, "q006": 0, "q008": 0}
    return dict(log0=log0, reads=reads, read_span=read_span or {}, group=group, shared=1024)


def _run():
    return dict(wall=[W0, W0 + 10.0], requests={
        "q000": _req("q000", 1.0, 3.0),   # the request the late write struck
        "q001": _req("q001", 1.5, 3.0),   # group 1, whose only write the fence dropped
        "q002": _req("q002", 2.2, 2.8),   # group 0, ran while pages 10-11 were damaged
        "q003": _req("q003", 2.6, 3.5),   # another sequence's KV in its own page
        "q004": _req("q004", 4.0, 5.0),   # group 0, arrived after the repair
        "q005": _req("q005", 3.5, 4.5),   # a check of it failed
        "q006": _req("q006", 1.8, 2.1)})  # group 0, but it read other pages (a reused copy)


def test_logged_pages_decide_who_read_damage():
    reads = {"sf_q000": set(range(1, 13)), "sf_q001": set(range(13, 25)), "sf_q002": set(range(1, 13)) | {50},
             "sf_q003": set(range(13, 25)) | {40}, "sf_q004": set(range(1, 13)) | {60},
             "sf_q005": set(range(13, 25)), "sf_q006": set(range(80, 92))}
    ctx, run = _ctx(reads), _run()
    windows = RA.damage_windows("sf", run, ctx)
    assert [(w["kind"], sorted(w["pages"]), w["end"] - W0, w["own"]) for w in windows] == [
        ("late_bypass", [10, 11], 2.5, False), ("foreign", [40], 3.25, True)]  # the fenced write changed nothing
    judged = {q: RA.judge(q, run, ctx, windows, {"sf_q005"}) for q in run["requests"]}
    assert {q: j["read_damage"] for q, j in judged.items()} == {
        "q000": True, "q001": False, "q002": True, "q003": True, "q004": False, "q005": True, "q006": False}
    assert judged["q002"]["damage_read"] == [dict(rid="sf_q002", fault="sf_q000", kind="late_bypass", pages=[10, 11])]
    assert judged["q005"]["failed_checks"] == ["sf_q005"] and judged["q004"]["pages_logged"]
    # q008 was submitted at 2.4 s, while pages 10-11 were damaged, but its first check (right after its prefill)
    # came at 2.6 s, after the repair: it read the mended pages. Without that time it would count as a reader.
    run["requests"]["q008"] = _req("q008", 2.4, 3.0)
    reads["sf_q008"] = set(range(1, 13))
    for span, want in ((None, True), ((W0 + 2.6, W0 + 3.0), False)):
        ctx = _ctx(reads, {"sf_q008": span} if span else None)
        assert RA.judge("q008", run, ctx, windows, set())["read_damage"] is want, span


def test_records_without_pages_are_judged_by_group():
    ctx, run = _ctx({}), _run()
    windows = RA.damage_windows("sf", run, ctx)
    judged = {q: RA.judge(q, run, ctx, windows, set()) for q in run["requests"]}
    # group 0's requests that ran between 2.0 and 2.5 s read the late write, q006 included (no pages to say
    # otherwise); the own-page fault reaches only q003; group 1 had no damage
    assert {q: j["read_damage"] for q, j in judged.items()} == {
        "q000": True, "q001": False, "q002": True, "q003": True, "q004": False, "q005": False, "q006": True}
    assert not judged["q004"]["pages_logged"]


REF = list(range(100, 164))  # the fault-free text of every request below


def _att(rid, ids, finish="length", released=None, submit=1.0, end=2.0):
    return dict(rid=rid, submit=submit, first=submit + 0.01, end=end, finish=finish, ids=list(ids),
                **({} if released is None else dict(released=released)))


def _repair_ctx():
    """sf_q012's check at output 16 found a burst on rank 1, which rebuilt 16 pieces bit for bit (rank 0's record
    lists the pieces but does not mark them), kept 12 tokens and regenerated 4; sf_q016, another reader, kept 8 and
    regenerated 2. rf_q020's check aborted it."""
    pieces = [dict(rank=1, layer=1, page=2 + i % 8) for i in range(16)]
    rank0 = dict(op="release_check", rid="sf_q012", step=50, n_out=16, rank=0, found_pieces=16, found={"2": {}},
                 time=W0 + 1.64, mend=dict(rebuilt=pieces, recomputed=[], max_abs_diff_vs_clean=0.0),
                 regenerate=dict(retracted=True, regenerated=4, kept_outputs=12),
                 other_readers=[dict(rid="sf_q016", retracted=True, regenerated=2, kept_outputs=8)])
    rank1 = dict(rank0, rank=1, mend=dict(rebuilt=[dict(p, rebuilt=True, bit_exact=True) for p in pieces],
                                          recomputed=[], max_abs_diff_vs_clean=0.0))
    abort = dict(op="release_check", rid="rf_q020", step=70, n_out=16, rank=0, found_pieces=4, found={"9": {}},
                 time=W0 + 1.7, aborted=True, poisoned_pages=[9], other_readers=[])
    by_key = {("sf_q012", 50, 16): {0: rank0, 1: rank1}, ("rf_q020", 70, 16): {0: abort}}
    return dict(log0=[rank0, abort], by_key=by_key, ref={q: [REF] for q in ("q012", "q016", "q020", "q024")})


def test_a_rebuild_is_counted_on_the_rank_that_did_it():
    ctx = _repair_ctx()
    assert RA.repair_of(ctx["log0"][0], ctx) == dict(rebuilt=16, rebuilt_bit_exact=16, recopied=0, recomputed=0,
                                                     max_abs_diff_vs_clean=0.0)
    reps = RA.repairs_of("sf", ctx)
    assert [(r["kept"], r["how"]) for r in reps["sf_q012"]] == [(12, ["rebuilt bit-exact (16 pieces)",
                                                                       "regenerated 4 tokens"])]
    assert [(r["kept"], r["how"]) for r in reps["sf_q016"]] == [(8, ["rebuilt bit-exact (16 pieces)",
                                                                      "regenerated 2 tokens"])]
    assert [(r["kept"], r["aborted"], r["how"]) for r in RA.repairs_of("rf", ctx)["rf_q020"]] == [
        (None, True, ["aborted, not mended; resubmitted"])]


def test_only_tokens_released_before_the_repair_decide():
    ctx = _repair_ctx()
    run = dict(wall=[W0, W0 + 10.0], requests={
        # its text changes at token 55, long after the 12 tokens released before the repair
        "q012": dict(attempts=[_att("sf_q012", REF[:55] + [0] * 9)]),
        # token 5, inside the 8 released before the repair, is wrong
        "q016": dict(attempts=[_att("sf_q016", REF[:5] + [0] * 59)]),
        # the stream delivered 13 tokens before the detection at 1.64 s, one more than the release rule had cleared,
        # and token 12 is wrong
        "q024": dict(attempts=[_att("sf_q012", REF[:12] + [0] * 52, released=[[1.2, 8], [1.5, 13], [1.9, 64]])])})
    sf = RA.repairs_of("sf", ctx)
    read = dict(read_damage=True, damage_read=[], failed_checks=[])
    for q, rid, want in (("q012", "sf_q012", "diverged after the repair"), ("q016", "sf_q016", "wrong before the repair"),
                         ("q024", "sf_q012", "wrong before the repair")):
        judged = dict(read, failed_checks=[rid])
        released = RA.before_repair(q, run, ctx, judged, sf)
        assert RA.verdict_of(judged, released) == want, (q, released)
    assert RA.before_repair("q024", run, ctx, dict(read, failed_checks=["sf_q012"]), sf)[0][
        "released_before_repair"] == 13
    assert RA.verdict_of(dict(read_damage=False, damage_read=[], failed_checks=[]), []) == "varied without damage"


def test_an_aborted_attempt_is_judged_on_what_it_released():
    ctx = _repair_ctx()
    rf = RA.repairs_of("rf", ctx)
    judged = dict(read_damage=True, damage_read=[], failed_checks=["rf_q020"])

    def verdict(first_ids):
        run = dict(wall=[W0, W0 + 10.0], requests={"q020": dict(attempts=[
            _att("rf_q020", first_ids, finish="abort", end=1.7), _att("rf_q020.1", REF[:30] + [0] * 34, submit=1.7,
                                                                      end=2.4)])})
        return RA.verdict_of(judged, RA.before_repair("q020", run, ctx, judged, rf))

    assert verdict(REF[:12]) == "diverged after the repair"  # the resubmission's text changes at token 30
    assert verdict(REF[:3] + [0] * 9) == "wrong before the repair"


def test_damage_with_no_repair_counts_every_token():
    run = dict(wall=[W0, W0 + 10.0], requests={"q012": dict(attempts=[_att("ff_q012", REF[:40] + [0] * 24)])})
    judged = dict(read_damage=True, damage_read=[dict(rid="ff_q012")], failed_checks=[])
    released = RA.before_repair("q012", run, dict(ref={"q012": [REF]}), judged, {})
    assert released == [dict(rid="ff_q012", released_before_repair=64, point="never repaired",
                             first_wrong_released=40, repair=[])]
    assert RA.verdict_of(judged, released) == "wrong before the repair"


def test_a_run_in_which_the_server_stood_still_is_cold():
    def run(marks):
        return dict(requests={f"q{i}": dict(attempts=[dict(submit=s, first=f, end=e)]) for i, (s, f, e) in
                              enumerate(marks)})

    steady = run([(0.1 * i, 0.1 * i + 0.03, 0.1 * i + 0.4) for i in range(20)])
    stalled = run([(0.0, 0.02, 41.0), (0.3, 40.6, 41.2), (5.0, 40.7, 41.3)])
    idle = run([(0.0, 0.02, 0.4), (30.0, 30.02, 30.4)])  # nothing in flight between them: not a stall
    assert RA.longest_stall(steady) < 0.2
    assert abs(RA.longest_stall(stalled) - 40.58) < 1e-6
    assert RA.longest_stall(idle) < 0.5
