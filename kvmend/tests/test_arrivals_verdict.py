"""run_arrivals.py's verdict tells a stable request whose text changed after it read damage from one whose text varied
without damage: from the pages the hook logged, or, in records without them, by group. And a run in which the server
stood still is cold (CPU only, no SGLang)."""
import run_arrivals as RA

W0 = 1000.0  # the run's start, wall clock


def _req(qid, submit, end, ids=(1, 2, 3)):
    return dict(qid=qid, arrive=submit, e2e=end - submit, ttft=0.01, ids=list(ids), finish="length",
                attempts=[dict(rid=f"sf_{qid}", submit=submit, first=submit + 0.01, end=end, finish="length")])


def _ctx(reads):
    log0 = [  # a late write into group 0's prefix pages 10-11 at 2.0 s, found and mended at 2.5 s; a fenced write in
        # group 1's prefix that the fence dropped; another sequence's KV in q003's own page 40, found at 3.25 s
        dict(op="fault", rid="sf_q000", kind="late_bypass", pages=[10, 11], start=960, time=W0 + 2.0),
        dict(op="release_check", rid="sf_q000", found_pieces=4, found={"10": {}, "11": {}}, time=W0 + 2.5),
        dict(op="fault", rid="sf_q001", kind="late_store", pages=[20, 21], start=992, fenced=True, written_tokens=0,
             time=W0 + 2.125),
        dict(op="fault", rid="sf_q003", kind="foreign", pages=[40], start=1024, time=W0 + 3.0),
        dict(op="release_check", rid="sf_q003", found_pieces=2, found={"40": {}}, time=W0 + 3.25),
        dict(op="release_check", rid="sf_q005", found_pieces=1, found={"77": {}}, time=W0 + 4.0)]
    group = {"q000": 0, "q001": 1, "q002": 0, "q003": 1, "q004": 0, "q005": 1, "q006": 0}
    return dict(log0=log0, reads=reads, group=group, shared=1024)


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


def test_records_without_pages_are_judged_by_group():
    ctx, run = _ctx({}), _run()
    windows = RA.damage_windows("sf", run, ctx)
    judged = {q: RA.judge(q, run, ctx, windows, set()) for q in run["requests"]}
    # group 0's requests that ran between 2.0 and 2.5 s read the late write, q006 included (no pages to say
    # otherwise); the own-page fault reaches only q003; group 1 had no damage
    assert {q: j["read_damage"] for q, j in judged.items()} == {
        "q000": True, "q001": False, "q002": True, "q003": True, "q004": False, "q005": False, "q006": True}
    assert not judged["q004"]["pages_logged"]


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
