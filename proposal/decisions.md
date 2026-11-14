# Decisions

> [!info] About this note
> Owner replies are quoted exactly, with the row text of the question each one answers. Entries are grouped by date; use Obsidian's Outline pane to jump between decisions.
>
> Related note: [[PROPOSAL|KVMend proposal]]

## 2026-11-05

### Testing on cluster ja · 00:02 CST

> [!quote] Owner · Mid-turn message
> "to test for now you can use scp to transfer a file into yes cluster and run with slrum"

**Context:** This answers the request to test KVMend on cluster ja through the login node yes.

### Decision 6 · 00:02 CST

> [!quote] Owner
> "yes can do it. you can create a folder call kvmed and use uv to init and then use uv to install package i think i already install uv"

**Downloads on cluster ja:** One CPU-only Slurm job on octave (no GPU, ≤1 hour, 4 CPUs, 16 GB RAM) that:

- installs SGLang 0.5.1 with the Tsinghua PyPI mirror;
- downloads model weights from hf-mirror.com into `/home/tae/models`.

### Decision 7 · 00:02 CST

> [!quote] Owner
> "Both jobs (Recommended)"

**GPU test jobs on octave** (partition `a100`, account `a100`):

1. A smoke test of the KVMend SGLang hook on 4 A100s with TinyLlama, ≤30 min (≤2 GPU-hours).
2. After that, the running example on 8 A100s with Llama-3.1-8B at TP=8 and a 32K prompt, ≤50 min (≤6.7 GPU-hours).

**Total:** ≤ about 9 GPU-hours. **Start time:** whenever the GPUs free up.

### Cluster control plane · 00:08 CST

> [!quote] Owner · Mid-turn message
> "btw you know the rule right you need to use login node as a control plane and use slrum to submit a job to octave or ja clauster"

### Login-node installs · 00:08 CST

> [!quote] Owner · Mid-turn message
> "also for installing staff you can use login node to install directilly no need to submit a task"

### Decision 8 · 09:21 CST

> [!quote] Owner
> "Yes, one more attempt (Recommended)"

One more attempt of the 4-GPU smoke test (TinyLlama on octave), capped at 30 minutes (≤2 GPU-hours), starting when the GPUs free up. The two attempts so far used 2.27 GPU-hours (0.27 + 2.0).

### Decision 9 · 12:49 CST

> [!quote] Owner
> "Yes, and the 8-GPU run if it passes"

One more 4-GPU smoke test attempt (TinyLlama on octave), ≤30 minutes (≤2 GPU-hours), starting when GPUs free up.

**Option text:** "Same smoke attempt; if it passes, queue the approved 8-GPU Llama-3.1-8B run (decision 7, ≤50 min) right after, without asking again."

Smoke attempts so far total 2.34 GPU-hours.

### Decision 10 · 14:50 CST

> [!quote] Owner
> "Yes, and the 8-GPU run if it passes (Recommended)"

One more 4-GPU smoke attempt (TinyLlama, ≤30 min, ≤2 GPU-hours) when GPUs free up.

**Option text:** "Same as decision 9: the smoke attempt with an automatic pass check, and the approved 8-GPU Llama-3.1-8B run (≤50 min) chained to start only if every check passes."

Smoke attempts so far total 2.41 GPU-hours.

### Decision 11 · 18:03 CST

> [!quote] Owner
> "Smoke again, then 8-GPU (Recommended)"

**Option text:** "One more 4-GPU TinyLlama run (≤30 min, about 2 min of real work) to check the new comparison on GPUs, with the 8-GPU Llama-3.1-8B run (≤50 min) chained to start only if every check passes."

Smoke attempts so far total 2.48 GPU-hours.

## 2026-11-06

### Decision 12 · 11:30 CST

> [!quote] Owner
> "4-GPU check, then 8-GPU (Recommended)"

**Comparison with GhostServe on octave:** GhostServe re-implemented in our SGLang patch; part 1 the same fault under five approaches, part 2 fault-free throughput.

**Option text:** "A 4-GPU TinyLlama check of the new code (≤30 min, ≤2 GPU-hours), chained to the 8-GPU Llama-3.1-8B comparison (≤50 min, ≤6.7 GPU-hours) only if it passes. Total ≤8.7 GPU-hours, starting when the code is ready and the GPUs free up."

### Decision 13 · 12:54 CST

> [!quote] Owner
> "yes"

Add the comparison results to the proposal as a "Preliminary results" section (no GPU time).

### Decision 14 · 13:07 CST

> [!quote] Owner
> "yes"

Split the merged abstract paragraph of `PROPOSAL.md` back into short paragraphs, with no wording changes, so the length check passes.

### Decision 15 · 14:20 CST

> [!quote] Owner
> "a"

**Option a:** Rewrite the design and evaluation text of the proposal with fixes 1–6 and the smaller fixes from the re-read; no GPU time.

### Decision 16 · 14:27 CST

> [!quote] Owner
> "yes"

Add the attention-kernel check (checks on read) and the release rule to Figures 5 and 9 of the proposal.

### Decision 17 · 14:40 CST

> [!quote] Owner
> "a"

**Option a:** Reframe the proposal as a correctness watchdog for the KV cache and keep the name KVMend:

- title and abstract in the four-part style, with the Problem paragraph unchanged;
- introduction and contributions around watch, guard and mend;
- liveness watchdogs added to related work;
- section titles and Figure 1 step labels in the new words;
- mechanisms, evaluation, results and other figures stay.

### Decision 18 · 19:13 CST

> [!quote] Owner
> "b"

**Option b:** Rewrite the text (section 7 as a feasibility test, GhostServe rows reported qualitatively, abort-and-retry as the main baseline, fairness rules, the additive tag), then one new proof-of-concept run of the fair test:

- KVMend with strict release against abort-and-retry, both returning the right text;
- fluent-but-wrong flips and KVMend without parity;
- about 3–4 GPU-hours on octave, launch to be asked separately.

### Decision 19 · 19:52 CST

> [!quote] Owner
> "a"

**Option a:** Submit now the 4-GPU TinyLlama check of the fair test on octave (at most 15 min, 1.0 GPU-hour), chained to the 8-GPU Llama-3.1-8B fair test (at most 25 min, 3.3 GPU-hours), which starts only if the check passes and all 8 cards are free.

**Total:** at most 4.3 GPU-hours.

## 2026-11-07

### Decision 20 · 10:07 CST

> [!quote] Owner
> "we can fix all and you can pioritize it which one should come first"

**Question:** Which limitation to tackle first.

**Options:**

- **a)** The motivating fault: build the three missing checks (slot generations, token keys, tag coverage) and test the incident's faults.
- **b)** The fast checks: the fused checker first, then the release rule.
- **c)** Write the limitations into section 8 only, and leave the code as it is.

**Limitations listed:**

1. The motivating fault is untested.
2. The fast checks are not built.
3. Parity's value is unproven.
4. Parity needs a model split across GPUs.
5. The tests are light.
6. Wrong computation (state only).
7. Our hardware (state only).

### Decision 21 · 11:41 CST

> [!quote] Owner
> "a"

**Option a:** Submit now a 2-GPU TinyLlama check (TP=2) of the incident test on the two free A100s of octave, at most 15 minutes (at most 0.5 GPU-hours), chained to the 8-GPU Llama-3.1-8B incident run, at most 30 minutes (at most 4.0 GPU-hours), which starts only if the check passes and all 8 cards are free (possibly not before 17:47 CST).

**Total:** at most 4.5 GPU-hours.

### Decision 22 · 14:20 CST

> [!quote] Owner
> "a"

**Option a:** Submit now the 2-GPU TinyLlama check, at most 25 minutes (at most 0.85 GPU-hours), which runs:

1. the CPU tests;
2. the incident test;
3. then, not gating, the GPU kernel tests and the streaming release-rule test.

Chain it to the 8-GPU Llama-3.1-8B incident run, at most 30 minutes (at most 4.0 GPU-hours), which starts only if the check passes and all 8 cards are free (possibly not before 17:46 CST).

**Total:** at most 4.85 GPU-hours.

### Decision 23 · 14:43 CST

> [!quote] Owner
> "a"

**Option a:** Submit now the full test chain on octave, each 8-GPU step running only if the same step passed on TinyLlama:

1. **Check job:** 2 GPUs, TinyLlama, at most 45 minutes (at most 1.5 GPU-hours), running every test once plus the stripes-within-one-GPU test on Llama-3.1-8B on a single GPU.
2. **Main job:** 8 GPUs, Llama-3.1-8B at TP=8, at most 50 minutes (at most 6.7 GPU-hours): the incident test, the release rule, the burst sweep and the loaded test, only if the check job's gate passes.
3. **Benchmark job:** 8 GPUs, at most 45 minutes (at most 6 GPU-hours): checker speed, fault-free throughput and time per output token, after the main job ends and only if the check job passed.

**Expected:** about 8 GPU-hours. **Maximum:** 14.2 GPU-hours.

### Decision 24 · 14:58 CST

> [!quote] Owner
> "a"

**Option a:** Reframe the proposal with:

- cheap always-on checks plus repair by recomputing only the damaged pages as the core;
- the guard (holding tokens back one step) as an option for requests that must not be wrong;
- parity (across GPUs or within one) as an optional extra for bursts and very long prompts.

The motivation becomes rare but silent damage, more likely as KV moves between GPUs, host memory and machines.

### Decision 25 · 15:46 CST

> [!quote] Owner
> "a"

**Option a:** Replace the two waiting 8-GPU jobs (326400, 326401):

1. Cancel them and stage the fixed code.
2. Run a new 2-GPU check now on the free cards: the full suite on TinyLlama plus the one-page stripe rebuild on Llama-3.1-8B on one GPU; at most 45 minutes, at most 1.5 GPU-hours.
3. Resubmit the main job (at most 50 minutes, at most 6.7 GPU-hours, now with the release-rule and loaded tests) and the benchmark job (at most 45 minutes, at most 6 GPU-hours), both starting when all 8 cards free up, possibly not before 23:00.

**Expected:** about 9 GPU-hours. **Maximum:** 14.2 GPU-hours.

### Decision 26 · 20:58 CST

> [!quote] Owner
> "a"

**Option a:** After both 8-GPU jobs (326412, 326413) end, copy the new check (all layers in one kernel launch) to the cluster, then run one 2-GPU job on octave:

- the CPU tests;
- the GPU kernel tests;
- the checker speed test;
- the streamed release-rule test on TinyLlama.

At most 25 minutes (at most 0.85 GPU-hours, about 0.2 expected), starting when the 8-GPU jobs end, about 04:10 CST at the earliest.

## 2026-11-08

### Decision 27 · 00:37 CST

> [!quote] Owner
> "our method is practical right and for 28 b"

**Option b:**

1. Read the whole proposal through and fix what I find, with no cluster time.
2. Prepare the 8-GPU run:
   - Llama-3.1-8B;
   - the streamed test with damage that changes its text and the one-launch check;
   - the 24-request test with more fault-free runs.
3. Ask the owner to approve its launch with the exact cost.

No decision 28 was open; decision 27 was the only open one. **Next free number:** 29.

### Decisions 29–31 · 00:47 CST

> [!quote] Owner
> "29 a, 30 i will do it later, 31 a"

#### Decision 29 · Option a

Copy the three changed test scripts and submit now:

- a 2-GPU TinyLlama check (at most 20 minutes, 0.67 GPU-hours) on free cards;
- chained to the 8-GPU Llama-3.1-8B run (at most 25 minutes, 3.3 GPU-hours), which starts only if the check passes and all 8 cards are free, likely not before 07:08.

**Total:** at most 4.0 GPU-hours, about 1.6 expected.

#### Decision 30 · Deferred

Put the experiment code in git:

- **a:** create a private GitHub repository, commit and push, later runs use a named commit;
- **b:** keep copying files and recording checksums.

#### Decision 31 · Option a

Draw Figure 13 with two panels:

- **(a)** Extra time to the right text, KVMend against abort and retry, for each kind of damage.
- **(b)** Repair time against burst size, rebuilding from parity against recomputing.

Add one line to `CLAUDE.md`'s list of plot styles: a dotted blue line with hollow markers for KVMend recomputing pages.

**Next free number:** 32.

### Decision 32 · 11:35 CST

> [!quote] Owner
> "a"

Restated after job 326431 had run.

**Option a:** Copy the fixed release rule and its tests, then run a 2-GPU TinyLlama check (at most 20 minutes, 0.67 GPU-hours), chained to the streamed test on Llama-3.1-8B with 8 GPUs (at most 15 minutes, 2.0 GPU-hours), only if the check passes it. All 8 cards free now.

**Total:** at most 2.7 GPU-hours, about 0.8 expected. **Next free number:** 33.

### Decisions 33–34 · 12:02 CST

> [!quote] Owner
> "33 a, 34 a"

#### Decision 33 · Option a

Copy the four changed files and submit now:

1. A 2-GPU TinyLlama check: CPU tests and the incident test at both damage positions; at most 25 minutes, 0.83 GPU-hours.
2. Chained to it, only if its incident test passes, the 8-GPU Llama-3.1-8B run: the incident test with the damage at the end of the prompt and at page 64, each with the fair retry and the damaged tags and parity; at most 45 minutes, 6.0 GPU-hours.

**Total:** at most 6.8 GPU-hours, about 4 expected. All 8 cards free.

#### Decision 34 · Option a

Commit the four changed files as one commit, "Add a fair retry, damaged tags and parity, and a damage position to the incident test"; the runs name that commit:

- `kvmend/hook.py`
- `run_incident.py`
- `tests/test_incident_gloo.py`
- `slurm/suite.sh`

**Next free number:** 35.

### Decision 37 · 12:58 CST

> [!quote] Owner
> "a"

This was the decision my last message laid out; 35 and 36 were also open.

**Option a:** Commit `proposal/PROPOSAL.md` and `kvmend/decisions.md` as one commit, "Answer 'why not something simpler?' in the proposal".

**Next free number:** 38.

### Decisions 35, 36, 38 and 39 · 13:38 CST

> [!quote] Owner
> "35 a, 36 a, 38 a, 39 keep"

#### Decision 35 · Option a

Commit the incident test's change (19 lines in `run_incident.py`) as one commit, "Give the fair retry a fault-free control on its own path"; the rerun names that commit.

#### Decision 36 · Option a

Copy the fixed incident test and resubmit the open-parts chain:

1. A 2-GPU TinyLlama check, at most 25 minutes, 0.83 GPU-hours.
2. Chained to it, only if its incident test passes, the 8-GPU Llama-3.1-8B run with damage at the end of the prompt and at page 64, each with the fair retry and damaged tags and parity, at most 45 minutes, 6.0 GPU-hours.

**Total:** at most 6.8 GPU-hours, about 4 expected.

#### Decision 38 · Option a

Rewrite the message of commit `77cc09c` to list only what it holds (local only; no remote).

#### Decision 39 · Keep

The open parts not built yet:

- checks inside the attention kernel;
- loads from host memory that land on only some GPUs;
- requests that arrive over time.

**Suggestion:** Keep them in the proposal as planned evaluation work.

**Next free number:** 40.

### Decision 39 — Revised · 13:38 CST

> [!quote] Owner · Mid-turn message
> "39 i change my mind we can do it now"

The open parts not built yet are now approved:

- checks inside the attention kernel;
- loads from host memory that land on only some GPUs;
- requests that arrive over time.

> [!important] Supersedes the earlier reply
> This replaces the earlier "39 keep".

### Decisions 40–45 · 14:59 CST

> [!quote] Owner
> "40 a, 41 a, 42 a, 43 a, 44 a, 45 a"

#### Decision 40 · Option a

Commit the proposal changes (text, Figure 13, its data and its script) as one commit on main: "Report the fair retry by damage position, and correct claims after the final read-through".

#### Decision 41 · Option a

Keep the arrivals build's extra fix: when a strict request's rollback must wait a step, its prompt KV computed on top of the damage is now dropped, not reused.

#### Decision 42 · Option a

Land the builds: commit each on its own branch, merge all three into main resolving the 10 conflicting spots, then rerun every CPU test on the merged code.

#### Decision 43 · Option a

GPU checks from the merged commit on octave:

1. A 2-GPU TinyLlama check (CPU tests, GPU kernel tests and the three new steps), at most 45 minutes (1.5 GPU-hours).
2. Chained after it, each step only if it passed on TinyLlama, an 8-GPU Llama-3.1-8B run of the three steps, at most 55 minutes (7.3 GPU-hours).

Starts about 20 minutes after the yes on 42.

#### Decision 44 · Option a

Build now the check right after the prefill for every request, not only strict ones, and a rerun of page 64 without the test's own check. The launch is asked separately.

#### Decision 45 · Option a

Measure the tags' cost on one A100 with Llama-3.1-8B, plain against tags alone, on both workloads, on card 0, after decision 43's runs. At most 30 minutes (0.5 GPU-hours).

**Next free number:** 46.

### Decision 46 · 15:41 CST

> [!quote] Owner
> "a"

**Option a:** Run one short 8-GPU check now on octave:

- Llama-3.1-8B, decode-heavy;
- tags alone and plain taking turns (tags, plain, tags, plain);
- NCCL's setup log on;
- same commit `ae09c14`, no code change;
- at most 20 minutes (2.7 GPU-hours), expected 8–10 minutes.

**Next free number:** 47.

### Decisions 47–49 · 15:56 CST

> [!quote] Owner
> "47 a, 48 a, 49 a"

#### Decision 47 · Option a

Commit each fix on its own branch, merge all three into main, and rerun every CPU test on the merged code.

#### Decision 48 · Option a

Rerun on octave from the merged commit:

1. A 2-GPU TinyLlama check (CPU tests, the incident test at both positions using the hook's own check, requests over time, host-memory loads), at most 30 minutes (1.0 GPU-hours).
2. Chained after it, each step only if it passed, an 8-GPU Llama-3.1-8B run of page 64 with the hook's check, requests over time and host-memory loads, at most 45 minutes (6.0 GPU-hours).

Starts about 20 minutes after the yes on 47.

#### Decision 49 · Option a

Build now the fix for skipped writes during the prefill: when the check after the prefill fails, also recompute every later prompt chunk that read the damaged page. Add a test fault that skips a write during the prefill. The launch is asked separately.

**Next free number:** 50.

### Decisions 50–52 · 16:51 CST

> [!quote] Owner
> "50 a, 51 a, 52 a"

#### Decision 50 · Option a

Remove the co-author and session-link lines from the 12 commits that carry them, by rewriting main from `d8e9273` on. Keep the old history on a local branch so the job records still point to real commits, and add an old-to-new ID table to the results file.

#### Decision 51 · Option a

Commit the refined verdict of the requests-over-time test (4 files, short message, no trailer lines), with the retry mode judged by the same rule as strict release, merge it into main, and rerun every CPU test.

#### Decision 52 · Option a

Rerun the requests-over-time test on octave now, from decision 51's commit:

1. A 2-GPU TinyLlama check (CPU tests, then the test), at most 20 minutes (0.67 GPU-hours).
2. Then chained, only if it passes, the test on Llama-3.1-8B with 8 GPUs, at most 20 minutes (2.7 GPU-hours).

**Next free number:** 53.

### Decisions 53–55 · 17:18 CST

> [!quote] Owner
> "53 a, 54 a, 55 a"

#### Decision 53 · Option a

Keep SGLang for every measured result, and state "one engine" as a limit in the proposal.

#### Decision 54 · Option a

Commit both changes onto main, one commit each with short messages and no trailer lines, then rerun every CPU test on main:

- "Judge arrivals misses by when the wrong token left"
- "Recompute the prompt chunks that read a skipped write"

#### Decision 55 · Option a

Rerun on octave from decision 54's commits:

1. A 2-GPU TinyLlama check (CPU tests, an early-page incident with KVMend's own check, the skipped-write test with small prompt chunks, requests over time), at most 20 minutes (0.67 GPU-hours).
2. Chained, each step only if it passed, the same three tests on Llama-3.1-8B with 8 GPUs at page 64, at most 35 minutes (4.7 GPU-hours).

**Next free number:** 56.

### Decision 56 · 18:18 CST

> [!quote] Owner
> "56 a"

**Option a:** Apply the checked proposal update:

1. Copy the new source into the proposal's build folder (not tracked by git).
2. Commit the regenerated `PROPOSAL.md` on main with a short message and no trailer lines, "Add today's results to the proposal".

**Next free number:** 57.

### Decisions 57–58 · 19:19 CST

> [!quote] Owner
> "57 a, 58 a"

#### Decision 57 · Option a

Delete the leftovers from the day's builds:

- 7 build checkouts (58 MB);
- the 6 local branches that point at them (merge-builds and five build branches);
- two 8 KB scratch folders in the shared `/tmp`.

Keep `main`, `production` and `before-trailer-removal`.

#### Decision 58 · Option a

Make a failed check inside attention start the repair (today it only logs), then run it on both models. The GPU run is asked separately.

**Next free number:** 59.

### Decisions 59–60 · 20:52 CST

> [!quote] Owner
> "59 a, 60 a"

#### Decision 59 · Option a

Commit both changes on main as two short commits with no trailer lines:

1. "Evict KV cached right after a strict request's last good check"
2. "Repair damage that checks on read find, in the same step"

Rerun every CPU test on main.

#### Decision 60 · Option a

Run on octave from decision 59's commits:

1. A 2-GPU TinyLlama check (CPU tests, the attention-repair test, which must show a wrong token avoided, then the release-rule and requests-over-time tests), at most 30 minutes (1.0 GPU-hours).
2. Chained, each step only if it passed, the attention-repair and requests-over-time tests on Llama-3.1-8B with 8 GPUs, at most 30 minutes (4.0 GPU-hours).

**Next free number:** 61.

### Decision 61 · 22:37 CST

> [!quote] Owner
> "61 a"

**Option a:** Apply the checked proposal update for checks on read that repair:

1. Copy the new source into the proposal's build folder.
2. Commit the regenerated `PROPOSAL.md` on main as "Add checks on read that repair to the proposal", short, with no trailer lines.

**Next free number:** 62.

### Decision 62 · 22:40 CST

> [!quote] Owner
> "62 a"

**Option a:** Delete:

- the last two scratch checkouts:
  - `.claude/worktrees/agent-af530143e052117d7`, 8.3 MB, its 8 uncommitted files identical to main;
  - `.claude/worktrees/split-check`, 8.5 MB, clean at main's tip;
- the branch `worktree-agent-af530143e052117d7` (at `127e8f4`, already on main).

**Next free number:** 63.
