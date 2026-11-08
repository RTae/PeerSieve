# KVMend proof of concept

KVMend finds silent damage in an LLM server's KV cache, rebuilds only the damaged piece, and recomputes only the
tokens that the damage could have reached. This folder tests the idea as a small patch on SGLang 0.5.1, loaded the way
GhostServe loads its hook: a `sitecustomize.py` on `PYTHONPATH` patches SGLang's classes as they are imported. SGLang
itself is not edited.

## What the patch changes

| File | What it does |
|---|---|
| `kvmend/tags.py` | the tag of a piece (one 16-token page, one layer, one GPU): a 32-bit sum of mixed words, updated one token at a time |
| `kvmend/hook.py` | the patch: running tags in `MHATokenToKVPool.set_kv_buffer`, dropped tags in the paged allocator's `free`, and a step after every scheduler iteration that runs the plan's actions (encode, scrub, inject, repair, retract, dump, and for the comparison repair_share and evict) and the background encode and scrub; with HiCache, tags that travel with host copies and a check of every loaded layer on arrival |
| `kvmend/meta.py` | names every record: Slurm job, node, GPUs and the SHA-256 of each source file |
| `hook_site/sitecustomize.py` | loads the patch into every SGLang process when `KVMEND_ENABLE=1` |
| `run_poc.py` | the driver: request R under each response to one fault, on one engine, then a summary and the comparison table |
| `run_hicache.py` | loads from host memory (HiCache) that land on only some GPUs: found on arrival or by the reader's check, copied again from the host copy |
| `bench_overhead.py` | fault-free throughput of plain SGLang, parity alone and KVMend, one engine per mode |
| `run_arrivals.py` | requests that arrive over time, under fast release, strict release and abort-and-retry (below) |
| `slurm/poc.sh`, `slurm/compare.sh` | the GPU jobs: the proof of concept, and the GhostServe comparison with the throughput benchmark |
| `tests/` | CPU tests: tag and parity math, running tags, the plan and the log, and the collective actions on 4 gloo processes |

The actions run on every tensor-parallel rank at the same point of the same iteration:

- **encode:** sealed pages go in chunks to a rotating encoder rank. It XORs the pieces into one parity piece and keeps
  it in pinned host memory. Each rank checks its own pieces' tags before it sends them.
- **scrub:** each rank recomputes every written piece's tag from its bytes. A mismatch names the rank, layer and page.
- **repair:** the other ranks' pieces, XORed with the parity, rebuild the damaged piece in place. The result must match
  its tag.
- **retract:** SGLang's own retract steps for one request. Its KV after the cached prefix is freed, and SGLang
  re-prefills only those tokens.

## The five variants

Each variant runs R (a fixed prompt, greedy decoding) on an empty cache. At output 1, each variant encodes the
prompt's pages and scrubs once, which sets the last good check.

| Variant | What happens | What it shows |
|---|---|---|
| clean | no fault | the reference output |
| control | no fault; R is rolled back at output 41 | the re-prefill path alone |
| silent | at output 10, bit 14 of one K value of R's page 64 (tokens 1,024–1,039), layer 12, rank 2 flips; nothing checks | what silent damage does |
| kvmend | the same fault; at output 41 the scrub finds it, parity rebuilds the piece, R is rolled back | KVMend |
| repaironly | the same fault, found and rebuilt, but no rollback | why the suspect tokens must be recomputed |
| forced | no fault; the prompt is cached, then extended by kvmend's first 41 output tokens, and the rest is decoded | the healthy server that produced kvmend's text |

**Pass condition.** kvmend's output equals forced's, token for token.

That means that after the repair and the rollback, the KV cache and everything generated next are exactly what a healthy server gives for the same text.

kvmend need not equal control. A fault can change the tokens generated before its detection, and KVMend keeps tokens already generated, as the proposal says.

The other checks:
- the scrub names exactly the injected piece;
- the rebuilt piece matches its tag and the clean copy, bit for bit;
- the re-prefill is small;
- no errors occur.

## Comparison with GhostServe

GhostServe's code runs only on Hopper GPUs, so the patch re-implements its parity on SGLang 0.5.1. The same fault
(one flipped bit in R's page 64, layer 12, rank 2, at output 10) meets five responses:

| Response | Run | What happens at output 41 |
|---|---|---|
| GhostServe as published | silent | nothing: GhostServe keeps parity but checks nothing |
| GhostServe, told the GPU | gs_gpu | an oracle names rank 2; GhostServe rebuilds rank 2's share of every encoded page, all layers; no rollback |
| GhostServe, told the GPU and the chunk | gs_chunk | the same, for the 2,048-token chunk that holds the damage only |
| Detect, then re-prefill the whole request | reprefill_all | the scrub finds the fault; R is rolled back and its cached prompt dropped, as when a checksum mismatch aborts the request and the retry recomputes it |
| KVMend | kvmend | the scrub finds the piece; parity rebuilds it; R's suspect tokens are re-prefilled |

Each response is scored on four things:
- whether it notices the fault;
- the data it reads to repair;
- the tokens it recomputes;
- the recovery time, from the first recovery action to the first token after it.

Its output is also compared with a fault-free run that produced the same text on the same compute path:
- `forced` for KVMend and the responses that do not roll back;
- `forced_full` for reprefill_all.

`control` equal to `clean` shows that the re-prefill path alone does not change the output.

**Fault-free cost** (`bench_overhead.py`). Each mode runs in its own engine, because the tag update is captured into SGLang's decode CUDA graphs:
- **plain:** SGLang without the patch.
- **parity:** GhostServe-style parity alone, encoding sealed pages every 16 batch steps.
- **kvmend:** tags, parity with a tag check before each encode, and a background scrub at several rates.

The workload is 64 requests × 1,024 prompt tokens × 256 output tokens, sent at once; each figure is the median of 3 runs.

The proposal's target (H7) is throughput within 3% of parity alone.

## Requests that arrive over time

`run_arrivals.py` runs the shared-prefix workload of `run_load.py` with requests that arrive by a Poisson process
(`--rate` per second, `--seed`), streamed. Three modes run in one engine, each without and with `run_load.py`'s four
faults:
- **fast:** no checks;
- **strict:** KVMend's release rule, a check every `--every` tokens; damage is mended, and the unreleased tokens are
  generated again;
- **retry:** the same checks, but a failed check aborts the request and every other reader of the damaged pages, and
  the client resubmits.

The abort is a hook option: an entry `rid:every:abort` in `KVMEND_STRICT`, or `KVMEND_ON_FAIL=abort` for every strict
request. An id that ends in `*` names every id that starts with what comes before it. The damaged pages leave the
prefix cache, with everything cached on top of them, once no running request holds them. The node that holds the first
damaged page is split there, so its clean head stays cached and a resubmission recomputes from the damaged page on.

Each run reports time to first token and end-to-end latency (median and 99th percentile, from the arrival to the end of
the last attempt), throughput, goodput, wrong outputs on stable requests, aborts and resubmissions, and checks per
step (`summary.json`, `arrivals.md`). The suite step is `arrivals`.

## Loads from host memory (HiCache)

SGLang's hierarchical cache copies KV pages to a host pool (`HiCacheController.write`, on a write thread) and back into new device pages on a prefix hit (`HiCacheController.load`, on a load thread, layer by layer). Neither copy goes through `set_kv_buffer`, so the patch carries the tags along:

- **backup:** the tags, coverage and token keys of each copied page are recorded under its host page (`PoolState.carry_out`);
- **load:** the new device pages get that record before the load thread copies a byte (`carry_in`), never tags recomputed from the bytes that land;
- **check on arrival:** in the load thread, each loaded layer is tested against those tags before attention reads it (`load_layer`); a piece that fails is copied again from the host copy, which must then pass;
- **backstop:** what still fails (a damaged host copy, or a load not checked on arrival) fails the reader's check, and `mend` copies it again from the host copy or recomputes it in place.

`run_hicache.py` evicts the prompt's device copy (op `offload`) and lets a request load it back with one GPU's copy of some pages left as zeros (fault `partial_load`), then compares the text with a fault-free load.

## Run it on cluster ja

```bash
# once: the uv project and the models (slurm/setup_env.sh, a CPU-only job)
# smoke test, 4 A100s, TinyLlama-1.1B at TP=4
sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:4 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  --open-mode=append -J kvmend-smoke -o /home/tae/kvmend/logs/%x-%j.out \
  --export=ALL,MODEL=/home/tae/models/TinyLlama-1.1B-Chat-v1.0,TP=4,PROMPT_LEN=1536 /home/tae/kvmend/slurm/poc.sh
```

Each run writes `runs/<job>/`:
- `plan.json`;
- `rank<r>.jsonl`, one line per action per rank;
- `outputs.json`;
- `summary.json`, holding the comparisons, the fault, the detection, the repair, the rollback, the re-prefill size and the spread of the damage across layers.

## Limits of this proof of concept

- **Tags** are a position-keyed 32-bit sum, not CRC-32C. PyTorch has scatter-add but no scatter-XOR. Any change inside one 32-bit word is always caught.
- **Encode** gathers every chunk to one rank with NCCL's gather. Nothing is fused or overlapped yet.
- **Rollback** re-prefills all of R's decode tokens, not only those after the last good check. Here the two sets are the same, because the last good check is at output 1.
- **Not built yet:** the scrub runs only when the plan asks for it, not as a rate-limited background stream. There are no generations, so stale reads are not caught. There is no Reed–Solomon code, and finished requests' cached KV is not evicted.
- **Backend:** SGLang runs with the Triton attention backend, so no FlashInfer kernels need compiling.
- **GhostServe re-implemented:** XOR parity per page in plain PyTorch and NCCL, with synchronous host copies. GhostServe
  uses fused kernels and asynchronous copies, so the parity-alone throughput here is a lower bound for GhostServe's.
  The oracle that names the GPU does not exist in GhostServe; it stands for the best case of a detector it lacks.
- **Logs** are kept in memory and written when the scheduler is idle or flushes: the cluster's `/home` is NFS, where
  one append took 0.25-0.5 s and lengthened the timed steps of the first proof-of-concept runs.
