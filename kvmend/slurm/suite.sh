#!/bin/bash
# Decision 20's tests on octave, one Slurm job: the steps named in STEPS, in this order, each with its own cap.
# Gates for the job's exit code (what an afterok chain waits for): the CPU tests, the incident test and the HiCache
# test. Every other step is logged with its exit code and does not stop the job.
#   tests            CPU tests (one thread per process, at most 7 minutes)
#   gputests         the fused Triton kernels against the PyTorch tags, in CUDA graphs too
#   incident         run_incident.py at TP=$TP (the incident's faults; gate)
#   incident_local   run_incident.py at TP=1 on the job's first GPU with stripes within the GPU ($LOCAL_MODEL)
#   incident_early   run_incident.py at TP=$TP with the damage early in the prompt ($EARLY_PAGES pages from
#                    position $EARLY_POS), where KVMend's saving over a retry from the damaged page is largest
#   hicache          run_hicache.py at TP=$TP: SGLang's hierarchical cache (write_through) backs the prompt up to host
#                    memory and loads it back; the load does not land on one GPU ($HICACHE_PAGES pages, $HICACHE_MODE)
#                    (gate)
#   release          run_release.py: the release rule on a streamed request
#   burst            run_burst.py: a burst on one GPU, rebuilt from parity or recomputed ($BURST_SIZES)
#   load             run_load.py: many requests, shared prefixes, four faults, detection by the release rule
#   checker          bench_checker.py on the job's first GPU
#   overhead         bench_overhead.py, modes $OVERHEAD_MODES, then the decode-heavy workload, modes $DECODE_MODES
# Environment: MODEL TP PROMPT_LEN [MAX_TOTAL] [LOCAL_MODEL LOCAL_PROMPT_LEN] [BURST_SIZES BURST_PROMPT_LEN]
#              [LOAD_SHARED LOAD_CLEAN_RUNS] [RELEASE_KIND RELEASE_LATE_AT] [OVERHEAD_MODES DECODE_MODES]
#              [EARLY_POS EARLY_PAGES] [COMMIT: the local git commit the copied files match]
#              [REQUIRE_OK_FROM] [LOCAL_PAGES] [HICACHE_PROMPT_LEN HICACHE_PAGES HICACHE_MODE]
# Each step that exits 0 is listed in $RUN/steps_ok. With REQUIRE_OK_FROM=<an earlier run folder>, a step other than
# the CPU tests runs only if that folder's steps_ok lists it: a check job on a small model gates each step of a
# bigger job, so a bug that does not depend on the model size cannot spend the bigger job's GPU time.
set -uo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=/home/tae/kvmend
cd "$ROOT" || exit 1
RUN=$ROOT/runs/${SLURM_JOB_NAME:-suite}-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/.cache"
export TRITON_CACHE_DIR=$ROOT/.cache/triton XDG_CACHE_HOME=$ROOT/.cache TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_ext
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
unset PYTHONPATH KVMEND_ENABLE KVMEND_PLAN KVMEND_TAGS KVMEND_CTRL KVMEND_ENCODE_EVERY KVMEND_SCRUB_PIECES KVMEND_FENCE \
  KVMEND_FUSED KVMEND_STRICT KVMEND_STRICT_EVERY KVMEND_STRIPE KVMEND_LOAD_CHECK
MAX_TOTAL=${MAX_TOTAL:-65536}
FIRST_GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z'); steps: $STEPS; run folder $RUN; commit ${COMMIT:-unknown}"
echo "${COMMIT:-unknown}" > "$RUN/commit.txt"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,driver_version --format=csv
scontrol show job -d "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'IDX:[0-9,-]*' || true
sha256sum kvmend/*.py hook_site/sitecustomize.py run_*.py bench_*.py tests/*.py tests_gpu/*.py slurm/suite.sh \
  | tee "$RUN/code.sha256"

GATE=0
step() {  # step NAME CAP_SECONDS GATING COMMAND...
  local name=$1 cap=$2 gating=$3
  shift 3
  echo "== $name ($(date '+%T'), at most $cap s)"
  timeout "$cap" "$@"
  local rc=$?
  echo "== $name exit $rc ($(date '+%T'))$([ $rc -eq 124 ] && echo '; the cap stopped it')"
  if [ "$gating" = 1 ] && [ $rc -ne 0 ]; then GATE=$rc; fi
  if [ $rc -eq 0 ]; then echo "$name" >> "$RUN/steps_ok"; fi
  return $rc
}
allowed() {  # allowed STEP: true unless REQUIRE_OK_FROM is set and its steps_ok does not list STEP
  [ -z "${REQUIRE_OK_FROM:-}" ] || [ "$1" = tests ] || grep -qx "$1" "$REQUIRE_OK_FROM/steps_ok" 2>/dev/null
}
PY="uv run --frozen python"

for S in $STEPS; do
  if ! allowed "$S"; then echo "== $S skipped: it did not pass in $REQUIRE_OK_FROM"; continue; fi
  case $S in
    tests)
      step tests 420 1 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -m pytest -v -p no:cacheprovider \
        -o faulthandler_timeout=150 --durations=0 tests || { echo "== CPU tests failed: stopping"; exit 2; } ;;
    gputests)
      step gputests 300 0 $PY -m pytest -v -p no:cacheprovider tests_gpu ;;
    incident)
      step incident 1200 1 $PY run_incident.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
        --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/incident" ;;
    incident_local)
      step incident_local 1200 0 env CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY run_incident.py --model "${LOCAL_MODEL:-$MODEL}" \
        --tp 1 --stripe local --prompt-len "${LOCAL_PROMPT_LEN:-$PROMPT_LEN}" --pages "${LOCAL_PAGES:-4}" \
        --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/incident_local" ;;
    incident_early)
      step incident_early 1500 0 $PY run_incident.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
        --pos "${EARLY_POS:-1024}" --pages "${EARLY_PAGES:-1}" \
        --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/incident_early" ;;
    hicache)
      step hicache 1200 1 $PY run_hicache.py --model "$MODEL" --tp "$TP" \
        --prompt-len "${HICACHE_PROMPT_LEN:-$PROMPT_LEN}" --pages "${HICACHE_PAGES:-4}" --mode "${HICACHE_MODE:-zeros}" \
        --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/hicache" ;;
    release)
      step release 900 0 $PY run_release.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
        --kind "${RELEASE_KIND:-late_bypass}" --late-at "${RELEASE_LATE_AT:-10}" \
        --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/release" ;;
    burst)
      step burst 1200 0 $PY run_burst.py --model "$MODEL" --tp "$TP" --prompt-len "${BURST_PROMPT_LEN:-$PROMPT_LEN}" \
        --sizes "${BURST_SIZES:-1x1,8x1,64x1,64x8}" --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/burst" ;;
    load)
      step load 1200 0 $PY run_load.py --model "$MODEL" --tp "$TP" --shared "${LOAD_SHARED:-1024}" \
        --clean-runs "${LOAD_CLEAN_RUNS:-3}" --max-total-tokens "$MAX_TOTAL" --require-pass --out "$RUN/load" ;;
    checker)
      step checker 600 0 env CUDA_VISIBLE_DEVICES=$FIRST_GPU $PY bench_checker.py --out "$RUN/checker" ;;
    overhead)
      OK=1
      for M in ${OVERHEAD_MODES:-plain parity kvmend fused tags_fused}; do
        step "overhead:$M" 900 0 $PY bench_overhead.py --model "$MODEL" --tp "$TP" --mode "$M" --repeats 2 \
          --scrub 0 --out "$RUN/overhead" || OK=0
      done
      step overhead:report 120 0 $PY bench_overhead.py --report --out "$RUN/overhead" || OK=0
      for M in ${DECODE_MODES:-plain parity fused}; do
        step "decode:$M" 900 0 $PY bench_overhead.py --model "$MODEL" --tp "$TP" --mode "$M" --repeats 2 \
          --scrub 0 --requests 32 --prompt-len 128 --max-new 512 --out "$RUN/decode" || OK=0
      done
      step decode:report 120 0 $PY bench_overhead.py --report --out "$RUN/decode" || OK=0
      if [ $OK = 1 ]; then echo overhead >> "$RUN/steps_ok"; fi ;;
    *) echo "== unknown step $S" ;;
  esac
done
echo "== done at $(date '+%F %T %Z'); gate exit $GATE"
exit $GATE
