#!/bin/bash
# The incident's faults (decision 20): KV that is well formed but wrong on every GPU (a never-written page, a late
# write around or through the store path, another sequence's KV), against no checks, abort and retry, GhostServe
# told the GPU, and KVMend (reader's check, recompute in place, strict release). Submit from the login node yes:
#   C=$(sbatch --parsable -p a100 -A a100 --nodelist=octave --gres=gpu:a100:4 --cpus-per-task=16 --mem=64G \
#       --time=00:15:00 --open-mode=append -J kvmend-incident4 -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/TinyLlama-1.1B-Chat-v1.0,TP=4,PROMPT_LEN=1536 /home/tae/kvmend/slurm/incident.sh)
#   sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:8 --cpus-per-task=32 --mem=160G --time=00:30:00 \
#       --dependency=afterok:$C --kill-on-invalid-dep=yes --open-mode=append -J kvmend-incident8 \
#       -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/Meta-Llama-3.1-8B-Instruct,TP=8,PROMPT_LEN=32768,MAX_TOTAL=49152 \
#       /home/tae/kvmend/slurm/incident.sh
set -uo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=/home/tae/kvmend
cd "$ROOT" || exit 1
RUN=$ROOT/runs/${SLURM_JOB_NAME:-incident}-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/.cache"
export TRITON_CACHE_DIR=$ROOT/.cache/triton XDG_CACHE_HOME=$ROOT/.cache TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_ext
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH KVMEND_ENABLE KVMEND_PLAN KVMEND_TAGS KVMEND_CTRL KVMEND_ENCODE_EVERY KVMEND_SCRUB_PIECES KVMEND_FENCE

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z'); run folder $RUN"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,driver_version --format=csv
scontrol show job -d "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'IDX:[0-9,-]*' || true
sha256sum kvmend/*.py hook_site/sitecustomize.py run_poc.py run_fair.py run_incident.py tests/*.py tests_gpu/*.py \
  slurm/incident.sh | tee "$RUN/code.sha256"

# CPU tests: one thread per process (the gloo tests start four ranks on a few CPUs), unbuffered, each test named
# with its time, a stack dump if one runs past 150 s, and at most 7 minutes for the whole step.
echo "== CPU tests ($(date '+%T'))"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 timeout 420 uv run --frozen python -m pytest -v \
  -p no:cacheprovider -o faulthandler_timeout=150 --durations=0 tests
TRC=$?
if [ $TRC -ne 0 ]; then echo "== CPU tests failed (exit $TRC; 124 means the 7-minute cap)"; exit 2; fi

echo "== the incident's faults ($(date '+%T'))"
uv run --frozen python run_incident.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
  --max-total-tokens "${MAX_TOTAL:-65536}" --variant-timeout "${VARIANT_TIMEOUT:-600}" --require-pass --out "$RUN/incident"
RC=$?

# The fused Triton tag kernels against the PyTorch tags (step 2 of decision 20): logged, not a gate for this test.
echo "== GPU kernel tests ($(date '+%T'))"
PYTHONUNBUFFERED=1 timeout 300 uv run --frozen python -m pytest -v -p no:cacheprovider tests_gpu \
  || echo "== GPU kernel tests failed (exit $?)"

# With RELEASE=1: the release rule on a streamed request, with the fused kernels (step 2); logged, not a gate.
if [ "${RELEASE:-0}" = "1" ]; then
  echo "== the release rule ($(date '+%T'))"
  timeout 480 uv run --frozen python run_release.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
    --max-total-tokens "${MAX_TOTAL:-65536}" --require-pass --out "$RUN/release" \
    || echo "== the release rule test failed (exit $?)"
fi
echo "== done at $(date '+%F %T %Z'); exit $RC"
exit $RC
