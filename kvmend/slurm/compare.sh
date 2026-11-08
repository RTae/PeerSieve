#!/bin/bash
# GhostServe comparison (decision 12). Part 1 puts one silent fault under GhostServe-style responses and KVMend
# (run_poc.py); part 2 measures the fault-free throughput of plain SGLang, parity alone and KVMend (bench_overhead.py).
# Submit from the login node yes: the 4-GPU check, then the 8-GPU comparison only if the check passes.
#   C=$(sbatch --parsable -p a100 -A a100 --nodelist=octave --gres=gpu:a100:4 --cpus-per-task=16 --mem=64G \
#       --time=00:30:00 --open-mode=append -J kvmend-cmp4 -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/TinyLlama-1.1B-Chat-v1.0,TP=4,PROMPT_LEN=1536 /home/tae/kvmend/slurm/compare.sh)
#   sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:8 --cpus-per-task=32 --mem=160G --time=00:50:00 \
#       --dependency=afterok:$C --kill-on-invalid-dep=yes --open-mode=append -J kvmend-cmp8 \
#       -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/Meta-Llama-3.1-8B-Instruct,TP=8,PROMPT_LEN=32768,MAX_TOTAL=49152,VARIANT_TIMEOUT=900 \
#       /home/tae/kvmend/slurm/compare.sh
# The job exits 0 only if part 1's pass checks hold and every part 2 run finishes.
set -uo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=/home/tae/kvmend
cd "$ROOT" || exit 1
RUN=$ROOT/runs/${SLURM_JOB_NAME:-compare}-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/.cache"
export TRITON_CACHE_DIR=$ROOT/.cache/triton XDG_CACHE_HOME=$ROOT/.cache TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_ext
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH KVMEND_ENABLE KVMEND_PLAN KVMEND_TAGS KVMEND_CTRL KVMEND_ENCODE_EVERY KVMEND_SCRUB_PIECES

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z'); run folder $RUN"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,driver_version --format=csv
scontrol show job -d "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'IDX:[0-9,-]*' || true
sha256sum kvmend/*.py hook_site/sitecustomize.py run_poc.py bench_overhead.py slurm/compare.sh | tee "$RUN/code.sha256"

echo "== CPU tests ($(date '+%T'))"
uv run --frozen python -m pytest -q -p no:cacheprovider tests || { echo "== CPU tests failed"; exit 2; }

echo "== part 1: one silent fault under five responses ($(date '+%T'))"
uv run --frozen python run_poc.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
  --max-total-tokens "${MAX_TOTAL:-65536}" --variant-timeout "${VARIANT_TIMEOUT:-600}" \
  --variants clean,control,silent,kvmend,repaironly,gs_gpu,gs_chunk,reprefill_all --require-pass --out "$RUN/compare"
P1=$?
echo "== part 1 exit $P1 ($(date '+%T'))"

echo "== part 2: fault-free throughput ($(date '+%T'))"
P2=0
for MODE in plain parity kvmend; do
  uv run --frozen python bench_overhead.py --model "$MODEL" --tp "$TP" --mode "$MODE" --out "$RUN/overhead" ${BENCH:-}
  RC=$?
  [ "$RC" -ne 0 ] && P2=$RC
  echo "== $MODE exit $RC ($(date '+%T'))"
done
uv run --frozen python bench_overhead.py --report --out "$RUN/overhead" || P2=4
echo "== done at $(date '+%F %T %Z'); part 1 exit $P1, part 2 exit $P2"
[ "$P1" -eq 0 ] && [ "$P2" -eq 0 ]
