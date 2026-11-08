#!/bin/bash
# The fair test (decision 18): KVMend with strict release against abort and retry, both required to return the right
# text, plus a sweep of single-bit faults and the cost of recomputing a page. Submit from the login node yes:
#   C=$(sbatch --parsable -p a100 -A a100 --nodelist=octave --gres=gpu:a100:4 --cpus-per-task=16 --mem=64G \
#       --time=00:15:00 --open-mode=append -J kvmend-fair4 -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/TinyLlama-1.1B-Chat-v1.0,TP=4,PROMPT_LEN=1536 /home/tae/kvmend/slurm/fair.sh)
#   sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:8 --cpus-per-task=32 --mem=160G --time=00:25:00 \
#       --dependency=afterok:$C --kill-on-invalid-dep=yes --open-mode=append -J kvmend-fair8 \
#       -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/Meta-Llama-3.1-8B-Instruct,TP=8,PROMPT_LEN=32768,MAX_TOTAL=49152 \
#       /home/tae/kvmend/slurm/fair.sh
set -uo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=/home/tae/kvmend
cd "$ROOT" || exit 1
RUN=$ROOT/runs/${SLURM_JOB_NAME:-fair}-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/.cache"
export TRITON_CACHE_DIR=$ROOT/.cache/triton XDG_CACHE_HOME=$ROOT/.cache TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_ext
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH KVMEND_ENABLE KVMEND_PLAN KVMEND_TAGS KVMEND_CTRL KVMEND_ENCODE_EVERY KVMEND_SCRUB_PIECES

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z'); run folder $RUN"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,driver_version --format=csv
scontrol show job -d "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'IDX:[0-9,-]*' || true
sha256sum kvmend/*.py hook_site/sitecustomize.py run_poc.py run_fair.py slurm/fair.sh | tee "$RUN/code.sha256"

echo "== CPU tests ($(date '+%T'))"
uv run --frozen python -m pytest -q -p no:cacheprovider tests || { echo "== CPU tests failed"; exit 2; }

echo "== the fair test ($(date '+%T'))"
uv run --frozen python run_fair.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
  --max-total-tokens "${MAX_TOTAL:-65536}" --variant-timeout "${VARIANT_TIMEOUT:-600}" --require-pass --out "$RUN/fair"
RC=$?
echo "== done at $(date '+%F %T %Z'); exit $RC"
exit $RC
