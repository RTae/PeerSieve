#!/bin/bash
# KVMend PoC GPU job (decision 7). Submit from the login node yes:
#   smoke test, 4 A100s, TinyLlama:
#     sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:4 --cpus-per-task=16 --mem=64G --time=00:30:00 \
#       --open-mode=append -J kvmend-smoke -o /home/tae/kvmend/logs/%x-%j.out \
#       --export=ALL,MODEL=/home/tae/models/TinyLlama-1.1B-Chat-v1.0,TP=4,PROMPT_LEN=1536 /home/tae/kvmend/slurm/poc.sh
#   running example, 8 A100s, Llama-3.1-8B at TP=8, 32K prompt:
#     sbatch ... --gres=gpu:a100:8 --cpus-per-task=32 --mem=160G --time=00:50:00 -J kvmend-r8b ... \
#       --export=ALL,MODEL=/home/tae/models/Meta-Llama-3.1-8B-Instruct,TP=8,PROMPT_LEN=32768,MAX_TOTAL=49152 ...
set -euo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=/home/tae/kvmend
cd "$ROOT"
RUN=$ROOT/runs/${SLURM_JOB_NAME:-poc}-${SLURM_JOB_ID:-local}
mkdir -p "$RUN" "$ROOT/.cache"
export TRITON_CACHE_DIR=$ROOT/.cache/triton XDG_CACHE_HOME=$ROOT/.cache TORCH_EXTENSIONS_DIR=$ROOT/.cache/torch_ext
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z'); run folder $RUN"
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used,memory.total,driver_version --format=csv
scontrol show job -d "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -o 'IDX:[0-9,-]*' || true

echo "== CPU tests"
uv run --frozen python -m pytest -q -p no:cacheprovider tests

echo "== PoC"
uv run --frozen python run_poc.py --model "$MODEL" --tp "$TP" --prompt-len "$PROMPT_LEN" \
  --max-total-tokens "${MAX_TOTAL:-65536}" --out "$RUN" ${EXTRA:-}
echo "== done at $(date '+%F %T %Z')"
