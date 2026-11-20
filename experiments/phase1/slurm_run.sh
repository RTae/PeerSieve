#!/usr/bin/env bash
# Phase 1 Slurm runner for TrustGraph-DFL CIFAR-10 reproduction.
# Supports multi-GPU parallel sharding across all GPUs allocated to the job.
#
# Example (after explicit approval):
#   mkdir -p logs
#   sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
#     --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
#     experiments/phase1/slurm_run.sh --mode suite --suite full --skip-existing --rounds 200 --eval-every 10 --seeds 42 43 44

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

echo "=== Phase 1 Slurm Run Record ==="
date -Is
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Git status:"
git status --short || true

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "--- scontrol show job -d ---"
  scontrol show job -d "$SLURM_JOB_ID" || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "--- nvidia-smi topo -m ---"
  nvidia-smi topo -m || true
  echo "--- nvidia-smi query-gpu ---"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv || true
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_CMD=("$PYTHON_BIN")
elif command -v uv >/dev/null 2>&1 && [[ -d .venv ]]; then
  PY_CMD=(uv run --no-sync python)
elif [[ -x .venv/bin/python ]]; then
  PY_CMD=(.venv/bin/python)
else
  PY_CMD=(python3)
fi

echo "--- Python / PyTorch (${PY_CMD[*]}) ---"
"${PY_CMD[@]}" -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"

# Determine GPU list assigned to this Slurm job.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_COUNT=$("${PY_CMD[@]}" -c "import torch; print(max(1, torch.cuda.device_count()))")
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi

NUM_SHARDS="${#GPU_IDS[@]}"
echo "--- Launching ${NUM_SHARDS} parallel shard worker(s) on GPUs: ${GPU_IDS[*]} ---"

if [[ "$NUM_SHARDS" -le 1 ]]; then
  exec "${PY_CMD[@]}" -m experiments.phase1.run --device cuda --no-download "$@"
fi

PIDS=()
for idx in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$idx]}"
  shard_log="logs/phase1-${SLURM_JOB_ID:-local}-shard${idx}.log"
  echo "Starting shard ${idx}/${NUM_SHARDS} on physical/visible GPU ${gpu} -> ${shard_log}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PY_CMD[@]}" -m experiments.phase1.run \
    --device cuda --no-download \
    --shard-index "$idx" --num-shards "$NUM_SHARDS" "$@" \
    > "$shard_log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for idx in "${!PIDS[@]}"; do
  pid="${PIDS[$idx]}"
  if ! wait "$pid"; then
    echo "ERROR: shard ${idx} (pid=${pid}) exited non-zero!"
    FAILED=1
  else
    echo "Shard ${idx} (pid=${pid}) completed cleanly."
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more shards failed. Check logs/phase1-${SLURM_JOB_ID:-local}-shard*.log"
  exit 1
fi

echo "=== All ${NUM_SHARDS} shards completed cleanly at $(date -Is) ==="
