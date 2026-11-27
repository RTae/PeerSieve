#!/usr/bin/env bash
# Phase 1 Slurm runner for TrustGraph-DFL CIFAR-10 experiments (Typer subcommands).
# Supports two execution styles:
#   1. Sharded multi-GPU sweeps:   slurm_run.sh suite --suite paper_resnet18_50k ...
#   2. Multi-process distributed:  slurm_run.sh distributed --n-nodes 8 ...
#
# Examples (after explicit approval):
#   mkdir -p logs
#   # Sharded suite across 8x A100 GPUs:
#   sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
#     --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
#     experiments/phase1/slurm_run.sh suite --suite paper_resnet18_50k --skip-existing --eval-every 5 --seeds 42,43,44
#
#   # Real 8-GPU distributed FL (one peer process per GPU over TcpPeerTransport):
#   sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
#     --cpus-per-task=32 --mem=128G --time=00:15:00 --open-mode=append --output=logs/%x-%j.out \
#     experiments/phase1/slurm_run.sh distributed --model resnet18_pretrained --dataset cifar10 \
#     --samples-per-node 2500 --val-split iid --lr 0.005 --lr-schedule cosine --augment --grad-clip-norm 10.0 \
#     --trustgraph-clip-norm --trustgraph-clip-mode model --trustgraph-score-threshold 0.003 \
#     --topology random --attack model_poison --byzantine-frac 0.25 --rounds 5 --eval-every 5 --seed 42

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

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_COUNT=$("${PY_CMD[@]}" -c "import torch; print(max(1, torch.cuda.device_count()))")
  GPU_IDS=()
  for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_IDS+=("$i")
  done
fi

NUM_GPUS="${#GPU_IDS[@]}"
SUBCMD="${1:-suite}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ "$SUBCMD" == "distributed" ]]; then
  DIST_WORLD_SIZE=""
  PREV_ARG=""
  for arg in "$@"; do
    if [[ "$PREV_ARG" == "--n-nodes" ]]; then
      DIST_WORLD_SIZE="$arg"
    elif [[ "$arg" == --n-nodes=* ]]; then
      DIST_WORLD_SIZE="${arg#--n-nodes=}"
    fi
    PREV_ARG="$arg"
  done
  if [[ -z "$DIST_WORLD_SIZE" ]]; then
    DIST_WORLD_SIZE="$NUM_GPUS"
  fi
  DIST_RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
  export DIST_RUN_ID
  echo "--- Launching ${DIST_WORLD_SIZE} distributed peer processes (run_id=${DIST_RUN_ID}) across ${NUM_GPUS} GPUs: ${GPU_IDS[*]} ---"
  PIDS=()
  for ((idx = 0; idx < DIST_WORLD_SIZE; idx++)); do
    gpu="${GPU_IDS[$((idx % NUM_GPUS))]}"
    peer_log="logs/phase1-${SLURM_JOB_ID:-local}-peer${idx}.log"
    echo "Starting peer rank ${idx}/${DIST_WORLD_SIZE} on visible GPU ${gpu} -> ${peer_log}"
    RANK="$idx" WORLD_SIZE="$DIST_WORLD_SIZE" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES="$gpu" \
      "${PY_CMD[@]}" -m experiments.phase1.run distributed --device cuda --no-download "$@" \
      > "$peer_log" 2>&1 &
    PIDS+=("$!")
  done
  FAILED=0
  for idx in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$idx]}"; then
      echo "ERROR: peer rank ${idx} exited non-zero!"
      FAILED=1
    else
      echo "Peer rank ${idx} completed cleanly."
    fi
  done
  if [[ "$FAILED" -ne 0 ]]; then
    echo "One or more distributed peers failed. Check logs/phase1-${SLURM_JOB_ID:-local}-peer*.log"
    exit 1
  fi
  echo "=== All ${DIST_WORLD_SIZE} distributed peers completed cleanly at $(date -Is) ==="
  exit 0
fi

if [[ "$SUBCMD" == "suite" && "$NUM_GPUS" -gt 1 ]]; then
  echo "--- Launching ${NUM_GPUS} parallel suite shard workers on GPUs: ${GPU_IDS[*]} ---"
  PIDS=()
  for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    shard_log="logs/phase1-${SLURM_JOB_ID:-local}-shard${idx}.log"
    echo "Starting shard ${idx}/${NUM_GPUS} on visible GPU ${gpu} -> ${shard_log}"
    CUDA_VISIBLE_DEVICES="$gpu" "${PY_CMD[@]}" -m experiments.phase1.run suite \
      --device cuda --no-download \
      --shard-index "$idx" --num-shards "$NUM_GPUS" "$@" \
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

  echo "=== All ${NUM_GPUS} shards completed cleanly at $(date -Is) ==="
  exit 0
fi

echo "--- Launching single command '${SUBCMD}' on GPU(s): ${GPU_IDS[*]} ---"
exec "${PY_CMD[@]}" -m experiments.phase1.run "$SUBCMD" --device cuda --no-download "$@"
