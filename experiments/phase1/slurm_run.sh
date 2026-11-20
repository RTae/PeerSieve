#!/usr/bin/env bash
# Phase 1 Slurm runner for TrustGraph-DFL CIFAR-10 reproduction.
# Submit with explicit partition, account, node/GPU, time, and log flags after approval:
#   mkdir -p logs
#   sbatch -p <partition> -A <account> --nodes=1 --gres=gpu:1 --time=00:45:00 \
#     --open-mode=append --output=logs/%x-%j.out \
#     experiments/phase1/slurm_run.sh --mode suite --suite main_attacks --seeds 42 43 44

set -euo pipefail

echo "=== Phase 1 Slurm Run Record ==="
date -Is
echo "Host: $(hostname)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-unset}"
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

echo "--- Python / PyTorch ---"
uv run python -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'avail:', torch.cuda.is_available())"

echo "--- Running Phase 1 ---"
exec uv run python -m experiments.phase1.run --device cuda "$@"
