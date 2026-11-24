#!/usr/bin/env bash
# Stage 2 Slurm runner: measure batched FP32 TrustGraph-DFL memory, communication,
# and per-edge wasted-bandwidth bottlenecks on octave (8x A100-PCIE-40GB).
#
# Submit ONLY after explicit owner approval:
#   mkdir -p logs
#   sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
#     --cpus-per-task=32 --mem=128G --time=00:30:00 --job-name=tg_stage2_prof \
#     --open-mode=append --output=logs/%x-%j.out experiments/phase1/slurm_profile.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
OUT_DIR="${1:-outputs/phase1_motivation}"
mkdir -p "$OUT_DIR" logs

echo "=== Stage 2 System Bottleneck Profile Run Record ==="
date -Is
echo "Host: $(hostname)"
echo "Workdir: $(pwd)"
echo "Slurm Job ID: ${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
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

if [[ -x .venv/bin/python ]]; then
  PY_BIN=".venv/bin/python"
  TORCHRUN_BIN=".venv/bin/torchrun"
else
  PY_BIN="python3"
  TORCHRUN_BIN="torchrun"
fi

echo "--- Python / PyTorch ---"
"$PY_BIN" -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'count:', torch.cuda.device_count())"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  GPU_IDS=(0 1 2 3 4 5 6 7)
fi

echo "=== [1/3] GPU Receive-Buffer Memory Scaling & OOM Cliff (GPU ${GPU_IDS[0]}) ==="
CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY_BIN" -m experiments.phase1.profile_motivation \
  --mode memory \
  --degrees 1 2 4 6 8 12 \
  --out "$OUT_DIR/memory.json"

echo "=== [2/3] 8-GPU torch.distributed Batched FP32 Communication & Round-Time Breakdown ==="
"$TORCHRUN_BIN" --standalone --nproc_per_node="${#GPU_IDS[@]}" \
  -m experiments.phase1.profile_motivation \
  --mode distributed \
  --degree 6 --warmup 1 --repeats 5 \
  --out "$OUT_DIR/distributed.json"

echo "=== [3/3] 4-GPU Parallel Per-Edge Wasted Bandwidth & Repeat-Rejection Audit (50k CIFAR-10 ResNet-18, 25 rounds) ==="
ATTACKS=(none label_flip_all model_poison random_noise)
PIDS=()
for idx in "${!ATTACKS[@]}"; do
  atk="${ATTACKS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  log_f="logs/stage2-per-edge-${SLURM_JOB_ID:-local}-${atk}.log"
  echo "Launching per_edge audit attack=${atk} on GPU ${gpu} -> ${log_f}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY_BIN" -m experiments.phase1.profile_motivation \
    --mode per_edge \
    --attack "$atk" --dataset cifar10 --model resnet18_pretrained \
    --rounds 25 --samples-per-node 2500 --seed 42 \
    --out "$OUT_DIR/per_edge_${atk}.json" > "$log_f" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for idx in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$idx]}"; then
    echo "ERROR: per_edge attack=${ATTACKS[$idx]} failed!"
    FAILED=1
  fi
done
if [[ "$FAILED" -ne 0 ]]; then
  exit 1
fi

"$PY_BIN" -m experiments.phase1.profile_motivation \
  --mode merge \
  --merge-dir "$OUT_DIR" \
  --out "$OUT_DIR/phase1_motivation_profile.json"

echo "=== Stage 2 Profile Completed Cleanly at $(date -Is) ==="
