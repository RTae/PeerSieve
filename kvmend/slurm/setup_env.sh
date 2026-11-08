#!/bin/bash
# CPU-only setup job for the KVMend PoC on cluster ja (decision 6). Holds no GPU.
#   sbatch -p a100 -A a100 --nodelist=octave --cpus-per-task=4 --mem=16G --time=01:00:00 --open-mode=append \
#     -J kvmend-setup -o /home/tae/kvmend/logs/%x-%j.out /home/tae/kvmend/slurm/setup_env.sh
# It makes a uv project in /home/tae/kvmend, adds SGLang 0.5.1 from the Tsinghua PyPI mirror, and downloads the
# model weights from hf-mirror.com into /home/tae/models. Every step prints what it did, so the log is the record.
set -euo pipefail
export PATH=/home/tae/.local/bin:$PATH
ROOT=${KVMEND_ROOT:-/home/tae/kvmend}
MODELS=${KVMEND_MODELS:-/home/tae/models}
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}
export UV_HTTP_TIMEOUT=300
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$ROOT" "$MODELS" "$ROOT/logs"
cd "$ROOT"

echo "== job ${SLURM_JOB_ID:-local} on $(hostname) at $(date '+%F %T %Z')"
echo "== uv $(uv --version)"
echo "== network"
for url in "$UV_DEFAULT_INDEX/sglang/" "$HF_ENDPOINT" https://pypi.org/simple/ https://download.pytorch.org/whl/; do
  code=$(curl -sS -m 15 -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "fail")
  echo "   $code  $url"
done

echo "== uv project"
if [ ! -f pyproject.toml ]; then
  uv init --bare --python 3.12 --name kvmend
fi
uv add "sglang[srt]==0.5.1"
uv run python - <<'EOF'
import importlib.metadata as md
import torch
for p in ("sglang", "torch", "transformers", "sgl-kernel", "flashinfer-python", "triton"):
    try:
        print(f"   {p} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"   {p} not installed")
print(f"   torch CUDA build {torch.version.cuda}")
EOF

echo "== models"
uv run python - <<EOF
import os
from huggingface_hub import snapshot_download
for repo in ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "NousResearch/Meta-Llama-3.1-8B-Instruct"):
    path = snapshot_download(repo, local_dir=os.path.join("$MODELS", repo.split("/")[1]),
                             allow_patterns=["*.json", "*.safetensors", "tokenizer*"], max_workers=8)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
    print(f"   {repo} -> {path} ({size / 2**30:.2f} GiB)")
EOF

if [ -d "$ROOT/tests" ]; then
  echo "== CPU tests"
  uv run python -m pytest -q "$ROOT/tests" || echo "   CPU tests failed"
fi
echo "== done at $(date '+%F %T %Z')"
