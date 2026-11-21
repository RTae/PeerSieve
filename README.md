# PeerSieve

Byzantine-resilient decentralized federated learning (DFL) research codebase built with PyTorch and managed with `uv`.

- **Phase 1 (Complete + Phase 1c Paper-Match Calibration Ready):**
  - Literal 3-seed CIFAR-10 reproduction audit (`225` runs) of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026): [report/phase1_final.md](report/phase1_final.md).
  - Stabilized working reproduction (`110` runs across stabilized 2-conv CNN + pretrained ResNet-18 using descent sign, model-space norm clipping, and $\tau=0.02$): [report/phase1_stabilized_ab.md](report/phase1_stabilized_ab.md).
- **Phase 2 (Core Mechanisms Verified Locally; 8-GPU LLM Evaluation Planned):**
  - Core PeerSieve modules implemented & verified against Proposal Table 2 (including 4-process `torch.distributed` Gloo tests): [report/phase2_local_verification.md](report/phase2_local_verification.md).
  - Planned next: eight-GPU TinyLlama-1.1B / Dolly-15k evaluation on `octave`.

---

## Repository Layout

```text
.
├── pyproject.toml              # uv project and dependency specification
├── src/                        # Reusable DFL & PeerSieve core modules
│   ├── aggregators.py          # FedAvg, TrustGraph-DFL (+ clip/ablations), BALANCE, Krum, Trimmed Mean
│   ├── attacks.py              # none, label_flip (0<->1), label_flip_all, model_poison, random_noise
│   ├── audit.py                # PeerSieve two-pass audited streaming aggregation & receiver-norm cap
│   ├── compression.py          # Blockwise INT4 quantization + error feedback, tiled decode, Top-k baseline
│   ├── data.py                 # CIFAR-10 & synthetic loaders, balanced Dirichlet split (local/iid val)
│   ├── models.py               # CNNs (cnn2x2, cnn2x2_gn, lenet2x2), ResNet-18, augmentation, vector helpers
│   ├── quarantine.py           # Per-edge EMA backoff probation, permanent hard-fault bans, & referrals
│   ├── simulator.py            # Synchronous DFL simulator, metrics, and JSON/JSONL logging
│   ├── sketch.py               # Deterministic layer-proportional FP32->FP16 CountSketch (32 KiB)
│   ├── topology.py             # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
│   └── transport.py            # Token-bucket receive limiter & multi-process Gloo exchange protocol
├── experiments/
│   ├── phase1/
│   │   ├── run.py              # CLI runner (single runs, named suites, multi-GPU sharding, --skip-existing)
│   │   ├── summarize.py        # Aggregates summary.json files into Markdown tables (mean ± std, NaNs)
│   │   └── slurm_run.sh        # Multi-GPU Slurm launcher with hardware/git run-record capture
│   └── phase2/
│       └── verify_mechanisms.py # Local pre-GPU verification suite for Proposal Table 2
├── tests/
│   ├── test_phase1.py          # Phase 1 unit and offline synthetic DFL integration tests
│   └── test_phase2.py          # Phase 2 unit and 4-process Gloo quarantine/referral tests
└── report/
    ├── phase1_smoke.md              # Local CPU 5-round CIFAR-10 smoke test & diagnostics
    ├── phase1_step_a.md             # Single-seed (seed 42) 75-run A100 literal reproduction report
    ├── phase1_final.md              # Final 3-seed (225-run) A100 literal reproduction report
    ├── phase1_stabilized_ab.md      # Option C (110-run) stabilized CNN + pretrained ResNet-18 report
    └── phase2_local_verification.md # Pre-GPU verification of all 4 PeerSieve mechanisms + Gloo tests
```

---

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Local install
uv sync

# On the THU cluster (faster downloads via THU TUNA mirror when syncing without --frozen,
# or warm the uv cache once):
uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

Run the unit and offline integration test suite:

```bash
uv run pytest -v
```

---

## Running Phase 1 Experiments

### 1. Local CIFAR-10 Smoke Test

```bash
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke
```

### 2. Single Custom Run

```bash
uv run python -m experiments.phase1.run \
  --mode single --dataset cifar10 \
  --topology random --attack model_poison --byzantine-frac 0.3 \
  --aggregator trustgraph --val-grad-sign descent_neg --trustgraph-clip-norm \
  --rounds 200 --eval-every 10 --seeds 42 43 44
```

Key CLI toggles in `experiments/phase1/run.py`:
- `--suite`: `smoke`, `sign_check`, `main_attacks`, `topologies`, `fpr_sweep`, `ablations`, `full` (75 literal configs; `step_a` alias), `stabilized_ab` (110 Option C stabilized CNN + pretrained ResNet-18 configs).
- `--model`: `cnn2x2`, `cnn2x2_gn`, `lenet2x2`, `resnet18_pretrained`, `resnet18_scratch`.
- `--val-grad-sign`: `descent_neg` ($-\nabla F_i$, corrected descent direction) or `paper_pos` ($+\nabla F_i$, literal TrustGraph-DFL Eq. 3 / Alg. 1).
- `--val-split`: `local` (literal paper: 10% held out from each node's non-IID shard) or `iid` (balanced 50-sample validation diagnostic).
- `--trustgraph-clip-norm`, `--trustgraph-clip-mode` (`update | model`), `--trustgraph-clip-factor`, `--trustgraph-threshold`: PeerSieve receiver-norm cap and noise-floor cosine threshold ($\tau$).
- `--lr-schedule` (`constant | cosine`), `--augment`, `--balance-gamma`: training stabilization and BALANCE distance multiplier (`0.3` literal; calibrated `0.15` / `0.015`).
- `--shard-index`, `--num-shards`, `--skip-existing`: resumable parallel sharding across GPUs.

### 3. Summarize Outputs to Markdown

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_smoke \
  --title "Phase 1 Summary" --out report/summary.md
```

### 4. Cluster Execution (Slurm on `octave`)

Stage CIFAR-10 under `data/` and create `logs/` on the login node before submitting:

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --job-name=peersieve_p1 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh \
  --mode suite --suite full --skip-existing \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42 43 44 \
  --output-root outputs/phase1
```

`slurm_run.sh` automatically logs the git commit, `scontrol show job -d`, `nvidia-smi topo -m`, `nvidia-smi --query-gpu=...`, and PyTorch/CUDA versions, then launches one worker process per allocated GPU.

---

## Phase 1 Headline Results

Full tables, hardware provenance, and analysis are in [report/phase1_final.md](report/phase1_final.md) (`225` literal 3-seed runs) and [report/phase1_stabilized_ab.md](report/phase1_stabilized_ab.md) (`110` Option C stabilized runs on `octave` A100 GPUs):

- **Literal reproduction gap ([report/phase1_final.md](report/phase1_final.md)):** On `10,000` CIFAR-10 images ($\alpha_{\mathrm{dir}}=0.5$, 2-conv 2-FC CNN, 200 rounds), attack-free `FedAvg` reaches `53.27 ± 0.86%` mean benign accuracy (`56.52 ± 0.41%` consensus), far below the `93–95%` claimed in Fig. 3(a). Literal `paper_pos` ($+\nabla F_i$) is inverted relative to SGD updates, unclipped cosine hits `NaN` on `model_poison` and `random_noise`, and local non-IID validation yields `50.3–53.5%` honest FPR rather than `<10%`.
- **Working stabilized reproduction ([report/phase1_stabilized_ab.md](report/phase1_stabilized_ab.md)):** Combining corrected `descent_neg` ($-\nabla F_i$), model-space receiver-norm clipping (`mclip`, $2 r_i$), and a noise-floor threshold ($\tau=0.02$) produces a strong defense across both stabilized `cnn2x2_gn` and ImageNet-pretrained `ResNet-18`:
  - On `cnn2x2_gn`, `trustgraph(stab)` is the only method surviving `model_poison` ($\lambda=10$) at both $\alpha_{\mathrm{dir}}=0.5$ (`45.32%` / `54.99%` cons) and $\alpha_{\mathrm{dir}}=5.0$ (`57.20%` / `58.61%` cons), and cuts `random_noise` Byzantine acceptance to `0.00%`.
  - On pretrained `ResNet-18` ($\alpha_{\mathrm{dir}}=0.5$, 50 rounds, clean ceiling `90.34–90.73%`), `trustgraph(stab)` beats all baselines on `label_flip_all` at `30%` Byz (`83.73%` / `88.36%` cons vs. `Krum` `77.68%`, `Trimmed Mean` `74.49%`, `FedAvg` `65.75%`), holds **`84.41%` / `87.67%` cons at `50%` Byz** (while `Krum` collapses to `33.53%` / `75.8%` FPR), and is the sole clean winner on `random_noise` (`82.91%` / `88.23%` cons, `0.00%` Byz accept).
