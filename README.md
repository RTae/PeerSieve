# PeerSieve

Byzantine-resilient decentralized federated learning (DFL) research codebase built with PyTorch and managed with `uv`.

- **Phase 1 (Complete):** 3-seed CIFAR-10 reproduction and diagnostic audit of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) against `FedAvg`, `Trimmed Mean`, `Krum`, and `BALANCE`, plus an early test of PeerSieve's receiver-norm clipping cap. See [report/phase1_final.md](report/phase1_final.md).
- **Phase 2 (Planned):** PeerSieve mechanisms (CountSketch pre-transfer screening, blockwise INT4 compression with error feedback, two-pass audited streaming, per-edge quarantine) and eight-GPU TinyLlama-1.1B / Dolly-15k evaluation on `octave`.

---

## Repository Layout

```text
.
├── pyproject.toml              # uv project and dependency specification
├── src/                        # Reusable DFL core modules
│   ├── aggregators.py          # FedAvg, TrustGraph-DFL (+ clip/ablations), BALANCE, Krum, Trimmed Mean
│   ├── attacks.py              # none, label_flip (0<->1), label_flip_all, model_poison, random_noise
│   ├── data.py                 # CIFAR-10 & synthetic loaders, balanced Dirichlet split (local/iid val)
│   ├── models.py               # 2-conv 2-FC CNNs (cnn2x2, lenet2x2) and parameter-vector helpers
│   ├── simulator.py            # Synchronous DFL simulator, metrics, and JSON/JSONL logging
│   └── topology.py             # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
├── experiments/
│   └── phase1/
│       ├── run.py              # CLI runner (single runs, named suites, multi-GPU sharding, --skip-existing)
│       ├── summarize.py        # Aggregates summary.json files into Markdown tables (mean ± std, NaNs)
│       └── slurm_run.sh        # Multi-GPU Slurm launcher with hardware/git run-record capture
├── tests/
│   └── test_phase1.py          # Unit and offline synthetic DFL integration tests
└── report/
    ├── phase1_smoke.md         # Local CPU 5-round CIFAR-10 smoke test & diagnostics
    ├── phase1_step_a.md        # Single-seed (seed 42) 75-run A100 reproduction report
    └── phase1_final.md         # Final 3-seed (225-run) A100 reproduction report
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
- `--suite`: `smoke`, `sign_check`, `main_attacks`, `topologies`, `fpr_sweep`, `ablations`, `full` (75 configs; `step_a` kept as alias).
- `--val-grad-sign`: `descent_neg` ($-\nabla F_i$, corrected descent direction) or `paper_pos` ($+\nabla F_i$, literal TrustGraph-DFL Eq. 3 / Alg. 1).
- `--val-split`: `local` (literal paper: 10% held out from each node's non-IID shard) or `iid` (balanced 50-sample validation diagnostic).
- `--trustgraph-clip-norm`: enables PeerSieve's receiver-norm update cap $a_{ij}^{(t)} = \min(1, r_i^{(t)} / \|\Delta\theta_j^{(t)}\|_2)$.
- `--balance-gamma`: BALANCE distance multiplier (default `0.3`; calibrated `0.15` tested in diagnostics).
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

Full tables, hardware provenance, and analysis are in [report/phase1_final.md](report/phase1_final.md) (`225` runs: `75` configs $\times$ `3` seeds on `octave` A100 GPUs):

- **Reproducibility gap vs. `TrustGraph-DFL.pdf`:** On `10,000` CIFAR-10 images ($\alpha_{\mathrm{dir}}=0.5$, 2-conv 2-FC CNN, 200 rounds), even attack-free `FedAvg` reaches `53.27 ± 0.86%` mean benign accuracy (`56.52 ± 0.41%` consensus), far below the `93–95%` claimed in Fig. 3(a). With local non-IID validation sets, `trustgraph(descent_neg)` exhibits `50.3–53.5%` honest FPR rather than `<10%`.
- **Qualitative behaviors confirmed:** `Krum` FPR scales from `8.28%` (10% Byz) to `52.36 ± 1.50%` (50% Byz), while `trustgraph(descent_neg)` remains invariant (`51.23 ± 3.51%` at 50% Byz). Corrected `descent_neg` consistently beats literal `paper_pos` across all Byzantine fractions (`+1.0` to `+2.5 pp` consensus accuracy; `10–24 pp` lower Byzantine acceptance).
- **Norm clipping prevents `NaN` explosions:** Every unclipped `TrustGraph-DFL` variant hit `NaN` on `1–3/3` seeds under `model_poison` ($\lambda=10$) and `3/3` seeds under `random_noise` ($\sigma=0.5$). Adding PeerSieve's self-update norm cap eliminated all `NaN`s (`0/3` on both attacks: `26.39 ± 0.33%` on `model_poison`, `28.76 ± 0.66%` mean / `40.56 ± 0.44%` consensus on `random_noise`).
