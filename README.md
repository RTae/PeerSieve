# TrustGraph-DFL CIFAR-10 Reproduction & FPR Study

PyTorch decentralized federated learning (DFL) reproduction, robustness audit, and four-lever honest False Positive Rate (FPR) study of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026), managed with `uv`.

**Unified Final Report:** [report/phase1_report.md](report/phase1_report.md) (`927` A100 runs across all stages, figures in [report/figures/](report/figures/), datasets in [report/data/](report/data/)).

---

## Headline Results (`927` A100 Runs on `octave`)

| Stage (`Runs`, `NaNs`) | Setup | Key Verified Results |
|---|---|---|
| **Literal 2-Conv CNN Audit** (`225` runs, 3 seeds) | `10k` CIFAR-10 (`500`/node, $\alpha_{\mathrm{dir}}=0.5$, `val=local`), `SimpleCIFARCNN`, `200` rounds | Attack-free ceiling `53.27%` peer / `56.52%` cons (`FedAvg`); `descent_neg` ($-\nabla F_i$) beats `paper_pos` ($+\nabla F_i$) at every 10–50% Byz fraction; unclipped cosine hits `3/3` NaNs on `model_poison` ($\lambda=10$) and `random_noise` ($\sigma=0.5$). |
| **Locked Full-`50k` 3-Seed Baseline** (`150` runs, **`0/150` NaNs**) | `50k` CIFAR-10 (`2,500`/node, `250` `val=iid`), `resnet18_pretrained`, `25` rounds, `grad_clip_norm=10.0`, `mclip 2 r_i`, $\tau=0.003$ | Clean consensus **`92.87%`** (`FedAvg` **`93.29%`**); sole defense holding **`91.5–92.9%` consensus** across `label_flip_all` (`92.38%`), `model_poison` (`91.53%`), and `random_noise` (`92.33%`, **`0.00%` Byz accept**); #1 per-peer accuracy on **Ring (`84.70%`), Random (`89.28%`), Small-World (`89.48%`)**, and **50% Byz (`89.53%`)**; §5.5 ablations (`no_relu` **`-35.9 pp`**, `euclidean` **`-9.0 pp`**). |
| **Four-Lever Honest FPR Study** (`40` runs, **`0/40` NaNs**) | Full `50k` ResNet-18, `seed 42`, testing L1 (`rxdelta`), L1b (`rxdelta` $\tau$ sweep), L2 (`steps/round`), L3 (`layer_mean` / subspaces), L4 (`ema`) one by one | **L1 (`rxdelta`)** cuts honest FPR `38.8–43.9% → 7.4–9.5%` (matching Fig. 3b `3–9%`, `+0.5–1.5 pp` peer acc); **L2 & L3** stay at `38.1–45.0%` FPR; **L4 (`ema=0.85`, strict `sender`)** halves FPR to `21.0–22.2%` and cuts `model_poison` Byz accept to **`14.32%`**. |

*(Plus `512` intermediate calibration / Option-C `10k` A100 runs establishing `grad_clip_norm=10.0` and $\tau=0.003$.)*

---

## Repository Layout

```text
.
├── AGENTS.md                       # Repository scope, editing/testing, and lab Slurm conventions
├── TrustGraph-DFL.pdf              # Target paper (Xu et al., CIBDA 2026)
├── pyproject.toml                  # uv project and dependency specification
├── src/
│   ├── aggregators.py              # FedAvg, TrustGraph-DFL (sender/rxdelta/scoped/EMA/mclip), BALANCE, Krum, Trimmed Mean
│   ├── attacks.py                  # none, label_flip, label_flip_all, model_poison, random_noise
│   ├── data.py                     # CIFAR-10 & synthetic loaders, balanced Dirichlet split (local/iid val)
│   ├── models.py                   # SimpleCIFARCNN, GroupNormCIFARCNN, PretrainedResNet18CIFAR
│   ├── simulator.py                # Synchronous DFL simulator, metrics, and JSON/JSONL logging
│   └── topology.py                 # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
├── experiments/phase1/
│   ├── suites.py                   # Named suites (smoke, full/literal, paper_match_50k/baseline_50k, fpr_lever1-4)
│   ├── run.py                      # CLI runner with multi-GPU sharding & --skip-existing
│   ├── summarize.py                # Aggregates summary.json files into Markdown tables
│   ├── plot_figures.py             # Generates all 6 final PNG+PDF figures in report/figures/
│   ├── figstyle.py                 # figures4papers house style (unedited skill copy)
│   ├── figroles.py                 # Fixed method color/marker/dash encodings
│   └── slurm_run.sh                # Multi-GPU Slurm launcher with hardware/git run-record capture
├── tests/
│   └── test_phase1.py              # Unit and offline synthetic DFL integration tests (21 tests)
└── report/
    ├── phase1_report.md            # Unified final Phase 1 report
    ├── data/                       # Committed JSON datasets (paper_match_50k 3-seed + fpr_levers)
    └── figures/                    # 300 dpi PNG + header-free PDF figures (Fig. 2, 3a, Table 1, Fig. 3b, 4, 5)
```

---

## Setup, Tests, and Experiment Suites

Install dependencies and run all unit + offline integration tests:

```bash
uv sync
uv run pytest -v
```

### Available Suites (`experiments/phase1/suites.py`)
- `smoke` (`12` runs): fast 2-attack (`label_flip`, `model_poison`) × 6-method local sanity sweep.
- `full` (aliases `literal`, `step_a`; `75` runs/seed): literal `10k` CIFAR-10 2-conv CNN audit matrix.
- `paper_match_50k` (alias `baseline_50k`; `50` runs/seed): locked full-`50k` ResNet-18 4-panel baseline.
- `fpr_lever1`, `fpr_lever1b`, `fpr_lever2`, `fpr_lever3`, `fpr_lever4` (`8` runs each): four-lever honest FPR diagnostics.

### Key CLI Options (`experiments/phase1/run.py`)
- `--mode`: `single` or `suite`; `--suite <name>`; `--seeds 42 43 44`; `--shard-index` / `--num-shards`; `--skip-existing`.
- `--model`: `cnn2x2`, `cnn2x2_gn`, `resnet18_pretrained`, `resnet18_scratch`.
- `--val-split`: `local` (10% local non-IID shard) or `iid` (balanced validation set); `--grad-clip-norm`.
- `--val-grad-sign`: `descent_neg` ($-\nabla F_i$) or `paper_pos` ($+\nabla F_i$).
- `--trustgraph-clip-norm`, `--trustgraph-clip-mode` (`update | model`), `--trustgraph-clip-factor`, `--trustgraph-score-threshold`.
- `--trustgraph-delta-mode` (`sender | receiver`), `--trustgraph-score-scope` (`global | layer_mean | backbone_only | head_only`), `--trustgraph-ema-beta`.

### Local Commands & Cluster Execution

```bash
# 5-round local CIFAR-10 smoke test
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke

# Summarize run directories and regenerate all 6 final figures
uv run python -m experiments.phase1.summarize outputs/phase1_smoke --title "Phase 1 Smoke"
uv run python -m experiments.phase1.plot_figures

# Cluster execution (Slurm on octave, after explicit owner approval)
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --job-name=peersieve_p1 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh \
  --mode suite --suite paper_match_50k --skip-existing \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/phase1_50k
```
