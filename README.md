# TrustGraph-DFL CIFAR-10 Reproduction & FPR Study

PyTorch decentralized federated learning (DFL) reproduction and robustness audit of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026), managed with `uv`.

- **Final Phase 1 Report (`927` A100 runs across literal audit, 3-seed `50k` baseline, and 4-lever FPR diagnostic):** [report/phase1_report.md](report/phase1_report.md)
  - Locked `50k` ResNet-18 3-seed baseline (`150` runs, `0/150` NaNs): `92.87%` clean consensus (`FedAvg` `93.29%`), sole defense holding `91.5–92.9%` consensus across `label_flip_all`, `model_poison`, and `random_noise` (`0.00%` Byz accept), and #1 per-peer accuracy across Ring (`84.7%`), Random (`89.3%`), Small-World (`89.5%`), and `50%` Byzantine (`89.5%`).
  - Four-lever honest FPR study (`40` runs, `0/40` NaNs): receiver-anchored displacement (`rxdelta`) reaches `6.5–9.5%` honest FPR (`val=iid`), while multi-round EMA (`beta=0.85`) halves strict `sender` FPR to `21.0–22.2%` and cuts `model_poison` Byzantine acceptance to `14.32%`.

---

## Repository Layout

```text
.
├── pyproject.toml                  # uv project and dependency specification
├── CLAUDE.md                       # figures4papers figure rules & encodings
├── src/
│   ├── aggregators.py              # FedAvg, TrustGraph-DFL (sender/rxdelta/scoped/EMA/mclip), BALANCE, Krum, Trimmed Mean
│   ├── attacks.py                  # none, label_flip, label_flip_all, model_poison, random_noise
│   ├── data.py                     # CIFAR-10 & synthetic loaders, balanced Dirichlet split (local/iid val)
│   ├── models.py                   # SimpleCIFARCNN, GroupNormCIFARCNN, PretrainedResNet18CIFAR
│   ├── simulator.py                # Synchronous DFL simulator, metrics, and JSON/JSONL logging
│   └── topology.py                 # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
├── experiments/phase1/
│   ├── suites.py                   # Experiment suite definitions (smoke, full, paper_match_50k, fpr_lever1-4)
│   ├── run.py                      # CLI runner with multi-GPU sharding & --skip-existing
│   ├── summarize.py                # Aggregates summary.json files into Markdown tables
│   ├── plot_figures.py             # Generates all 6 final PNG+PDF figures in report/figures/
│   ├── figstyle.py                 # figures4papers house style (unedited skill copy)
│   ├── figroles.py                 # Fixed method color/marker/dash encodings
│   └── slurm_run.sh                # Multi-GPU Slurm launcher with hardware/git run-record capture
├── tests/
│   └── test_phase1.py              # Unit and offline synthetic DFL integration tests
└── report/
    ├── phase1_report.md            # Unified final Phase 1 report
    ├── data/                       # Committed JSON datasets (paper_match_50k 3-seed + fpr_levers)
    └── figures/                    # 300 dpi PNG + header-free PDF figures (Fig. 2, Table 1, Fig. 3a/3b, 4, 5)
```

---

## Setup & Usage

```bash
# Install dependencies and run unit tests
uv sync
uv run pytest -q tests/test_phase1.py

# Run 5-round local CIFAR-10 smoke test
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke

# Regenerate all 6 final figures from committed report/data/*.json
uv run python -m experiments.phase1.plot_figures

# Cluster execution (Slurm on octave, after explicit approval)
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --job-name=peersieve_p1 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh \
  --mode suite --suite paper_match_50k --skip-existing \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/phase1_50k
```
