# PeerSieve

This repository reproduces [TrustGraph-DFL.pdf](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) in PyTorch on CIFAR-10 and measures its real multi-node Ethernet communication bottlenecks to motivate PeerSieve.

| Document | Contents |
|---|---|
| [report/phase1/reproduce_report.md](report/phase1/reproduce_report.md) | TrustGraph-DFL reproduction results, accuracy and FPR gaps, and figures |
| [report/phase1/motivation_report.md](report/phase1/motivation_report.md) | Real multi-node V100 `1 Gb/s` Ethernet bandwidth and round-time bottlenecks |
| [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md) | Full 50k/10k suites, Slurm commands, and figure reproduction steps |

## Quick start

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and an NVIDIA CUDA GPU for full ResNet-18 training (a CPU is enough for unit tests and short smoke checks).

### Step 1. Install dependencies and run tests

```bash
uv sync
uv run --no-sync pytest -v
```

### Step 2. Download data and pretrained weights (optional for online runs)

Online runs download CIFAR-10 and ResNet-18 weights automatically on first use. If you plan to run offline or on Slurm compute nodes with `--no-download`, run these two commands once on a machine with internet access:

```bash
# Save CIFAR-10 into data/cifar-10-batches-py/
uv run python -c "from src.data import load_cifar10_tensors; load_cifar10_tensors('data', download=True)"

# Cache ImageNet-pretrained ResNet-18 weights in ~/.cache/torch/hub/checkpoints/
uv run python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

*(Small CNN models `cnn2x2`, `cnn2x2_gn`, and `resnet18_scratch` do not need pretrained weights.)*

### Step 3. Run a quick local smoke check

Run a 5-round CPU check across 4 small configurations to confirm training, logging, and file outputs work end to end:

```bash
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke
```

Every completed run creates four files inside its output directory:
- `config.json`: full hyperparameters and graph summary
- `history.jsonl`: per-round accuracy, FPR, and timing records
- `summary.json`: final aggregated metrics for the run
- `run.log`: timestamped `loguru` log for that run

### Step 4. Summarize results and build figures

```bash
# Log a Markdown summary table for any output folder (add --out table.md to save it)
uv run python -m experiments.phase1.summarize outputs/phase1_smoke --title "Smoke Check"

# Rebuild all 6 PNG and header-free PDF figures in report/figures/ from report/data/
uv run python -m experiments.phase1.plot_figures
```

## Running experiments

[experiments/phase1/run.py](experiments/phase1/run.py) supports three training modes depending on what you want to test:

| Mode | How it runs | When to use it |
|---|---|---|
| `--mode single` | All peers in one process (`LocalTransport`) | Test one custom attack, topology, or hyperparameter setting |
| `--mode suite` | Runs a named grid from [suites.py](experiments/phase1/suites.py) | Reproduce paper tables and sweeps (supports multi-GPU sharding) |
| `--mode distributed` | One OS process per peer over real TCP sockets (`TcpPeerTransport`) | Measure real multi-GPU or multi-node Ethernet communication |

### Single custom run

```bash
uv run python -m experiments.phase1.run \
  --mode single --dataset cifar10 --model resnet18_pretrained \
  --topology random --attack model_poison --byzantine-frac 0.3 \
  --val-split iid --val-grad-sign descent_neg --trustgraph-clip-norm \
  --rounds 25 --seeds 42 --output-root outputs/single_run
```

### Named benchmark suite

```bash
uv run python -m experiments.phase1.run \
  --mode suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --skip-existing \
  --output-root outputs/phase1_50k
```

### Multi-process distributed run (real TCP sockets)

Launch one peer process per GPU locally with `torchrun` (or across multiple physical servers over Ethernet with Slurm `srun` and `--dist-iface enp68s0f0`; see [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md)):

```bash
uv run torchrun --standalone --nproc_per_node=4 -m experiments.phase1.run \
  --mode distributed --dataset cifar10 --model resnet18_pretrained \
  --n-nodes 4 --samples-per-node 2500 --val-split iid \
  --lr 0.005 --lr-schedule cosine --augment --grad-clip-norm 10.0 \
  --trustgraph-clip-norm --trustgraph-clip-mode model --trustgraph-score-threshold 0.003 \
  --topology random --er-p 1.0 --attack model_poison --byzantine-frac 0.25 \
  --rounds 25 --eval-every 5 --seeds 42 --output-root outputs/dist_run
```

For full Slurm scripts (`slurm_run.sh`), multi-node V100 Ethernet runs, and 10k/50k reproduction commands, see [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md).

## Reference

### Named suites

Defined in [experiments/phase1/suites.py](experiments/phase1/suites.py):

| Suite | Runs per seed | Purpose |
|---|---:|---|
| `smoke` | 4 | Fast sanity check across two attacks and two gradient signs |
| `paper_cnn_10k` | 31 | Literal Section 5 setup on 10k CIFAR-10 and 2-conv CNN (200 rounds) |
| `paper_resnet18_50k` | 14 | Full 50k ResNet-18 reproduction (Fig. 2, Fig. 3a, Table 1, Fig. 3b, Section 5.5) |
| `fig3a_attacks` | 4 | No-attack and 30% Byzantine attacks on 50k ResNet-18 |
| `table1_topologies` | 3 | Ring, Random, and Small-World under 30% label flip |
| `fig3b_byz_sweep` | 5 | 10% to 50% Byzantine stress sweep under label flip |
| `sec5_5_ablations` | 5 | Ablations (full, self-weights 0.3 and 0.7, Euclidean, No ReLU) |
| `fig3b_fpr_rxdelta` | 8 | Receiver-anchored scoring (`rxdelta`, $\tau=0.003$) |
| `fig3b_fpr_rxdelta_tau` | 8 | Receiver-anchored threshold sweep ($\tau \in \{0.020, 0.035\}$) |

### Main CLI flags

Common options for `python -m experiments.phase1.run`:

| Group | Flags |
|---|---|
| Execution and logging | `--mode`, `--suite`, `--seeds`, `--device auto \| cpu \| cuda`, `--skip-existing`, `--log-level DEBUG \| INFO \| WARNING \| ERROR`, `--log-file`, `--quiet` |
| Distributed and sharding | `--n-nodes`, `--dist-iface`, `--dist-port`, `--shard-index`, `--num-shards` |
| Model and data | `--dataset cifar10 \| synthetic`, `--model cnn2x2 \| cnn2x2_gn \| resnet18_pretrained \| resnet18_scratch`, `--samples-per-node`, `--val-split local \| iid`, `--augment`, `--grad-clip-norm` |
| TrustGraph scoring | `--val-grad-sign descent_neg \| paper_pos`, `--trustgraph-delta-mode sender \| receiver`, `--trustgraph-score-threshold`, `--trustgraph-clip-norm`, `--trustgraph-clip-mode update \| model`, `--trustgraph-clip-factor` |

## Project structure

```text
.
├── AGENTS.md                       # Repository scope, testing rules, and cluster conventions
├── TrustGraph-DFL.pdf              # Target paper (Xu et al., CIBDA 2026)
├── pyproject.toml                  # uv project dependencies
├── src/
│   ├── aggregators.py              # TrustGraph-DFL aggregation (sender/receiver delta and clipping)
│   ├── attacks.py                  # none, label_flip, label_flip_all, model_poison, random_noise
│   ├── data.py                     # CIFAR-10 and synthetic loaders, Dirichlet partitioning
│   ├── distributed.py              # Multi-process and multi-node DFL peer runner
│   ├── logging_utils.py            # Loguru formatting, host/rank context, and per-run log sinks
│   ├── metrics.py                  # Accuracy, FPR, Byzantine acceptance, and wire/timing tracker
│   ├── models.py                   # SimpleCIFARCNN, GroupNormCIFARCNN, PretrainedResNet18CIFAR
│   ├── peer.py                     # Self-contained PeerNode runtime shared by local and distributed runs
│   ├── simulator.py                # Single-process synchronous DFL runner over LocalTransport
│   ├── topology.py                 # Ring, Erdos-Renyi random, and Watts-Strogatz small-world graphs
│   └── transport.py                # PeerTransport, LocalTransport, TcpPeerTransport, TCP benchmark
├── experiments/phase1/
│   ├── REPRODUCE.md                # Full experiment, Slurm, and figure reproduction guide
│   ├── suites.py                   # Named benchmark suites
│   ├── run.py                      # Unified CLI runner (single, suite, distributed, transfer_bench)
│   ├── summarize.py                # Aggregates summary.json files into Markdown tables and JSON
│   ├── plot_figures.py             # Generates PNG and header-free PDF figures in report/figures/
│   ├── figstyle.py                 # Shared matplotlib house style (leave unedited)
│   ├── figroles.py                 # Fixed colors, markers, and line styles
│   └── slurm_run.sh                # Slurm launcher for sharded sweeps and distributed runs
├── tests/
│   └── test_phase1.py              # Unit and offline synthetic DFL integration suite
└── report/
    ├── phase1/
    │   ├── reproduce_report.md     # TrustGraph-DFL reproduction and FPR report
    │   └── motivation_report.md    # Real multi-node V100 Ethernet motivation report
    ├── data/                       # Committed 50k 3-seed, rxdelta FPR, and V100 Ethernet JSON datasets
    └── figures/                    # 300 dpi PNG and header-free PDF figures
```

## Development

- Run `uv run --no-sync pytest -v` after editing any code under [src/](src/), [experiments/](experiments/), or [tests/](tests/).
- Regenerate figures strictly from [report/data/](report/data/) with `uv run python -m experiments.phase1.plot_figures`, leave [experiments/phase1/figstyle.py](experiments/phase1/figstyle.py) unchanged, and verify exported PDFs have no header text via `pdftotext`.
- Keep `outputs/`, `data/`, and `logs/` out of Git.
