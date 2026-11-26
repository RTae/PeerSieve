# TrustGraph-DFL

This repository implements [TrustGraph-DFL.pdf](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) in PyTorch for Byzantine-resilient decentralized federated learning on CIFAR-10.

| Document | Contents |
|---|---|
| [report/phase1/reproduce_report.md](report/phase1/reproduce_report.md) | TrustGraph-DFL reproduction findings, gap table, and figures |
| [report/phase1/motivation_report.md](report/phase1/motivation_report.md) | Real multi-node V100 `1 Gb/s` Ethernet communication gaps (wasted bandwidth and FP32 transfer time) motivating Phase 2 |
| [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md) | Full experiment suites, Slurm launch commands, and figure regeneration steps |

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) package manager
- An NVIDIA CUDA GPU for full 50k ResNet-18 training (a CPU is enough for unit tests and short smoke runs)

### Dataset and pretrained model

Training requires the CIFAR-10 dataset, stored under `data/cifar-10-batches-py/`. Runs with `--model resnet18_pretrained` also require ImageNet-pretrained `ResNet-18` weights (`ResNet18_Weights.DEFAULT` from `torchvision`), cached under `~/.cache/torch/hub/checkpoints/` or `$TORCH_HOME/hub/checkpoints/`. Models `cnn2x2`, `cnn2x2_gn`, and `resnet18_scratch` need no pretrained weights.

Online runs download both files on first use. On offline machines or Slurm compute nodes using `--no-download`, run these commands once on an internet-connected node after `uv sync`.

```bash
# Download and extract CIFAR-10 into data/cifar-10-batches-py/
uv run python -c "from src.data import load_cifar10_tensors; load_cifar10_tensors('data', download=True)"

# Cache ImageNet-pretrained ResNet-18 weights (for --model resnet18_pretrained)
uv run python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

## Installation

Install runtime and development dependencies into a local `.venv` and run the test suite to verify the setup.

```bash
uv sync
uv run pytest -v
```

When working offline after setup, run `uv run --no-sync pytest -v`.

## How to run

Run a 5-round local CPU smoke suite to check end-to-end training and logging.

```bash
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke
```

Run a single custom experiment with `--mode single`.

```bash
uv run python -m experiments.phase1.run \
  --mode single --dataset cifar10 --model resnet18_pretrained \
  --topology random --attack model_poison --byzantine-frac 0.3 \
  --val-split iid --val-grad-sign descent_neg --trustgraph-clip-norm \
  --rounds 25 --seeds 42 --output-root outputs/single_run
```

Run multi-process, multi-GPU decentralized FL with `--mode distributed` (one or more independent peer processes per GPU over persistent topology-scoped TCP sockets via `torchrun` or `slurm_run.sh`; pass `--n-nodes 20` to pack all 20 peers round-robin across 8 GPUs).

```bash
uv run torchrun --standalone --nproc_per_node=8 -m experiments.phase1.run \
  --mode distributed --dataset cifar10 --model resnet18_pretrained \
  --samples-per-node 2500 --val-split iid --lr 0.005 --lr-schedule cosine \
  --augment --grad-clip-norm 10.0 --trustgraph-clip-norm --trustgraph-clip-mode model \
  --trustgraph-score-threshold 0.003 --topology random --attack model_poison \
  --byzantine-frac 0.25 --rounds 25 --eval-every 5 --seeds 42 --output-root outputs/dist_run
```

### Available suites

[experiments/phase1/suites.py](experiments/phase1/suites.py) defines all named experiment suites.

| Suite | Runs per seed | Paper target |
|---|---:|---|
| `smoke` | 4 | Fast local sanity check across two attacks and two gradient signs |
| `paper_cnn_10k` | 31 | Literal Section 5 setup on 10k CIFAR-10 and 2-conv CNN (200 rounds) |
| `paper_resnet18_50k` | 14 | Full 50k ResNet-18 reproduction (Fig. 2, Fig. 3a, Table 1, Fig. 3b, Section 5.5) |
| `fig3a_attacks` | 4 | Fig. 2 and Fig. 3a no-attack and 30% Byzantine attacks (50k ResNet-18) |
| `table1_topologies` | 3 | Table 1 Ring, Random, and Small-World under 30% label flip (50k ResNet-18) |
| `fig3b_byz_sweep` | 5 | Fig. 3b 10% to 50% Byzantine stress sweep (50k ResNet-18) |
| `sec5_5_ablations` | 5 | Section 5.5 ablations (full, self-weights 0.3 and 0.7, Euclidean, No ReLU) |
| `fig3b_fpr_rxdelta` | 8 | Fig. 3b FPR with receiver-anchored delta (`rxdelta`, $\tau=0.003$) |
| `fig3b_fpr_rxdelta_tau` | 8 | Fig. 3b receiver-anchored threshold sweep ($\tau \in \{0.020, 0.035\}$) |

### Common CLI flags

[experiments/phase1/run.py](experiments/phase1/run.py) accepts the following main options.

| Flag group | Options |
|---|---|
| Execution | `--mode single \| suite \| distributed \| transfer_bench`, `--suite <name>`, `--seeds 42 43 44`, `--device auto \| cpu \| cuda`, `--skip-existing`, `--log-level`, `--log-file` |
| Sharding and transport | `--shard-index`, `--num-shards`, `--dist-iface`, `--dist-port` |
| Model and data | `--model cnn2x2 \| cnn2x2_gn \| resnet18_pretrained \| resnet18_scratch`, `--val-split local \| iid`, `--grad-clip-norm` |
| TrustGraph scoring | `--val-grad-sign descent_neg \| paper_pos`, `--trustgraph-score-threshold`, `--trustgraph-delta-mode sender \| receiver` |
| Norm clipping | `--trustgraph-clip-norm`, `--trustgraph-clip-mode update \| model`, `--trustgraph-clip-factor` |

Summarize completed runs into Markdown tables and regenerate publication figures from committed data.

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_smoke --title "TrustGraph-DFL Smoke"
uv run python -m experiments.phase1.plot_figures
```

For full 50k, 10k, `rxdelta`, and multi-GPU Slurm reproduction commands, see [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md).

## Experiment results

See [report/phase1/reproduce_report.md](report/phase1/reproduce_report.md) for the TrustGraph-DFL reproduction results, adjustments, and remaining accuracy or FPR gaps, and [report/phase1/motivation_report.md](report/phase1/motivation_report.md) for the real multi-node V100 `1 Gb/s` Ethernet bandwidth and round-time gaps motivating Phase 2. The committed JSON datasets are in [report/data/](report/data/), and the PNG and PDF figures are in [report/figures/](report/figures/).

## Project structure

```text
.
├── AGENTS.md                       # Repository scope, testing rules, and lab Slurm conventions
├── pyproject.toml                  # uv project and dependency specification
├── src/
│   ├── aggregators.py              # TrustGraph-DFL aggregation (sender/receiver delta and clipping)
│   ├── attacks.py                  # none, label_flip, label_flip_all, model_poison, random_noise
│   ├── data.py                     # CIFAR-10 and synthetic test loaders, Dirichlet split
│   ├── distributed.py              # Multi-process and multi-GPU DFL peer runner
│   ├── logging_utils.py            # Loguru configuration, host/rank formatting, and per-run log sinks
│   ├── metrics.py                  # Per-round accuracy, FPR, rejection streak, and system-cost tracker
│   ├── models.py                   # SimpleCIFARCNN, GroupNormCIFARCNN, PretrainedResNet18CIFAR
│   ├── peer.py                     # Self-contained PeerNode runtime shared by local and distributed runs
│   ├── simulator.py                # Single-process synchronous DFL runner over LocalTransport
│   ├── topology.py                 # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
│   └── transport.py                # PeerTransport, LocalTransport, TcpPeerTransport, TCP benchmark
├── experiments/phase1/
│   ├── REPRODUCE.md                # Full experiment, Slurm, and figure reproduction guide
│   ├── suites.py                   # Named TrustGraph-DFL suites
│   ├── run.py                      # Slim CLI runner (single, suite, distributed, transfer_bench)
│   ├── summarize.py                # Aggregates summary.json files into Markdown tables and JSON
│   ├── plot_figures.py             # Generates PNG and PDF figures in report/figures/
│   ├── figstyle.py                 # Shared matplotlib house style
│   ├── figroles.py                 # Fixed colors, markers, and line styles
│   └── slurm_run.sh                # Multi-GPU Slurm launcher (sharded sweeps and distributed peers)
├── tests/
│   └── test_phase1.py              # Unit and offline synthetic DFL integration suite
└── report/
    ├── phase1/
    │   ├── reproduce_report.md     # TrustGraph-DFL reproduction and FPR report
    │   └── motivation_report.md    # Real multi-node V100 Ethernet gaps motivating Phase 2
    ├── data/                       # Committed 50k 3-seed, rxdelta FPR, and multi-node V100 JSON datasets
    └── figures/                    # 300 dpi PNG and header-free PDF figures
```

## Development

- **Running tests.** Run `uv run --no-sync pytest -v` after any edit under [src/](src/), [experiments/](experiments/), or [tests/](tests/).
- **Adding experiments.** Define new benchmark grids in [experiments/phase1/suites.py](experiments/phase1/suites.py) and keep core simulator, attack, model, topology, and aggregation modules in [src/](src/).
- **Updating figures.** Generate figures from committed JSON datasets via `uv run python -m experiments.phase1.plot_figures`, leave [experiments/phase1/figstyle.py](experiments/phase1/figstyle.py) unchanged, and confirm exported PDFs contain no header or footnote text using `pdftotext`.
- **Untracked runtime files.** Keep `outputs/`, `data/`, and `logs/` out of Git.
