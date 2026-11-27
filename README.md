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
uv run python -m experiments.phase1.run suite \
  --suite smoke --dataset cifar10 --rounds 5 --eval-every 0 \
  --device cpu --seeds 42 --output-root outputs/phase1_smoke
```

Every completed run writes four files inside its output directory:
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

## CLI guide

All command-line tools are built with [Typer](https://typer.tiangolo.com/) and `loguru`. Pass `--help` to any command or subcommand to inspect its exact flags, defaults, and allowed choices:

```bash
uv run python -m experiments.phase1.run --help
uv run python -m experiments.phase1.run single --help
uv run python -m experiments.phase1.run suite --help
uv run python -m experiments.phase1.run distributed --help
uv run python -m experiments.phase1.run transfer-bench --help
uv run python -m experiments.phase1.summarize --help
uv run python -m experiments.phase1.plot_figures --help
```

### 1. Experiment runner (`experiments.phase1.run`)

| Subcommand | How it runs | When to use it |
|---|---|---|
| `single` | All peers in one process (`LocalTransport`) | Test one custom attack, topology, or hyperparameter setting |
| `suite` | Runs a named grid from [suites.py](experiments/phase1/suites.py) | Reproduce paper tables and sweeps (supports multi-GPU sharding) |
| `distributed` | One OS process per peer over TCP (`TcpPeerTransport`) | Measure real multi-GPU or multi-node Ethernet communication |
| `transfer-bench` | Point-to-point TCP payload timer between two hosts | Measure raw cluster Ethernet goodput without GPU training |
| `list-suites` | Prints registered suites and run counts | Check available `--suite` names before launching a sweep |

#### `single` — run one custom experiment

Outputs are written under `<output-root>/single/<tag>/seed_<id>/`. Pass multiple seeds as a comma-separated list (`--seeds 42,43,44`) or by repeating `--seed`.

```bash
uv run python -m experiments.phase1.run single \
  --dataset cifar10 --model resnet18_pretrained \
  --n-nodes 20 --samples-per-node 2500 --val-split iid \
  --rounds 25 --lr 0.005 --lr-schedule cosine --augment --grad-clip-norm 10.0 \
  --topology random --er-p 0.3 --attack model_poison --byzantine-frac 0.3 \
  --val-grad-sign descent_neg --trustgraph-clip-norm --trustgraph-clip-mode model \
  --trustgraph-score-threshold 0.003 --seeds 42,43,44 --skip-existing \
  --output-root outputs/single_run
```

#### `suite` — run a named benchmark suite

Outputs are written under `<output-root>/<suite>/<run-name>/seed_<id>/`, plus `<output-root>/<suite>/suite_summaries.json`. For multi-GPU parallel execution, split jobs evenly with `--shard-index` and `--num-shards` (or let [slurm_run.sh](experiments/phase1/slurm_run.sh) shard automatically across allocated GPUs).

```bash
# List all available suites
uv run python -m experiments.phase1.run list-suites

# Run the 14-config 50k ResNet-18 suite across 3 seeds
uv run python -m experiments.phase1.run suite \
  --suite paper_resnet18_50k --dataset cifar10 --eval-every 5 \
  --seeds 42,43,44 --skip-existing --output-root outputs/phase1_50k
```

#### `distributed` — run real multi-process peers over TCP

Each peer runs in its own OS process (`RANK` / `WORLD_SIZE` from `torchrun`, or `SLURM_PROCID` / `SLURM_NTASKS` from Slurm `srun`). Rank `0` writes `config.json`, `history.jsonl`, `summary.json`, and `run.log` under `<output-root>/distributed/<tag>/seed_<id>/`.

- Omit `--dist-iface` for single-node loopback TCP (`127.0.0.1`).
- Pass `--dist-iface enp68s0f0` for inter-node Ethernet TCP across separate servers.
- Set `--trustgraph-delta-mode sender` (`89.45 MB`/neighbor, 2 vectors) or `receiver` (`44.73 MB`/neighbor, 1 vector).

```bash
# Single-node 4-GPU launch with torchrun
uv run torchrun --standalone --nproc_per_node=4 -m experiments.phase1.run distributed \
  --dataset cifar10 --model resnet18_pretrained --n-nodes 4 --samples-per-node 2500 \
  --val-split iid --rounds 25 --eval-every 5 --lr 0.005 --lr-schedule cosine \
  --augment --grad-clip-norm 10.0 --trustgraph-clip-norm --trustgraph-clip-mode model \
  --trustgraph-score-threshold 0.003 --topology random --er-p 1.0 \
  --attack model_poison --byzantine-frac 0.25 --seed 42 --output-root outputs/dist_run
```

#### `transfer-bench` — measure raw TCP link speed

Run with `--role server` on node A and `--role client --host <node-A-ip>` on node B (or `--role auto` under `srun --ntasks=2 --ntasks-per-node=1`):

```bash
uv run python -m experiments.phase1.run transfer-bench \
  --role client --host 172.23.15.13 --port 29501 \
  --repeats 25 --burst-repeats 10 --out outputs/transfer_bench.json
```

### 2. Result summarizer (`experiments.phase1.summarize`)

Recursively finds every `summary.json` under `ROOT`, deduplicates `(config, seed)`, logs Markdown accuracy and system-cost tables via `loguru`, and optionally writes a Markdown file (`--out`) or aggregated JSON (`--export-system-json`).

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k \
  --title "TrustGraph-DFL 50k Summary" \
  --out outputs/phase1_50k/summary_table.md
```

| Argument / Flag | Default | Description |
|---|---|---|
| `ROOT` | *(required)* | Directory searched recursively for `summary.json` files |
| `--out` | `None` | Write the rendered Markdown report to this file path |
| `--title` | `"Phase 1 Experiment Summary"` | Top-level Markdown heading |
| `--export-system-json` | `None` | Export aggregated system metrics across runs to JSON |
| `--transfer-bench-json` | `None` | Embed a `transfer-bench` JSON report inside exported JSON |
| `--job-id`, `--commit` | `None` | Optional Slurm job ID and git hash metadata for exported JSON |
| `--log-level` | `INFO` | Minimum `loguru` log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### 3. Figure builder (`experiments.phase1.plot_figures`)

Generates all 6 publication 300 dpi PNG and header-free PDF figures strictly from committed JSON datasets under `report/data/`.

```bash
uv run python -m experiments.phase1.plot_figures \
  --data-50k report/data/phase1_paper_match_50k_3seed.json \
  --data-rxdelta report/data/phase1_fpr_rxdelta.json \
  --data-v100 report/data/phase1_v100_distributed_2node.json \
  --out-dir report/figures
```

For full Slurm scripts (`slurm_run.sh`), multi-node V100 Ethernet commands, and 10k/50k reproduction recipes, see [experiments/phase1/REPRODUCE.md](experiments/phase1/REPRODUCE.md).

## CLI option reference

### Model and data (`single`, `distributed`, plus base flags on `suite`)

| Flag | Default (`single` / `distributed`) | Allowed values / type | Description |
|---|---|---|---|
| `--dataset` | `cifar10` | `cifar10`, `synthetic` | Training dataset (`synthetic` runs fast offline tests) |
| `--data-dir` | `data` | path | Directory storing or receiving CIFAR-10 files |
| `--no-download` | `False` | flag | Fail instead of downloading CIFAR-10 (for offline nodes) |
| `--model` | `cnn2x2` / `resnet18_pretrained` | `cnn2x2`, `cnn2x2_gn`, `resnet18_pretrained`, `resnet18_scratch` | Neural network architecture |
| `--n-nodes` | `20` / `4` | int (`>= 2`) | Number of peers (`distributed` uses `WORLD_SIZE`) |
| `--samples-per-node` | `500` / `2500` | int (`>= 10`) | Total train + validation samples per peer |
| `--alpha-dir` | `0.5` | float | Dirichlet non-IID skew (smaller = more heterogeneous) |
| `--val-frac` | `0.1` | float | Fraction of each peer's samples held out for validation |
| `--val-split` | `local` / `iid` | `local`, `iid` | `local` non-IID shard validation vs. stratified balanced `iid` |

### Training (`single`, `distributed`)

| Flag | Default (`single` / `distributed`) | Allowed values / type | Description |
|---|---|---|---|
| `--rounds` | `200` / `25` | int (`>= 1`) | Total decentralized communication rounds |
| `--local-epochs` | `1` | int (`>= 1`) | Local SGD epochs per round on benign peers |
| `--lr` | `0.01` / `0.005` | float | Initial SGD learning rate |
| `--lr-schedule` | `constant` / `cosine` | `constant`, `cosine` | Learning rate decay schedule across rounds |
| `--momentum` | `0.9` | float | SGD momentum factor |
| `--weight-decay` | `1e-4` | float | L2 weight decay during local SGD |
| `--batch-size` | `64` | int (`>= 1`) | Mini-batch size for local SGD |
| `--augment` | `False` | flag | Random crop and horizontal flip on CIFAR-10 |
| `--grad-clip-norm` | `0.0` | float | Local gradient L2 norm cap (`0.0` disables; 50k uses `10.0`) |
| `--val-grad-weight-decay` | `False` | flag | Add L2 weight decay term to validation gradient |

### Topology and attacks (`single`, `distributed`)

| Flag | Default (`single` / `distributed`) | Allowed values / type | Description |
|---|---|---|---|
| `--topology` | `random` | `ring`, `random`, `small_world` | Peer graph family |
| `--er-p` | `0.3` | float | Edge probability for `random` Erdos-Renyi (`1.0` = complete) |
| `--ws-k`, `--ws-p` | `4`, `0.3` | int, float | Degree and rewiring probability for `small_world` |
| `--attack` | `label_flip` / `model_poison` | `none`, `label_flip`, `label_flip_all`, `model_poison`, `random_noise` | Byzantine attack applied on malicious peers |
| `--byzantine-frac` | `0.3` / `0.25` | float (`0.0` to `1.0`) | Fraction of peers marked Byzantine |
| `--byz-local-epochs` | `0` | int (`>= 0`) | Extra local epochs for label-flip attackers (`0` uses `--local-epochs`) |
| `--poison-lambda` | `10.0` | float | Negative scaling multiplier for `model_poison` |
| `--noise-sigma` | `0.5` | float | Gaussian std for `random_noise` |

### TrustGraph scoring and clipping (`single`, `distributed`)

| Flag | Default (`single` / `distributed`) | Allowed values / type | Description |
|---|---|---|---|
| `--alpha` | `0.5` | float (`0.0` to `1.0`) | Self-weight kept on local model during aggregation |
| `--val-grad-sign` | `descent_neg` | `descent_neg`, `paper_pos` | Reference direction (`-grad` descent vs. literal paper `+grad`) |
| `--trustgraph-delta-mode` | `sender` | `sender`, `receiver` | Neighbor anchor: `sender` (2 vectors) vs. `receiver` (`rxdelta`, 1 vector) |
| `--trustgraph-score-threshold` | `0.0` / `0.003` | float | Minimum cosine score $\tau$ before ReLU zeroing |
| `--trustgraph-metric` | `cosine` | `cosine`, `euclidean` | Similarity metric for neighbor trust scores |
| `--no-relu` | `False` | flag | Keep negative scores instead of zeroing with ReLU |
| `--trustgraph-clip-norm` | `False` | flag | Clip accepted neighbor step norm relative to local update |
| `--trustgraph-clip-mode` | `update` / `model` | `update`, `model` | Clip sender update delta or model displacement from receiver |
| `--trustgraph-clip-factor` | `2.0` | float | Clipping cap multiplier relative to receiver's own update norm |

### Execution, sharding, TCP, and logging

| Flag | Applies to | Default | Description |
|---|---|---|---|
| `--seed`, `--seeds` | `single`, `suite`, `distributed` | `42` | Random seed(s); `single`/`suite` accept comma-separated (`42,43,44`) or repeated flags |
| `--device` | `single`, `suite`, `distributed` | `auto` | Compute device (`auto`, `cpu`, `cuda`, `mps`) |
| `--eval-every` | `single`, `suite`, `distributed` | `5` | Evaluate test accuracy every N rounds (`0` = final round only) |
| `--skip-existing` | `single`, `suite` | `False` | Skip runs that already have `summary.json` |
| `--shard-index`, `--num-shards` | `suite` | `0`, `1` | Partition suite jobs across parallel GPU workers |
| `--dist-iface` | `distributed` | `None` (loopback) | Linux NIC name for inter-node TCP (e.g. `enp68s0f0`) |
| `--dist-port` | `distributed` | `29600` | Base TCP port; rank `r` listens on `dist_port + r` |
| `--output-root` | `single`, `suite`, `distributed` | `outputs/phase1` | Root folder for run outputs and `run.log` |
| `--log-level` | all commands | `INFO` | Minimum `loguru` level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--log-file` | `run` subcommands | `None` | Extra file receiving all CLI logs in addition to per-run `run.log` |
| `--quiet` | `single`, `suite`, `distributed` | `False` | Raise stderr threshold to `WARNING` while keeping file logs at `--log-level` |

### Named suites (`experiments/phase1/suites.py`)

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

## Project structure

```text
.
├── AGENTS.md                       # Repository scope, testing rules, and cluster conventions
├── TrustGraph-DFL.pdf              # Target paper (Xu et al., CIBDA 2026)
├── pyproject.toml                  # uv project dependencies (torch, loguru, typer)
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
│   ├── run.py                      # Typer CLI (single, suite, distributed, transfer-bench, list-suites)
│   ├── summarize.py                # Typer CLI aggregating summary.json files into Markdown and JSON
│   ├── plot_figures.py             # Typer CLI generating PNG and header-free PDF figures
│   ├── figstyle.py                 # Shared matplotlib house style (leave unedited)
│   ├── figroles.py                 # Fixed colors, markers, and line styles
│   └── slurm_run.sh                # Slurm launcher for sharded suites and distributed runs
├── tests/
│   └── test_phase1.py              # Unit, CLI, logging, and offline synthetic DFL test suite
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
