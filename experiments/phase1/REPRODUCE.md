# Phase 1 experiments

This guide covers environment checks, paper-aligned experiment suites, multi-GPU and multi-node Slurm execution, and figure regeneration for [report/phase1/reproduce_report.md](../../report/phase1/reproduce_report.md) and [report/phase1/motivation_report.md](../../report/phase1/motivation_report.md).

## Environment and quick check

Install dependencies with `uv`, run the test suite, inspect available suites, and run the 4-run smoke suite.

```bash
uv sync
uv run --no-sync pytest -v

# Inspect registered suites and Typer CLI options
uv run python -m experiments.phase1.run list-suites
uv run python -m experiments.phase1.run suite --help

# Quick 4-run smoke check
uv run python -m experiments.phase1.run suite \
  --suite smoke --dataset cifar10 --rounds 5 --eval-every 0 --device cpu \
  --seeds 42 --output-root outputs/phase1_smoke
```

## Running the experiment suites

[suites.py](suites.py) defines all paper-aligned experiment suites, and `experiments.phase1.run suite` runs them.

```bash
# Full 50k ResNet-18 suite covering Fig. 2, Fig. 3a, Table 1, Fig. 3b, and Section 5.5 (14 runs per seed, 42 total runs)
uv run python -m experiments.phase1.run suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42,43,44 --skip-existing --output-root outputs/phase1_50k

# Run an individual paper section on 50k ResNet-18 (fig3a_attacks, table1_topologies, fig3b_byz_sweep, sec5_5_ablations)
uv run python -m experiments.phase1.run suite --suite fig3a_attacks \
  --dataset cifar10 --eval-every 5 --seeds 42,43,44 --output-root outputs/fig3a_attacks

# Literal Section 5 setup on 10k CIFAR-10 and 2-conv CNN (31 configs, 200 rounds across seeds 42, 43, 44)
uv run python -m experiments.phase1.run suite --suite paper_cnn_10k \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42,43,44 --output-root outputs/phase1_10k

# Fig. 3b FPR runs with receiver-anchored delta (8 runs each: fig3b_fpr_rxdelta, fig3b_fpr_rxdelta_tau)
uv run python -m experiments.phase1.run suite --suite fig3b_fpr_rxdelta \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/fig3b_fpr_rxdelta
```

On the cluster, [slurm_run.sh](slurm_run.sh) wraps either sharded multi-GPU sweeps (`suite`) or multi-process distributed DFL (`distributed`, launching `--n-nodes` peer processes round-robin across the allocated GPUs over persistent TCP sockets), while recording job metadata, GPU topology, and `nvidia-smi` output.

```bash
mkdir -p logs
# Sharded 20-peer suite across 8x A100 GPUs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42,43,44 --skip-existing --output-root outputs/phase1_50k

# Real 8-GPU distributed DFL (one peer process per A100 GPU over TcpPeerTransport)
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:15:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh distributed --preset resnet18_50k \
  --n-nodes 8 --topology random --attack model_poison --byzantine-frac 0.25 \
  --rounds 5 --seed 42 --output-root outputs/phase1_dist_smoke
```

## Multi-node V100 Ethernet motivation runs

Run real multi-node distributed TrustGraph-DFL across two V100 servers (`ja3` and `ja4`) over physical `1 Gb/s` Ethernet (`enp68s0f0`) with a CUDA `11.8` PyTorch environment matching driver `525.116.04`.

```bash
# Example: 4 peers (2 per V100 node) over physical 1 Gb/s Ethernet on partition open
srun -p open --nodes=2 --ntasks=4 --ntasks-per-node=2 --gres=gpu:v100:1 --time=00:10:00 \
  ~/venvs/peersieve-v100/bin/python -m experiments.phase1.run distributed \
  --preset resnet18_50k --device cuda --no-download --n-nodes 4 \
  --topology random --set er_p=1.0 --attack model_poison --byzantine-frac 0.25 \
  --delta-mode sender --dist-iface enp68s0f0 --seed 42 \
  --output-root outputs/phase1_v100_2node
```

## Summarizing results and regenerating figures

Generate Markdown accuracy and system-cost tables from any run output directory with `summarize.py` (pass `--out` to write a file), and regenerate all six PNG and header-free PDF figures in `report/figures/` from the committed JSON datasets with `plot_figures.py`.

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k \
  --title "TrustGraph-DFL Summary" \
  --out outputs/phase1_50k/summary_table.md

uv run python -m experiments.phase1.plot_figures
```

Run `pdftotext` on the exported PDFs to confirm they contain no header or footnote text.
