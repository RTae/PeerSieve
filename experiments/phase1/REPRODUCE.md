# Phase 1 experiments

This guide covers environment checks, paper-aligned experiment suites, multi-GPU Slurm execution, and figure regeneration for [report/phase1/reproduce_report.md](../../report/phase1/reproduce_report.md) and [report/phase1/motivation_report.md](../../report/phase1/motivation_report.md).

## Environment and quick check

Install dependencies with `uv`, run the unit and synthetic integration tests, and run the 4-run smoke suite.

```bash
uv sync
uv run --no-sync pytest -v

# Quick 4-run smoke check
uv run python -m experiments.phase1.run --mode suite --suite smoke --dataset cifar10 --output-root outputs/phase1_smoke
```

## Running the experiment suites

[suites.py](suites.py) defines all paper-aligned experiment suites, and `experiments.phase1.run` runs them.

```bash
# Full 50k ResNet-18 suite covering Fig. 2, Fig. 3a, Table 1, Fig. 3b, and Section 5.5 (14 runs per seed, 42 total runs)
uv run python -m experiments.phase1.run --mode suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/phase1_50k

# Run an individual paper section on 50k ResNet-18 (fig3a_attacks, table1_topologies, fig3b_byz_sweep, sec5_5_ablations)
uv run python -m experiments.phase1.run --mode suite --suite fig3a_attacks \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/fig3a_attacks

# Literal Section 5 setup on 10k CIFAR-10 and 2-conv CNN (31 configs, 200 rounds across seeds 42, 43, 44)
uv run python -m experiments.phase1.run --mode suite --suite paper_cnn_10k \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42 43 44 --output-root outputs/phase1_10k

# Fig. 3b FPR runs with receiver-anchored delta (8 runs each: fig3b_fpr_rxdelta, fig3b_fpr_rxdelta_tau)
uv run python -m experiments.phase1.run --mode suite --suite fig3b_fpr_rxdelta \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/fig3b_fpr_rxdelta
```

On the cluster, [slurm_run.sh](slurm_run.sh) runs either sharded multi-GPU sweeps (`--mode suite`) or multi-process distributed DFL (`--mode distributed`, launching `--n-nodes` peer processes round-robin across the allocated GPUs over persistent TCP sockets), while recording job metadata, GPU topology, and `nvidia-smi` output.

```bash
mkdir -p logs
# Sharded 20-peer suite across 8x A100 GPUs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh --mode suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/phase1_50k

# Real 8-GPU distributed DFL (one peer process per A100 GPU over TcpPeerTransport)
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:15:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh --mode distributed --dataset cifar10 --model resnet18_pretrained \
  --samples-per-node 2500 --val-split iid --lr 0.005 --lr-schedule cosine --augment --grad-clip-norm 10.0 \
  --trustgraph-clip-norm --trustgraph-clip-mode model --trustgraph-score-threshold 0.003 \
  --topology random --attack model_poison --byzantine-frac 0.25 --rounds 5 --eval-every 5 \
  --seeds 42 --output-root outputs/phase1_dist_smoke
```

## Inter-node TCP transfer benchmark

Measure point-to-point ResNet-18 FP32 TCP transfer times between two cluster nodes (CPU-only) using `--mode transfer_bench`.

```bash
# Run on server node (rank 0) and client node (rank 1) targeting the 1 Gb/s Ethernet IP
srun --ntasks=2 --ntasks-per-node=1 uv run --no-sync python -m experiments.phase1.run \
  --mode transfer_bench --bench-role auto --bench-host 172.23.15.13 --bench-port 29501 \
  --bench-out outputs/phase1_transfer_bench/ja3_ja4_transfer.json
```

## Summarizing results and regenerating figures

Generate Markdown accuracy and system-cost tables from any run output directory with `summarize.py`, and optionally export system metrics to JSON with `--export-system-json`. Regenerate all six PNG and header-free PDF figures in `report/figures/` from the committed JSON datasets with `plot_figures.py`.

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k --title "TrustGraph-DFL Summary"
uv run python -m experiments.phase1.summarize outputs/phase1_motivation \
  --title "TrustGraph-DFL System Cost Summary" \
  --export-system-json report/data/phase1_system_metrics_50k.json \
  --transfer-bench-json outputs/phase1_transfer_bench/ja3_ja4_transfer.json
uv run python -m experiments.phase1.plot_figures
```

Run `pdftotext` on the exported PDFs to confirm they contain no header or footnote text.
