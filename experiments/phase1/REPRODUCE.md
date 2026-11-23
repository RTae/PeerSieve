# Phase 1 experiments

This guide covers environment checks, paper-aligned experiment suites, multi-GPU Slurm execution, and figure regeneration for [report/phase1_report.md](../../report/phase1_report.md).

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
# Full 50k ResNet-18 suite covering Fig. 2, Fig. 3a, Table 1, Fig. 3b, and Section 5.5 (14 runs/seed, 42 total runs)
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

On the cluster, [slurm_run.sh](slurm_run.sh) shards a suite across all GPUs allocated to the Slurm job and records job metadata, GPU topology, and `nvidia-smi` output.

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh --mode suite --suite paper_resnet18_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/phase1_50k
```

## Summarizing results and regenerating figures

Generate Markdown summary tables from any run output directory with `summarize.py`. Regenerate all four PNG and header-free PDF figures in `report/figures/` from the committed JSON datasets with `plot_figures.py`.

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k --title "TrustGraph-DFL Summary"
uv run python -m experiments.phase1.plot_figures
```

Run `pdftotext` on the exported PDFs to confirm they contain no header or footnote text.
