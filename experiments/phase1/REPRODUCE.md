# Reproducing TrustGraph-DFL Phase 1 experiments

This guide covers environment checks, experiment suites, multi-GPU Slurm execution, and figure regeneration for [report/phase1_report.md](../../report/phase1_report.md).

## Environment and quick check

Install dependencies with `uv`, verify the unit and synthetic integration tests, and run the fast smoke suite to confirm the setup.

```bash
uv sync
uv run --no-sync pytest -v

# Quick 4-run smoke check
uv run python -m experiments.phase1.run --mode suite --suite smoke --dataset cifar10 --output-root outputs/phase1_smoke
```

## Running the experiment suites

All TrustGraph-DFL Phase 1 suites live in [suites.py](suites.py) and execute through `experiments.phase1.run`.

```bash
# Locked 14-config TrustGraph-DFL 50k suite (per seed, 42 total runs across seeds 42, 43, 44)
uv run python -m experiments.phase1.run --mode suite --suite paper_match_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/phase1_50k

# Literal 10k 2-conv CNN audit (200 rounds across seeds 42, 43, 44)
uv run python -m experiments.phase1.run --mode suite --suite full \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42 43 44 --output-root outputs/phase1_10k

# Four-lever honest FPR diagnostics (8 runs each; suites are fpr_lever1, fpr_lever1b, fpr_lever2, fpr_lever3, fpr_lever4)
uv run python -m experiments.phase1.run --mode suite --suite fpr_lever1 \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/phase1_fpr_lever1
```

On the cluster, [slurm_run.sh](slurm_run.sh) shards a suite across all GPUs allocated to the Slurm job and saves job metadata, GPU topology, and `nvidia-smi` records.

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh --mode suite --suite paper_match_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/phase1_50k
```

## Summarizing results and regenerating figures

Summarize any output directory into aggregated tables, and regenerate all four PNG and header-free PDF figures in `report/figures/` from the committed datasets.

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k --title "TrustGraph-DFL Summary"
uv run python -m experiments.phase1.plot_figures
```

Verify that exported PDFs contain no header or footnote text by inspecting them with `pdftotext`.
