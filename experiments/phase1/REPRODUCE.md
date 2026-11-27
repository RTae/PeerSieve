# Experiment reproduction guide

This guide walks step by step through reproducing the CIFAR-10 TrustGraph-DFL experiments, multi-node `1 Gb/s` Ethernet communication measurements, summary tables, and publication figures. It uses the same Typer CLI and layered configuration workflow documented in [README.md](../../README.md).

| Target report | Data source | Figures |
|---|---|---|
| [reproduce_report.md](../../report/phase1/reproduce_report.md) | [phase1_paper_match_50k_3seed.json](../../report/data/phase1_paper_match_50k_3seed.json), [phase1_fpr_rxdelta.json](../../report/data/phase1_fpr_rxdelta.json) | `fig1` to `fig4` |
| [motivation_report.md](../../report/phase1/motivation_report.md) | [phase1_v100_distributed_2node.json](../../report/data/phase1_v100_distributed_2node.json) | `fig5`, `fig6` |

## Step 1. Set up the environment and offline caches

Install dependencies with `uv` and run the test suite to verify the local installation:

```bash
uv sync
uv run --no-sync pytest -v
```

If you run on offline Slurm compute nodes with `--no-download`, fetch CIFAR-10 and the ImageNet-pretrained ResNet-18 checkpoint once on a node with internet access:

```bash
# Save CIFAR-10 into data/cifar-10-batches-py/
uv run python -c "from src.data import load_cifar10_tensors; load_cifar10_tensors('data', download=True)"

# Cache ImageNet-pretrained ResNet-18 weights in ~/.cache/torch/hub/checkpoints/
uv run python -c "from torchvision.models import ResNet18_Weights, resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

*(Small CNN models `cnn2x2`, `cnn2x2_gn`, and `resnet18_scratch` do not need pretrained weights.)*

## Step 2. Run a quick local sanity check

Inspect the registered suites and run the 4-config CPU smoke suite before launching long GPU sweeps:

```bash
uv run python -m experiments.phase1.run list-suites

uv run python -m experiments.phase1.run suite \
  --suite smoke --dataset cifar10 --rounds 5 --eval-every 0 \
  --device cpu --seeds 42 --output-root outputs/phase1_smoke
```

Every completed run writes four files inside its seed folder (`config.json`, `history.jsonl`, `summary.json`, and `run.log`). You can pass any saved `config.json` back into `single --config` or `distributed --config` to replay or tweak that exact run.

## Step 3. Reproduce the TrustGraph-DFL accuracy and FPR results

[suites.py](suites.py) defines named benchmark grids covering every table and figure in [reproduce_report.md](../../report/phase1/reproduce_report.md):

| Suite | Runs / seed | Seeds | Target figure or table |
|---|---:|---:|---|
| `paper_resnet18_50k` | 14 | 3 (`42,43,44`) | Full 50k ResNet-18 reproduction (`fig1`, `fig2`, `fig3`) |
| `fig3a_attacks` | 4 | 3 (`42,43,44`) | Attack regime subset (`fig1`) |
| `table1_topologies` | 3 | 3 (`42,43,44`) | Topology comparison subset (`fig2a`) |
| `fig3b_byz_sweep` | 5 | 3 (`42,43,44`) | 10% to 50% Byzantine sweep subset (`fig2b`) |
| `sec5_5_ablations` | 5 | 3 (`42,43,44`) | Component ablations subset (`fig3a`) |
| `fig3b_fpr_rxdelta` | 8 | 1 (`42`) | Receiver-anchored scoring (`rxdelta`, $\tau=0.003$, `fig4`) |
| `fig3b_fpr_rxdelta_tau` | 8 | 1 (`42`) | Receiver-anchored threshold sweep ($\tau \in \{0.020, 0.035\}$, `fig4`) |
| `paper_cnn_10k` | 31 | 3 (`42,43,44`) | Literal 10k CIFAR-10 / 2-conv CNN reproduction (`ascent` vs. `descent`) |

### Local or single-GPU execution

```bash
# Full 50k ResNet-18 suite (14 configs x 3 seeds = 42 runs)
uv run python -m experiments.phase1.run suite \
  --suite paper_resnet18_50k --dataset cifar10 --eval-every 5 \
  --seeds 42,43,44 --skip-existing --output-root outputs/phase1_50k

# Receiver-anchored (rxdelta) FPR suites (8 runs each)
uv run python -m experiments.phase1.run suite \
  --suite fig3b_fpr_rxdelta --dataset cifar10 --eval-every 5 \
  --seeds 42 --skip-existing --output-root outputs/fig3b_fpr_rxdelta

uv run python -m experiments.phase1.run suite \
  --suite fig3b_fpr_rxdelta_tau --dataset cifar10 --eval-every 5 \
  --seeds 42 --skip-existing --output-root outputs/fig3b_fpr_rxdelta_tau

# Literal 10k / 2-conv CNN suite (31 configs x 3 seeds, 200 rounds)
uv run python -m experiments.phase1.run suite \
  --suite paper_cnn_10k --dataset cifar10 --rounds 200 --eval-every 10 \
  --seeds 42,43,44 --skip-existing --output-root outputs/phase1_10k
```

### Multi-GPU Slurm execution (sharded across 8x A100 on `octave`)

[slurm_run.sh](slurm_run.sh) automatically records `scontrol show job -d`, `nvidia-smi topo -m`, and per-GPU CSV metadata into the Slurm log, then shards `suite` jobs evenly across every allocated GPU using `--shard-index` and `--num-shards`:

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh suite \
  --suite paper_resnet18_50k --dataset cifar10 --no-download --eval-every 5 \
  --seeds 42,43,44 --skip-existing --output-root outputs/phase1_50k
```

## Step 4. Reproduce the real multi-node Ethernet measurements

The communication bottleneck measurements in [motivation_report.md](../../report/phase1/motivation_report.md) run one OS process per peer over real TCP sockets (`TcpPeerTransport`) via `experiments.phase1.run distributed` using the `resnet18_50k` preset (`2,500` images/peer, `25` rounds, `val=iid`, `descent`).

- **Single-node loopback check.** Omit `--dist-iface` to test multi-GPU distributed training over `127.0.0.1` (`torchrun` or `slurm_run.sh distributed`).
- **Inter-node `1 Gb/s` Ethernet.** Pass `--dist-iface enp68s0f0` so peers exchange FP32 payloads across separate V100 servers (`ja3` and `ja4`, partition `open`, driver `525.116.04`, CUDA `11.8` environment `~/venvs/peersieve-v100`).
- **Two topologies measured.** `2` peers (`1` dedicated V100 + NIC per peer, degree `1`) and `4` peers (`2` peers per V100 across 2 nodes, complete graph `--set er_p=1.0`, degree `3`), comparing `--delta-mode sender` (`89.45 MB`/neighbor, 2 vectors) against `--delta-mode receiver` (`44.73 MB`/neighbor, 1 vector).

### Single-node 8-GPU distributed check (`octave`)

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:15:00 \
  --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh distributed \
  --preset resnet18_50k --no-download --n-nodes 8 --topology random \
  --attack model_poison --byzantine-frac 0.25 --rounds 5 --seed 42 \
  --output-root outputs/phase1_dist_smoke
```

### Multi-node 2-V100 inter-node Ethernet runs (`ja3` + `ja4`)

```bash
# 2 peers (1 dedicated V100 + 1 Gb/s NIC per peer, degree 1, sender mode)
srun -p open --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:v100:1 --time=00:10:00 \
  ~/venvs/peersieve-v100/bin/python -m experiments.phase1.run distributed \
  --preset resnet18_50k --device cuda --no-download --n-nodes 2 \
  --topology random --set er_p=1.0 --attack model_poison --byzantine-frac 0.5 \
  --delta-mode sender --dist-iface enp68s0f0 --seed 42 \
  --output-root outputs/phase1_v100_2node/2p_mp_sender

# 4 peers (2 peers per V100 across 2 nodes, degree 3, sender vs. receiver mode)
srun -p open --nodes=2 --ntasks=4 --ntasks-per-node=2 --gres=gpu:v100:1 --time=00:10:00 \
  ~/venvs/peersieve-v100/bin/python -m experiments.phase1.run distributed \
  --preset resnet18_50k --device cuda --no-download --n-nodes 4 \
  --topology random --set er_p=1.0 --attack model_poison --byzantine-frac 0.25 \
  --delta-mode sender --dist-iface enp68s0f0 --seed 42 \
  --output-root outputs/phase1_v100_2node/4p_mp_sender
```

Switch `--delta-mode sender` to `--delta-mode receiver` (or vary `--attack none`, `label_flip_all --set byz_local_epochs=2`, `random_noise`) to reproduce all 8 runs recorded in [phase1_v100_distributed_2node.json](../../report/data/phase1_v100_distributed_2node.json). To measure raw inter-node TCP goodput without GPU training, run `experiments.phase1.run transfer-bench` across two nodes with `--role auto`.

## Step 5. Summarize results, rebuild figures, and verify PDFs

Aggregate any completed run tree into Markdown tables (and optional JSON) with `summarize`:

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_50k \
  --title "TrustGraph-DFL 50k Summary" \
  --out outputs/phase1_50k/summary_table.md
```

Regenerate all six 300 dpi PNG and header-free PDF figures strictly from the committed JSON datasets in [report/data/](../../report/data/), then confirm no header or footnote text appears in the exported PDFs:

```bash
uv run python -m experiments.phase1.plot_figures

# Verify exported PDFs are header-free and footnote-free
for pdf in report/figures/*.pdf; do
  echo "=== $pdf ==="
  pdftotext "$pdf" - | head -n 5
done
```
