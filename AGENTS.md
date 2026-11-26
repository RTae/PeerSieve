# Peersieve

## Scope and source files

This repository contains the PyTorch CIFAR-10 reproduction, robustness audit, receiver-anchored (`rxdelta`) false positive rate (FPR) study, and system-cost measurements of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026), managed with `uv`.

- **`src/`.** Core decentralized FL components: [peer.py](src/peer.py) (`PeerNode`), [transport.py](src/transport.py) (`LocalTransport`, `TcpPeerTransport`, `run_transfer_benchmark`), [metrics.py](src/metrics.py) (`MetricsTracker`), [distributed.py](src/distributed.py) (`run_distributed_experiment`), [simulator.py](src/simulator.py) (`run_experiment`), [aggregators.py](src/aggregators.py) (`aggregate_trustgraph`), [attacks.py](src/attacks.py), [data.py](src/data.py), [models.py](src/models.py) (`SimpleCIFARCNN`, `GroupNormCIFARCNN`, `PretrainedResNet18CIFAR`), and [topology.py](src/topology.py).
- **`experiments/phase1/`.** Suite definitions ([suites.py](experiments/phase1/suites.py)), CLI runner supporting local, sharded suite, distributed multi-GPU, and TCP transfer benchmark modes ([run.py](experiments/phase1/run.py)), summary aggregator ([summarize.py](experiments/phase1/summarize.py)), publication figure generator ([plot_figures.py](experiments/phase1/plot_figures.py)), house style ([figstyle.py](experiments/phase1/figstyle.py), leave unedited), fixed visual encodings ([figroles.py](experiments/phase1/figroles.py)), reproduction guide ([REPRODUCE.md](experiments/phase1/REPRODUCE.md)), and Slurm launcher ([slurm_run.sh](experiments/phase1/slurm_run.sh)).
- **`tests/`.** Unit and offline synthetic DFL integration suite ([test_phase1.py](tests/test_phase1.py)).
- **`report/`.** Phase 1 reports under `report/phase1/` ([reproduce_report.md](report/phase1/reproduce_report.md) and [motivation_report.md](report/phase1/motivation_report.md)), committed TrustGraph-DFL 3-seed, `rxdelta` FPR, and system-metrics datasets under `report/data/`, and 300 dpi PNG plus header-free PDF figures under `report/figures/`.

Keep `outputs/`, `data/`, and `logs/` out of Git. Do not restore deleted intermediate reports or obsolete datasets unless the owner asks.

## Editing, testing, and figures

- Run `uv run pytest -v` after modifying any code under `src/`, `experiments/`, or `tests/`.
- Keep these five methodological distinctions clear in code, reports, and explanations:
  - **Validation gradient sign.** Compare literal `paper_pos` ($+\nabla F_i$) against corrected `descent_neg` ($-\nabla F_i$).
  - **Magnitude control.** Compare unclipped cosine against model-space receiver-norm clipping (`mclip 2 r_i`) and `grad_clip_norm=10.0`.
  - **Validation split.** Compare literal `val=local` (10% local non-IID shard) against balanced `val=iid` (250 stratified samples per node on 50k).
  - **Reference anchor.** Compare strict `sender` ($\theta_j^{t+1/2}-\theta_j^t$) against receiver-anchored `rxdelta` ($\theta_j^{t+1/2}-\theta_i^t$).
  - **Accuracy metrics.** Report per-peer benign mean (`Final Acc`) alongside consensus parameter average (`Cons Acc`), FPR, and Byzantine acceptance rate.
- Leave `experiments/phase1/figstyle.py` unchanged when drawing figures unless explicitly instructed. Generate figures strictly from committed `report/data/*.json` via `uv run python -m experiments.phase1.plot_figures`, visually inspect generated PNGs for layout clarity, and verify exported PDFs contain no header or footnote text (`pdftotext`).

## Lab hardware and Slurm

- Local edits, tests, and figure builds do not need cluster access. Commit or push only with the owner's explicit approval. Use short commit messages without trailers.
- Get explicit approval for each experiment or Slurm allocation. Before asking, specify the runs, node, GPU type and count, planned start window, wall time, and maximum GPU-hours.

### Cluster testbed

Full 50k CIFAR-10 ResNet-18 evaluations run on `octave`, sharding parallel runs across allocated GPUs (`experiments/phase1/slurm_run.sh`). The inventory below is from lab notes collected on November 4, 2026, not a live hardware check. Verify the assigned devices, topology, and software inside each approved job.

- **`octave`.** Eight NVIDIA A100 PCIe 40 GB GPUs (about 39.52 GiB usable per card), two AMD EPYC 7742 CPUs (256 logical CPUs total), and 512 GB host RAM. The lab recorded eight NUMA nodes with one GPU per NUMA node.
- **Links.** PCIe Gen4 x16, with GPU pairs reported as `SYS` in `nvidia-smi topo -m`, and no NVLink. The external network was 1 Gb/s Ethernet without RDMA.
- **Host transfers.** On 2026-09-30, one GPU at a time, the lab recorded pinned host-copy rates of 22.8 to 23.2 GiB/s on buses `01:00.0` through `61:00.0` and 19.6 to 21.6 GiB/s on buses `81:00.0` through `e1:00.0`. These are historical hardware measurements, not experiment results. Use matching devices and report the PCI bus IDs when comparing timings.
- **Software snapshot.** On 2026-09-28, the lab recorded NVIDIA driver 610.57.4 and CUDA 13.3 on `octave`. These are not pinned dependencies. Record the actual driver and CUDA runtime used by the training environment.
- **Smaller nodes.** The lab listed `ja1` through `ja4` with one PCIe V100 16 GB each. Use them only for approved small checks, not as substitutes for `octave` A100 runs.

Prior `octave` device map. Slurm indices are not ordered by PCI bus, and the order within the `4 to 6` group was not recorded. Recheck the map instead of assuming an index always selects the same card.

| Slurm index | PCI bus |
|---|---|
| 0 | `61:00.0` |
| 1 | `41:00.0` |
| 2 | `01:00.0` |
| 3 | `25:00.0` |
| 4 to 6 | `81:00.0`, `c1:00.0`, `e1:00.0` (order unknown) |
| 7 | `a1:00.0` |

### Access and scheduling

- Use the documented login alias `ssh yes` for cluster `ja`. Use the login node for file and job management only. Run compute through Slurm. Do not SSH directly to compute nodes or disable host-key checking.
- The earlier lab configuration used `-p a100 -A a100` for `octave` with a one-hour limit. It listed `open` as a ten-minute partition for V100s on `ja3` and `ja4`. Check current access, limits, and storage rules rather than assuming these values still apply. The lab references are in `~/workspace/thu/lab/boltz-cp`: `CLAUDE.md`, `note/5-reference/cluster-use.md`, `note/5-reference/cluster-hardware.md`, and `benchmarks/cluster/README.md`.
- Use `sinfo -a -e -o '%P|%a|%l|%D|%C|%G|%N'` and `sacctmgr show assoc user="$USER" format=User,Account,Partition,QOS` to check the partition and account. Use `squeue -u "$USER"`, `scontrol show job -d <job-id>`, and `scontrol show node octave -d` to inspect jobs and node allocation.
- For approved `sbatch` submissions, specify the confirmed partition and account (e.g., `--nodes=1 --nodelist=octave --gres=gpu:a100:8`). Set an explicit `--time` and use `--open-mode=append` with a fixed log location such as `--output=logs/%x-%j.out`. Create the log directory before submission. Keep at most ten of our jobs pending or running, or fewer if the lab limit is lower. Avoid running our other jobs on the node during timing runs.
- Use Git and a clean checkout pinned to the experiment commit for runs of record. Verify the commit before submitting. Do not force a pull over local changes. Check free space and get approval before writing large data. Keep datasets, logs, and caches in approved storage rather than filling shared `/home`.

### Run records

Capture these from inside the approved job and save the raw output with the job log (`experiments/phase1/slurm_run.sh` runs them automatically):

```bash
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,uuid,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv
```

Record the commit, Slurm job ID, node, peer-to-GPU mapping, software versions, configuration, and seeds. For timings, report the median, range, and repeat count. Record timeouts separately from out-of-memory failures.
