# KVMend project instructions

## Project overview

KVMend is a correctness watchdog for the KV cache in LLM serving. It detects silent KV damage, repairs only the affected pages, and recomputes tokens that may have read damaged data. The proof of concept patches SGLang 0.5.1 through `sitecustomize.py`; SGLang itself is not edited.

- `kvmend/kvmend/`: tags, the SGLang hook, attention checks, Triton tag kernels, and run metadata.
- `kvmend/hook_site/`: the opt-in loader, enabled with `KVMEND_ENABLE=1` on `PYTHONPATH`.
- `kvmend/run_*.py`: proof-of-concept and experiment drivers.
- `kvmend/bench_*.py`: checker and fault-free throughput benchmarks.
- `kvmend/tests/`: CPU tests, including multi-process gloo tests.
- `kvmend/tests_gpu/`: GPU kernel tests.
- `kvmend/slurm/`: environment setup and Slurm experiment scripts.
- `kvmend/results/`: measured records used by the proposal figures; do not treat these as source code.
- `proposal/decisions.md`: the owner's decisions and experiment approvals. Check it before assuming that a launch is approved.
- `proposal/PROPOSAL.md`: the generated proposal.
- `proposal/build/`: the keyed proposal template, assembler, and consistency checker.
- `proposal/figures/`: figure generators, shared styles, CSV data, and rendered figures.

Read `kvmend/README.md` before changing the proof of concept. It documents the mechanisms, experiment variants, pass conditions, and known limitations.

## Development and validation

Match the existing Python style and the SGLang 0.5.1 interfaces already used by the hook. Keep experiments reproducible: records must identify the code and hardware that produced them.

The local source snapshot has no `pyproject.toml` or lockfile. On the configured cluster checkout, `slurm/setup_env.sh` creates the uv environment and `slurm/suite.sh` is the canonical test and experiment entry point.

Run CPU tests from `kvmend/` in that configured environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --frozen python -m pytest -v -p no:cacheprovider \
  -o faulthandler_timeout=150 --durations=0 tests
```

Run GPU kernel tests only inside an approved Slurm GPU job:

```bash
uv run --frozen python -m pytest -v -p no:cacheprovider --durations=0 tests_gpu
```

The suite's `tests` step gates later steps. Other suite steps cover incidents, chunked prefill, HiCache, release, arrivals, attention repair, checker speed, and throughput; see `kvmend/slurm/suite.sh` for their environment variables and pass conditions.

## Proposal folder notes

Run the commands in this section from `proposal/` unless a command says otherwise.

### Figures

- **Plots (matplotlib).** Style module: `figures/figstyle.py`, an unedited copy from the paper-figures skill. Series roles: `figures/figroles.py`. Import both; never hard-code colors.
- **Python for plots.** `/opt/homebrew/bin/python3.8` has matplotlib (3.5.1); the default `python3` does not. Run `python3.8 -I -B figures/plot_expected.py`.
- **Data.** `figures/plot_expected.py` writes its estimates to `figures/expected_costs.csv`, prints every number it plots, and draws `fig11_expected.png` and `.pdf` from that file.
- **Measured data.** `figures/plot_poc.py` reads the proof of concept's records in `../kvmend/results/kvmend-cmp8-326326/`, writes `figures/poc_results.csv`, prints every number it plots, and draws `fig12_poc.png` and `.pdf`. Run `python3.8 -I -B figures/plot_poc.py`. The checker matches its numbers with Tables 10 and 11.
- **Figure 13.** `figures/plot_results.py` reads the incident and burst records, writes `figures/results_tp8.csv`, and draws `fig13_results.png` and `.pdf`. Run `python3.8 -I -B figures/plot_results.py`. The checker matches its numbers with Tables 14 and 16.
- **Encodings in plots:**
  - solid blue: KVMend, damage on one GPU;
  - dashed lighter blue: KVMend, damage on every GPU;
  - dotted blue line with hollow markers: KVMend recomputing pages (instead of rebuilding them from parity);
  - solid red: re-prefill from the damaged page;
  - gray (a dashed line, or a bar): re-prefill the whole request (the line is drawn on top, because it nearly coincides with red on a log axis);
  - pink: GhostServe as we re-implemented it, told the GPU, and parity alone; lighter pink: GhostServe told the GPU and the chunk;
  - hatched bar: the next tokens are still wrong after recovery;
  - gold marker: the running example, the one callout per figure.
- **Drawings (Excalidraw).** `figures/build_kvmend_figure.py` builds one scene, `figures/kvmend.excalidraw`, and renders Figures 1–10 as views of its frames:
  - Run `python3 -I -B figures/build_kvmend_figure.py --render`.
  - Style: Excalifont; every element has an `index` key; text at least 17 units in a 1,560-unit frame (about 5 pt at page width).
  - Colors, one meaning each: blue for KV on a GPU (dark blue: just written); yellow for host memory and parity; green for repaired or clean; red for damage and failed checks; violet for tags and checks; orange for suspect tokens; gray for free, retired or busy.
  - Draw the mechanism (data, operation, state before and after), not boxes of sentences.
- **Proposal build.** `PROPOSAL.md` is assembled from the keyed template `build/kvmend_template.md`, which numbers citations by first use and builds the reference list:
  - Run `python3 -I -B build/assemble_kvmend.py build/kvmend_template.md PROPOSAL.md`, then `python3 -I -B build/check_kvmend.py .`.
  - The checker tests paragraph length (240 characters at most), citation order, figure and table references, and that numbers drawn in the figures match the text.
  - Reassembling overwrites `PROPOSAL.md`. Carry any hand edits to it (for example from Obsidian) into the template first.

## Cluster `ja` (login node `yes`) and Slurm

Copied on 2026-11-04 from the user's lab repository `~/workspace/thu/lab/boltz-cp`:
- `CLAUDE.md`, "Git, and getting code onto the cluster" and "Hardware";
- `note/5-reference/cluster-use.md`;
- `note/5-reference/cluster-hardware.md`, "Cluster ja";
- `benchmarks/cluster/README.md`.

Those files are the source, so re-read them if a fact here looks stale. Their helper scripts (`gpu_env.sh`, the card guard, `throttle_submit.sh`, `queue_to_chain.sh`) exist only in boltz-cp.

### Reach the cluster

- `ssh yes` reaches the login node of cluster `ja`. `ssh ja` does not work.
- The login node only controls jobs. Run git, file listings and copies, `squeue`, `scontrol`, `sbatch` and `salloc` there, plus the read-only `cat`, `head`, `tail`, `grep`, `wc`, `df` and `du`.
- The login node is the control plane: submit every compute task to `octave` or the `ja` nodes with Slurm.
- **This project's exceptions (the user, 2026-11-05):**
  - Installs may run directly on the login node, for example `uv add` in `/home/tae/kvmend`, with no Slurm job.
  - For testing, files may go to `yes` by `scp`. The KVMend PoC lives in `/home/tae/kvmend`, a uv project; models sit in `/home/tae/models`.
  - Python that computes still runs only inside Slurm jobs.
- Never log into a compute node: a login joins the running job and can disturb a timed run. Never pass `StrictHostKeyChecking=no`.

### Hardware

- **`octave`:** 8× NVIDIA A100-PCIE-40GB (39.52 GiB usable on each), 2× AMD EPYC 7742 (256 CPUs), 512 GB RAM. It has 8 NUMA nodes, with one GPU on each.
- **Links:** no NVLink. Every GPU pair is `SYS`: PCIe Gen4 x16 through the CPU interconnect. No RDMA; only 1 Gb/s Ethernet.
- **Host copy speed per card:** pinned host copies ran at 22.8–23.2 GiB/s on the cards at buses `01:00.0`–`61:00.0`, and 19.6–21.6 GiB/s on `81:00.0`–`e1:00.0` (one card at a time, 2026-09-30).
- **`ja1`–`ja4`:** one V100-PCIE-16GB each. Slurm version 24.11.0.
- **Driver:** 610.57.4 with CUDA 13.3 on `octave` (recorded 2026-09-28). Node software changes, so read `nvidia-smi` inside the job.

### Partitions and account

| partition | limit | GPUs | for us |
|---|---|---|---|
| `a100` | 1 hour | A100 on `octave` | the only A100 partition for our account: submit with `-p a100 -A a100` |
| `open` | 10 minutes | V100 (`ja3`, `ja4`) | the default; takes every account |
| `octave` | 10 minutes | A100 | refuses account `a100` |
| `long` | 8 hours | V100, A100 | refuses account `a100` |
| `verylong` | 1 day | V100 | refuses account `a100` |
| `ja` | 1 hour | V100 | refuses account `a100` |

Our account `a100` may use only `a100` and `open`. A100 work longer than 1 hour must split into jobs of 1 hour or less.

### Submit a job

Ask for GPUs by type, pass a tight `--time`, and keep `--open-mode=append` with a log folder fixed at submission, so a requeued job keeps one log. `<...>` marks values that each job sets:

```bash
# one A100, batch job, from the login node
sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:1 --cpus-per-task=<n> --mem=<size> \
  --time=00:30:00 --open-mode=append -J <name> -o <log dir>/%x-%j.out <job script>
# all eight A100s, for example TP=8 serving
sbatch -p a100 -A a100 --nodelist=octave --gres=gpu:a100:8 --cpus-per-task=<n> --mem=<size> \
  --time=00:55:00 --open-mode=append -J <name> -o <log dir>/%x-%j.out <job script>
# a short interactive check; exit releases the allocation
salloc -A a100 -p a100 -w octave -N1 -n1 --gres=gpu:a100:1 --cpus-per-task=1 --mem=1G --time=00:05:00
srun -N1 -n1 nvidia-smi
exit
```

- Keep at most 10 of our jobs, running plus pending, on the cluster. It is shared.
- Hold the card between sequential jobs. Chain them with `--dependency=afterany:<previous job ID>`, or bundle several runs into one job within the 1-hour limit. A gap between two jobs lets another user take the card.
- Name every record: each job writes its job ID, commit, node and GPU indices into the data it produces.
- Use one script for every node, taking the node and GPU as options or variables. Never copy a script for one node.
- For every failed run, record whether the time limit or an out-of-memory error stopped it. The time limit has killed runs that looked like memory failures.

### Which A100 a job gets

Slurm numbers `octave`'s cards by device file, not by PCI bus, and gives a job the lowest free index:

| Slurm index | PCI bus |
|---|---|
| 0 | `61:00.0` |
| 1 | `41:00.0` |
| 2 | `01:00.0` |
| 3 | `25:00.0` |
| 4, 5, 6 | `81:00.0`, `c1:00.0`, `e1:00.0` (order unknown) |
| 7 | `a1:00.0` |

- The A100 card of record for timing is index 0, at bus `61:00.0`. Only index 0 always reaches its card whatever the other cards hold.
- A time job runs with no other job of ours on the node. Copy speed differs by card, so every copy-time number names its bus.
- Check the map again after any GPU or driver change:

```bash
ssh yes 'grep octave /etc/slurm/gres.conf; scontrol show node octave -d | grep GresUsed'
# inside a job on octave
scontrol show job -d "$SLURM_JOB_ID" | grep -o 'IDX:[0-9,-]*'
nvidia-smi --query-gpu=pci.bus_id,uuid --format=csv
```

### Get code onto the cluster

- Use git for source. Never use scp or rsync for source of record: a job that cannot name its commit cannot be reproduced. The testing exception above permits `scp` only for tests, not for a result of record.
- Push locally, then pull on the cluster with agent forwarding. The cluster's own SSH key does not reach GitHub, so `ssh-add -l` must list the local key. Check the commit before every `sbatch`:

```bash
git push origin main                                   # local
ssh -A yes 'cd <repo on ja> && git pull origin main'   # cluster ja
ssh yes 'cd <repo on ja> && git rev-parse --short HEAD && git status --porcelain | wc -l'
```

- HTTPS to github.com times out on ja's login node, so use an SSH remote (`git@github.com:...`).
- If a pull would overwrite an uncommitted file on the cluster, commit or discard that file first. Do not force.
- Run each job of record from a git worktree at its commit, and print the commit in the job log. boltz-cp uses `/home/tae/checkouts/<name>_<commit>` on ja.
- Commit and push only with the user's explicit approval.

### Environment and files

- Build Python environments with `uv` inside a Slurm job, except for the login-node install permission above. A uv `.venv` records absolute paths, so a checkout that moves needs its `.venv` repaired (boltz-cp `benchmarks/cluster/README.md`, "If a checkout moves, repair its virtual environment").
- ja loads no spack CUDA; the PyTorch wheel brings its own CUDA libraries.
- boltz-cp's rule is to keep data, run output and caches off `/home`, which is full and shared. Its notes name no other data disk on ja, and its ja logs sit under `/home/tae/sf_data/logs`.
- Check `df -h` and ask the user before writing large data. In a job, point `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR` and `MPLCONFIGDIR` at a job folder.

### Watch jobs

- `squeue -u $USER`, `scontrol show job -d <job ID>` and `scontrol show node octave -d | grep GresUsed`.
- `sinfo -a -e -o '%P|%a|%l|%D|%C|%G|%N'` lists the partitions. `sacctmgr show assoc user=$USER format=User,Account,Partition,QOS` shows our account and partitions.

### Before an experiment

- Launch no job without the user's yes on that launch. Name the runs, the node, the GPUs, the GPU-hours and the start time.
- Before a hardware-sensitive experiment, collect the hardware state again inside a job. Keep the raw output and the Slurm job ID with every published number:

```bash
salloc -A a100 -p a100 -w octave -N1 -n1 --gres=gpu:a100:8 --cpus-per-task=1 --mem=1G --time=00:05:00
srun -N1 -n1 nvidia-smi topo -m
srun -N1 -n1 nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv
exit
```

- Time runs warm: run an untimed warm-up first, synchronize the GPU before reading the clock, and report the median and the number of repeats.
