# PeerSieve project instructions

## Scope and source files

PeerSieve is a research proposal for Byzantine-resilient decentralized LLM tuning. It screens CountSketch announcements before requesting compressed updates, then audits and aggregates the accepted payloads. This repository contains the proposal, not an implementation or experiment results. KVMend is out of scope.

- `proposal_peersieve/latex/main.tex` is the only source of truth for the proposal. Edit it directly.
- `proposal_peersieve/latex/references.bib` holds the bibliography. `neurips_2025.sty` in the same directory is the vendored NeurIPS style.
- `proposal_peersieve/latex/figures/` contains the PNGs used by the paper. Keep their relative paths and filenames in sync with `main.tex`.

Keep the proposal directory source-only. Do not restore the old Markdown draft, README, Makefile, or figure generator unless the owner asks. Keep rendered PDFs and TeX intermediates outside the checkout.

## Editing the proposal

- Use LaTeX labels and BibTeX keys, not hard-coded reference numbers. Keep `amsmath` before `neurips_2025` so the style's `\And` author separator works.
- When changing a method, number, or assumption, check the corresponding equations, tables, captions, and figure artwork. In diagrams, show what changes state and keep each color's meaning consistent. Inspect edited figures at their printed size.
- Preserve the method's key distinctions: compare after-minus-before updates with the validation descent direction, audit before accumulation, and cap accepted payloads using the receiver's own update norm. Distinguish recoverable probation from hard protocol bans. A matching sketch is not proof of honesty or a general robustness guarantee.
- Label timing and memory estimates as projections, not measurements. Distinguish receive-side buffers from total training memory. Treat the CIFAR-10 baseline reproduction and the eight-GPU TinyLlama-1.1B/Dolly-15k evaluation as planned work.

## Build and validate

Use `latexmk` with a TeX installation that provides PDFLaTeX and BibTeX. Run this from the repository root. It builds in a fresh scratch directory without relying on a Makefile or local `latexmk` config:

```bash
(
  set -e
  src=proposal_peersieve/latex
  build_dir=$(mktemp -d "${JCODE_SCRATCH_DIR:-${TMPDIR:-/tmp}}/peersieve-latex.XXXXXX")
  cp "$src"/main.tex "$src"/neurips_2025.sty "$src"/references.bib "$build_dir"/
  cp -R "$src"/figures "$build_dir"/
  cd "$build_dir"
  latexmk -norc -pdf -interaction=nonstopmode -halt-on-error main.tex
  printf 'PDF: %s/main.pdf\n' "$build_dir"
)
```

Check the final `main.log` for unresolved citations or references and layout warnings. Inspect the changed PDF pages and verify that the expected figures, tables, equations, and references are present. Check `git status --short` to make sure the build left no artifacts in the checkout.

## Lab hardware and Slurm

- Local LaTeX edits and builds do not need cluster access. Commit or push only with the owner's explicit approval. Use short commit messages without trailers.
- Get explicit approval for each experiment or Slurm allocation. Before asking, specify the runs, node, GPU type and count, planned start window, wall time, and maximum GPU-hours.

### PeerSieve testbed

The planned TinyLlama-1.1B/Dolly-15k evaluation uses `octave` with one peer per GPU. The inventory below is from the lab notes collected on November 4, 2026, not a live hardware check. Verify the assigned devices, topology, and software inside each approved job.

- **`octave`:** eight NVIDIA A100 PCIe 40 GB GPUs (about 39.52 GiB usable per card), two AMD EPYC 7742 CPUs (256 logical CPUs total), and 512 GB host RAM. The lab recorded eight NUMA nodes with one GPU per NUMA node.
- **Links:** PCIe Gen4 x16, with GPU pairs reported as `SYS` in `nvidia-smi topo -m`, and no NVLink. The external network was 1 Gb/s Ethernet without RDMA. The proposal emulates receive bandwidth with a token bucket on the single-node testbed, so do not treat the capped rates as measurements of PCIe or the external network.
- **Host transfers:** the lab recorded pinned host-copy rates of 22.8–23.2 GiB/s on buses `01:00.0` through `61:00.0` and 19.6–21.6 GiB/s on buses `81:00.0` through `e1:00.0` on 2026-09-30, one GPU at a time. These are historical hardware measurements, not PeerSieve results. Use matching devices and report the PCI bus IDs when comparing timings.
- **Software snapshot:** on 2026-09-28 the lab recorded NVIDIA driver 610.57.4 and a reported CUDA version of 13.3 on `octave`. These are not pinned dependencies. Record the actual driver and the CUDA runtime used by the training environment.
- **Smaller nodes:** the lab listed `ja1` through `ja4` with one PCIe V100 16 GB each. If available, use them only for approved small checks, not as substitutes for the eight-A100 evaluation.

Prior `octave` device map. Slurm indices are not ordered by PCI bus, and the order within the `4–6` group was not recorded. Recheck the map instead of assuming an index always selects the same card.

| Slurm index | PCI bus |
|---|---|
| 0 | `61:00.0` |
| 1 | `41:00.0` |
| 2 | `01:00.0` |
| 3 | `25:00.0` |
| 4–6 | `81:00.0`, `c1:00.0`, `e1:00.0` (order unknown) |
| 7 | `a1:00.0` |

### Access and scheduling

- Use the documented login alias `ssh yes` for cluster `ja`. Use the login node for file and job management only. Run compute through Slurm. Do not SSH directly to compute nodes or disable host-key checking.
- The earlier lab configuration used `-p a100 -A a100` for `octave` with a one-hour limit. It listed `open` as a ten-minute partition for V100s on `ja3` and `ja4`. Check current access, limits, and storage rules rather than assuming these values still apply. The lab references are in `~/workspace/thu/lab/boltz-cp`: `CLAUDE.md`, `note/5-reference/cluster-use.md`, `note/5-reference/cluster-hardware.md`, and `benchmarks/cluster/README.md`.
- Use `sinfo -a -e -o '%P|%a|%l|%D|%C|%G|%N'` and `sacctmgr show assoc user="$USER" format=User,Account,Partition,QOS` to check the partition and account. Use `squeue -u "$USER"`, `scontrol show job -d <job-id>`, and `scontrol show node octave -d` to inspect jobs and node allocation.
- For approved `sbatch` submissions, specify the confirmed partition and account. For the planned eight-peer run, request a single `octave` node (`--nodes=1 --nodelist=octave --gres=gpu:a100:8`). Set an explicit `--time` and use `--open-mode=append` with a fixed log location such as `--output=logs/%x-%j.out`. Create the log directory before submission. Keep at most ten of our jobs pending or running, or fewer if the lab limit is lower. Avoid running our other jobs on the node during timing runs.
- Use Git and a clean checkout pinned to the experiment commit for runs of record. Verify the commit before submitting. Do not force a pull over local changes. Check free space and get approval before writing large data. Keep datasets, logs, and caches in approved storage rather than filling shared `/home`.

### Run records

Capture these from inside the approved job and save the raw output with the job log:

```bash
scontrol show job -d "$SLURM_JOB_ID"
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,uuid,memory.total,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,driver_version --format=csv
```

Record the commit, Slurm job ID, node, peer-to-GPU mapping, software versions, configuration, and seeds. For timings, warm up and synchronize the GPU before reading the clock, then report the median, range, and repeat count. Record timeouts separately from out-of-memory failures.
