# PeerSieve project instructions

## Scope and source files

PeerSieve is a research proposal for Byzantine-resilient decentralized LLM tuning. It screens CountSketch announcements before requesting compressed updates, then audits and aggregates the accepted payloads. This repository contains the proposal, not an implementation or experiment results. KVMend is out of scope.

- `proposal_peersieve/latex/main.tex` is the only source of truth for the proposal. Edit it directly.
- `proposal_peersieve/latex/references.bib` holds the bibliography. `neurips_2025.sty` in the same directory is the vendored NeurIPS style.
- `proposal_peersieve/latex/figures/` contains the PNGs used by the paper. Keep their relative paths and filenames in sync with `main.tex`.

Keep the proposal directory source-only. Do not restore the old Markdown draft, README, Makefile, or figure generator unless the owner asks. Keep rendered PDFs and TeX intermediates outside the checkout.

## Editing the proposal

- Use LaTeX labels and BibTeX keys, not hard-coded reference numbers. Keep `amsmath` before `neurips_2025` so the style's `\And` author separator works.
- When changing a method, number, or assumption, check the corresponding equations, tables, captions, and figure artwork. Inspect edited figures at their printed size.
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

## Approval and reproducibility

- Local LaTeX edits and builds do not need cluster access. Commit or push only with the owner's explicit approval. Use short commit messages without trailers.
- Get explicit approval for each experiment or Slurm allocation. Before asking, specify the runs, node, GPU type and count, planned start window, wall time, and maximum GPU-hours.
- For approved lab work, verify the current account, partition, limits, and storage rules before submitting. Use the `ja` login node (`yes`) for job control and Slurm for compute. Do not SSH directly to compute nodes or disable host-key checking. Keep at most ten of our jobs pending or running, or fewer if the lab limit is lower.
- Run experiments of record from a clean checkout. Keep the commit, Slurm job ID, node, GPU IDs and PCI mapping, software versions, configuration, seeds, and raw logs. Verify the hardware inside the job. For timings, warm up, synchronize the GPU, and report the median and repeat count. Record timeouts separately from out-of-memory failures.
