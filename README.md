# PeerSieve

Byzantine-resilient decentralized federated learning (DFL) research codebase built with PyTorch and managed with `uv`.

- **Phase 1 (Complete — 721 A100 Runs Across 3-Seed Audits & Paper-Panel Reproduction):**
  - Literal 3-seed CIFAR-10 reproduction audit (`225` runs) of [TrustGraph-DFL](TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026): [report/phase1_final.md](report/phase1_final.md).
  - Stabilized 3-seed Option C reproduction (`330` runs across `cnn2x2_gn` + unclipped `resnet18_pretrained`): [report/phase1_stabilized_ab.md](report/phase1_stabilized_ab.md).
  - Phase 1c gradient-clipping & validation-regime calibration (`16` runs): [report/phase1c_calibration.md](report/phase1c_calibration.md).
  - Calibrated 3-seed paper-panel reproduction (`150` runs, `0/150` NaNs, publication-style Fig. 2–4 PNG + PDF): [report/phase1_paper_match.md](report/phase1_paper_match.md).
- **Phase 2 (Core Mechanisms Verified Locally; 8-GPU LLM Evaluation Planned):**
  - Core PeerSieve modules implemented & verified against Proposal Table 2 (including 4-process `torch.distributed` Gloo tests): [report/phase2_local_verification.md](report/phase2_local_verification.md).
  - Planned next: eight-GPU TinyLlama-1.1B / Dolly-15k evaluation on `octave`.

---

## Repository Layout

```text
.
├── pyproject.toml              # uv project and dependency specification
├── src/                        # Reusable DFL & PeerSieve core modules
│   ├── aggregators.py          # FedAvg, TrustGraph-DFL (+ clip/ablations), BALANCE, Krum, Trimmed Mean
│   ├── attacks.py              # none, label_flip (0<->1), label_flip_all, model_poison, random_noise
│   ├── audit.py                # PeerSieve two-pass audited streaming aggregation & receiver-norm cap
│   ├── compression.py          # Blockwise INT4 quantization + error feedback, tiled decode, Top-k baseline
│   ├── data.py                 # CIFAR-10 & synthetic loaders, balanced Dirichlet split (local/iid val)
│   ├── models.py               # CNNs (cnn2x2, cnn2x2_gn, lenet2x2), ResNet-18, augmentation, vector helpers
│   ├── quarantine.py           # Per-edge EMA backoff probation, permanent hard-fault bans, & referrals
│   ├── simulator.py            # Synchronous DFL simulator, metrics, and JSON/JSONL logging
│   ├── sketch.py               # Deterministic layer-proportional FP32->FP16 CountSketch (32 KiB)
│   ├── topology.py             # Ring, Erdos-Renyi (p=0.3), Watts-Strogatz (k=4, p=0.3)
│   └── transport.py            # Token-bucket receive limiter & multi-process Gloo exchange protocol
├── experiments/
│   ├── phase1/
│   │   ├── run.py              # CLI runner (single runs, named suites, multi-GPU sharding, --skip-existing)
│   │   ├── summarize.py        # Aggregates summary.json files into Markdown tables (mean ± std, NaNs)
│   │   ├── plot_figures.py     # Draws 300 dpi PNG + header-free PDF figures from report/data/*.json
│   │   ├── figstyle.py         # figures4papers house style (unedited skill copy)
│   │   ├── figroles.py         # Fixed method color/marker/dash encodings
│   │   └── slurm_run.sh        # Multi-GPU Slurm launcher with hardware/git run-record capture
│   └── phase2/
│       └── verify_mechanisms.py # Local pre-GPU verification suite for Proposal Table 2
├── tests/
│   ├── test_phase1.py          # Phase 1 unit and offline synthetic DFL integration tests
│   └── test_phase2.py          # Phase 2 unit and 4-process Gloo quarantine/referral tests
└── report/
    ├── data/                        # Committed 3-seed aggregated JSON datasets
    ├── figures/                     # Publication-style 300 dpi PNG + header-free PDF figures
    ├── phase1_smoke.md              # Local CPU 5-round CIFAR-10 smoke test & diagnostics
    ├── phase1_step_a.md             # Single-seed (seed 42) 75-run A100 literal reproduction report
    ├── phase1_final.md              # Final 3-seed (225-run) A100 literal reproduction report
    ├── phase1_stabilized_ab.md      # 3-seed (330-run) stabilized CNN + unclipped ResNet-18 report
    ├── phase1c_calibration.md       # Phase 1c grad-clip, validation-regime, & BALANCE calibration report
    ├── phase1_paper_match.md        # 3-seed (150-run, 0-NaN) paper-panel reproduction & figures
    └── phase2_local_verification.md # Pre-GPU verification of all 4 PeerSieve mechanisms + Gloo tests
```

---

## Setup

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
# Local install
uv sync

# On the THU cluster (faster downloads via THU TUNA mirror when syncing without --frozen,
# or warm the uv cache once):
uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

Run the unit and offline integration test suite:

```bash
uv run pytest -v
```

---

## Running Phase 1 Experiments

### 1. Local CIFAR-10 Smoke Test

```bash
uv run python -m experiments.phase1.run \
  --mode suite --suite smoke \
  --dataset cifar10 --rounds 5 --eval-every 0 --device cpu --seeds 42 \
  --output-root outputs/phase1_smoke
```

### 2. Single Custom Run

```bash
uv run python -m experiments.phase1.run \
  --mode single --dataset cifar10 \
  --topology random --attack model_poison --byzantine-frac 0.3 \
  --aggregator trustgraph --val-grad-sign descent_neg --trustgraph-clip-norm \
  --rounds 200 --eval-every 10 --seeds 42 43 44
```

Key CLI toggles in `experiments/phase1/run.py`:
- `--suite`: `smoke`, `sign_check`, `main_attacks`, `topologies`, `fpr_sweep`, `ablations`, `full` (75 literal configs; `step_a` alias), `stabilized_ab` (110 Option C stabilized CNN + pretrained ResNet-18 configs), `paper_match_calib` (16 calibration configs), `paper_match` (50 calibrated paper-panel configs).
- `--model`: `cnn2x2`, `cnn2x2_gn`, `lenet2x2`, `resnet18_pretrained`, `resnet18_scratch`.
- `--val-grad-sign`: `descent_neg` ($-\nabla F_i$, corrected descent direction) or `paper_pos` ($+\nabla F_i$, literal TrustGraph-DFL Eq. 3 / Alg. 1).
- `--val-split`: `local` (literal paper: 10% held out from each node's non-IID shard) or `iid` (balanced 50-sample validation diagnostic).
- `--trustgraph-clip-norm`, `--trustgraph-clip-mode` (`update | model`), `--trustgraph-clip-factor`, `--trustgraph-score-threshold`: PeerSieve receiver-norm cap and noise-floor cosine threshold ($\tau$).
- `--lr-schedule` (`constant | cosine`), `--augment`, `--grad-clip-norm`, `--balance-gamma`: training stabilization and BALANCE distance multiplier (`0.3` literal; calibrated `0.15` / `0.025` / `0.015`).
- `--shard-index`, `--num-shards`, `--skip-existing`: resumable parallel sharding across GPUs.

### 3. Summarize Outputs to Markdown

```bash
uv run python -m experiments.phase1.summarize outputs/phase1_smoke \
  --title "Phase 1 Summary" --out report/summary.md
```

### 4. Cluster Execution (Slurm on `octave`)

Stage CIFAR-10 under `data/` and create `logs/` on the login node before submitting:

```bash
mkdir -p logs
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --job-name=peersieve_p1 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh \
  --mode suite --suite full --skip-existing \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42 43 44 \
  --output-root outputs/phase1
```

`slurm_run.sh` automatically logs the git commit, `scontrol show job -d`, `nvidia-smi topo -m`, `nvidia-smi --query-gpu=...`, and PyTorch/CUDA versions, then launches one worker process per allocated GPU.

---

## Phase 1 Headline Results

Full tables, hardware provenance, and publication-style figures (`report/figures/`) span `721` A100 runs on `octave`:

- **Literal reproduction gap ([report/phase1_final.md](report/phase1_final.md), `225` runs, 3 seeds):** On `10,000` CIFAR-10 images ($\alpha_{\mathrm{dir}}=0.5$, 2-conv 2-FC CNN, 200 rounds), attack-free `FedAvg` reaches `53.27 ± 0.86%` mean benign accuracy (`56.52 ± 0.41%` consensus), far below the `93–95%` claimed in Fig. 3(a). Literal `paper_pos` ($+\nabla F_i$) is inverted relative to SGD updates, unclipped cosine hits `NaN` on `model_poison` and `random_noise`, and local non-IID validation yields `50.3–53.5%` honest FPR rather than `<10%`.
- **Stabilized Option C ([report/phase1_stabilized_ab.md](report/phase1_stabilized_ab.md), `330` runs, 3 seeds):** Combining `descent_neg` ($-\nabla F_i$), model-space receiver-norm clipping (`2 r_i`), and $\tau=0.02$ makes `trustgraph(stab)` the only method surviving `model_poison` ($\lambda=10$) on `cnn2x2_gn` at both $\alpha_{\mathrm{dir}}=0.5$ (`43.71 ± 1.20%` / `52.34 ± 1.92%` cons, `0/3` NaNs) and $\alpha_{\mathrm{dir}}=5.0$ (`57.64 ± 1.08%` / `58.71 ± 0.94%` cons, `0/3` NaNs), while exposing sporadic isolated-peer frozen-BN SGD overflows on unclipped `resnet18_pretrained` (`grad_clip_norm=0.0`).
- **Calibrated paper-panel reproduction ([report/phase1_paper_match.md](report/phase1_paper_match.md), `150` runs, 3 seeds, `0/150` NaNs):** Adding `grad_clip_norm=10.0` and `balance_gamma=0.025` to pretrained `ResNet-18` ($\alpha_{\mathrm{dir}}=0.5$, 50 rounds) eliminates all NaNs (`0/150`) and preserves the `90.60 ± 0.02%` clean ceiling. `trustgraph(stab)` is the sole method staying at **`86.4–89.9%` consensus accuracy across every regime** — beating all baselines on `model_poison` (`87.62 ± 0.49%` cons vs. `Krum` `36.03%`, `BALANCE`/`Trimmed Mean`/`FedAvg` `10.00%`), rejecting `100%` of `random_noise` (`88.90 ± 0.41%` cons, `0.00%` Byz accept), holding **`86.43 ± 0.56%` cons at `50%` Byzantine** (while `Krum` collapses to `19.98 ± 3.06%` / `71.3%` FPR), and confirming all Section 5.5 ablations (`no_relu` `-18.9 pp`, `euclidean` `-12.3 pp`).
