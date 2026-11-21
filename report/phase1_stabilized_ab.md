# Phase 1b Report — 3-Seed Stabilized TrustGraph-DFL Reproduction (Option C: Parts A & B)

**Date:** 2026-11-21  
**Slurm Job IDs:** `327284` (seed `42`, `110` runs, `COMPLETED` in `00:38:19`, `MaxRSS=20.3G`, commit `5c649ed`) · `327296` (seed `43`, `110` runs, `COMPLETED` in `00:36:30`, `MaxRSS=20.0G`, commit `18c7e2e`) · `327297` (seed `44`, `110` runs, `COMPLETED` in `00:37:30`, `MaxRSS=16.4G`, commit `18c7e2e`)  
**Hardware:** `octave`, 8× NVIDIA A100-PCIE-40GB (`01:00.0` through `E1:00.0`, PCIe Gen4 x16 `SYS`, `32` CPUs, `128G` RAM, driver `610.57.04`)  
**Software:** Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`, `torchvision 0.29.2`  
**Scope:** `330 / 330` completed runs (`180` Part A `cnn2x2_gn` + `150` Part B `resnet18_pretrained`, `3` seeds `42, 43, 44`, `eval_every=10`, `grad_clip_norm=0.0`)  
**Failures:** `0` timeouts, `0` OOMs  
**Timing:** Part A `cnn2x2_gn` ($n=180$, 200 rounds): median `144.8s` (range `72.2s–212.4s`); Part B `resnet18_pretrained` ($n=150$, 50 rounds): median `163.0s` (range `113.7s–197.7s`).  
**Aggregated Data:** [data/phase1_stabilized_ab_3seed.json](data/phase1_stabilized_ab_3seed.json)  
**Related Reports:** [phase1_final.md](phase1_final.md) (literal 3-seed audit) · [phase1c_calibration.md](phase1c_calibration.md) (gradient-clipping calibration) · [phase1_paper_match.md](phase1_paper_match.md) (3-seed calibrated `grad_clip_norm=10.0` paper-panel reproduction, `0/150` NaNs)

---

## Executive Summary

Upgrading Option C from 1 seed to **all `3` seeds (`330 / 330` runs)** confirms two central results with tight `mean ± std` error bars:

1. **Part A (`cnn2x2_gn`, 2-conv 2-FC CNN + GroupNorm + augmentation + cosine LR, `180` runs, `0/3` NaNs on `tg_stab` across every cell):**
   - `tg_stab` (`descent_neg`, model-space clip `2 r_i`, $\tau = 0.02$) is the **only defense that survives `model_poison` ($\lambda=10$)** across all 3 seeds at both $\alpha_{\mathrm{dir}}=0.5$ (**`43.71 ± 1.20%` / `52.34 ± 1.92%` cons**, `0/3` NaN; every other method collapses to `~10%` with `3/3` NaNs for `FedAvg`, `Krum`, `Trimmed Mean`, and unclipped `TrustGraph`) and $\alpha_{\mathrm{dir}}=5.0$ (**`57.64 ± 1.08%` / `58.71 ± 0.94%` cons**, `0/3` NaN).
   - On `random_noise` ($\sigma=0.5$), $\tau=0.02$ cuts Byzantine acceptance to **`0.00%` on all 3 seeds**, achieving **`48.17 ± 0.67%` / `57.70 ± 1.09%` cons** ($\alpha_{\mathrm{dir}}=0.5$) and **`60.95 ± 0.82%` / `62.19 ± 0.69%` cons** ($\alpha_{\mathrm{dir}}=5.0$), vs. `13.35 ± 2.72%` for unclipped `TrustGraph`.
   - At **`50%` Byzantine `label_flip_all` ($\alpha_{\mathrm{dir}}=5.0$)**, `tg_stab` holds **`59.88 ± 0.86%` (`60.96 ± 0.77%` cons, `43.0%` FPR)** while `Krum` collapses to `27.51 ± 5.33%` (`27.18 ± 13.14%` cons, `59.7%` FPR).
2. **Part B (`resnet18_pretrained`, `grad_clip_norm=0.0`, `balance_gamma=0.015`, `150` runs) — Direct 3-Seed Ablation of Local Gradient Clipping:**
   - Clean ceiling reaches **`90.23–90.56%`** (`tg_stab`: `87.25 ± 0.43%` / `89.31 ± 0.30%` cons, `0/3` NaN), and `tg_stab` dominates `50%` Byzantine `label_flip_all` (**`80.86 ± 2.52%` / `86.21 ± 1.12%` cons**, `0/3` NaN vs. `Krum` `42.32 ± 22.29%` cons / `66.6%` FPR).
   - However, **without local gradient clipping (`grad_clip_norm=0.0`)**, isolated benign peers under frozen-BN SGD (`lr=0.005`, `momentum=0.9`) suffer sporadic single-peer overflows (`2/3` seeds on `model_poison`, `1/3` on `random_noise` and `ring`, and `1–3/3` seeds on over-tight `BALANCE(g=0.015)` even under `none`). Adding `grad_clip_norm=10.0` + `balance_gamma=0.025` in [phase1_paper_match.md](phase1_paper_match.md) completely eliminates all overflows (**`0 / 150` NaNs**) while preserving `90.60 ± 0.02%` clean consensus and lifting `tg_stab` on `model_poison` to **`87.62 ± 0.49%` cons**.

---

## 1. Part A — Stabilized 2-Conv 2-FC CNN (`cnn2x2_gn`, 200 Rounds, 10k CIFAR-10, 3 Seeds)

Architecture: 2 conv layers (`3->64`, `64->128`) with GroupNorm (`8` groups), Dropout (`0.25`), 2 FC layers (`8192->256->10`, `2,176,010` params), vectorized `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`, cosine LR decay from `0.01` to `0.0005`. Calibrated `BALANCE` uses $\gamma=0.15$.

### 1.1 Main Attacks at $\alpha_{\mathrm{dir}} = 0.5$ (Random ER $p=0.3$, 30% Byzantine, 3 Seeds)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | fedavg | 60.50 ± 0.32 | 61.94 ± 0.49 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | krum | 60.50 ± 0.33 | 61.92 ± 0.51 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | balance($\gamma=0.15$) | 58.52 ± 0.24 | 62.73 ± 0.29 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | trimmed_mean | 56.94 ± 0.34 | 58.98 ± 0.48 | 50.31 ± 1.70 | 0.00 | n/a | 0/3 |
| none | trustgraph(`descent_neg`) | 54.85 ± 0.62 | 61.01 ± 0.30 | 52.24 ± 3.15 | 0.00 | -0.008 / n/a | 0/3 |
| none | trustgraph(`stab`) | 53.08 ± 1.05 | 60.63 ± 0.21 | 59.15 ± 2.05 | 0.00 | -0.010 / n/a | 0/3 |
| **label_flip_all** | krum | 50.55 ± 3.71 | 55.76 ± 2.74 | 14.35 ± 3.90 | 48.76 | n/a | 0/3 |
| label_flip_all | balance($\gamma=0.15$) | 49.27 ± 2.00 | **57.11 ± 1.64** | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |
| label_flip_all | trustgraph(`descent_neg`) | 46.22 ± 1.22 | 56.36 ± 0.82 | 49.92 ± 4.16 | 37.12 | -0.000 / -0.058 | 0/3 |
| label_flip_all | trustgraph(`stab`) | 44.89 ± 1.13 | 56.46 ± 0.56 | 56.60 ± 4.62 | **32.13** | -0.002 / -0.053 | 0/3 |
| label_flip_all | fedavg | 47.14 ± 3.27 | 51.84 ± 3.89 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |
| label_flip_all | trimmed_mean | 45.56 ± 3.03 | 49.63 ± 3.23 | 47.64 ± 3.40 | 45.58 | n/a | 0/3 |
| **model_poison** ($\lambda=10$) | **trustgraph(`stab`)** | **43.71 ± 1.20** | **52.34 ± 1.92** | 63.73 ± 1.46 | **22.76** | -0.026 / -0.079 | **0/3** |
| model_poison | trustgraph(`descent_neg`) | 10.00 ± 0.00 | 10.00 ± 0.00 | 76.99 ± 13.32 | 2.16 | `NaN` | 3/3 |
| model_poison | balance($\gamma=0.15$) | 9.94 ± 0.06 | 9.94 ± 0.09 | 0.00 ± 0.00 | 83.76 | n/a | 0/3 |
| model_poison | fedavg / krum / trimmed_mean | 10.00 ± 0.00 | 10.00 ± 0.00 | 0 / 31.1 / 53.0 | 100 / 71.2 / 46.1 | n/a | 3/3 |
| **random_noise** ($\sigma=0.5$) | balance($\gamma=0.15$) | **56.39 ± 1.21** | **60.64 ± 1.21** | 0.00 ± 0.00 | **0.00** | n/a | 0/3 |
| random_noise | **trustgraph(`stab`)** | **48.17 ± 0.67** | **57.70 ± 1.09** | 60.49 ± 1.84 | **0.00** | -0.015 / -0.000 | **0/3** |
| random_noise | trimmed_mean | 22.53 ± 10.44 | 25.09 ± 12.20 | 33.79 ± 4.85 | 14.86 | n/a | 0/3 |
| random_noise | krum | 20.17 ± 14.61 | 21.99 ± 17.66 | 4.19 ± 3.13 | 24.73 | n/a | 0/3 |
| random_noise | trustgraph(`descent_neg`) | 12.82 ± 1.93 | 13.35 ± 2.72 | 56.88 ± 2.86 | 49.78 | -0.030 / +0.000 | 0/3 |
| random_noise | fedavg | 9.90 ± 0.14 | 10.00 ± 0.00 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |

### 1.2 Main Attacks at $\alpha_{\mathrm{dir}} = 5.0$ (Random ER $p=0.3$, 30% Byzantine, 3 Seeds)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | fedavg | 65.79 ± 0.30 | 65.97 ± 0.24 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | krum | 65.73 ± 0.32 | 65.93 ± 0.26 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | balance($\gamma=0.15$) | 64.69 ± 0.30 | 65.20 ± 0.26 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | trimmed_mean | 64.88 ± 0.47 | 65.16 ± 0.42 | 50.32 ± 1.58 | 0.00 | n/a | 0/3 |
| none | trustgraph(`stab`) | 63.04 ± 0.35 | 64.07 ± 0.25 | 49.81 ± 1.07 | 0.00 | +0.022 / n/a | 0/3 |
| **label_flip_all** | **trustgraph(`stab`)** | **60.72 ± 0.35** | **61.80 ± 0.38** | 44.98 ± 1.05 | **28.59** | +0.034 / -0.043 | 0/3 |
| label_flip_all | balance($\gamma=0.15$) | 59.83 ± 1.53 | 61.55 ± 0.92 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |
| label_flip_all | krum | 57.44 ± 3.61 | 60.65 ± 2.97 | 16.54 ± 5.88 | 47.38 | n/a | 0/3 |
| label_flip_all | trimmed_mean | 55.99 ± 1.88 | 57.60 ± 2.31 | 47.09 ± 2.81 | 44.35 | n/a | 0/3 |
| label_flip_all | fedavg | 53.67 ± 2.44 | 56.27 ± 3.81 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |
| **model_poison** ($\lambda=10$) | **trustgraph(`stab`)** | **57.64 ± 1.08** | **58.71 ± 0.94** | 51.33 ± 0.59 | **22.28** | +0.020 / -0.058 | **0/3** |
| model_poison | balance($\gamma=0.15$) | 9.90 ± 0.14 | 10.00 ± 0.00 | 0.00 ± 0.00 | 84.42 | n/a | 0/3 |
| model_poison | fedavg / krum / trimmed_mean | 10.00 ± 0.00 | 10.00 ± 0.00 | 0 / 31.1 / 53.0 | 100 / 71.2 / 46.0 | n/a | 3/3 |
| **random_noise** ($\sigma=0.5$) | balance($\gamma=0.15$) | **62.70 ± 1.05** | **63.45 ± 0.78** | 0.00 ± 0.00 | **0.00** | n/a | 0/3 |
| random_noise | **trustgraph(`stab`)** | **60.95 ± 0.82** | **62.19 ± 0.69** | 47.19 ± 0.38 | **0.00** | +0.031 / -0.000 | **0/3** |
| random_noise | trimmed_mean | 24.66 ± 11.83 | 27.31 ± 14.17 | 33.79 ± 4.86 | 14.86 | n/a | 0/3 |
| random_noise | krum | 21.29 ± 15.95 | 22.64 ± 17.86 | 4.19 ± 3.13 | 24.73 | n/a | 0/3 |
| random_noise | fedavg | 9.91 ± 0.13 | 10.00 ± 0.00 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |

### 1.3 Part A Stress Sweep ($\alpha_{\mathrm{dir}} = 5.0$, `label_flip_all`, 10% $\to$ 50% Byzantine, 3 Seeds)

Entries show `Final Acc% ± std (Cons Acc% ± std) / FPR%`:

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| **trustgraph(`stab`)** | 62.36 ± 0.44 (63.54 ± 0.23) / 49.1% | 61.55 ± 0.39 (62.55 ± 0.26) / 48.8% | **60.72 ± 0.35 (61.80 ± 0.38)** / 45.0% | **61.01 ± 0.20 (62.07 ± 0.25)** / 44.1% | **59.88 ± 0.86 (60.96 ± 0.77)** / **43.0%** |
| balance($\gamma=0.15$) | 63.59 ± 0.24 (64.32 ± 0.12) / 0.0% | 61.71 ± 0.72 (63.07 ± 0.06) / 0.0% | 59.83 ± 1.53 (61.55 ± 0.92) / 0.0% | 60.08 ± 1.08 (61.37 ± 0.68) / 0.0% | 59.35 ± 1.02 (60.43 ± 0.82) / 0.0% |
| trimmed_mean | 63.23 ± 0.24 (63.34 ± 0.14) / 49.1% | 60.72 ± 0.33 (61.51 ± 0.27) / 48.4% | 55.99 ± 1.88 (57.60 ± 2.31) / 47.1% | 53.10 ± 5.13 (57.87 ± 1.74) / 48.4% | 46.15 ± 4.59 (52.75 ± 4.51) / 47.6% |
| krum | **64.71 ± 0.50 (65.01 ± 0.50)** / 5.3% | **62.88 ± 1.01 (64.11 ± 0.29)** / 6.1% | 57.44 ± 3.61 (60.65 ± 2.97) / 16.5% | 54.35 ± 7.84 (59.89 ± 3.36) / 22.3% | 27.51 ± 5.33 (27.18 ± 13.14) / **59.7%** |

---

## 2. Part B — Unclipped Pretrained `ResNet-18` ($\alpha_{\mathrm{dir}} = 0.5$, 50 Rounds, `grad_clip_norm=0.0`, 3 Seeds)

> **Contrast with [phase1_paper_match.md](phase1_paper_match.md):** Part B runs `resnet18_pretrained` **without** local gradient clipping (`grad_clip_norm=0.0`, `balance_gamma=0.015`). This serves as a direct 3-seed ablation demonstrating why isolated-peer frozen-BN SGD overflows happen on unclipped ResNet-18 (even on clean `BALANCE` runs with `0.00%` Byzantine acceptance) and how `grad_clip_norm=10.0` in [phase1_paper_match.md](phase1_paper_match.md) reduces NaNs from `20+` runs to **`0 / 150`**.

### 2.1 Main Attacks (Random ER $p=0.3$, 30% Byzantine, 3 Seeds)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | krum | 90.25 ± 0.19 | 90.56 ± 0.25 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | fedavg | 90.23 ± 0.17 | 90.48 ± 0.23 | 0.00 ± 0.00 | 0.00 | n/a | 0/3 |
| none | trimmed_mean | 89.87 ± 0.22 | 90.17 ± 0.27 | 50.31 ± 1.59 | 0.00 | n/a | 0/3 |
| none | trustgraph(`descent_neg`) | 88.18 ± 0.15 | 89.72 ± 0.23 | 44.10 ± 1.70 | 0.00 | +0.020 / n/a | 0/3 |
| none | trustgraph(`stab`) | 87.25 ± 0.43 | 89.31 ± 0.30 | 52.05 ± 1.72 | 0.00 | +0.020 / n/a | 0/3 |
| none | balance($\gamma=0.015$) | 66.48 ± 2.08 | 33.37 ± 33.05 | 58.52 ± 7.16 | 0.00 | n/a | 2/3 |
| **label_flip_all** | **trustgraph(`stab`)** | **82.92 ± 1.07** | **87.82 ± 0.74** | 50.02 ± 3.31 | **33.02** | +0.026 / -0.020 | **0/3** |
| label_flip_all | krum | 82.36 ± 3.64 | 87.09 ± 2.36 | 9.24 ± 2.78 | 37.34 | n/a | 0/3 |
| label_flip_all | trustgraph(`descent_neg`) | 80.99 ± 0.70 | 86.82 ± 1.15 | 39.93 ± 2.88 | 38.75 | +0.036 / -0.046 | 0/3 |
| label_flip_all | trimmed_mean | 77.47 ± 3.19 | 81.36 ± 2.92 | 46.27 ± 3.37 | 42.26 | n/a | 0/3 |
| label_flip_all | fedavg | 69.48 ± 3.76 | 73.17 ± 4.68 | 0.00 ± 0.00 | 100.00 | n/a | 0/3 |
| label_flip_all | balance($\gamma=0.015$) | 59.51 ± 1.47 | 25.30 ± 21.63 | 59.92 ± 8.84 | 36.73 | n/a | 2/3 |
| **model_poison** ($\lambda=10$) | trustgraph(`descent_neg`) | 75.92 ± 6.99 | 10.00 ± 0.00* | 55.46 ± 7.52 | **0.16** | `NaN`* | 3/3* |
| model_poison | trustgraph(`stab`) | 72.75 ± 2.30 | 31.76 ± 30.77* | 60.85 ± 3.05 | 4.28 | `NaN`* | 2/3* |
| model_poison | balance($\gamma=0.015$) | 60.90 ± 11.14 | 48.83 ± 28.41* | 52.07 ± 19.25 | 4.91 | n/a | 1/3* |
| model_poison | fedavg / krum / trimmed_mean | 10.00 ± 0.00 | 10.00 ± 0.00 | 0 / 31.2 / 53.1 | 100 / 71.3 / 46.4 | n/a | 3/3 |
| **random_noise** ($\sigma=0.5$) | **trustgraph(`stab`)** | **78.65 ± 3.02** | **59.59 ± 35.21*** | 56.39 ± 6.34 | **0.00** | `NaN`* | 1/3* |
| random_noise | balance($\gamma=0.015$) | 58.38 ± 1.26 | 26.68 ± 23.59* | 69.56 ± 6.86 | 0.00 | n/a | 2/3* |
| random_noise | trustgraph(`descent_neg`) | 10.00 ± 0.00 | 10.00 ± 0.00 | 95.51 ± 1.48 | 4.02 | `NaN` | 3/3 |
| random_noise | fedavg / krum / trimmed_mean | 10.00 ± 0.00 | 10.00 ± 0.00 | 0 / 30.8 / 52.6 | 100 / 70.6 / 45.6 | n/a | 3/3 |

*\*Isolated-peer frozen-BN SGD overflow (`grad_clip_norm=0.0`); completely eliminated (`0/150` NaNs) by `grad_clip_norm=10.0` in [phase1_paper_match.md](phase1_paper_match.md).*

### 2.2 Topologies Under `label_flip_all` (Unclipped `resnet18_pretrained`, 30% Byzantine, 3 Seeds)

Entries show `Final Acc% ± std (Cons Acc% ± std)`:

| Method | Ring | Random | Small-World |
|---|---:|---:|---:|
| **trustgraph(`stab`)** | 66.98 ± 3.30 (57.54 ± 33.66*, 1/3 NaN) | **82.92 ± 1.07 (87.82 ± 0.74)** | 80.69 ± 0.52 (87.40 ± 0.77) |
| krum | 60.85 ± 5.19 (66.42 ± 8.78) | 82.36 ± 3.64 (87.09 ± 2.36) | **82.16 ± 3.06 (88.17 ± 0.90)** |
| trimmed_mean | 72.62 ± 1.51 (**85.47 ± 0.89**) | 77.47 ± 3.19 (81.36 ± 2.92) | 78.12 ± 1.91 (84.57 ± 0.93) |
| fedavg | **72.65 ± 1.48** (85.46 ± 0.80) | 69.48 ± 3.76 (73.17 ± 4.68) | 72.76 ± 1.48 (79.85 ± 1.48) |
| balance($\gamma=0.015$) | 47.87 ± 0.51 (10.00*, 3/3 NaN) | 59.51 ± 1.47 (25.30*, 2/3 NaN) | 57.66 ± 7.31 (24.01*, 2/3 NaN) |

### 2.3 Part B Stress Sweep (`label_flip_all`, Unclipped `resnet18_pretrained`, 10% $\to$ 50% Byzantine, 3 Seeds)

Entries show `Final Acc% ± std (Cons Acc% ± std) / FPR%`:

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| **trustgraph(`stab`)** | 86.60 ± 0.84 (89.08 ± 0.30) / 52.0% | 84.72 ± 0.91 (88.45 ± 0.70) / 53.3% | **82.92 ± 1.07 (87.82 ± 0.74)** / 50.0% | 82.62 ± 2.20 (87.40 ± 0.48) / 51.8% | **80.86 ± 2.52 (86.21 ± 1.12)** / **48.9%** |
| trimmed_mean | **88.98 ± 0.13 (89.63 ± 0.24)** / 48.7% | 86.86 ± 1.38 (88.71 ± 0.71) / 47.6% | 77.47 ± 3.19 (81.36 ± 2.92) / 46.3% | 74.94 ± 8.19 (77.97 ± 7.56) / 47.8% | 55.93 ± 5.12 (58.83 ± 5.82) / 48.4% |
| krum | 88.41 ± 0.61 (89.62 ± 0.62) / 5.8% | **87.93 ± 1.49 (89.45 ± 0.98)** / 6.5% | 82.36 ± 3.64 (87.09 ± 2.36) / 9.2% | **84.79 ± 5.64 (88.05 ± 2.06)** / 15.0% | 41.68 ± 16.67 (42.32 ± 22.29) / **66.6%** |
| balance($\gamma=0.015$) | 66.76 ± 4.87 (31.35*, 2/3 NaN) / 51.4% | 67.17 ± 2.66 (54.70*, 1/3 NaN) / 58.0% | 59.51 ± 1.47 (25.30*, 2/3 NaN) / 59.9% | 55.24 ± 3.54 (26.40*, 2/3 NaN) / 64.0% | 53.89 ± 3.57 (42.55*, 1/3 NaN) / 53.2% |
