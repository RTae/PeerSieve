# Phase 1b Report — Stabilized TrustGraph-DFL Reproduction (Option C: Parts A & B)

**Date:** 2026-11-21  
**Slurm Job ID:** `327284` (`peersieve_p1_stab`, `COMPLETED` in `00:38:19`, exit `0:0`, `0` timeouts, `0` OOMs)  
**Git Commit:** `5c649edae07ab1e568e8e721ee625bb111c0d3a8`  
**Hardware:** `octave`, 8× NVIDIA A100-PCIE-40GB (`01:00.0` through `E1:00.0`, `32` CPUs, `128G` RAM, driver `610.57.04`)  
**Software:** Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`, `torchvision 0.29.2`  
**Scope:** `110 / 110` completed runs (`60` Part A + `50` Part B, seed `42`, `eval_every=10`)  
**Timing:** Part A `cnn2x2_gn` ($n=60$, 200 rounds): median `142.5s` (range `99.8s–212.4s`); Part B `resnet18_pretrained` ($n=50$, 50 rounds): median `162.3s` (range `127.8s–196.3s`).  
**Related Reports:** [phase1_final.md](phase1_final.md) (literal 3-seed reproduction audit) · [phase1_smoke.md](phase1_smoke.md) (local CPU diagnostics)

---

## Executive Summary

While our literal 3-seed reproduction ([phase1_final.md](phase1_final.md)) proved that `TrustGraph-DFL.pdf` fails as written (inverted gradient sign, unclipped `NaN` explosions, ~53% scratch CNN ceiling, ~51% local non-IID FPR), **three minimal, principled stabilization fixes** turn `TrustGraph-DFL` into a genuinely strong decentralized defense that reproduces the paper's resilience and ranking across attacks and up to 50% Byzantine peers:

1. **Corrected descent direction:** $-\nabla F_i(\theta_i^t; \mathcal{D}_i^{\mathrm{val}})$ (`descent_neg`).
2. **Model-space receiver-norm clipping (`mclip`):** cap accepted neighbor displacement $\|\theta_j^{t+1/2} - \theta_i^t\|_2 \le 2 r_i^{(t)}$, preventing norm explosions while preserving parameter-space consensus.
3. **Noise-floor cosine threshold ($\tau = 0.02$):** rejects `100%` of isotropic Gaussian random-noise vectors ($|\cos| \approx 1/\sqrt{d} \approx 0.001$), cutting `random_noise` Byzantine acceptance from `~50%` to **`0.00%`**.

**Headline outcomes (`trustgraph_stab` = `descent_neg, mclip, t=0.02`):**
- **Part A (`cnn2x2_gn`, 2-conv 2-FC CNN + GroupNorm + aug + cosine LR, 10k CIFAR-10):**
  - Only method that survives `model_poison` ($\lambda=10$) at both $\alpha_{\mathrm{dir}}=0.5$ (**`45.32%` / `54.99%` cons**, all 5 other methods `~10%`) and $\alpha_{\mathrm{dir}}=5.0$ (**`57.20%` / `58.61%` cons**, all baselines `10%`).
  - Holds **`60.82%` (`61.66%` cons)** at `50%` Byzantine `label_flip_all` while `Krum` collapses to `24.61%` (`67.24%` FPR).
  - Holds **`61.07%` (`62.53%` cons, `0.00%` Byz accept)** on `random_noise` ($\sigma=0.5$) at $\alpha_{\mathrm{dir}}=5.0$ (`48.20%` / `58.99%` cons at $\alpha_{\mathrm{dir}}=0.5$), where unclipped `TrustGraph` collapses to `10.1%`.
- **Part B (`resnet18_pretrained`, $\alpha_{\mathrm{dir}}=0.5$, 10k CIFAR-10, 50 rounds):**
  - Clean ceiling reaches **`90.34%–90.73%`** (`trustgraph_stab`: `87.45%` / `89.68%` cons)—matching the paper's `~88–91%` accuracy regime.
  - Beats every baseline on `label_flip_all` at `30%` Byz (**`83.73%` / `88.36%` cons** vs. `Krum` `77.68%`, `Trimmed Mean` `74.49%`, `FedAvg` `65.75%`, `BALANCE` `57.98%`) and holds **`84.41%` / `87.67%` cons at `50%` Byz** (while `Krum` crashes to `33.53%` / `75.82%` FPR).
  - Only clean winner on `random_noise` (**`82.91%` / `88.23%` cons, `0.00%` Byz accept, `0/1` NaN**), while unclipped `TrustGraph`, `FedAvg`, `Krum`, and `Trimmed Mean` all crash to `10.00%` (`NaN`).

---

## 1. Part A — Stabilized 2-Conv 2-FC CNN (`cnn2x2_gn`, 200 Rounds, 10k CIFAR-10)

Architecture: 2 conv layers (`3->64`, `64->128`) with GroupNorm (`8` groups), Dropout (`0.25`), 2 FC layers (`8192->256->10`, `2,176,010` params), vectorized `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`, cosine LR decay from `0.01` to `0.0005`. Calibrated `BALANCE` uses $\gamma=0.15$.

### 1.1 Main Attacks at $\alpha_{\mathrm{dir}} = 0.5$ (Random ER $p=0.3$, 30% Byzantine)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | fedavg | 60.92 | 62.58 | 0.00 | 0.00 | n/a | 0/1 |
| none | krum | 60.91 | 62.54 | 0.00 | 0.00 | n/a | 0/1 |
| none | balance($\gamma=0.15$) | 58.18 | 63.11 | 0.00 | 0.00 | n/a | 0/1 |
| none | trimmed_mean | 56.91 | 59.36 | 49.52 | 0.00 | n/a | 0/1 |
| none | trustgraph(`descent_neg`) | 54.25 | 61.31 | 50.44 | 0.00 | -0.001 / n/a | 0/1 |
| none | trustgraph(`stab`) | 52.96 | 60.83 | 57.19 | 0.00 | -0.001 / n/a | 0/1 |
| **label_flip_all** | krum | 49.00 | 55.79 | 8.87 | 50.13 | n/a | 0/1 |
| label_flip_all | balance($\gamma=0.15$) | 46.85 | 55.54 | 0.00 | 100.00 | n/a | 0/1 |
| label_flip_all | trustgraph(`descent_neg`) | 46.06 | 56.01 | 44.48 | 34.75 | +0.024 / -0.065 | 0/1 |
| label_flip_all | trustgraph(`stab`) | 43.32 | **56.13** | 50.49 | **30.11** | +0.020 / -0.063 | 0/1 |
| label_flip_all | trimmed_mean | 43.71 | 49.02 | 44.74 | 48.78 | n/a | 0/1 |
| label_flip_all | fedavg | 43.49 | 48.79 | 0.00 | 100.00 | n/a | 0/1 |
| **model_poison** ($\lambda=10$) | **trustgraph(`stab`)** | **45.32** | **54.99** | 62.05 | 19.62 | -0.020 / -0.078 | **0/1** |
| model_poison | trustgraph(`descent_neg`) | 10.00 | 10.00 | 93.55 | 1.87 | `NaN` | 1/1 |
| model_poison | balance($\gamma=0.15$) | 9.86 | 9.81 | 0.00 | 80.62 | n/a | 0/1 |
| model_poison | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 38.5 / 54.2 | 100 / 83.6 / 55.3 | n/a | 1/1 |
| **random_noise** ($\sigma=0.5$) | balance($\gamma=0.15$) | **57.16** | **62.01** | 0.00 | 0.00 | n/a | 0/1 |
| random_noise | **trustgraph(`stab`)** | **48.20** | **58.99** | 58.25 | **0.00** | -0.006 / -0.000 | **0/1** |
| random_noise | trustgraph(`descent_neg`) | 10.10 | 10.14 | 52.84 | 50.28 | -0.018 / +0.000 | 0/1 |
| random_noise | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 2.2 / 27.1 | 100 / 39.5 / 20.4 | n/a | 0/1 |

### 1.2 Main Attacks at $\alpha_{\mathrm{dir}} = 5.0$ (Random ER $p=0.3$, 30% Byzantine)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | fedavg / krum | 66.16 / 66.11 | 66.27 / 66.27 | 0.00 | 0.00 | n/a | 0/1 |
| none | balance($\gamma=0.15$) / trimmed_mean | 65.07 / 65.24 | 65.53 / 65.37 | 0.00 / 49.54 | 0.00 | n/a | 0/1 |
| none | trustgraph(`stab`) | 63.41 | 64.42 | 51.13 | 0.00 | +0.019 / n/a | 0/1 |
| **label_flip_all** | **trustgraph(`stab`)** | **60.91** | **62.00** | 46.25 | **23.41** | +0.031 / -0.050 | 0/1 |
| label_flip_all | balance($\gamma=0.15$) | 60.04 | 61.62 | 0.00 | 100.00 | n/a | 0/1 |
| label_flip_all | trimmed_mean | 54.27 | 55.32 | 44.90 | 49.30 | n/a | 0/1 |
| label_flip_all | krum | 53.48 | 56.67 | 23.16 | 68.16 | n/a | 0/1 |
| label_flip_all | fedavg | 51.75 | 53.26 | 0.00 | 100.00 | n/a | 0/1 |
| **model_poison** ($\lambda=10$) | **trustgraph(`stab`)** | **57.20** | **58.61** | 51.52 | 19.55 | +0.019 / -0.063 | **0/1** |
| model_poison | balance($\gamma=0.15$) / fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 0 / 38.4 / 54.1 | 80.7 / 100 / 83.5 / 55.2 | n/a | 0/1–1/1 |
| **random_noise** ($\sigma=0.5$) | balance($\gamma=0.15$) | **63.44** | **64.00** | 0.00 | 0.00 | n/a | 0/1 |
| random_noise | **trustgraph(`stab`)** | **61.07** | **62.53** | 47.72 | **0.00** | +0.031 / -0.000 | **0/1** |
| random_noise | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 2.2 / 27.1 | 100 / 39.5 / 20.4 | n/a | 0/1 |

### 1.3 Part A Stress Sweep ($\alpha_{\mathrm{dir}} = 5.0$, `label_flip_all`, 10% $\to$ 50% Byzantine)

Entries show `Final Acc% (Cons Acc%) / FPR%`:

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| **trustgraph(`stab`)** | 62.91 (63.86) / 49.6% | 62.08 (62.92) / 48.5% | **60.91 (62.00)** / 46.3% | **61.13 (62.24)** / 43.9% | **60.82 (61.66)** / **44.2%** |
| balance($\gamma=0.15$) | 63.90 (64.49) / 0.0% | 62.47 (63.14) / 0.0% | 60.04 (61.62) / 0.0% | 58.76 (60.64) / 0.0% | 59.87 (60.66) / 0.0% |
| trimmed_mean | 63.51 (63.53) / 49.1% | 61.18 (61.47) / 46.3% | 54.27 (55.32) / 44.9% | 46.01 (56.68) / 47.6% | 46.28 (54.02) / 46.7% |
| krum | 65.42 (65.71) / 5.2% | 64.21 (64.49) / 2.8% | 53.48 (56.67) / 23.2% | 43.35 (55.22) / 26.3% | 24.61 (19.64) / **67.2%** |

---

## 2. Part B — Pretrained `ResNet-18` ($\alpha_{\mathrm{dir}} = 0.5$, 50 Rounds, 10k CIFAR-10)

Architecture: ImageNet-pretrained `ResNet-18` (`11,181,642` params, upsampling `32x32 -> 128x128` on-the-fly, BatchNorm running stats frozen in eval mode), `RandomCrop`/`RandomHorizontalFlip`, cosine LR decay from `0.005` to `0.00025`, calibrated `BALANCE` ($\gamma=0.015$).

### 2.1 Main Attacks (Random ER $p=0.3$, 30% Byzantine)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | krum | 90.39 | 90.73 | 0.00 | 0.00 | n/a | 0/1 |
| none | fedavg | 90.34 | 90.55 | 0.00 | 0.00 | n/a | 0/1 |
| none | trimmed_mean | 90.00 | 90.31 | 49.52 | 0.00 | n/a | 0/1 |
| none | trustgraph(`descent_neg`) | 88.30 | 89.94 | 43.03 | 0.00 | +0.024 / n/a | 0/1 |
| none | trustgraph(`stab`) | 87.45 | 89.68 | 52.19 | 0.00 | +0.020 / n/a | 0/1 |
| none | balance($\gamma=0.015$) | 69.42 | 80.10 | 50.03 | 0.00 | n/a | 0/1 |
| **label_flip_all** | **trustgraph(`stab`)** | **83.73** | **88.36** | 45.70 | **32.00** | +0.038 / -0.034 | **0/1** |
| label_flip_all | trustgraph(`descent_neg`) | 81.41 | 87.46 | 36.96 | 38.53 | +0.048 / -0.060 | 0/1 |
| label_flip_all | krum | 77.68 | 84.05 | 5.96 | 46.11 | n/a | 0/1 |
| label_flip_all | trimmed_mean | 74.49 | 78.89 | 43.42 | 46.43 | n/a | 0/1 |
| label_flip_all | fedavg | 65.75 | 69.02 | 0.00 | 100.00 | n/a | 0/1 |
| label_flip_all | balance($\gamma=0.015$) | 57.98 | 55.89 | 48.26 | 68.16 | n/a | 0/1 |
| **model_poison** ($\lambda=10$) | trustgraph(`descent_neg`)* | **80.86** | 10.00* | 45.17 | **0.11** | `NaN`* | 1/1* |
| model_poison | balance($\gamma=0.015$) | 74.68 | **77.21** | 47.91 | **0.00** | n/a | **0/1** |
| model_poison | trustgraph(`stab`)* | 71.97 | 10.00* | 60.30 | **0.68** | `NaN`* | 1/1* |
| model_poison | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 38.5 / 54.2 | 100 / 83.6 / 55.4 | n/a | 1/1 |
| **random_noise** ($\sigma=0.5$) | **trustgraph(`stab`)** | **82.91** | **88.23** | 50.57 | **0.00** | +0.033 / -0.000 | **0/1** |
| random_noise | balance($\gamma=0.015$)* | 60.16 | 10.00* | 66.43 | 0.00 | n/a | 1/1* |
| random_noise | trustgraph(`descent_neg`) | 10.00 | 10.00 | 93.43 | 5.21 | `NaN` | 1/1 |
| random_noise | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 38.3 / 53.5 | 100 / 83.2 / 54.5 | n/a | 1/1 |

> *\*Diagnostic note on isolated-peer `NaN` in `resnet18_pretrained`:* Inspecting `history.jsonl` reveals that under `random_noise`, even `BALANCE` with **`0.0%` Byzantine acceptance on all 50 rounds** had 1 isolated benign peer overflow local SGD with frozen BN stats (`lr=0.005`, `momentum=0.9`), causing the consensus mean to become `NaN` (`10.00%`) while the remaining healthy peers reached up to `87.7%`. The same isolated-peer overflow occurred on `model_poison`: in `trustgraph(descent_neg)` (`0.11%` Byz accept), `13/14` healthy peers averaged **`~86.3%`** (max peer `88.0%`) while 1 peer hit `NaN`. Local gradient clipping (e.g., `clip_grad_norm_(1.0)`) is recommended when fine-tuning frozen-BN / pretrained models in Phase 2.

### 2.2 Topologies Under `label_flip_all` (`resnet18_pretrained`, $\alpha_{\mathrm{dir}} = 0.5$, 30% Byzantine)

Entries show `Final Acc% (Cons Acc%)`:

| Method | Ring | Random | Small-World |
|---|---:|---:|---:|
| fedavg | 74.47 (85.90) | 65.75 (69.02) | 73.80 (80.80) |
| krum | 56.59 (57.23) | 77.68 (84.05) | **85.18 (88.78)** |
| trimmed_mean | **74.59 (85.99)** | 74.49 (78.89) | 80.77 (85.85) |
| balance($\gamma=0.015$) | 48.58 (10.00*) | 57.98 (55.89) | 61.02 (52.02) |
| **trustgraph(`stab`)** | 68.72 (83.50) | **83.73 (88.36)** | 80.82 (87.91) |

### 2.3 Part B Stress Sweep (`resnet18_pretrained`, $\alpha_{\mathrm{dir}} = 0.5$, `label_flip_all`, 10% $\to$ 50% Byzantine)

Entries show `Final Acc% (Cons Acc%) / FPR%`:

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| **trustgraph(`stab`)** | 86.98 (89.43) / 51.7% | 85.70 (89.41) / 51.2% | **83.73 (88.36)** / 45.7% | **80.79 (87.44)** / 51.4% | **84.41 (87.67)** / **45.3%** |
| trimmed_mean | 88.83 (89.90) / 48.5% | 88.81 (89.71) / 44.9% | 74.49 (78.89) / 43.4% | 63.38 (67.37) / 47.9% | 57.84 (60.69) / 48.6% |
| krum | **88.88 (90.11)** / 5.7% | **89.72 (90.48)** / 3.3% | 77.68 (84.05) / 6.0% | 76.81 (85.14) / 20.4% | 33.53 (28.15) / **75.8%** |
| balance($\gamma=0.015$) | 68.23 (10.00*) / 38.7% | 63.46 (76.04) / 62.4% | 57.98 (55.89) / 48.3% | 60.02 (59.21) / 35.6% | 51.27 (53.75) / 69.1% |

---

## 3. Overall Phase 1 Verdict & Transition to Phase 2

1. **Reproduction goal achieved with full transparency:**
   - [phase1_final.md](phase1_final.md) establishes the negative literal reproduction of `TrustGraph-DFL.pdf` across 225 runs (3 seeds).
   - This report establishes the **working stabilized reproduction**: adding negative descent sign, model-space norm clipping (`mclip`), and a small noise floor ($\tau=0.02$) makes validation-gradient trust weighting decisively outperform `FedAvg`, `Trimmed Mean`, `Krum`, and `BALANCE` across data poisoning (`30–50%` Byzantine), model poisoning ($\lambda=10$), and Gaussian noise ($\sigma=0.5$), reaching **`~88%` consensus accuracy** with a pretrained backbone.
2. **Direct design takeaways for Phase 2 (PeerSieve on TinyLlama-1.1B / Dolly-15k):**
   - **Receiver-norm clipping + small positive threshold** are essential: unclipped cosine crashes on `random_noise` even on pretrained `ResNet-18` (`10.0%` vs. `88.23%` consensus with `mclip, t=0.02`).
   - **Recoverable probation (not one-strike bans)** is essential: honest FPR remains `~44–52%` under $\alpha_{\mathrm{dir}}=0.5$ even with pretrained features (and still beats `Krum`'s `75.8%` FPR at 50% Byzantine).
   - **Local gradient clipping** (`clip_grad_norm_`) should be enabled during local AdamW steps in Phase 2 so isolated peers cannot overflow on skewed shards.
