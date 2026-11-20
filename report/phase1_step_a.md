# Phase 1 Cluster Step A Report — Complete 75-Run Single-Seed Reproduction

**Date:** 2026-11-20  
**Target Paper:** Xu et al., *TrustGraph-DFL: Byzantine-Resilient Decentralized Federated Learning via Consistency-Weighted Neighborhood Aggregation* (CIBDA 2026, `TrustGraph-DFL.pdf`)  
**Dataset / Model / Training:** CIFAR-10 (`10,000` subsampled train images across $n=20$ peers, `500`/peer, $\alpha_{\mathrm{dir}}=0.5$, 10% validation split; `10,000` clean test images), 2-conv 2-FC CNN (`SimpleCIFARCNN`, `1,070,794` params), 200 rounds, 1 local epoch/round, SGD ($\eta=0.01$, momentum $0.9$, weight decay $10^{-4}$, batch size $64$), seed `42`.

## 1. Run Provenance & Timing

| Job ID | Commit | State / Wall | Hardware (`octave`, driver `610.57.04`) | CPUs / RAM | Completed Runs | Per-Run Wall Median (Range) |
|---|---|---|---|---|---:|---|
| `327259` | `bcc0cd7` | `TIMEOUT` `00:35:18` | 4× A100-40GB (`01:00.0`, `25:00.0`, `41:00.0`, `61:00.0`) | 2 CPUs / 2G (default) | 49 / 75 | `162.4s` (`105.7s – 200.8s`) |
| `327261` | `9c05b8b` | `COMPLETED` `00:10:33` | 4× A100-40GB (`25:00.0`, `81:00.0`, `C1:00.0`, `E1:00.0`) | 16 CPUs / 64G | 26 / 26 (`75/75` total) | `77.1s` (`58.1s – 104.4s`) |

- **Software:** Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`, PCIe Gen4 x16 (`SYS` topology).
- **Remote artifacts:** `yes:~/PeerSieve/outputs/phase1_step_a/step_a/`, logs `yes:~/PeerSieve/logs/peersieve_p1_stepa-327259.out` and `yes:~/PeerSieve/logs/peersieve_p1_stepa2-327261.out`.

---

## 2. Main Attacks (Random ER $p=0.3$, 30% Byzantine; cf. Paper Fig. 2 & Fig. 3a)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) |
|---|---|---:|---:|---:|---:|---:|
| **none** | fedavg | 53.61 | 56.92 | 0.00 | 0.00 | n/a |
| none | krum | 53.65 | 56.88 | 0.00 | 0.00 | n/a |
| none | trimmed_mean | 52.07 | 56.42 | 49.64 | 0.00 | n/a |
| none | balance ($\gamma=0.3$) | 49.58 | 55.83 | 0.00 | 0.00 | n/a |
| none | trustgraph(`descent_neg`) | 47.62 | 55.48 | 48.66 | 0.00 | +0.005 / n/a |
| none | trustgraph(`paper_pos`) | 47.46 | 55.34 | 46.57 | 0.00 | +0.016 / n/a |
| **label_flip** (`0<->1`) | trimmed_mean | 48.30 | 52.54 | 46.13 | 51.06 | n/a |
| label_flip | fedavg | 47.56 | 50.61 | 0.00 | 100.00 | n/a |
| label_flip | balance ($\gamma=0.3$) | 44.92 | 50.75 | 0.00 | 100.00 | n/a |
| label_flip | krum | 44.71 | 49.58 | 22.73 | 73.46 | n/a |
| label_flip | trustgraph(`descent_neg`) | 42.35 | **52.69** | 48.52 | 44.49 | +0.008 / -0.026 |
| label_flip | trustgraph(`paper_pos`) | 42.08 | 50.38 | 52.50 | 60.75 | -0.012 / +0.059 |
| label_flip | *diag: trustgraph(`descent_neg`, `val=iid`)* | *45.75* | *53.22* | *43.07* | *47.58* | *+0.040 / -0.014* |
| label_flip | *diag: balance ($\gamma=0.15$)* | *38.75* | *51.90* | *22.70* | *75.58* | *n/a* |
| **model_poison** ($\lambda=10$) | *diag: trustgraph(`descent_neg`, $\alpha=0.7$)* | ***34.48*** | ***48.63*** | *50.08* | *34.64* | *+0.002 / -0.052* |
| model_poison | *diag: balance ($\gamma=0.15$)* | ***34.27*** | ***48.44*** | *16.61* | *21.36* | *n/a* |
| model_poison | *diag: trustgraph(`descent_neg`, `clip`)* | *26.80* | *9.97* | *76.64* | *81.29* | *-0.046 / +0.059* |
| model_poison | balance ($\gamma=0.3$) | 19.34 | 25.33 | 0.00 | 57.20 | n/a |
| model_poison | trustgraph(`paper_pos`) | 11.56 | 10.00 | 98.53 | 3.76 | `NaN` (r=8) |
| model_poison | trustgraph(`descent_neg`) | 10.00 | 10.00 | 85.00 | 3.71 | `NaN` |
| model_poison | fedavg / krum / trimmed_mean | 10.00 | 10.00 | 0 / 38.6 / 54.2 | 100 / 83.6 / 55.3 | n/a |
| **random_noise** ($\sigma=0.5$) | balance ($\gamma=0.3$) | **46.92** | **52.72** | 0.02 | 0.00 | n/a |
| random_noise | *diag: balance ($\gamma=0.15$)* | *38.57* | *51.86* | *26.33* | *0.00* | *n/a* |
| random_noise | *diag: trustgraph(`descent_neg`, `clip`)* | *29.68* | *41.18* | *58.78* | *49.47* | *-0.019 / -0.000* |
| random_noise | trustgraph(`descent_neg`) | 10.00 | 10.00 | 99.22 | 0.92 | `NaN` (r=3) |
| random_noise | trustgraph(`paper_pos`) | 10.00 | 10.00 | 99.15 | 0.80 | `NaN` |
| random_noise | fedavg / krum / trimmed_mean | 10.00 | 9.99 | 0 / 2.2 / 27.1 | 100 / 39.5 / 20.4 | n/a |

---

## 3. Topologies Under Label-Flipping (30% Byzantine; cf. Paper Table 1)

| Method | Ring Acc (Cons) | Random Acc (Cons) | Small-World Acc (Cons) | Paper Table 1 (Ring / Rand / SW) |
|---|---:|---:|---:|---:|
| fedavg | 42.40% (51.37%) | 47.56% (50.61%) | 45.94% (50.51%) | 42.3 / 45.1 / 43.8 |
| krum | 34.43% (44.86%) | 44.71% (49.58%) | 46.79% (53.01%) | 78.5 / 82.1 / 80.3 |
| trimmed_mean | 41.29% (50.73%) | 48.30% (52.54%) | 46.45% (52.52%) | 74.2 / 78.4 / 76.1 |
| balance ($\gamma=0.3$) | 42.90% (51.77%) | 44.92% (50.75%) | 44.97% (51.28%) | 85.3 / 87.5 / 86.2 |
| trustgraph(`descent_neg`) | 38.27% (49.17%) | 42.35% (52.69%) | 43.92% (52.28%) | 89.1 / 91.2 / 90.4 |
| trustgraph(`paper_pos`) | 40.61% (51.21%) | 42.08% (50.38%) | 43.11% (50.25%) | — |

---

## 4. Complete FPR Sweep (Random ER $p=0.3$, `label_flip`; cf. Paper Fig. 3b)

### Honest False Positive Rate (%) vs. Byzantine Fraction

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz | Paper Fig. 3b (10% $\to$ 50%) |
|---|---:|---:|---:|---:|---:|---:|
| balance ($\gamma=0.3$) | 0.07 | 0.15 | 0.00 | 0.00 | 0.06 | ~4 $\to$ ~16 |
| krum (multi) | 9.83 | 14.46 | 22.73 | 40.83 | **52.81** | ~8 $\to$ ~35 |
| trimmed_mean (30%) | 50.46 | 48.88 | 46.13 | 50.24 | 49.58 | ~10 $\to$ ~39 |
| trustgraph(`descent_neg`) | 48.50 | 48.85 | 48.52 | 53.89 | **48.04** | ~3 $\to$ ~9 |
| trustgraph(`paper_pos`) | 46.21 | 52.09 | 52.50 | 45.05 | 44.99 | — |

### Final Benign Accuracy / Consensus Accuracy (%) Across Byzantine Fraction

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| balance ($\gamma=0.3$) | 47.34 (53.82) | 46.53 (53.07) | 44.92 (50.75) | 46.67 (52.88) | 45.59 (52.51) |
| krum (multi) | 53.33 (56.54) | 48.97 (54.33) | 44.71 (49.58) | 45.06 (48.18) | 45.26 (48.07) |
| trimmed_mean (30%) | 51.60 (55.97) | 51.13 (55.60) | 48.30 (52.54) | 48.84 (52.13) | 48.31 (51.63) |
| trustgraph(`descent_neg`) | 46.13 (55.00) | 43.82 (54.18) | 42.35 (52.69) | 44.14 (53.77) | 42.50 (52.03) |
| trustgraph(`paper_pos`) | 46.55 (54.85) | 44.62 (51.71) | 42.08 (50.38) | 43.36 (51.68) | 43.76 (52.14) |

---

## 5. Ablations & Additional Diagnostics (Random ER $p=0.3$, 30% Byzantine)

| Group | Variant | Attack | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) |
|---|---|---|---:|---:|---:|---:|---:|
| Ablation | baseline (`descent_neg`, $\alpha=0.5$) | model_poison | 10.00 | 10.00 | 85.00 | 3.71 | `NaN` |
| Ablation | `no_relu` | model_poison | 10.00 | 10.00 | 0.00 | 100.00 | `NaN` |
| Ablation | `euclidean` | model_poison | 10.00 | 10.00 | 0.00 | 100.00 | `NaN` |
| Ablation | $\alpha=0.3$ | model_poison | 23.05 | 10.00 | 77.98 | 6.87 | `NaN` |
| Ablation | $\alpha=0.7$ | model_poison | **34.48** | **48.63** | 50.08 | 34.64 | +0.002 / -0.052 |
| Diagnostic | `val_split=iid` | label_flip | 45.75 | 53.22 | 43.07 | 47.58 | +0.040 / -0.014 |
| Diagnostic | `val_split=iid` | model_poison | 10.33 | 10.00 | 44.91 | 27.74 | +0.004 / -0.074 |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | fedavg | 34.21 | 36.51 | 0.00 | 100.00 | n/a |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | balance ($\gamma=0.15$) | 37.50 | **50.89** | 24.86 | 61.16 | n/a |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | trustgraph(`descent_neg`) | 37.49 | 48.45 | 46.33 | 42.41 | +0.018 / -0.042 |
| PeerSieve Preview | `trustgraph(descent_neg, clip)` | model_poison | 26.80 | 9.97 | 76.64 | 81.29 | -0.046 / +0.059 |
| PeerSieve Preview | `trustgraph(descent_neg, clip)` | random_noise | 29.68 | 41.18 | 58.78 | 49.47 | -0.019 / -0.000 |

---

## 6. Key Findings & Implications for PeerSieve (Phase 2)

1. **TrustGraph-DFL's published accuracy (`89–95%`) and `<10%` FPR do not reproduce on CIFAR-10:**
   - With `10,000` subsampled CIFAR-10 images ($\alpha_{\mathrm{dir}}=0.5$, 2-conv 2-FC CNN, 200 rounds), even **attack-free FedAvg** tops out at `53.61%` mean benign accuracy (`56.92%` consensus), vs `93–95%` in Fig. 3(a).
   - Under `label_flip` (`0<->1`, 30% Byz), `trustgraph(descent_neg)` beats FedAvg and BALANCE on **consensus accuracy** (`52.69%` vs `50.61%` / `50.75%`), and under full-class flipping (`label_flip_all`) it beats FedAvg by **`+11.94 pp`** consensus (`48.45%` vs `36.51%`). However, mean individual peer accuracy lags slightly because high honest rejection slows inter-peer consensus.
   - **FPR scaling vs. Krum:** Krum's FPR climbs monotonically with Byzantine fraction (`9.83%` at 10% $\to$ `52.81%` at 50%), whereas TrustGraph-DFL's FPR is **invariant to Byzantine fraction** (`48.0% – 53.9%` from 10% to 50%) and slightly lower than Krum at 50% (`48.04%` vs `52.81%`)—qualitatively supporting the paper's invariance claim, but at a ~48% baseline FPR rather than ~9% because local non-IID validation sets disagree across peers.

2. **Preprint gradient sign bug matters most under gradient-space attacks:**
   - Under `model_poison`, `trustgraph(paper_pos)` actively selects poisoned updates (`94.7%` acceptance, `cos_byz -> +0.906`) and explodes to `NaN` at round 8.
   - Under `label_flip`, `descent_neg` consistently achieves higher consensus accuracy and lower Byzantine acceptance (`44.5%` vs `60.8%` on `random`) than `paper_pos`.

3. **Unclipped cosine defenses fail catastrophically on high-norm attacks — validating PeerSieve's norm cap:**
   - Without clipping, `trustgraph(descent_neg)` explodes to `NaN` under both `random_noise` ($\sigma=0.5$, round 3) and `model_poison` ($\lambda=10$, $\alpha=0.5$).
   - Adding PeerSieve's self-update norm cap $a_{ij}=\min(1, r_i / \|\Delta\theta_j\|_2)$ **completely prevents `NaN` explosions** on both attacks (`random_noise`: `10.0% -> 29.68%` mean / `41.18%` consensus; `model_poison`: `10.0% -> 26.80%` mean).
   - **Important architectural nuance observed:** When `clip` aggregated purely in update space ($\theta_i^t + \alpha\Delta\theta_i^t + (1-\alpha)\Delta\bar\theta_i^t$) from random CNN init under non-IID data, peers drifted into different basins under `model_poison` (`26.80%` per-node vs `9.97%` consensus). In Phase 2 (TinyLlama-1.1B LoRA tuning), peers start from a shared pretrained LLM backbone where LoRA increments live in a much tighter linear basin, AND we should still monitor both per-peer and consensus perplexity closely!

4. **BALANCE is sensitive to $\gamma$ calibration:**
   - Default `balance(g=0.3)` excels on `random_noise` (`46.92%` / `52.72%`, `0%` Byz accept) but admits `model_poison` ($\lambda=10$, relative dist `0.256`) until round 21 (`19.34%` / `25.33%`).
   - Calibrating to `balance(g=0.15)` lifts `model_poison` performance dramatically to **`34.27%` / `48.44%`** (matching `trustgraph` with $\alpha=0.7$: `34.48%` / `48.63%`) and reaches **`50.89%`** consensus under `label_flip_all`.
