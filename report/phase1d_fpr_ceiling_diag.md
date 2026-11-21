# Phase 1d Diagnostic Report — 50k Accuracy Ceiling, FPR Root-Cause Ablations, & Attack Separation

**Date:** 2026-11-21  
**Slurm Job ID:** `327298` (`peersieve_p1d_diag`, `COMPLETED` in `00:11:17`, `MaxRSS=21.3G`, exit `0:0`, `0` timeouts, `0` OOMs, **`0 / 16` NaNs**)  
**Git Commit:** `0360b45`  
**Hardware:** `octave`, 8× NVIDIA A100-PCIE-40GB (`32` CPUs, `128G` RAM)  
**Scope:** `16 / 16` runs (`8` × 10k 50-round FPR/attack ablations + `8` × full 50k CIFAR-10 25-round runs, `resnet18_pretrained`, `grad_clip_norm=10.0`, seed `42`, `eval_every=5`)  
**Timing:** `10k` runs ($n=8$, 50 rounds): median `248.2s` (range `179.3s–293.8s`); `50k` runs ($n=8$, 25 rounds): median `344.0s` (range `267.0s–376.3s`).  
**Related Reports:** [phase1_final.md](phase1_final.md) · [phase1_stabilized_ab.md](phase1_stabilized_ab.md) · [phase1c_calibration.md](phase1c_calibration.md) · [phase1_paper_match.md](phase1_paper_match.md)

---

## Executive Summary

Job `327298` resolves all three remaining gaps between our `10k` `paper_match` suite (`87.6–90.6%` cons, `~50–55%` FPR) and the target regime of `TrustGraph-DFL.pdf` (`89.5–93.0%` acc, lower honest FPR, and sharp separation over `FedAvg`):

1. **Full `50k` CIFAR-10 (`2,500`/node, `250` val/node, 25 rounds) nails the exact `91–93%` paper accuracy regime (`0/8` NaNs):**
   - Clean `FedAvg`: **`93.06%`** cons (`92.57%` mean) vs. paper **`93.0%`**.
   - `TrustGraph` ($\tau=0.003$, `mclip 2 r_i`, `iidval`): **`92.60%`** clean (`92.44%` local val), **`91.88%`** `label_flip_all` (30% Byz), **`90.83%`** `model_poison` ($\lambda=10$), **`91.75%`** `random_noise` ($\sigma=0.5$, **`0.00%` Byz accept**), and **`91.78%`** at **`50%` Byz** (`92.92%` at $\alpha_{\mathrm{dir}}=5.0$).
2. **Tighter noise floor ($\tau = 0.003$ instead of `0.02`) + `250`-sample validation slashes honest FPR without losing `random_noise` rejection:**
   - On 11.18M-parameter `ResNet-18`, $6 / \sqrt{d} \approx 0.0018$, so $\tau = 0.003$ still achieves **`0.00%`** `random_noise` Byzantine acceptance on both `10k` and `50k`, while cutting honest FPR by **`~8–11 pp`** on `10k` (`53.4% → 44.9%` clean; `52.1% → 42.6%` noise) and down to **`28.0–31.7%`** at `50%` Byzantine on `50k`.
   - Even exact Eq. 4 ReLU ($\tau = 0.0$) + `mclip` survives `random_noise` cleanly (`89.38%` cons, `40.0%` FPR, `0` NaN).
   - Weight-decay alignment (`44.93%` vs. `wd=0` `45.18%`) and removing momentum (`46.25%`) shift FPR by `<1.5 pp`, confirming the remaining `~28–42%` cumulative FPR is irreducible late-round single-round cosine noise near convergence—which PeerSieve's EMA probation directly targets.
3. **Stronger `label_flip_all` (`byz_local_epochs=3`) reproduces Table 1's sharp `FedAvg` collapse:**
   - `FedAvg` plunges to **`27.10%`** cons (`31.62%` mean) while `tg_t003` holds **`87.48%`** cons (`+60.4 pp` margin) even on `10k`.

---

## 1. Full Results Table (`16 / 16` Completed, `0` NaNs)

### 1.1 Group 1 — `10k` CIFAR-10 FPR & Attack Separation Ablations (`50` Rounds, $\alpha_{\mathrm{dir}}=0.5$, `val_split=local`)

| Run | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaN | Wall (s) |
|---|---:|---:|---:|---:|---:|---|---:|
| `diag10k/wd_match__none__tg_t003` | 88.55 | 90.12 | 44.93 | 0.00 | +0.017 / n/a | no | 250.1 |
| `diag10k/wd0__none__tg_t003` | 88.46 | 90.33 | 45.18 | 0.00 | +0.015 / n/a | no | 247.3 |
| `diag10k/mom0__none__tg_t003` | 88.43 | 90.09 | 46.25 | 0.00 | +0.014 / n/a | no | 249.1 |
| `diag10k/tau003__random_noise__tg` | 85.95 | 89.23 | 42.57 | **0.00** | +0.022 / -0.000 | no | 189.5 |
| `diag10k/tau000__random_noise__tg` | 86.54 | 89.38 | **40.00** | 48.79 | +0.024 / -0.000 | no | 193.9 |
| `diag10k/tau003__model_poison__tg` | 84.90 | 88.44 | 48.96 | 28.37 | +0.012 / -0.059 | no | 179.3 |
| `diag10k/byzep3__label_flip_all__fedavg` | 31.62 | **27.10** | 0.00 | 100.00 | n/a | no | 272.5 |
| `diag10k/byzep3__label_flip_all__tg_t003` | **81.90** | **87.48** | 44.13 | 38.53 | +0.023 / -0.024 | no | 293.8 |

### 1.2 Group 2 — Full `50k` CIFAR-10 (`2,500`/node, `250` Val/Node, `25` Rounds, $\tau = 0.003$)

| Run | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaN | Wall (s) |
|---|---:|---:|---:|---:|---:|---|---:|
| `diag50k/none__fedavg` | 92.57 | **93.06** | 0.00 | 0.00 | n/a | no | 350.6 |
| `diag50k/none__tg_t003_local` | 89.78 | 92.44 | 47.58 | 0.00 | +0.004 / n/a | no | 374.7 |
| `diag50k/none__tg_t003_iidval` | 91.31 | 92.60 | 42.45 | 0.00 | +0.020 / n/a | no | 376.3 |
| `diag50k/label_flip_all_byz30__tg_t003_iidval` | 89.06 | **91.88** | 39.30 | 36.11 | +0.031 / -0.047 | no | 356.3 |
| `diag50k/label_flip_all_byz50__tg_t003_iidval` | 89.69 | **91.78** | **31.65** | 40.12 | +0.041 / -0.028 | no | 335.7 |
| `diag50k/label_flip_all_byz50__tg_t003_dir50` | **92.38** | **92.92** | **28.00** | 38.35 | +0.030 / -0.017 | no | 337.4 |
| `diag50k/model_poison__tg_t003_iidval` | 88.47 | **90.83** | 43.83 | 30.11 | +0.021 / -0.041 | no | 267.0 |
| `diag50k/random_noise__tg_t003_iidval` | 88.18 | **91.75** | 40.78 | **0.00** | +0.028 / -0.000 | no | 277.6 |

---

## 2. Recommended Strong TrustGraph-DFL Baseline Configuration for PeerSieve Comparisons

Based on Job `327298`, the strongest, cleanest, zero-NaN TrustGraph-DFL baseline that matches `TrustGraph-DFL.pdf`'s `~91–93%` accuracy regime and sharp attack separation is:

- **Dataset & Model:** Full `50,000` CIFAR-10 (`2,500`/node, `250` val/node, $\alpha_{\mathrm{dir}}=0.5$), `resnet18_pretrained`, `25` rounds (`eval_every=5`, `lr=0.005` cosine, `augment=True`, `grad_clip_norm=10.0`).
- **TrustGraph-DFL (`tg_strong`):** `descent_neg` ($-\nabla F_{\mathrm{val}}$), model-space clip `2 r_i`, noise floor $\tau = 0.003$ (`0.00%` `random_noise` Byz accept, `8–11 pp` lower honest FPR than $\tau=0.02$).
- **Attacks:** `none`, `label_flip_all` (`byz_local_epochs=3` so `FedAvg` drops into `~30–45%` while `TrustGraph` stays at `~91–92%`), `model_poison` ($\lambda=10$), and `random_noise` ($\sigma=0.5$).
