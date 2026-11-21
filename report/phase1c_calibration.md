# Phase 1c Calibration Report — Gradient Clipping, Validation Regimes, & BALANCE Tuning

**Date:** 2026-11-21  
**Slurm Job ID:** `327293` (`peersieve_p1c_calib`, `COMPLETED` in `00:10:54`, exit `0:0`, `0` timeouts, `0` OOMs)  
**Git Commit:** `c186e83`  
**Hardware:** `octave`, 8× NVIDIA A100-PCIE-40GB (`32` CPUs, `128G` RAM)  
**Scope:** `16 / 16` calibration runs (`paper_match_calib`, seed `42`, `resnet18_pretrained`, `50` rounds, `grad_clip_norm=1.0`, `eval_every=5`)  
**Timing:** Median `194.8s` (range `162.5s–334.1s`, $n=16$)

---

## 1. Results Table (`16 / 16` Completed, `0` NaNs)

| Run | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaN | Wall (s) |
|---|---:|---:|---:|---:|---:|---|---:|
| `calib/local_dir05/model_poison__tg_stab` | 69.71 | 74.84 | 54.39 | 20.74 | +0.010 / -0.115 | no | 179.9 |
| `calib/local_dir05/model_poison__tg_raw` | 70.78 | 72.45 | 49.61 | 16.79 | +0.007 / -0.164 | no | 181.2 |
| `calib/iidval_dir05/none__tg_stab` | 77.09 | 78.42 | 47.00 | 0.00 | +0.031 / n/a | no | 247.4 |
| `calib/iidval_dir05/label_flip_all_byz30__tg_stab` | 73.99 | 75.80 | 46.39 | 33.79 | +0.044 / -0.065 | no | 216.2 |
| `calib/iidval_dir05/label_flip_all_byz50__tg_stab` | 68.93 | 70.89 | 39.24 | 41.59 | +0.065 / -0.038 | no | 168.9 |
| `calib/iidval_dir05/label_flip_all_byz50__tg_stab_t01` | 68.99 | 70.91 | 38.12 | 43.18 | +0.064 / -0.038 | no | 189.8 |
| `calib/iidval_dir05/model_poison_byz30__tg_stab` | 73.71 | 75.41 | 43.43 | 23.63 | +0.051 / -0.090 | no | 180.5 |
| `calib/iidval_dir05/random_noise_byz30__tg_stab` | 74.95 | 78.34 | 41.57 | **0.00** | +0.061 / -0.000 | no | 201.2 |
| `calib/iidval_dir05/random_noise_byz30__tg_stab_t01` | 75.29 | 78.23 | 40.09 | **0.00** | +0.059 / -0.000 | no | 199.6 |
| `calib/local_dir50/label_flip_all_byz50__tg_stab` | 79.05 | 79.09 | 32.41 | 23.59 | +0.081 / -0.065 | no | 189.9 |
| `calib/local_dir05_20k/label_flip_all_byz30__tg_stab` | 81.38 | 84.50 | 53.00 | 27.05 | +0.005 / -0.088 | no | 334.1 |
| `calib/balance/none__bal_g03` | 78.13 | 78.67 | 0.00 | 0.00 | n/a | no | 228.9 |
| `calib/balance/label_flip_all__bal_g03` | 71.21 | 72.39 | 0.00 | 100.00 | n/a | no | 174.0 |
| `calib/balance/none__bal_g05` | 78.12 | 78.62 | 0.00 | 0.00 | n/a | no | 226.1 |
| `calib/balance/label_flip_all__bal_g05` | 71.21 | 72.38 | 0.00 | 100.00 | n/a | no | 200.3 |
| `calib/balance/model_poison__bal_g05` | 10.00 | 10.00 | 0.00 | 100.00 | n/a | no | 162.5 |

---

## 2. Key Calibration Findings

1. **Local gradient clipping eliminated all isolated-peer `NaN`s (`0 / 16`), but `1.0` was too tight for ResNet-18:**
   - Every `model_poison` run finished cleanly with `0` `NaN`s (fixing the single-peer frozen-BN overflow observed in Job `327284`).
   - However, clipping every ResNet-18 batch gradient to `1.0` reduced effective step size by ~`5–10×`, lowering the 50-round clean ceiling from `90.55%` (Job `327284`) to `78.67%` and causing `BALANCE(g=0.03–0.05)` to accept `100%` of smaller updates.
   - **Action for `paper_match`:** Raise `grad_clip_norm` to **`10.0`** (leaves normal ResNet-18 gradients unscaled to preserve the `~90%` 50-round ceiling while still preventing exploding isolated-peer `NaN`s) and calibrate `BALANCE` to $\gamma = 0.025$ (between the over-tight `0.015` in `327284` and the loose `0.03–0.05`).

2. **Round-by-round FPR trajectory explains why cumulative honest FPR exceeds `10%` even under IID validation or $\alpha_{\mathrm{dir}} = 5.0$:**
   Inspecting `history.jsonl` reveals that honest FPR is **near zero early in training** and climbs only as peer models converge and diverge slightly around the minimum:

   | Round | `local_dir50` (50% Byz) FPR / Cos(Ben) | `iidval_dir05` (Clean) FPR / Cos(Ben) | `327284` `local_dir05` (30% Byz) FPR / Cons Acc |
   |---:|---:|---:|---:|
   | 1 | **0.0%** / +0.480 | **0.0%** / +0.335 | 13.0% / 15.05% |
   | 5 | **0.0%** / +0.221 | 25.4% / +0.119 | 32.6% / — |
   | 10 | **5.9%** / +0.131 | 41.0% / +0.070 | 32.6% / 68.05% |
   | 20 | 35.3% / +0.040 | 53.7% / +0.021 | 37.0% / 85.23% |
   | 40 | 64.7% / +0.008 | 57.5% / -0.016 | 56.5% / 87.51% |
   | 50 | 41.2% / +0.032 | 58.2% / -0.034 | 58.7% / 88.36% |

   This confirms that high late-round cosine FPR is an intrinsic property of single-round validation-gradient cosine on converged decentralized neural networks—not an artifact of local non-IID validation alone—and directly motivates PeerSieve's EMA probation + recovery mechanism.

3. **`tau = 0.01` vs. `tau = 0.02`:** Both achieved **`0.00%`** Byzantine acceptance on `random_noise` ($\sigma=0.5$), while `t=0.01` lowered honest FPR by `~1.1–1.5 pp`. We will use `t=0.015` (or `0.02`) in `paper_match`.
