# Phase 1 Final Report — 3-Seed TrustGraph-DFL Reproduction on CIFAR-10

**Date:** 2026-11-20  
**Target Paper:** Xu et al., *TrustGraph-DFL: Byzantine-Resilient Decentralized Federated Learning via Consistency-Weighted Neighborhood Aggregation* (CIBDA 2026, `TrustGraph-DFL.pdf`)  
**Scope:** `225` complete 200-round CIFAR-10 runs (`75` configurations $\times$ `3` paired seeds: `42, 43, 44`)  
**Related Reports:** [phase1_smoke.md](phase1_smoke.md) (local 5-round CPU diagnostics) · [phase1_step_a.md](phase1_step_a.md) (seed-42 single-seed report)

---

## Executive Summary

We reproduced the CIFAR-10 experimental suite of `TrustGraph-DFL.pdf` across three paired seeds (`42, 43, 44`) on `octave` (NVIDIA A100 40 GB), testing both the literal preprint validation gradient sign (`paper_pos`, $+\nabla F_i$) and the corrected descent direction (`descent_neg`, $-\nabla F_i$), along with `FedAvg`, `Trimmed Mean (30%)`, `Multi-Krum`, `BALANCE`, TrustGraph ablations, and targeted diagnostics (`val_split=iid`, `balance_gamma=0.15`, `label_flip_all`, and PeerSieve's receiver-norm clipping cap).

| Metric / Claim | `TrustGraph-DFL.pdf` Claim | Measured (3 Seeds, `mean ± std`) | Verdict |
|---|---|---|---|
| Clean ceiling (`random`, `none`, FedAvg) | `93.0%` (Fig. 3a) | `53.27 ± 0.86%` mean / `56.52 ± 0.41%` cons | **Does not reproduce** (2-conv CNN on 10k CIFAR-10) |
| `label_flip` (`0<->1`, 30% Byz, `random`) | `91.2%` vs. `87.5%` BALANCE / `45.1%` FedAvg | TG `42.07 ± 0.23%` (`52.83 ± 0.16%` cons) vs. BAL `45.54 ± 0.44%` (`52.21%`) / FedAvg `48.59 ± 1.18%` (`51.97%`) | **Numbers do not reproduce**; TG leads slightly on consensus accuracy |
| Honest FPR at 50% Byz (`label_flip`) | TG `~9%` vs. Krum `~35%` (Fig. 3b) | TG `51.23 ± 3.51%` vs. Krum `52.36 ± 1.50%` | **Invariance holds**, magnitude ~51% (not ~9%) due to local Dir(0.5) val skew |
| Gradient sign (`descent_neg` vs. `paper_pos`) | Uses $+\nabla F_i$ (Eq. 3 / Alg. 1) | `descent_neg` beats `paper_pos` across all 10–50% Byz (`+1.0` to `+2.5 pp` cons; `10–24 pp` less Byz accept) | **Sign bug confirmed** (`paper_pos` hits `NaN` at round 8 under `model_poison`) |
| High-norm attacks (`model_poison` $\lambda=10$, `random_noise` $\sigma=0.5$) | TG `89.5%` / `93.0%` (Fig. 3a) | Unclipped TG: `3/3` `NaN` on both; PeerSieve `clip`: **`0/3` `NaN`** (`26.39 ± 0.33%` / `28.76 ± 0.66%`) | **Unclipped cosine fails**; PeerSieve norm cap prevents divergence |

---

## 1. Experimental Setup & Cluster Run Records

- **Dataset & Partitioning:** CIFAR-10 (`10,000` stratified training images across $n=20$ peers, exact `500` samples/peer via balanced Dirichlet $\alpha_{\mathrm{dir}}=0.5$, `10%` (`50` samples/peer) reserved as private local validation set $\mathcal{D}_i^{\mathrm{val}}$; `10,000` clean test images).
- **Model & Training:** 2-conv, 2-FC CNN (`SimpleCIFARCNN`, `1,070,794` parameters), `200` communication rounds, `1` local epoch/round, SGD ($\eta=0.01$, momentum $0.9$, weight decay $10^{-4}$, batch size $64$), self-weight $\alpha=0.5$, evaluated every `10` rounds (`eval_every=10`) on all benign peers (`Final Acc`) and their parameter average (`Cons Acc`).
- **Topologies ($n=20$):** `ring` (cycle, degree `2`), `random` (Erdős–Rényi $p=0.3$), `small_world` (Watts–Strogatz $k=4, p=0.3$).
- **Software:** Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`, NVIDIA driver `610.57.04`, `torchvision 0.29.2`, `networkx 3.7`.
- **Hardware (`octave`):** 8× NVIDIA A100-PCIE-40GB (`PCIe Gen4 x16`, `SYS` topology, no NVLink). Observed Slurm-index to PCI-bus mapping inside job `327264`:
  `0: 01:00.0`, `1: 25:00.0`, `2: 41:00.0`, `3: 61:00.0`, `4: 81:00.0`, `5: A1:00.0`, `6: C1:00.0`, `7: E1:00.0`.

| Job ID | Commit | Seeds | GPUs / CPUs / RAM | Outcome & Wall | Runs | Per-Run Wall Median (Range, $n$) | Failures |
|---|---|---|---|---|---:|---|---|
| `327259` | `bcc0cd7` | `42` | 4× A100 / 2 CPUs / 2G (default) | `TIMEOUT` `00:35:18` | `49` | `162.4s` (`105.7s – 200.8s`, $n=49$) | 1 timeout (CPU-throttled); 0 OOM |
| `327261` | `9c05b8b` | `42` | 4× A100 / 16 CPUs / 64G | `COMPLETED` `00:10:33` | `26` | `77.1s` (`58.1s – 104.4s`, $n=26$) | 0 timeouts; 0 OOM |
| `327264` | `9c05b8b` | `43, 44` | 8× A100 / 32 CPUs / 128G | `COMPLETED` `00:29:50` | `150` | `92.3s` (`61.7s – 112.3s`, $n=150$) | 0 timeouts; 0 OOM |

- **Raw cluster artifacts:** `yes:~/PeerSieve/outputs/phase1_step_a/step_a/` and Slurm logs `yes:~/PeerSieve/logs/peersieve_p1_step{a-327259,a2-327261,b-327264}.out`.

---

## 2. Main Attacks (Random ER $p=0.3$, 30% Byzantine, 3 Seeds; cf. Paper Fig. 2 & Fig. 3a)

All values are **mean ± std** across seeds `42, 43, 44`. `NaNs` reports how many of the 3 seeds diverged to non-finite weights/scores.

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) | NaNs |
|---|---|---:|---:|---:|---:|---:|---:|
| **none** | krum (multi) | 53.33 ± 0.80 | 56.61 ± 0.27 | 0.00 ± 0.00 | 0.00 ± 0.00 | n/a | 0/3 |
| none | fedavg | 53.27 ± 0.86 | 56.52 ± 0.41 | 0.00 ± 0.00 | 0.00 ± 0.00 | n/a | 0/3 |
| none | trimmed_mean (30%) | 52.22 ± 1.12 | 56.12 ± 0.72 | 50.37 ± 1.64 | 0.00 ± 0.00 | n/a | 0/3 |
| none | balance ($\gamma=0.3$) | 48.43 ± 1.22 | 54.95 ± 0.63 | 0.15 ± 0.22 | 0.00 ± 0.00 | n/a | 0/3 |
| none | trustgraph(`descent_neg`) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0.00 ± 0.00 | -0.000 / n/a | 0/3 |
| none | trustgraph(`paper_pos`) | 46.51 ± 1.42 | 54.70 ± 0.62 | 45.18 ± 2.80 | 0.00 ± 0.00 | +0.021 / n/a | 0/3 |
| **label_flip** (`0<->1`) | trimmed_mean (30%) | 48.97 ± 0.50 | **52.84 ± 0.46** | 49.28 ± 3.78 | 49.32 ± 1.30 | n/a | 0/3 |
| label_flip | fedavg | 48.59 ± 1.18 | 51.97 ± 2.02 | 0.00 ± 0.00 | 100.00 ± 0.00 | n/a | 0/3 |
| label_flip | krum (multi) | 46.69 ± 1.67 | 50.95 ± 1.82 | 27.22 ± 4.21 | 78.23 ± 4.79 | n/a | 0/3 |
| label_flip | balance ($\gamma=0.3$) | 45.54 ± 0.44 | 52.21 ± 1.11 | 0.16 ± 0.23 | 99.80 ± 0.29 | n/a | 0/3 |
| label_flip | trustgraph(`descent_neg`) | 42.07 ± 0.23 | **52.83 ± 0.16** | 51.38 ± 2.50 | 43.45 ± 1.64 | -0.003 / -0.029 | 0/3 |
| label_flip | trustgraph(`paper_pos`) | 43.41 ± 0.95 | 51.27 ± 1.61 | 46.68 ± 5.55 | 59.16 ± 5.73 | +0.012 / +0.055 | 0/3 |
| label_flip | *diag: trustgraph(`descent_neg`, `val=iid`)* | *45.67 ± 0.62* | *52.76 ± 0.48* | *42.28 ± 0.86* | *53.93 ± 4.55* | *+0.043 / +0.013* | *0/3* |
| label_flip | *diag: balance ($\gamma=0.15$)* | *39.09 ± 0.89* | *51.73 ± 1.28* | *24.18 ± 1.15* | *72.46 ± 2.33* | *n/a* | *0/3* |
| **model_poison** ($\lambda=10$) | *diag: balance ($\gamma=0.15$)* | ***35.87 ± 1.20*** | ***49.24 ± 1.04*** | *18.24 ± 5.52* | *20.93 ± 1.37* | *n/a* | ***0/3*** |
| model_poison | *diag: trustgraph(`descent_neg`, `clip`)* | ***26.39 ± 0.33*** | *13.97 ± 5.64* | *77.00 ± 0.78* | *79.87 ± 1.12* | *-0.050 / +0.058* | ***0/3*** |
| model_poison | balance ($\gamma=0.3$) | 20.30 ± 1.42 | 27.86 ± 2.65 | 0.01 ± 0.02 | 64.02 ± 7.26 | n/a | 0/3 |
| model_poison | trustgraph(`descent_neg`) | 12.01 ± 2.85 | 10.00 ± 0.00 | 88.18 ± 2.25 | 2.83 ± 0.87 | `NaN` | **3/3** |
| model_poison | trustgraph(`paper_pos`) | 10.52 ± 0.74 | 10.00 ± 0.00 | 96.83 ± 1.51 | 4.62 ± 0.79 | `NaN` | **3/3** |
| model_poison | fedavg | 10.00 ± 0.00 | 10.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | n/a | 0/3 |
| model_poison | krum (multi) | 10.00 ± 0.00 | 10.00 ± 0.00 | 31.08 ± 10.68 | 71.19 ± 18.88 | n/a | 0/3 |
| model_poison | trimmed_mean (30%) | 10.00 ± 0.00 | 10.00 ± 0.00 | 53.04 ± 2.40 | 46.11 ± 9.09 | n/a | 0/3 |
| **random_noise** ($\sigma=0.5$) | balance ($\gamma=0.3$) | **46.05 ± 0.78** | **51.55 ± 1.17** | 0.30 ± 0.37 | **0.00 ± 0.00** | n/a | **0/3** |
| random_noise | *diag: balance ($\gamma=0.15$)* | *37.64 ± 3.06* | *49.04 ± 2.00* | *23.71 ± 3.19* | *0.00 ± 0.00* | *n/a* | *0/3* |
| random_noise | *diag: trustgraph(`descent_neg`, `clip`)* | ***28.76 ± 0.66*** | ***40.56 ± 0.44*** | *62.22 ± 2.57* | *49.69 ± 0.18* | *-0.024 / -0.000* | ***0/3*** |
| random_noise | trimmed_mean (30%) | 21.09 ± 15.63 | 23.32 ± 18.85 | 39.32 ± 11.69 | 21.21 ± 9.42 | n/a | 0/3 |
| random_noise | krum (multi) | 20.12 ± 14.32 | 22.37 ± 17.50 | 8.88 ± 5.59 | 32.03 ± 13.94 | n/a | 0/3 |
| random_noise | trustgraph(`paper_pos`) | 10.95 ± 1.34 | 10.00 ± 0.00 | 96.47 ± 2.62 | 2.89 ± 2.10 | `NaN` | **3/3** |
| random_noise | trustgraph(`descent_neg`) | 10.00 ± 0.00 | 10.00 ± 0.00 | 99.40 ± 0.14 | 0.79 ± 0.10 | `NaN` | **3/3** |
| random_noise | fedavg | 10.00 ± 0.00 | 10.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | n/a | 0/3 |

---

## 3. Topologies Under Label-Flipping (30% Byzantine, 3 Seeds; cf. Paper Table 1)

Entries show `Final Acc ± std` (`Cons Acc ± std`). Note that on the degree-2 `ring`, $m = |\mathcal{N}_i|+1 = 3$ gives $\lfloor 0.3 \times 3 \rfloor = 0$, so 30% Trimmed Mean reduces to FedAvg (`FPR = 0.00%`).

| Method | Ring | Random | Small-World | Paper Table 1 (Ring / Rand / SW) |
|---|---:|---:|---:|---:|
| fedavg | 42.44 ± 0.99 (50.97 ± 0.72) | 48.59 ± 1.18 (51.97 ± 2.02) | 46.94 ± 0.71 (52.36 ± 1.31) | 42.3 / 45.1 / 43.8 |
| krum (multi) | 34.17 ± 0.71 (45.02 ± 0.12) | 46.69 ± 1.67 (50.95 ± 1.82) | 46.84 ± 1.12 (52.05 ± 1.28) | 78.5 / 82.1 / 80.3 |
| trimmed_mean (30%) | 42.37 ± 0.97 (51.25 ± 0.54) | 48.97 ± 0.50 (52.84 ± 0.46) | 47.27 ± 0.75 (52.62 ± 0.42) | 74.2 / 78.4 / 76.1 |
| balance ($\gamma=0.3$) | 41.59 ± 0.98 (50.69 ± 0.91) | 45.54 ± 0.44 (52.21 ± 1.11) | 45.14 ± 0.46 (52.15 ± 0.61) | 85.3 / 87.5 / 86.2 |
| trustgraph(`descent_neg`) | 38.83 ± 1.20 (50.56 ± 1.46) | 42.07 ± 0.23 (**52.83 ± 0.16**) | 43.22 ± 0.69 (**52.73 ± 0.44**) | 89.1 / 91.2 / 90.4 |
| trustgraph(`paper_pos`) | 38.99 ± 2.77 (51.05 ± 0.30) | 43.41 ± 0.95 (51.27 ± 1.61) | 43.14 ± 0.79 (51.27 ± 0.82) | — |

---

## 4. Honest FPR & Accuracy Sweep (Random ER $p=0.3$, `label_flip`, 3 Seeds; cf. Paper Fig. 3b)

### 4.1 Honest False Positive Rate (%) vs. Byzantine Fraction

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz | Paper Fig. 3b (10% $\to$ 50%) |
|---|---:|---:|---:|---:|---:|---:|
| balance ($\gamma=0.3$) | 0.12 ± 0.12 | 0.14 ± 0.10 | 0.16 ± 0.23 | 0.18 ± 0.25 | 0.02 ± 0.03 | ~4 $\to$ ~16 |
| krum (multi) | 8.28 ± 1.10 | 13.46 ± 2.52 | 27.22 ± 4.21 | 32.53 ± 7.39 | **52.36 ± 1.50** | ~8 $\to$ ~35 |
| trimmed_mean (30%) | 50.29 ± 1.66 | 50.22 ± 1.54 | 49.28 ± 3.78 | 50.70 ± 1.69 | 50.02 ± 0.83 | ~10 $\to$ ~39 |
| trustgraph(`descent_neg`) | 50.53 ± 2.52 | 50.31 ± 1.45 | 51.38 ± 2.50 | 53.46 ± 2.32 | **51.23 ± 3.51** | ~3 $\to$ ~9 |
| trustgraph(`paper_pos`) | 45.72 ± 3.50 | 47.97 ± 3.16 | 46.68 ± 5.55 | 43.92 ± 3.03 | 46.68 ± 4.55 | — |

### 4.2 Consensus Accuracy (%) / Byzantine Acceptance (%) Across Byzantine Fraction

| Method | 10% Byz | 20% Byz | 30% Byz | 40% Byz | 50% Byz |
|---|---:|---:|---:|---:|---:|
| trustgraph(`descent_neg`) | **54.75 ± 0.27** / 43.3% | **54.43 ± 0.31** / 45.4% | **52.83 ± 0.16** / 43.5% | **53.33 ± 0.74** / 45.9% | **52.06 ± 0.04** / 46.6% |
| trustgraph(`paper_pos`) | 53.72 ± 0.80 / 67.1% | 51.92 ± 0.55 / 64.4% | 51.27 ± 1.61 / 59.2% | 52.38 ± 0.84 / 55.5% | 51.01 ± 1.47 / 57.0% |
| balance ($\gamma=0.3$) | 53.96 ± 0.16 / 99.8% | 52.73 ± 0.97 / 99.9% | 52.21 ± 1.11 / 99.8% | 53.04 ± 0.12 / 99.9% | 51.75 ± 1.32 / 100.0% |
| krum (multi) | 55.98 ± 0.60 / 78.9% | 54.46 ± 0.88 / 67.4% | 50.95 ± 1.82 / 78.2% | 51.35 ± 2.30 / 63.3% | 49.49 ± 1.63 / 67.9% |
| trimmed_mean (30%) | 55.70 ± 1.19 / 46.9% | 55.28 ± 0.82 / 46.3% | 52.84 ± 0.46 / 49.3% | 53.41 ± 0.91 / 47.5% | 51.73 ± 0.44 / 49.0% |

---

## 5. Ablations & Additional Diagnostics (Random ER $p=0.3$, 30% Byzantine, 3 Seeds)

| Group | Variant | Attack | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | NaNs |
|---|---|---|---:|---:|---:|---:|---:|
| Ablation | baseline (`descent_neg`, $\alpha=0.5$) | model_poison | 12.01 ± 2.85 | 10.00 ± 0.00 | 88.18 ± 2.25 | 2.83 ± 0.87 | 3/3 |
| Ablation | `no_relu` | model_poison | 10.00 ± 0.00 | 10.00 ± 0.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | 3/3 |
| Ablation | `euclidean` | model_poison | 14.97 ± 7.03 | 18.49 ± 12.00 | 0.00 ± 0.00 | 100.00 ± 0.00 | 1/3 |
| Ablation | $\alpha=0.3$ | model_poison | 14.72 ± 5.91 | 10.00 ± 0.00 | 89.13 ± 7.89 | 3.20 ± 2.59 | 3/3 |
| Ablation | $\alpha=0.7$ | model_poison | 18.45 ± 11.34 | 22.88 ± 18.21 | 58.48 ± 10.09 | 28.78 ± 9.13 | 2/3 |
| Diagnostic | `val_split=iid` | label_flip | 45.67 ± 0.62 | 52.76 ± 0.48 | 42.28 ± 0.86 | 53.93 ± 4.55 | 0/3 |
| Diagnostic | `val_split=iid` | model_poison | 14.87 ± 5.02 | 10.00 ± 0.00 | 50.22 ± 9.78 | 26.20 ± 10.86 | 1/3 |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | fedavg | 36.14 ± 2.01 | 40.40 ± 2.97 | 0.00 ± 0.00 | 100.00 ± 0.00 | 0/3 |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | trustgraph(`descent_neg`) | 36.08 ± 1.13 | 46.63 ± 1.42 | 50.45 ± 3.79 | 43.73 ± 1.54 | 0/3 |
| Diagnostic | `label_flip_all` ($y \mapsto 9-y$) | balance ($\gamma=0.15$) | 37.19 ± 1.06 | **49.30 ± 1.56** | 27.24 ± 1.74 | 60.46 ± 0.75 | 0/3 |
| PeerSieve Preview | `trustgraph(descent_neg, clip)` | model_poison | **26.39 ± 0.33** | 13.97 ± 5.64 | 77.00 ± 0.78 | 79.87 ± 1.12 | **0/3** |
| PeerSieve Preview | `trustgraph(descent_neg, clip)` | random_noise | **28.76 ± 0.66** | **40.56 ± 0.44** | 62.22 ± 2.57 | 49.69 ± 0.18 | **0/3** |

---

## 6. Key Findings & Lessons for Phase 2 (PeerSieve)

1. **Why the preprint's `89–95%` accuracy and `<10%` FPR do not reproduce:**
   - **Data & model ceiling:** Subsampling `10,000` CIFAR-10 images (`500`/peer, `450` train) with $\alpha_{\mathrm{dir}}=0.5$ and a 2-conv 2-FC CNN yields an attack-free FedAvg ceiling of `53.27 ± 0.86%` mean (`56.52 ± 0.41%` consensus), roughly `40 pp` below Fig. 3(a).
   - **Non-IID validation skew:** Holding out `10%` of each peer's own $\text{Dir}(0.5)$ shard aligns strongly with that peer's own update (`+0.82` at round 0) but opposes neighbors with different dominant classes (`53.3%` negative at round 0), locking honest FPR near `~50%`. Even switching to a balanced 50-sample IID validation set (`val_split=iid`) only reduces FPR from `51.38%` to `42.28%`.
   - **Mild attack definition (`0<->1`):** Flipping only two classes on 30% of nodes leaves 80% of classes untouched, so FedAvg drops by only `~4.7 pp` (`53.27% -> 48.59%`). Under full-class flipping (`label_flip_all`), FedAvg drops sharply to `40.40%` consensus, where `trustgraph(descent_neg)` gains `+6.23 pp` (`46.63%`) and calibrated `balance(g=0.15)` gains `+8.90 pp` (`49.30%`).

2. **Sign bug confirmed systematically:**
   Using the corrected descent direction (`descent_neg`, $-\nabla F_i$) beats literal preprint `paper_pos` ($+\nabla F_i$) across every single Byzantine fraction (`10%–50%`) on `label_flip` (`+1.0` to `+2.5 pp` consensus accuracy; `10–24 pp` lower Byzantine acceptance). Under `model_poison`, `paper_pos` actively amplifies poisoned updates (`94.7%` acceptance, `cos_byz -> +0.906`) and explodes to `NaN` at round 8.

3. **Unclipped cosine defenses collapse on high-norm attacks — validating PeerSieve's norm cap:**
   Every unclipped `TrustGraph-DFL` configuration hit `NaN` on `1–3/3` seeds under `model_poison` ($\lambda=10$) and `3/3` seeds under `random_noise` ($\sigma=0.5$, exploding by round 3 when isotropic noise vectors of norm `~500` obtain small positive cosine). Adding PeerSieve's self-update norm cap $a_{ij}^{(t)} = \min(1, r_i^{(t)} / \|\Delta\theta_j^{(t)}\|_2)$ reduced `NaN` failures to **`0/3` on both attacks** with tight cross-seed stability (`± 0.33%` and `± 0.66%`).

4. **Design checklist carried into Phase 2 (TinyLlama-1.1B / Dolly-15k):**
   - Keep **receiver-norm clipping** active in PeerSieve Pass 2 and compare directly against unclipped batched `TrustGraph-DFL` under positive scale inflation and sign-flipping.
   - Calibrate `BALANCE`'s $\gamma$ explicitly on the LLM task (on CIFAR-10, $\gamma=0.15$ vs. `0.3` swung `model_poison` consensus from `27.86%` to `49.24%`).
   - Track **both per-peer and consensus metrics** alongside honest rejection and active degree under Dirichlet heterogeneity ($\alpha \in \{0.05, 0.1, 0.5\}$ vs. IID), because update-space aggregation without model-space consensus can let peers diverge into separate basins when starting from random weights (`model_poison` `clip`: `26.39%` per-peer vs. `13.97%` consensus).

---

## 7. Reproducibility Command

```bash
sbatch -p a100 -A a100 --nodes=1 --nodelist=octave --gres=gpu:a100:8 \
  --cpus-per-task=32 --mem=128G --time=00:45:00 \
  --job-name=peersieve_p1 --open-mode=append --output=logs/%x-%j.out \
  experiments/phase1/slurm_run.sh \
  --mode suite --suite full --skip-existing \
  --dataset cifar10 --rounds 200 --eval-every 10 --seeds 42 43 44 \
  --output-root outputs/phase1
```
