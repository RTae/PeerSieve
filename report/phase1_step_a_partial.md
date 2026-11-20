# Phase 1 Cluster Step A Report — Partial (49/73 Runs Completed)

**Date:** 2026-11-20  
**Slurm Job ID:** `327259` (`peersieve_p1_stepa`, status `TIMEOUT` at `00:35:18` due to default 2-CPU allocation throttling 4 GPU shards)  
**Git Commit:** `bcc0cd7c9a869236d8f484c9b0a5541f98f29d7a`  
**Hardware:** `octave`, 4× NVIDIA A100-PCIE-40GB (PCI bus IDs `01:00.0`, `25:00.0`, `41:00.0`, `61:00.0`, PCIe Gen4 x16, topology `SYS`), driver `610.57.04`  
**Software:** Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`  
**Timing (49 completed 200-round runs, 2 CPUs shared across 4 shards):** median `162.4s` (range `105.7s – 200.8s`)  
**Remote artifacts:** `yes:~/PeerSieve/outputs/phase1_step_a/step_a/` and `yes:~/PeerSieve/logs/peersieve_p1_stepa-327259.out`

## 1. Completed Results (Seed 42, 200 Rounds, CIFAR-10, 20 Peers, $\alpha_{\mathrm{dir}} = 0.5$)

All `main_attacks` (`24/24`), all `topologies` (`12/12`), and the first `13/25` of `fpr_sweep` completed before the `00:35:00` limit.

### 1.1 Main Attacks (Random ER $p=0.3$, 30% Byzantine)

| Attack | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) |
|---|---|---:|---:|---:|---:|---:|
| none | fedavg | 53.61 | 56.92 | 0.00 | 0.00 | n/a |
| none | krum | 53.65 | 56.88 | 0.00 | 0.00 | n/a |
| none | trimmed_mean | 52.07 | 56.42 | 49.64 | 0.00 | n/a |
| none | balance | 49.58 | 55.83 | 0.00 | 0.00 | n/a |
| none | trustgraph(descent_neg) | 47.62 | 55.48 | 48.66 | 0.00 | +0.005 / n/a |
| none | trustgraph(paper_pos) | 47.46 | 55.34 | 46.57 | 0.00 | +0.016 / n/a |
| label_flip | trimmed_mean | 48.25 | 52.59 | 46.13 | 51.06 | n/a |
| label_flip | fedavg | 47.56 | 50.61 | 0.00 | 100.00 | n/a |
| label_flip | krum | 44.83 | 49.66 | 22.64 | 73.25 | n/a |
| label_flip | balance | 44.64 | 49.64 | 1.17 | 99.29 | n/a |
| label_flip | trustgraph(descent_neg) | 43.12 | 52.86 | 49.61 | 44.47 | +0.004 / -0.027 |
| label_flip | trustgraph(paper_pos) | 43.22 | 51.07 | 53.02 | 60.66 | -0.013 / +0.058 |
| model_poison | balance | 19.34 | 25.33 | 0.00 | 57.20 | n/a |
| model_poison | fedavg | 10.00 | 10.00 | 0.00 | 100.00 | n/a |
| model_poison | krum | 10.00 | 10.00 | 38.57 | 83.64 | n/a |
| model_poison | trimmed_mean | 10.00 | 10.00 | 54.17 | 55.29 | n/a |
| model_poison | trustgraph(descent_neg) | 10.00 | 10.00 | 85.00 | 3.71 | +nan / +nan |
| model_poison | trustgraph(paper_pos) | 11.56 | 10.00 | 98.53 | 3.76 | +nan / +nan |
| random_noise | balance | 46.92 | 52.72 | 0.02 | 0.00 | n/a |
| random_noise | fedavg | 10.00 | 10.00 | 0.00 | 100.00 | n/a |
| random_noise | krum | 9.99 | 9.99 | 2.17 | 39.47 | n/a |
| random_noise | trimmed_mean | 10.07 | 9.98 | 27.06 | 20.42 | n/a |
| random_noise | trustgraph(descent_neg) | 10.00 | 10.00 | 99.22 | 0.92 | +nan / +nan |
| random_noise | trustgraph(paper_pos) | 10.00 | 10.00 | 99.15 | 0.80 | +nan / +nan |

### 1.2 Topologies Under Label-Flipping (30% Byzantine; cf. TrustGraph-DFL Table 1)

| Method | Ring Acc (Cons) | Random Acc (Cons) | Small-World Acc (Cons) | Paper Table 1 (Ring / Rand / SW) |
|---|---:|---:|---:|---:|
| fedavg | 42.40% (51.37%) | 47.56% (50.61%) | 45.94% (50.51%) | 42.3 / 45.1 / 43.8 |
| krum | 34.43% (44.86%) | 44.83% (49.66%) | 46.79% (53.01%) | 78.5 / 82.1 / 80.3 |
| trimmed_mean | 41.29% (50.73%) | 48.25% (52.59%) | 46.45% (52.52%) | 74.2 / 78.4 / 76.1 |
| balance | 42.90% (51.77%) | 44.64% (49.64%) | 44.97% (51.28%) | 85.3 / 87.5 / 86.2 |
| trustgraph(descent_neg) | 38.27% (49.17%) | 43.12% (52.86%) | 43.92% (52.28%) | 89.1 / 91.2 / 90.4 |
| trustgraph(paper_pos) | 40.61% (51.21%) | 43.22% (51.07%) | 43.11% (50.25%) | — |

### 1.3 Partial FPR Sweep (Random ER $p=0.3$, Label-Flipping)

| Byz% | Method | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) |
|---:|---|---:|---:|---:|---:|
| 10 | balance | 47.34 | 53.82 | 0.07 | 99.67 |
| 10 | krum | 53.33 | 56.54 | 9.83 | 87.04 |
| 10 | trimmed_mean | 51.60 | 55.97 | 50.46 | 50.31 |
| 10 | trustgraph(descent_neg) | 46.13 | 55.00 | 48.50 | 49.00 |
| 10 | trustgraph(paper_pos) | 46.55 | 54.85 | 46.21 | 59.25 |
| 20 | balance | 46.53 | 53.07 | 0.15 | 99.92 |
| 20 | krum | 48.97 | 54.33 | 14.46 | 67.88 |
| 20 | trimmed_mean | 51.13 | 55.60 | 48.88 | 47.99 |
| 20 | trustgraph(descent_neg) | 43.82 | 54.18 | 48.85 | 43.69 |
| 20 | trustgraph(paper_pos) | 44.62 | 51.71 | 52.09 | 70.54 |
| 30 | trustgraph(descent_neg) | 42.35 | 52.69 | 48.52 | 44.49 |
| 30 | trustgraph(paper_pos) | 42.08 | 50.38 | 52.50 | 60.75 |
| 40 | trustgraph(descent_neg) | 44.14 | 53.77 | 53.89 | 48.08 |

## 2. Key Scientific Findings

1. **Preprint accuracy claims (`89–95%`) do not reproduce on 10k CIFAR-10 with a 2-conv 2-FC CNN:**
   Even with **zero attackers**, decentralized FedAvg reaches `53.61%` mean benign accuracy (`56.92%` consensus) across 200 rounds on 10,000 subsampled CIFAR-10 images ($\alpha_{\mathrm{dir}} = 0.5$), versus `93.0–95.0%` reported in `TrustGraph-DFL.pdf` Fig. 3(a). Under `label_flip` (`0 <-> 1`, 30% Byzantine), `trustgraph(descent_neg)` achieves the highest *consensus* accuracy on `random` (`52.86%` vs `50.61%` FedAvg / `49.64%` BALANCE), but lags on mean per-node accuracy (`43.12%`) because ~50% honest rejection slows consensus across non-IID peers.

2. **Honest rejection (FPR) is ~48–54%, not `<10%`:**
   With each node's 10% validation set drawn from its own $\text{Dir}(0.5)$ shard (`val_split=local`), honest neighbors with different class mixes have cosine similarity distributed around `0.0`, producing `48.5% – 53.9%` FPR from 0% to 40% Byzantine peers. (Pending finish-up job includes `val_split=iid` to isolate validation-set skew.)

3. **Unclipped cosine scoring explodes to `NaN` under high-norm attacks (`random_noise` and `model_poison`):**
   - **`random_noise` ($\sigma=0.5$):** Isotropic Gaussian updates ($\|\Delta\theta\|_2 \approx 500$) have positive cosine with a peer's validation gradient ~50% of the time (`byz_acc = 47.4%` at rounds 0–1). Because TrustGraph-DFL has **no norm clipping**, blending even one unclipped $\sigma=0.5$ vector blows weights up to **`NaN` by round 3**. By contrast, `BALANCE` filters by Euclidean distance and rejects `100.0%` of noise payloads (`46.92%` mean / `52.72%` consensus, `0.02%` FPR).
   - **`model_poison` ($\lambda=10$):** Literal preprint sign `trustgraph(paper_pos)` accepts `94.7%` of poisoned updates (`cos_byz` rises to `+0.906`) and hits **`NaN` at round 8**. Corrected `trustgraph(descent_neg)` defends through early rounds (reaching `20.59%` at round 19), before intermittent acceptance of unclipped $\lambda=10$ updates on skewed nodes (`byz_acc ~ 20–30%`) also triggers `NaN` later in the run.
   - **`BALANCE` ($\gamma=0.3$) under `model_poison`:** Accepts `100%` of poisoned updates during rounds 0–20 (`0.256 <= 0.30 * exp(-t/200)`), then abruptly drops `byz_acc` to `0.0%` at round 21 as the threshold decays, partially recovering to `19.34%` (`25.33%` consensus). (Pending finish-up job includes `balance_gamma=0.15`.)
   - **Direct takeaway for PeerSieve:** These failures directly motivate PeerSieve's receiver-norm clipping cap $a_{ij}^{(t)} = \min(1, r_i^{(t)} / \|y_j^{(t)}\|_2)$, which prevents any single positive-cosine high-norm payload from destabilizing weights.

4. **Note on Ring Trimmed Mean:**
   On the degree-2 Ring ($m = |\mathcal{N}_i| + 1 = 3$), $\lfloor 0.3 \times 3 \rfloor = 0$, so 30% Trimmed Mean trims zero entries and reduces to FedAvg (`FPR = 0.00%`).

## 3. Remaining Runs (24/73)

Still pending from Step A (will run cleanly via `--skip-existing` with `--cpus-per-task=16 --mem=64G`):
- `fpr_sweep` (12 runs): `byz30` (`balance`, `krum`, `trimmed_mean`), `byz40` (`tg_paper_pos`, `balance`, `krum`, `trimmed_mean`), `byz50` (all 5 methods).
- `ablations` (4 runs): `no_relu`, `euclidean`, `alpha=0.3`, `alpha=0.7` under `model_poison`.
- `diagnostics` (8 runs): `val_split=iid` (`label_flip`, `model_poison`), `balance_gamma=0.15` (`label_flip`, `model_poison`, `random_noise`), and `label_flip_all` (`fedavg`, `balance_g015`, `tg_descent_neg`).
