# TrustGraph-DFL CIFAR-10 reproduction report

This report evaluates [TrustGraph-DFL.pdf](../TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) on CIFAR-10 across 20 decentralized peers. TrustGraph-DFL scores each neighbor update by taking the cosine similarity against the receiver's validation gradient, zeroing negative scores with ReLU, and averaging the accepted models. We test each accuracy, attack, topology, ablation, and false-positive claim from the paper, and finish with the exact adjustments required to reproduce them.

---

## Hardware and software setup

| Component | Configuration |
|---|---|
| Cluster node | `octave` |
| GPUs | 8x NVIDIA A100-PCIE-40GB (`01:00.0` to `E1:00.0`, PCIe Gen4 x16 `SYS`) |
| NVIDIA driver and CUDA | Driver `610.57.04`, CUDA `13.2` |
| Software stack | Python `3.12.13`, PyTorch `2.15.0+cu132`, `torchvision 0.29.2`, `networkx 3.7` |

---

## Metrics we aim to reproduce

- **Per-peer accuracy.** Average CIFAR-10 test accuracy across honest peers' local models at the end of training.
- **Consensus accuracy.** Test accuracy of the single parameter-averaged model across all honest peers.
- **Honest false positive rate (FPR).** Percentage of honest neighbor updates assigned zero trust weight and ignored. Lower is better.
- **Byzantine acceptance.** Percentage of malicious neighbor updates assigned positive trust weight. Lower is better.

| Metric | Paper target | Our result | Status |
|---|---|---|---|
| Clean accuracy (Fig. 2, Fig. 3a) | ~93% | 91.39% per-peer, 92.87% consensus | Reproduced on 50k ResNet-18 |
| Attack resilience at 30% Byzantine peers (Fig. 3a) | 89.5% to 93% | 91.53% to 92.38% consensus, 0/42 `NaN` runs | Reproduced with $-\nabla F_i$ and norm clipping |
| Topologies at 30% label flip (Table 1) | Ring 89.1%, Random 91.2%, Small-World 90.4% | 84.70% Ring, 89.28% Random, 89.48% Small-World per-peer (91.69% to 92.38% consensus) | Reproduced on Random and Small-World; Ring per-peer is 4.4 points lower |
| Resilience at 50% Byzantine peers (Fig. 3b) | High accuracy at 50% attackers | 89.53% per-peer, 92.03% consensus | Reproduced |
| Component ablations (Section 5.5) | ReLU, cosine, and $\alpha=0.5$ matter | Removing ReLU loses 35.9 points; Euclidean distance loses 9.0 points | Reproduced with 0 `NaN` runs |
| Honest FPR (Fig. 3b) | 3% to 9% | 36.2% to 43.9% (`sender`), 6.5% to 9.5% (`rxdelta` + `val=iid`), 21.0% to 22.2% (`sender` + EMA) | Requires scoring adjustment |

---

## 1. Clean accuracy

The paper reports around 93% CIFAR-10 accuracy without attackers. Running the literal 2-convolution CNN on 10,000 non-IID images reaches 46.46% per-peer and 54.84% consensus accuracy after 200 rounds. Moving to all 50,000 CIFAR-10 images with an ImageNet-pretrained ResNet-18 reaches 91.39% per-peer and 92.87% consensus accuracy in 25 rounds across 3 seeds (42, 43, and 44).

| Setup | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | `NaN` runs |
|---|---:|---:|---:|---:|
| Literal 10k 2-conv CNN (200 rounds, $-\nabla F_i$) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0/3 |
| 50k ResNet-18 (25 rounds, 3 seeds) | 91.39 ± 0.30 | 92.87 ± 0.22 | 41.18 ± 3.05 | 0/3 |

---

## 2. Attack resilience at 30% Byzantine peers

The paper reports 89.5% to 93% accuracy under 30% label-flipping, model-poisoning, and random-noise attacks on a Random Erdos-Renyi graph ($p = 0.3$). In the literal 10k unclipped setup, model poisoning ($\lambda = 10$) and random noise ($\sigma = 0.5$) overflow model weights and crash in 3/3 runs. With $-\nabla F_i$, gradient clipping, and update norm clipping on 50k ResNet-18, consensus accuracy stays between 91.53% and 92.38% across all three attacks with 0 `NaN` runs, and random noise has 0.00% Byzantine acceptance.

![Fig. 1 TrustGraph-DFL Attack Regimes (50k)](figures/fig1-attacks-50k.png)

| Attack on 50k Random ER ($p=0.3$) | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | Byzantine acceptance (%) | Cosine (benign / Byz) | `NaN` runs |
|---|---:|---:|---:|---:|---:|---:|
| Clean (0% Byzantine) | 91.39 ± 0.30 | 92.87 ± 0.22 | 41.18 ± 3.05 | 0.00 | +0.020 / None | 0/3 |
| All-class label flip (30%, 2 attacker epochs) | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 | 0/3 |
| Model poisoning (30%, $\lambda=10$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | +0.016 / -0.030 | 0/3 |
| Random noise (30%, $\sigma=0.5$) | 88.98 ± 1.31 | 92.33 ± 0.42 | 38.83 ± 3.33 | 0.00 | +0.023 / +0.000 | 0/3 |

---

## 3. Network topologies

Table 1 of the paper reports 89.1% on Ring, 91.2% on Random, and 90.4% on Small-World under 30% label flipping. On 50k ResNet-18 under 30% all-class label flipping, Random (89.28% per-peer, 92.38% consensus) and Small-World (89.48% per-peer, 92.36% consensus, lowest FPR at 27.84%) match the paper. Ring reaches 91.69% consensus accuracy but 84.70% per-peer accuracy because each peer connects to only two neighbors (on the 10k 2-conv CNN, per-peer accuracy was 38.8% Ring, 42.1% Random, and 43.2% Small-World).

![Fig. 2 TrustGraph-DFL Topologies and Byzantine Stress (50k)](figures/fig2-topologies-and-stress-50k.png)

| Topology (30% all-class label flip, 50k) | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | Byzantine acceptance (%) | Cosine (benign / Byz) | `NaN` runs |
|---|---:|---:|---:|---:|---:|---:|
| Ring (degree 2) | 84.70 ± 1.04 | 91.69 ± 0.07 | 48.03 ± 2.26 | 48.09 | +0.015 / -0.015 | 0/3 |
| Random (ER $p=0.3$) | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 | 0/3 |
| Small-World ($k=4, p=0.3$) | 89.48 ± 0.12 | 92.36 ± 0.27 | 27.84 ± 2.00 | 43.78 | +0.030 / -0.025 | 0/3 |

---

## 4. Scaling to 50% Byzantine peers

Figure 3b of the paper tests robustness as attacker share grows up to 50%. On 50k ResNet-18 (Figure 2b), raising all-class label-flipping attackers from 10% to 50% lowers consensus accuracy by 0.60 percentage points (92.63% to 92.03%) with zero `NaN` runs.

| Byzantine fraction (all-class label flip, Random) | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---:|---:|---:|---:|---:|---:|
| 10% | 90.93 ± 0.16 | 92.63 ± 0.22 | 43.06 ± 2.20 | 47.78 | 0/3 |
| 20% | 90.58 ± 0.43 | 92.75 ± 0.17 | 39.54 ± 3.90 | 41.68 | 0/3 |
| 30% | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | 0/3 |
| 40% | 89.89 ± 0.54 | 92.23 ± 0.03 | 38.68 ± 4.80 | 42.25 | 0/3 |
| 50% | 89.53 ± 0.38 | 92.03 ± 0.29 | 36.17 ± 3.70 | 44.30 | 0/3 |

---

## 5. Component ablations (Section 5.5)

Section 5.5 of the paper tests removing ReLU zeroing, replacing cosine similarity with Euclidean distance, and varying self-weight $\alpha$. Under 30% model poisoning ($\lambda = 10$) on 50k ResNet-18 (Figure 3a and 3b), dropping ReLU keeps negative-scoring attackers in the average and loses 35.9 percentage points (to 55.60% consensus). Replacing cosine similarity with Euclidean distance accepts 100% of attackers and loses 9.0 percentage points (to 82.57%). Raising self-weight to $\alpha = 0.7$ isolates peers and lowers per-peer accuracy to 82.49%, whereas $\alpha = 0.3$ reaches 91.67% consensus accuracy and cuts Byzantine acceptance from 33.07% to 21.92%.

![Fig. 3 TrustGraph-DFL Ablations and FPR Dynamics (50k)](figures/fig3-ablations-and-fpr-dynamics-50k.png)

| Ablation variant (30% model poison, $\lambda=10$) | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---|---:|---:|---:|---:|---:|
| Full TrustGraph-DFL ($\alpha = 0.5$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | 0/3 |
| Lower self-weight ($\alpha = 0.3$) | 89.18 ± 0.96 | 91.67 ± 0.30 | 45.69 ± 1.88 | 21.92 | 0/3 |
| Higher self-weight ($\alpha = 0.7$) | 82.49 ± 2.20 | 90.70 ± 0.44 | 46.92 ± 4.48 | 49.89 | 0/3 |
| Euclidean distance instead of cosine | 80.01 ± 5.58 | 82.57 ± 5.32 | 0.00 ± 0.00 | 100.00 | 0/3 |
| No ReLU zeroing (keep negative scores) | 66.55 ± 16.61 | 55.60 ± 33.40 | 0.00 ± 0.00 | 100.00 | 0/3 |

---

## 6. Honest false positive rate (FPR)

Figure 3b of the paper reports honest FPR around 3% to 9%. In default single-round `sender` mode ($\theta_j^{t+1/2} - \theta_j^t$), Figure 3c shows honest FPR starts near 0% in round 1 when peers share initial weights, then climbs to 36.2% to 43.9% as non-IID peers drift onto different weights and draw noisy local mini-batches. We tested four fixes across 40 seed-42 runs (0 `NaN` runs) to see what lowers honest FPR.

![Fig. 4 Four-Lever FPR Diagnostic (50k)](figures/fig4-fpr-levers-50k.png)

- **Lever 1 (receiver-anchored updates, `rxdelta`).** Scoring $\theta_j^{t+1/2} - \theta_i^t$ with balanced validation (`val=iid`) lowers honest FPR to 7.4% to 9.5% at 30% Byzantine peers (6.5% to 6.8% at 50%) and raises per-peer accuracy by 0.45 to 1.45 percentage points. It fails under `val=local` (36.9% to 39.2% FPR) and increases late-round attack acceptance (61.58% label flip, 41.26% model poison), which raising $\tau$ to 0.035 (`Lever 1b`) resolves while still beating `sender` on FPR (22.3% to 28.6%), attack rejection, and per-peer accuracy.
- **Lever 2 (fewer local SGD steps, ruled out).** Reducing local work to 9 steps per round (`batch_size=256`, 25 rounds) or 4 steps per round (`max_local_steps=4`, 100 rounds) leaves honest FPR at 40.6% to 45.0% and lowers clean consensus accuracy by about 2.5 percentage points.
- **Lever 3 (per-layer and subspace cosine, ruled out).** Although `backbone.fc` (0.05% of weights) holds 38.52% of squared gradient norm, `layer_mean`, `backbone_only`, and `head_only` all leave honest FPR at 38.1% to 43.0%. Using `head_only` lowers label-flip acceptance to 32.32%, while `layer_mean` leaks 0.53% of random-noise updates.
- **Lever 4 (multi-round score EMA).** Averaging `sender` scores across rounds ($R_{ij}^t = \beta R_{ij}^{t-1} + (1 - \beta) s_{ij}^t$) at $\beta = 0.85$ halves honest FPR to 21.0% to 22.2%, cuts model-poisoning acceptance from 33.07% to 14.32% (0.0% through round 16), keeps random-noise acceptance at 0.00%, and maintains 92.72% clean consensus accuracy.

| Method (`val=iid` unless noted) | Clean consensus (%) | Honest FPR range (%) | Model poison Byz accept (%) | Label flip Byz accept (%) | Random noise Byz accept (%) |
|---|---:|---:|---:|---:|---:|
| Baseline (`sender`, 3-seed) | 92.87 | 38.8 to 43.9 | 33.07 | 41.75 | 0.00 |
| Lever 1 (`rxdelta`, $\tau=0.003$) | 92.63 | 7.4 to 9.5 | 41.26 | 61.58 | 0.00 |
| Lever 1 (`rxdelta`, `val=local`) | 92.50 | 36.9 to 39.2 | Not tested | Not tested | Not tested |
| Lever 1b (`rxdelta`, $\tau=0.020$) | 92.71 | 13.7 to 16.7 | 34.63 | 50.84 | 0.00 |
| Lever 1b (`rxdelta`, $\tau=0.035$) | 92.45 | 22.3 to 28.6 | 31.68 | 39.16 | 0.00 |
| Lever 2 (`batch_size=256`, 25 rounds) | 90.08 | 41.0 to 42.0 | 29.05 | 40.95 | 0.00 |
| Lever 2 (`max_local_steps=4`, 100 rounds) | 90.38 | 40.6 to 45.0 | 31.39 | 34.08 | 0.00 |
| Lever 3 (`layer_mean`) | 92.58 | 38.1 to 41.4 | 30.11 | 40.95 | 0.53 |
| Lever 3 (`backbone_only`) | 92.52 | 41.5 to 43.0 | 32.53 | 42.00 | Not tested |
| Lever 3 (`head_only`) | Not tested | 40.3 | Not tested | 32.32 | Not tested |
| Lever 4 (EMA $\beta=0.70$) | 92.57 | 26.4 to 28.7 | 21.79 | 41.16 | 0.00 |
| Lever 4 (EMA $\beta=0.85$) | 92.72 | 21.0 to 22.2 | 14.32 | 44.53 | 0.00 |

---

## 7. What we adjusted to reach these results

Reproducing the paper's 91% to 93% accuracy and zero-`NaN` robustness required seven concrete differences from the literal paper description.

| Setting | Literal paper setup | Adjusted setup | Why the change was needed |
|---|---|---|---|
| Dataset | 10,000 CIFAR-10 images (500 per peer) | 50,000 CIFAR-10 images (2,500 per peer, crop+flip augmentation) | 10,000 images caps clean consensus accuracy near 55% |
| Model and schedule | 2-conv `SimpleCIFARCNN` (1.07M params), 200 rounds | ImageNet-pretrained `ResNet-18`, 25 rounds, cosine LR 0.005 | Needed to reach the 91% to 93% accuracy regime |
| Validation gradient sign | $+\nabla F_i$ (Algorithm 1 / Eq. 3) | $-\nabla F_i$ (descent direction) | $+\nabla F_i$ points opposite to SGD updates and diverges at round 8 under model poisoning |
| Magnitude bounds | Unclipped cosine similarity | Gradient clip 10.0, receiver update norm cap $2 r_i$, $\tau = 0.003$ | Unclipped cosine ignores attack scale and causes 3/3 `NaN` crashes on poisoning and noise |
| Validation split | `val=local` (10% skewed local shard) | `val=iid` (250 class-balanced images per peer) | Non-IID `val=local` keeps honest FPR at 37% to 53% in both 10k and 50k runs |
| Label-flipping attack | Two-class `0<->1` flip | All-class flip, 2 attacker epochs | `0<->1` flipping is mild; all-class flipping with 2 attacker epochs stresses the filter |
| Honest FPR rule | Single-round `sender` delta | `rxdelta` (7% to 9% FPR) or strict `sender` + EMA $\beta=0.85$ (21% FPR) | Non-IID single-round `sender` cosine keeps honest FPR near 40% |

Below is the literal 10k 2-conv CNN 3-seed audit (200 rounds, Random ER $p = 0.3$) that measured the capacity ceiling, the $+\nabla F_i$ sign bug, and the unclipped `NaN` crashes. Across 10% to 50% label flipping, $-\nabla F_i$ outperforms $+\nabla F_i$ by 1.0 to 2.5 percentage points in consensus accuracy (52.06% to 54.75%) with 10 to 24 percentage points lower Byzantine acceptance (43.3% to 46.6% versus 55.5% to 67.1%).

| Attack (30% Byzantine unless clean) | Validation variant | Per-peer accuracy (%) | Consensus accuracy (%) | Honest FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---|---|---:|---:|---:|---:|---:|
| Clean (`none`) | Corrected ($-\nabla F_i$) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0.00 | 0/3 |
| Clean (`none`) | Literal ($+\nabla F_i$) | 46.51 ± 1.42 | 54.70 ± 0.62 | 45.18 ± 2.80 | 0.00 | 0/3 |
| Label flip (`0<->1`) | Corrected ($-\nabla F_i$) | 42.07 ± 0.23 | 52.83 ± 0.16 | 51.38 ± 2.50 | 43.45 ± 1.64 | 0/3 |
| Label flip (`0<->1`) | Literal ($+\nabla F_i$) | 43.41 ± 0.95 | 51.27 ± 1.61 | 46.68 ± 5.55 | 59.16 ± 5.73 | 0/3 |
| Label flip (`0<->1`) | Corrected + balanced val (`val=iid`) | 45.67 ± 0.62 | 52.76 ± 0.48 | 42.28 ± 0.86 | 53.93 ± 4.55 | 0/3 |
| Model poison ($\lambda=10$) | Corrected, unclipped | 12.01 ± 2.85 | 10.00 ± 0.00 | 88.18 ± 2.25 | 2.83 ± 0.87 | 3/3 |
| Model poison ($\lambda=10$) | Corrected + norm clip | 26.39 ± 0.33 | 13.97 ± 5.64 | 77.00 ± 0.78 | 79.87 ± 1.12 | 0/3 |
| Random noise ($\sigma=0.5$) | Corrected, unclipped | 10.00 ± 0.00 | 10.00 ± 0.00 | 99.40 ± 0.14 | 0.79 ± 0.10 | 3/3 |
| Random noise ($\sigma=0.5$) | Corrected + norm clip | 28.76 ± 0.66 | 40.56 ± 0.44 | 62.22 ± 2.57 | 49.69 ± 0.18 | 0/3 |

---

## Run records and reproducibility

| Experiment batch | Scope and seeds | Slurm job IDs, wall times, and memory | Git commits | Runs / `NaN`s |
|---|---|---|---|---|
| Literal 10k 2-conv CNN audit | 200 rounds, seeds 42, 43, 44 | `327259`, `327261`, `327264` | `9c05b8b` | 3 seeds (`NaN`s on unclipped attacks only) |
| Locked 50k ResNet-18 evaluation (`paper_match_50k`) | 14 configurations, seeds 42, 43, 44 | `327299` (`00:42:46`, 20.7 GB), `327300` (`00:42:35`, 17.3 GB), `327301` (`00:41:26`, 17.3 GB) | `a0be515` | 42 runs, 0 `NaN`s |
| Four-lever FPR diagnostic (`fpr_lever1` through `fpr_lever4`) | 5 suites, seed 42 (timing median 338.6s, range 164.6s to 517.1s) | `327344` (`00:12:37`), `327496` (`00:08:14`), `327354` (`00:06:35`), `327486` (`00:13:47`), `327500` (`00:10:14`) | `3361b95`, `e26337e`, `cb28406`, `ce87e28` | 40 runs, 0 `NaN`s |

The committed datasets are [data/phase1_paper_match_50k_3seed.json](data/phase1_paper_match_50k_3seed.json) and [data/phase1_fpr_levers.json](data/phase1_fpr_levers.json), and the figure script is [../experiments/phase1/plot_figures.py](../experiments/phase1/plot_figures.py).

```bash
# Locked 14-config TrustGraph-DFL 50k suite (per seed)
uv run python -m experiments.phase1.run --mode suite --suite paper_match_50k \
  --dataset cifar10 --eval-every 5 --seeds 42 43 44 --output-root outputs/phase1_50k

# Any 8-run FPR diagnostic (fpr_lever1, fpr_lever1b, fpr_lever2, fpr_lever3, fpr_lever4)
uv run python -m experiments.phase1.run --mode suite --suite fpr_lever1 \
  --dataset cifar10 --eval-every 5 --seeds 42 --output-root outputs/phase1_fpr_lever1

# Summarize runs and regenerate all 4 TrustGraph-DFL figures
uv run python -m experiments.phase1.summarize outputs/phase1_50k --title "TrustGraph-DFL Summary"
uv run python -m experiments.phase1.plot_figures
```
