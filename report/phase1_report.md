# TrustGraph-DFL CIFAR-10 reproduction report

This report evaluates [TrustGraph-DFL.pdf](../TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) on CIFAR-10 across 20 decentralized peers. TrustGraph-DFL scores each neighbor update by taking the cosine similarity against the receiver's validation gradient, zeroing negative scores with ReLU, and averaging the accepted models. We test each accuracy, attack, topology, ablation, and false-positive claim from the paper, then list our adjustments and the remaining gaps.

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

- **Per-peer accuracy.** The average CIFAR-10 test accuracy across all honest peers' local models at the end of training.
- **Consensus accuracy.** The test accuracy of the single parameter-averaged model across all honest peers.
- **False positive rate (FPR).** The percentage of honest neighbor updates that a peer assigns zero trust weight and ignores. Lower is better.
- **Byzantine acceptance.** The percentage of malicious neighbor updates that a peer assigns positive trust weight. Lower is better.

| Metric | Paper target | Our result | Status |
|---|---|---|---|
| No-attack accuracy (Fig. 2, Fig. 3a) | About 93% | 91.39% per-peer, 92.87% consensus | Reproduced on 50k ResNet-18 |
| Attack resilience at 30% Byzantine peers (Fig. 3a) | 89.5% to 93% | 91.53% to 92.38% consensus, 0/42 `NaN` runs | Reproduced with $-\nabla F_i$ and norm clipping |
| Topologies at 30% label flip (Table 1) | Ring 89.1%, Random 91.2%, Small-World 90.4% | 84.70% Ring, 89.28% Random, 89.48% Small-World per-peer, 91.69% to 92.38% consensus | Reproduced on Random and Small-World; Ring per-peer is 4.4 points lower |
| Resilience at 50% Byzantine peers (Fig. 3b) | High accuracy at 50% attackers | 89.53% per-peer, 92.03% consensus | Reproduced |
| Component ablations (Section 5.5) | ReLU, cosine, and $\alpha=0.5$ matter | Removing ReLU loses 35.9 points; Euclidean distance loses 9.0 points | Reproduced with 0 `NaN` runs |
| False positive rate (Fig. 3b) | 3% to 9% | 36.2% to 43.9% (`sender`), 6.5% to 9.5% (`rxdelta` with `val=iid`) | Gap in default `sender` mode; reduced into 3% to 9% with `rxdelta` and balanced validation |

---

## 1. No-attack accuracy

The paper reports about 93% CIFAR-10 accuracy without attackers. Running the literal 2-convolution CNN on 10,000 non-IID images reaches 46.46% per-peer accuracy and 54.84% consensus accuracy after 200 rounds. Training an ImageNet-pretrained ResNet-18 on all 50,000 CIFAR-10 images reaches 91.39% per-peer accuracy and 92.87% consensus accuracy in 25 rounds across seeds 42, 43, and 44.

| Setup | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | `NaN` runs |
|---|---:|---:|---:|---:|
| Literal 10k 2-conv CNN (200 rounds, $-\nabla F_i$) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0/3 |
| 50k ResNet-18 (25 rounds, 3 seeds) | 91.39 ± 0.30 | 92.87 ± 0.22 | 41.18 ± 3.05 | 0/3 |

---

## 2. Attack resilience at 30% Byzantine peers

The paper reports 89.5% to 93% accuracy under 30% label-flipping, model-poisoning, and random-noise attacks on a Random Erdos-Renyi graph with $p = 0.3$. In the literal 10k unclipped setup, model poisoning ($\lambda = 10$) and random noise ($\sigma = 0.5$) overflow model weights and crash in 3/3 runs. With $-\nabla F_i$, gradient clipping, and update norm clipping on 50k ResNet-18, consensus accuracy stays between 91.53% and 92.38% across all three attacks with 0 `NaN` runs. Under random noise, peers accept 0.00% of Byzantine updates.

![Fig. 1 TrustGraph-DFL Attack Regimes (50k)](figures/fig1-attacks-50k.png)

| Attack on 50k Random ER ($p=0.3$) | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | Byzantine acceptance (%) | Cosine (benign / Byzantine) | `NaN` runs |
|---|---:|---:|---:|---:|---:|---:|
| No attack (0% Byzantine) | 91.39 ± 0.30 | 92.87 ± 0.22 | 41.18 ± 3.05 | 0.00 | +0.020 / None | 0/3 |
| All-class label flip (30%, 2 attacker epochs) | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 | 0/3 |
| Model poisoning (30%, $\lambda=10$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | +0.016 / -0.030 | 0/3 |
| Random noise (30%, $\sigma=0.5$) | 88.98 ± 1.31 | 92.33 ± 0.42 | 38.83 ± 3.33 | 0.00 | +0.023 / +0.000 | 0/3 |

---

## 3. Network topologies

Table 1 of the paper reports 89.1% on Ring, 91.2% on Random, and 90.4% on Small-World under 30% label flipping. On 50k ResNet-18 under 30% all-class label flipping, Random reaches 89.28% per-peer and 92.38% consensus accuracy, while Small-World reaches 89.48% per-peer accuracy, 92.36% consensus accuracy, and the lowest FPR at 27.84%. Ring reaches 91.69% consensus accuracy but 84.70% per-peer accuracy because each peer has only two neighbors. On the literal 10k 2-conv CNN, per-peer accuracy was 38.8% on Ring, 42.1% on Random, and 43.2% on Small-World.

![Fig. 2 TrustGraph-DFL Topologies and Byzantine Stress (50k)](figures/fig2-topologies-and-stress-50k.png)

| Topology (30% all-class label flip, 50k) | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | Byzantine acceptance (%) | Cosine (benign / Byzantine) | `NaN` runs |
|---|---:|---:|---:|---:|---:|---:|
| Ring (degree 2) | 84.70 ± 1.04 | 91.69 ± 0.07 | 48.03 ± 2.26 | 48.09 | +0.015 / -0.015 | 0/3 |
| Random (ER $p=0.3$) | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 | 0/3 |
| Small-World ($k=4, p=0.3$) | 89.48 ± 0.12 | 92.36 ± 0.27 | 27.84 ± 2.00 | 43.78 | +0.030 / -0.025 | 0/3 |

---

## 4. Scaling to 50% Byzantine peers

Figure 3b of the paper tests robustness as the attacker share grows up to 50%. In Figure 2b, raising all-class label-flipping attackers from 10% to 50% on 50k ResNet-18 lowers consensus accuracy by 0.60 percentage points, from 92.63% to 92.03%, with zero `NaN` runs.

| Byzantine fraction (all-class label flip, Random) | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---:|---:|---:|---:|---:|---:|
| 10% | 90.93 ± 0.16 | 92.63 ± 0.22 | 43.06 ± 2.20 | 47.78 | 0/3 |
| 20% | 90.58 ± 0.43 | 92.75 ± 0.17 | 39.54 ± 3.90 | 41.68 | 0/3 |
| 30% | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | 0/3 |
| 40% | 89.89 ± 0.54 | 92.23 ± 0.03 | 38.68 ± 4.80 | 42.25 | 0/3 |
| 50% | 89.53 ± 0.38 | 92.03 ± 0.29 | 36.17 ± 3.70 | 44.30 | 0/3 |

---

## 5. Component ablations (Section 5.5)

Section 5.5 of the paper tests removing ReLU zeroing, replacing cosine similarity with Euclidean distance, and varying self-weight $\alpha$. Under 30% model poisoning ($\lambda = 10$) on 50k ResNet-18 in Figure 3a, dropping ReLU keeps negative-scoring attackers in the average and drops consensus accuracy by 35.9 percentage points to 55.60%. Replacing cosine similarity with Euclidean distance accepts 100% of attackers and drops consensus accuracy by 9.0 percentage points to 82.57%. Raising self-weight to $\alpha = 0.7$ isolates peers and lowers per-peer accuracy to 82.49%. Lowering self-weight to $\alpha = 0.3$ reaches 91.67% consensus accuracy and cuts Byzantine acceptance from 33.07% to 21.92%.

![Fig. 3 TrustGraph-DFL Ablations and FPR Dynamics (50k)](figures/fig3-ablations-and-fpr-dynamics-50k.png)

| Ablation variant (30% model poison, $\lambda=10$) | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---|---:|---:|---:|---:|---:|
| Full TrustGraph-DFL ($\alpha = 0.5$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | 0/3 |
| Lower self-weight ($\alpha = 0.3$) | 89.18 ± 0.96 | 91.67 ± 0.30 | 45.69 ± 1.88 | 21.92 | 0/3 |
| Higher self-weight ($\alpha = 0.7$) | 82.49 ± 2.20 | 90.70 ± 0.44 | 46.92 ± 4.48 | 49.89 | 0/3 |
| Euclidean distance instead of cosine | 80.01 ± 5.58 | 82.57 ± 5.32 | 0.00 ± 0.00 | 100.00 | 0/3 |
| No ReLU zeroing (keep negative scores) | 66.55 ± 16.61 | 55.60 ± 33.40 | 0.00 ± 0.00 | 100.00 | 0/3 |

---

## 6. False positive rate (FPR)

Figure 3b of the paper reports FPR around 3% to 9%. Under the paper's default single-round `sender` scoring ($\theta_j^{t+1/2} - \theta_j^t$), a large gap appears. As Figure 3b shows, FPR starts at 0% in round 1 when all peers share initial weights, then rises to 36.2% to 43.9% as non-IID peers diverge to different weights and draw noisy local mini-batches.

Scoring updates relative to the receiver's weights (`rxdelta`, $\theta_j^{t+1/2} - \theta_i^t$) with balanced validation (`val=iid`) reduces FPR into the paper's range across 16 seed-42 runs with 0 `NaN` runs. At $\tau = 0.003$, `rxdelta` lowers FPR to 7.4% to 9.5% at 30% Byzantine peers and 6.5% to 6.8% at 50% Byzantine peers, while raising per-peer accuracy by 0.45 to 1.45 percentage points. Receiver anchoring requires balanced validation, because `val=local` leaves `rxdelta` FPR at 36.9% to 39.2%. At $\tau = 0.003$, receiver anchoring increases late-round Byzantine acceptance to 61.58% on label flipping and 41.26% on model poisoning. Raising the threshold to $\tau = 0.035$ beats the `sender` baseline on FPR (22.3% to 28.6%), attack rejection, and per-peer accuracy, while leaving an FPR gap above 3% to 9%.

![Fig. 4 Receiver-Anchored FPR on TrustGraph-DFL (50k)](figures/fig4-fpr-rxdelta-50k.png)

| Method (`val=iid` unless noted) | No-attack consensus (%) | FPR range (%) | Model poison Byzantine acceptance (%) | Label flip Byzantine acceptance (%) | Random noise Byzantine acceptance (%) |
|---|---:|---:|---:|---:|---:|
| Baseline (`sender`, 3-seed) | 92.87 | 38.8 to 43.9 | 33.07 | 41.75 | 0.00 |
| `rxdelta` ($\tau=0.003$) | 92.63 | 7.4 to 9.5 | 41.26 | 61.58 | 0.00 |
| `rxdelta` ($\tau=0.003$, `val=local`) | 92.50 | 36.9 to 39.2 | Not tested | Not tested | Not tested |
| `rxdelta` ($\tau=0.020$) | 92.71 | 13.7 to 16.7 | 34.63 | 50.84 | 0.00 |
| `rxdelta` ($\tau=0.035$) | 92.45 | 22.3 to 28.6 | 31.68 | 39.16 | 0.00 |

---

## 7. What we adjusted to reach these results

Reproducing the paper's 91% to 93% accuracy and zero-`NaN` robustness required seven changes from the literal paper setup.

| Setting | Literal paper setup | Adjusted setup | Why we changed it |
|---|---|---|---|
| Dataset | 10,000 CIFAR-10 images, 500 per peer | 50,000 CIFAR-10 images, 2,500 per peer with crop and flip augmentation | 10,000 images limits no-attack consensus accuracy to about 55% |
| Model and schedule | 2-conv `SimpleCIFARCNN` (1.07M parameters), 200 rounds | ImageNet-pretrained `ResNet-18`, 25 rounds, cosine learning rate 0.005 | Required to reach the 91% to 93% accuracy range |
| Validation gradient sign | $+\nabla F_i$ (Algorithm 1, Eq. 3) | $-\nabla F_i$ (descent direction) | $+\nabla F_i$ points opposite to SGD updates and diverges at round 8 under model poisoning |
| Magnitude bounds | Unclipped cosine similarity | Gradient clip 10.0, receiver update norm cap $2 r_i$, $\tau = 0.003$ | Unclipped cosine ignores attack scale and causes 3/3 `NaN` crashes on poisoning and noise |
| Validation split | `val=local`, 10% skewed local shard | `val=iid`, 250 class-balanced images per peer | Non-IID `val=local` keeps FPR at 37% to 53% in both 10k and 50k runs |
| Label-flipping attack | Two-class `0<->1` flip | All-class flip, 2 attacker epochs | `0<->1` flipping is mild; all-class flipping with 2 attacker epochs tests the filter under stronger attack |
| FPR scoring rule | Single-round `sender` delta | Receiver-anchored `rxdelta` ($\tau=0.003$ for 7% to 9% FPR, $\tau=0.035$ for lower attack acceptance) | Non-IID single-round `sender` cosine keeps FPR near 40% |

The table below lists the literal 10k 2-conv CNN results across 3 seeds and 200 rounds on a Random Erdos-Renyi graph with $p = 0.3$. Those runs measured the accuracy limit, the $+\nabla F_i$ sign bug, and the unclipped `NaN` crashes. Across 10% to 50% label flipping, $-\nabla F_i$ outperforms $+\nabla F_i$ by 1.0 to 2.5 percentage points in consensus accuracy, reaching 52.06% to 54.75%, with 10 to 24 percentage points lower Byzantine acceptance (43.3% to 46.6% versus 55.5% to 67.1%).

| Attack (30% Byzantine unless none) | Validation variant | Per-peer accuracy (%) | Consensus accuracy (%) | FPR (%) | Byzantine acceptance (%) | `NaN` runs |
|---|---|---:|---:|---:|---:|---:|
| No attack (`none`) | Corrected ($-\nabla F_i$) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0.00 | 0/3 |
| No attack (`none`) | Literal ($+\nabla F_i$) | 46.51 ± 1.42 | 54.70 ± 0.62 | 45.18 ± 2.80 | 0.00 | 0/3 |
| Label flip (`0<->1`) | Corrected ($-\nabla F_i$) | 42.07 ± 0.23 | 52.83 ± 0.16 | 51.38 ± 2.50 | 43.45 ± 1.64 | 0/3 |
| Label flip (`0<->1`) | Literal ($+\nabla F_i$) | 43.41 ± 0.95 | 51.27 ± 1.61 | 46.68 ± 5.55 | 59.16 ± 5.73 | 0/3 |
| Label flip (`0<->1`) | Corrected with balanced validation (`val=iid`) | 45.67 ± 0.62 | 52.76 ± 0.48 | 42.28 ± 0.86 | 53.93 ± 4.55 | 0/3 |
| Model poison ($\lambda=10$) | Corrected, unclipped | 12.01 ± 2.85 | 10.00 ± 0.00 | 88.18 ± 2.25 | 2.83 ± 0.87 | 3/3 |
| Model poison ($\lambda=10$) | Corrected with norm clip | 26.39 ± 0.33 | 13.97 ± 5.64 | 77.00 ± 0.78 | 79.87 ± 1.12 | 0/3 |
| Random noise ($\sigma=0.5$) | Corrected, unclipped | 10.00 ± 0.00 | 10.00 ± 0.00 | 99.40 ± 0.14 | 0.79 ± 0.10 | 3/3 |
| Random noise ($\sigma=0.5$) | Corrected with norm clip | 28.76 ± 0.66 | 40.56 ± 0.44 | 62.22 ± 2.57 | 49.69 ± 0.18 | 0/3 |

---

## 8. Current gaps and limitations

Five gaps and limitations remain relative to the literal paper.

- **Model and dataset accuracy gap.** The paper's 2-conv CNN on 10,000 CIFAR-10 images reaches at most 46.46% per-peer accuracy and 54.84% consensus accuracy, which is 38 to 47 percentage points below 93%. Reaching 91% to 93% requires all 50,000 training images and an ImageNet-pretrained ResNet-18.
- **Default `sender`-mode FPR gap and `rxdelta` trade-off.** Under strict `sender` updates ($\theta_j^{t+1/2} - \theta_j^t$), FPR remains at 36.2% to 43.9% rather than 3% to 9%. Switching to `rxdelta` ($\theta_j^{t+1/2} - \theta_i^t$) at $\tau = 0.003$ lowers FPR to 6.5% to 9.5%, but raises late-round Byzantine acceptance to 61.58% on label flipping and 41.26% on model poisoning. Raising $\tau$ to 0.035 reduces Byzantine acceptance below the `sender` baseline (31.68% on model poisoning and 39.16% on label flipping), but raises FPR back to 22.3% to 28.6%.
- **Balanced validation dependency.** Holding out 10% of each peer's skewed local data (`val=local`), as written in the paper, leaves FPR between 36.9% and 53.5% across both 10k and 50k runs, even with `rxdelta`. Low FPR requires a small class-balanced validation set (`val=iid`).
- **Ring topology per-peer accuracy gap.** Under 30% label flipping on a degree-2 Ring graph, consensus accuracy reaches 91.69%, while per-peer accuracy reaches 84.70%, 4.4 percentage points below the 89.1% reported in Table 1.
- **Single-seed `rxdelta` evaluation.** The 50k baseline, topology, sweep, and ablation results average seeds 42, 43, and 44, whereas the 16 `rxdelta` runs use seed 42 only.

For commands to reproduce every run and figure, see [experiments/phase1/REPRODUCE.md](../experiments/phase1/REPRODUCE.md).
