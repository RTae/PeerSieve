# TrustGraph-DFL on CIFAR-10: reproduction and false-positive study

This report evaluates [TrustGraph-DFL.pdf](../TrustGraph-DFL.pdf) (Xu et al., CIBDA 2026) on CIFAR-10 across 20 decentralized peers. TrustGraph-DFL lets each peer score neighbor updates by taking the cosine similarity against its own validation gradient, zeroing negative scores with ReLU, and averaging the trusted models.

We first ran the setup written in the paper (a small 2-convolution CNN on 10,000 images). That setup plateaus near 55% accuracy and crashes with `NaN` values under scaled poisoning or noise attacks. Switching to all 50,000 CIFAR-10 images with an ImageNet-pretrained ResNet-18, fixing the validation gradient sign, and adding update norm clipping reproduces the paper's 91% to 93% accuracy across attacks, topologies, and up to 50% Byzantine peers with zero `NaN` crashes. One gap remains in the default single-round scoring rule: peers reject 36% to 44% of honest neighbor updates instead of the 3% to 9% reported in the paper. Testing four candidate fixes explains why this happens and shows two ways to reduce those false rejections.

---

## Key terms and metrics

- **Per-peer accuracy (`Final Acc`)**: average CIFAR-10 test accuracy of individual honest peers' local models at the end of training.
- **Consensus accuracy (`Cons Acc`)**: test accuracy of the single parameter-averaged model across all honest peers.
- **Honest false positive rate (`Honest FPR`)**: percentage of honest neighbor updates assigned zero trust weight and ignored. Lower is better.
- **Byzantine acceptance (`Byz Accept`)**: percentage of malicious neighbor updates that receive positive trust weight. Lower is better.
- **Validation set (`val=local` vs. `val=iid`)**: each peer computes its reference validation gradient either on a 10% slice of its own skewed local data (`val=local`) or on a small 250-image class-balanced validation set (`val=iid`).
- **Update anchor (`sender` vs. `receiver` / `rxdelta`)**: the literal paper scores a neighbor's own local step from its starting weights (`sender`: $\theta_j^{t+1/2} - \theta_j^t$). The receiver-anchored alternative scores how far the neighbor's updated weights sit from the receiver's weights (`rxdelta`: $\theta_j^{t+1/2} - \theta_i^t$).

---

## Results at a glance

| Paper claim | Literal 10k 2-conv CNN (200 rounds, 3 seeds) | 50k ResNet-18 (25 rounds, 3 seeds) | Verdict |
|---|---|---|---|
| Clean accuracy around 93% (Fig. 2, Fig. 3a) | 46.46% per-peer, 54.84% consensus | 91.39% per-peer, 92.87% consensus | Reproduced on 50k ResNet-18; unreachable on 10k 2-conv CNN |
| Attack resilience at 30% Byzantine peers (~89.5% to 93%) | 52.83% consensus on 2-class label flip; 3/3 `NaN`s on unclipped poisoning and noise | 92.38% label flip, 91.53% model poisoning, 92.33% random noise (0/42 `NaN`s) | Reproduced after flipping validation gradient to $-\nabla F_i$ and adding norm clipping |
| Topologies under 30% label flip (Table 1: 89.1 / 91.2 / 90.4%) | 38.8% Ring, 42.1% Random, 43.2% Small-World per-peer | 84.70% Ring, 89.28% Random, 89.48% Small-World per-peer (91.69% to 92.38% consensus) | Closely reproduced on 50k ResNet-18 (Ring per-peer sits 4.4 percentage points lower) |
| Resilience at 50% Byzantine peers (Fig. 3b) | 52.06% consensus, 51.23% honest FPR | 89.53% per-peer, 92.03% consensus | Reproduced |
| Design ablations (Section 5.5) | Confounded by `NaN` crashes (1/3 to 3/3 runs) | Dropping ReLU loses 35.9 percentage points; Euclidean distance loses 9.0 percentage points | Reproduced cleanly with zero `NaN`s |
| Honest FPR of 3% to 9% (Fig. 3b) | 50.3% to 53.5% (`val=local`), 42.3% (`val=iid`) | 36.2% to 43.9% in strict `sender` mode | Reaches 6.5% to 9.5% with `rxdelta` and balanced validation; EMA halves strict-`sender` FPR to 21.0% to 22.2% |

---

## 1. Why the literal paper setup fails (10k 2-conv CNN audit)

We first implemented the paper's stated setup across 20 peers using a random Erdos-Renyi graph ($p = 0.3$), 10,000 non-IID CIFAR-10 images (Dirichlet $\alpha = 0.5$, 500 images per peer with 50 held out for local validation), learning rate $\eta = 0.01$, and a 2-convolution, 2-fully-connected CNN (`SimpleCIFARCNN`, 1,070,794 parameters) trained for 200 rounds across 3 random seeds (42, 43, 44). Three issues block reproduction in this regime:

1. **Model and data capacity ceiling.** Even with no attackers, the 2-conv CNN on 10,000 non-IID images reaches only 46.46% per-peer accuracy and 54.84% consensus accuracy, roughly 40 percentage points below the paper's 93% target.
2. **Reversed validation gradient sign.** Algorithm 1 and Equation (3) in the paper compute $+\nabla F_i$, which points in the loss-increasing direction opposite to stochastic gradient descent updates. Negating it to $-\nabla F_i$ fixes the direction: under 30% two-class label flipping (`0↔1`), consensus accuracy rises from 51.27% to 52.83% and Byzantine acceptance drops from 59.16% to 43.45%. Across 10% to 50% Byzantine peers, $-\nabla F_i$ beats $+\nabla F_i$ by 1.0 to 2.5 percentage points in consensus accuracy (52.06% to 54.75%) while cutting Byzantine acceptance by 10 to 24 percentage points (43.3% to 46.6% versus 55.5% to 67.1%).
3. **Unbounded attack magnitudes cause `NaN` crashes.** Cosine similarity ignores vector length. Under unclipped model poisoning ($\lambda = 10$) or Gaussian random noise ($\sigma = 0.5$), a single accepted bad update blows up the model weights and produces 3/3 `NaN` failures (`paper_pos` diverges by round 8). Clipping incoming updates eliminates all `NaN` crashes (0/3).

| Attack (30% Byzantine unless clean) | Validation variant | Per-peer acc (%) | Consensus acc (%) | Honest FPR (%) | Byz accept (%) | Cosine (benign / Byz) | `NaN` runs |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean (`none`) | Corrected ($-\nabla F_i$) | 46.46 ± 0.85 | 54.84 ± 0.47 | 50.54 ± 2.30 | 0.00 | -0.000 / n/a | 0/3 |
| Clean (`none`) | Literal ($+\nabla F_i$) | 46.51 ± 1.42 | 54.70 ± 0.62 | 45.18 ± 2.80 | 0.00 | +0.021 / n/a | 0/3 |
| Label flip (`0↔1`) | Corrected ($-\nabla F_i$) | 42.07 ± 0.23 | 52.83 ± 0.16 | 51.38 ± 2.50 | 43.45 ± 1.64 | -0.003 / -0.029 | 0/3 |
| Label flip (`0↔1`) | Literal ($+\nabla F_i$) | 43.41 ± 0.95 | 51.27 ± 1.61 | 46.68 ± 5.55 | 59.16 ± 5.73 | +0.012 / +0.055 | 0/3 |
| Label flip (`0↔1`) | Corrected + balanced val (`val=iid`) | 45.67 ± 0.62 | 52.76 ± 0.48 | 42.28 ± 0.86 | 53.93 ± 4.55 | +0.043 / +0.013 | 0/3 |
| Model poison ($\lambda=10$) | Corrected, unclipped | 12.01 ± 2.85 | 10.00 ± 0.00 | 88.18 ± 2.25 | 2.83 ± 0.87 | `NaN` | 3/3 |
| Model poison ($\lambda=10$) | Corrected + norm clip | 26.39 ± 0.33 | 13.97 ± 5.64 | 77.00 ± 0.78 | 79.87 ± 1.12 | -0.050 / +0.058 | 0/3 |
| Random noise ($\sigma=0.5$) | Corrected, unclipped | 10.00 ± 0.00 | 10.00 ± 0.00 | 99.40 ± 0.14 | 0.79 ± 0.10 | `NaN` | 3/3 |
| Random noise ($\sigma=0.5$) | Corrected + norm clip | 28.76 ± 0.66 | 40.56 ± 0.44 | 62.22 ± 2.57 | 49.69 ± 0.18 | -0.024 / -0.000 | 0/3 |

Across topologies on 10,000 images with 30% label flipping (`0↔1`), corrected $-\nabla F_i$ reaches 38.8% per-peer (50.6% consensus) on Ring, 42.1% (52.8%) on Random, and 43.2% (52.7%) on Small-World.

---

## 2. Reproducing 91% to 93% accuracy (50k ResNet-18, 3 seeds)

To reach the paper's reported accuracy regime without changing TrustGraph-DFL's cosine trust mechanism, we applied four targeted adjustments and evaluated all 14 configurations across 3 random seeds (42 runs total, 0 `NaN`s):

- **Full CIFAR-10 dataset**: all 50,000 training images (2,500 per peer, Dirichlet $\alpha = 0.5$) with random crop and horizontal flip augmentation.
- **Stronger backbone**: ImageNet-pretrained `ResNet-18` trained for 25 rounds with a cosine learning rate starting at 0.005.
- **Numerical stability and sign**: negative validation gradient ($-\nabla F_i$), local gradient clipping at norm 10.0, receiver-side model-space update norm cap at twice the peer's own update norm, and a small positive trust threshold ($\tau = 0.003$).
- **Balanced validation and stronger label flipping**: a 250-image class-balanced validation set per peer (`val=iid`), and all-class label flipping with 2 local attacker epochs instead of mild 2-class `0↔1` flipping.

### 2.1 Attack resilience and network topologies

Figure 1 and Figure 2 show that TrustGraph-DFL maintains over 91.5% consensus accuracy across every 30% attack and topology. Under random noise ($\sigma = 0.5$), cosine scores cluster at zero and TrustGraph-DFL rejects 100% of malicious updates (`0.00%` Byzantine acceptance). Random and Small-World topologies closely match Table 1 of the paper. On Ring graphs, where each peer connects to only two neighbors, consensus accuracy still reaches 91.69%, though per-peer accuracy is 84.70% (4.4 percentage points below the paper's 89.1%).

![Fig. 1 TrustGraph-DFL Attack Regimes (50k)](figures/fig1-attacks-50k.png)

![Fig. 2 TrustGraph-DFL Topologies and Byzantine Stress (50k)](figures/fig2-topologies-and-stress-50k.png)

All runs in the table below finished with 0/3 `NaN` failures across seeds 42, 43, and 44.

| Topology | Attack (30% Byzantine unless clean) | Per-peer acc (%) | Consensus acc (%) | Honest FPR (%) | Byz accept (%) | Cosine (benign / Byz) |
|---|---|---:|---:|---:|---:|---:|
| Random (ER $p=0.3$) | Clean (0% Byzantine) | 91.39 ± 0.30 | 92.87 ± 0.22 | 41.18 ± 3.05 | 0.00 | +0.020 / n/a |
| Random (ER $p=0.3$) | All-class label flip (2 attacker epochs) | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 |
| Random (ER $p=0.3$) | Model poisoning ($\lambda=10$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | +0.016 / -0.030 |
| Random (ER $p=0.3$) | Random noise ($\sigma=0.5$) | 88.98 ± 1.31 | 92.33 ± 0.42 | 38.83 ± 3.33 | 0.00 | +0.023 / +0.000 |
| Ring (degree 2) | All-class label flip (2 attacker epochs) | 84.70 ± 1.04 | 91.69 ± 0.07 | 48.03 ± 2.26 | 48.09 | +0.015 / -0.015 |
| Small-World ($k=4, p=0.3$) | All-class label flip (2 attacker epochs) | 89.48 ± 0.12 | 92.36 ± 0.27 | 27.84 ± 2.00 | 43.78 | +0.030 / -0.025 |

### 2.2 Scaling to 50% Byzantine peers and ablations

Figure 3 shows three findings. First, increasing the fraction of all-class label-flipping peers from 10% to 50% reduces consensus accuracy by less than 1 percentage point (92.63% down to 92.03%). Second, the Section 5.5 ablations under 30% model poisoning confirm why each component of TrustGraph-DFL matters: removing ReLU zeroing leaves negative-scoring malicious updates included and drops consensus accuracy by 35.9 percentage points (to 55.60%), replacing cosine similarity with Euclidean distance accepts 100% of malicious updates and loses 9.0 percentage points (to 82.57%), and raising self-weight to $\alpha = 0.7$ isolates peers and hurts per-peer accuracy (82.49%). Lowering self-weight to $\alpha = 0.3$ slightly improves accuracy over $\alpha = 0.5$ and cuts model-poisoning acceptance from 33.07% to 21.92%. Third, the right panel of Figure 3 reveals why honest FPR is high: it begins near 0% in round 1 when peers share the same starting weights, then climbs into the 36% to 44% range as peers specialize on non-IID local shards.

![Fig. 3 TrustGraph-DFL Ablations and FPR Dynamics (50k)](figures/fig3-ablations-and-fpr-dynamics-50k.png)

**Byzantine fraction sweep (all-class label flip, 2 attacker epochs, Random topology, 0/3 `NaN`s):**

| Byzantine fraction | Per-peer acc (%) | Consensus acc (%) | Honest FPR (%) | Byz accept (%) | Cosine (benign / Byz) |
|---:|---:|---:|---:|---:|---:|
| 10% | 90.93 ± 0.16 | 92.63 ± 0.22 | 43.06 ± 2.20 | 47.78 | +0.019 / -0.017 |
| 20% | 90.58 ± 0.43 | 92.75 ± 0.17 | 39.54 ± 3.90 | 41.68 | +0.022 / -0.016 |
| 30% | 89.28 ± 1.26 | 92.38 ± 0.32 | 40.39 ± 2.36 | 41.75 | +0.024 / -0.020 |
| 40% | 89.89 ± 0.54 | 92.23 ± 0.03 | 38.68 ± 4.80 | 42.25 | +0.025 / -0.012 |
| 50% | 89.53 ± 0.38 | 92.03 ± 0.29 | 36.17 ± 3.70 | 44.30 | +0.028 / -0.011 |

**Section 5.5 component ablations (30% model poisoning, $\lambda = 10$, Random topology, 0/3 `NaN`s):**

| Ablation variant | Per-peer acc (%) | Consensus acc (%) | Honest FPR (%) | Byz accept (%) | Cosine (benign / Byz) |
|---|---:|---:|---:|---:|---:|
| Full TrustGraph-DFL ($\alpha = 0.5$) | 88.88 ± 1.36 | 91.53 ± 0.58 | 43.93 ± 4.27 | 33.07 | +0.016 / -0.030 |
| Lower self-weight ($\alpha = 0.3$) | 89.18 ± 0.96 | 91.67 ± 0.30 | 45.69 ± 1.88 | 21.92 | +0.013 / -0.060 |
| Higher self-weight ($\alpha = 0.7$) | 82.49 ± 2.20 | 90.70 ± 0.44 | 46.92 ± 4.48 | 49.89 | +0.009 / -0.002 |
| Euclidean distance instead of cosine | 80.01 ± 5.58 | 82.57 ± 5.32 | 0.00 ± 0.00 | 100.00 | +0.535 / +0.340 |
| No ReLU zeroing (keep negative scores) | 66.55 ± 16.61 | 55.60 ± 33.40 | 0.00 ± 0.00 | 100.00 | +0.071 / -0.166 |

---

## 3. Why honest FPR stays near 40%, and four fixes we tested

In the default `sender` mode, each peer compares a neighbor's single-round local step ($\theta_j^{t+1/2} - \theta_j^t$) against its own validation gradient ($-\nabla F_i(\theta_i^t)$). Because non-IID peers drift onto slightly different local weights ($\theta_j^t \neq \theta_i^t$) and sample skewed batches each round, single-round cosine scores bounce around zero and roughly 40% fall below $\tau = 0.003$. We ran 40 controlled seed-42 experiments (0 `NaN`s) across four candidate fixes to isolate what actually lowers honest FPR.

![Fig. 4 Four-Lever FPR Diagnostic (50k)](figures/fig4-fpr-levers-50k.png)

### What each lever showed

- **Lever 1 (works, with a trade-off): score relative to the receiver's weights (`rxdelta`: $\theta_j^{t+1/2} - \theta_i^t$).** Switching the anchor from the sender's weights to the receiver's weights drops honest FPR from 38.8%–43.9% down to **7.4%–9.5%** at 30% Byzantine peers (and **6.5%–6.8%** at 50% Byzantine peers) while lifting per-peer accuracy by 0.45 to 1.45 percentage points. However, it depends on balanced validation (`val=iid`): with local validation (`val=local`), honest FPR remains at 39.19% clean and 36.94% at 50% label flipping. It also creates a late-round security trade-off: because $\theta_m^{t+1/2} - \theta_i^t = (\theta_m^{t+1/2} - \theta_m^t) + (\theta_m^t - \theta_i^t)$, a malicious update piggybacks on the honest starting-point difference, raising label-flipping Byzantine acceptance to 61.58% and model-poisoning acceptance to 41.26%. Raising the trust threshold (`Lever 1b`) tunes this smoothly: at $\tau = 0.035$, `rxdelta` beats the single-round `sender` baseline on honest FPR (22.3% to 28.6%), model-poisoning acceptance (31.68%), label-flipping acceptance (39.16%), and per-peer accuracy at the same time.
- **Lever 2 (ruled out): take fewer local SGD steps per round.** Cutting local work to 9 steps per round (`batch_size=256`, 25 rounds) or 4 steps per round (`max_local_steps=4`, 100 rounds) triples the average cosine gap between honest and malicious peers (`+0.044` up to `+0.137`), yet keeps honest FPR stuck at 40.6% to 45.0% and lowers clean consensus accuracy by about 2.5 percentage points. Local drift within a single round is not the bottleneck.
- **Lever 3 (ruled out): reweight or separate network layers.** Profiling ResNet-18 showed that the tiny final classifier layer (`backbone.fc`, 0.05% of parameters) accounts for 38.52% of total squared gradient norm ($\|\nabla\|^2$). Even so, averaging cosine scores across layers (`layer_mean`), scoring only the convolutional backbone (`backbone_only`), or scoring only the classifier head (`head_only`) all leave honest FPR between 38.1% and 43.0% (though `head_only` lowers label-flipping Byzantine acceptance to 32.32%).
- **Lever 4 (works in strict `sender` mode): average neighbor scores across rounds (EMA).** Keeping strict `sender` mode and smoothing each neighbor's trust score across rounds with a zero-initialized exponential moving average ($R_{ij}^t = \beta R_{ij}^{t-1} + (1 - \beta) s_{ij}^t$) cancels single-round batch noise without introducing starting-point piggybacking. At $\beta = 0.85$, honest FPR is cut in half to **21.0%–22.2%**, model-poisoning acceptance drops from 33.07% down to **14.32%** (remaining 0.0% through round 16), random-noise acceptance stays **0.00%**, and clean accuracy stays high (91.51% per-peer, 92.72% consensus).

### Clean accuracy and honest FPR across levers (balanced validation `val=iid` unless noted)

| Variant | Anchor mode | Clean per-peer (%) | Clean consensus (%) | Honest FPR range across tests (%) | Outcome |
|---|---|---:|---:|---:|---|
| Baseline (`sender`, 3-seed) | Sender | 91.39 | 92.87 | 38.8 to 43.9 | High FPR baseline |
| Lever 1 (`rxdelta`, $\tau = 0.003$) | Receiver | 91.84 | 92.63 | 7.4 to 9.5 | Hits paper FPR range; higher attack acceptance |
| Lever 1 (`rxdelta`, `val=local`) | Receiver | 89.09 | 92.50 | 36.9 to 39.2 | Confirms balanced validation is needed |
| Lever 1b (`rxdelta`, $\tau = 0.020$) | Receiver | 91.73 | 92.71 | 13.7 to 16.7 | Mid-range FPR vs. rejection trade-off |
| Lever 1b (`rxdelta`, $\tau = 0.035$) | Receiver | 91.66 | 92.45 | 22.3 to 28.6 | Strictly beats `sender` baseline on all axes |
| Lever 2 (`batch_size=256`, 25 rounds) | Sender | 88.53 | 90.08 | 41.0 to 42.0 | Ruled out (accuracy drops, FPR unchanged) |
| Lever 2 (`max_local_steps=4`, 100 rounds) | Sender | 89.87 | 90.38 | 40.6 to 45.0 | Ruled out (accuracy drops, FPR unchanged) |
| Lever 3 (`layer_mean`) | Sender | 91.49 | 92.58 | 38.1 to 41.4 | Ruled out (0.53% random noise leaks) |
| Lever 3 (`backbone_only`) | Sender | 91.37 | 92.52 | 41.5 to 43.0 | Ruled out (FPR unchanged) |
| Lever 4 (EMA $\beta = 0.70$) | Sender | 91.22 | 92.57 | 26.4 to 28.7 | Lower FPR and better poisoning rejection |
| Lever 4 (EMA $\beta = 0.85$) | Sender | 91.51 | 92.72 | 21.0 to 22.2 | Halves `sender` FPR and cuts poisoning accept to 14.32% |

### Attack robustness across levers (30% Byzantine peers, balanced validation `val=iid`)

| Variant | Model poison cons (%) | Model poison FPR (%) | Model poison Byz accept (%) | Label flip cons (%) | Label flip FPR (%) | Label flip Byz accept (%) | Random noise Byz accept (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline (`sender`, 3-seed) | 91.53 | 43.93 | 33.07 | 92.38 | 40.39 | 41.75 | 0.00 |
| Lever 1 (`rxdelta`, $\tau = 0.003$) | 91.08 | 7.39 | 41.26 | 92.05 | 7.83 | 61.58 | 0.00 |
| Lever 1b (`rxdelta`, $\tau = 0.020$) | 91.31 | 15.22 | 34.63 | 92.09 | 13.65 | 50.84 | 0.00 |
| Lever 1b (`rxdelta`, $\tau = 0.035$) | 91.33 | 25.65 | 31.68 | 92.19 | 22.35 | 39.16 | 0.00 |
| Lever 2 (`batch_size=256`, 25 rounds) | 87.65 | 41.04 | 29.05 | 88.76 | 42.00 | 40.95 | 0.00 |
| Lever 2 (`max_local_steps=4`, 100 rounds) | 87.13 | 44.98 | 31.39 | 89.08 | 40.91 | 34.08 | 0.00 |
| Lever 3 (`layer_mean`) | 90.72 | 41.39 | 30.11 | 92.04 | 40.17 | 40.95 | 0.53 |
| Lever 3 (`backbone_only`) | 90.80 | 42.96 | 32.53 | 91.88 | 41.48 | 42.00 | n/a |
| Lever 3 (`head_only`) | n/a | n/a | n/a | 92.07 | 40.26 | 32.32 | n/a |
| Lever 4 (EMA $\beta = 0.70$) | 91.79 | 27.57 | 21.79 | 92.01 | 26.43 | 41.16 | 0.00 |
| Lever 4 (EMA $\beta = 0.85$) | 91.72 | 21.13 | 14.32 | 91.92 | 21.57 | 44.53 | 0.00 |

---

## 4. Summary of gaps that cannot be reproduced literally

1. **Model and data capacity:** A 2-conv CNN trained on 10,000 non-IID CIFAR-10 images caps at 46.5% per-peer and 54.8% consensus accuracy. Reaching 91% to 93% requires all 50,000 images and an ImageNet-pretrained `ResNet-18`.
2. **Gradient sign and norm control:** The paper's literal $+\nabla F_i$ points opposite to descent updates, and unclipped cosine similarity diverges (`NaN`) on scaled poisoning and noise. Zero-`NaN` robustness requires $-\nabla F_i$, gradient clipping, update norm clipping, and $\tau > 0$.
3. **Single-round `sender` FPR floor:** Under Dirichlet $\alpha = 0.5$, single-round `sender` cosine keeps honest FPR at 36% to 44% even when cutting local SGD steps 9x or reweighting layers. Reaching the paper's 3% to 9% FPR requires receiver-anchored scoring (`rxdelta`) with balanced validation (`val=iid`), while score EMA cuts strict `sender` FPR in half to 21%.
4. **Local validation split and label-flipping stress:** Keeping a 10% non-IID local validation set locks honest FPR at 37% to 53% in both 10k and 50k experiments. Balanced validation (`val=iid`) is required for low FPR under `rxdelta`, and all-class label flipping (2 attacker epochs) is needed to meaningfully stress the defense compared to mild 2-class `0↔1` flipping.

---

## Appendix: hardware, run records, and reproducibility

**Hardware and software environment:** All cluster runs executed on node `octave` using 8x NVIDIA A100-PCIE-40GB GPUs (PCI buses `01:00.0` to `E1:00.0`, PCIe Gen4 x16 `SYS`, driver `610.57.04`), Python `3.12.13`, PyTorch `2.15.0+cu132`, CUDA `13.2`, `torchvision 0.29.2`, and `networkx 3.7`.

| Experiment batch | Scope and seeds | Slurm job IDs and wall times / memory | Git commits | Runs / `NaN`s |
|---|---|---|---|---|
| Literal 10k 2-conv CNN audit | 200 rounds, seeds 42, 43, 44 | `327259`, `327261`, `327264` | `9c05b8b` | 3 seeds (`NaN`s on unclipped attacks only) |
| Locked 50k ResNet-18 evaluation (`paper_match_50k`) | 14 configs x 3 seeds (42, 43, 44) | `327299` (`00:42:46`, 20.7 GB), `327300` (`00:42:35`, 17.3 GB), `327301` (`00:41:26`, 17.3 GB) | `a0be515` | 42 runs, 0 `NaN`s |
| Four-lever FPR diagnostic (`fpr_lever1` through `fpr_lever4`) | 5 suites, seed 42 (timing median 338.6s, range 164.6s to 517.1s) | `327344` (`00:12:37`), `327496` (`00:08:14`), `327354` (`00:06:35`), `327486` (`00:13:47`), `327500` (`00:10:14`) | `3361b95`, `e26337e`, `cb28406`, `ce87e28` | 40 runs, 0 `NaN`s |

**Committed data and figure script:** [data/phase1_paper_match_50k_3seed.json](data/phase1_paper_match_50k_3seed.json) · [data/phase1_fpr_levers.json](data/phase1_fpr_levers.json) · [../experiments/phase1/plot_figures.py](../experiments/phase1/plot_figures.py)

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
