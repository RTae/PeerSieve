# TrustGraph-DFL system cost and Phase 2 motivation

This report measures the communication, round-time, receive-memory, and repeat-transfer overhead of batched FP32 TrustGraph-DFL on our Phase 1 CIFAR-10 ResNet-18 setup. Accuracy, attack resilience, topology, ablation, and false positive rate (FPR) findings live in [reproduce_report.md](reproduce_report.md). Here we quantify the four system bottlenecks in our Phase 1 baseline so Phase 2 can target them directly.

---

## Hardware and measurement setup

We ran all 14 configurations of `paper_resnet18_50k` (seed 42) on `octave` using our Phase 1 simulator with synchronized GPU timers and exact FP32 transfer accounting. Every run trained `PretrainedResNet18CIFAR` (`11,181,642` FP32 parameters, `44.73 MB` or `42.65 MiB` per update) across 20 peers for 25 communication rounds with 0 `NaN` runs.

| Component | Configuration |
|---|---|
| Cluster node and Slurm job | `octave`, job `327539`, commit `3899c13`, elapsed `00:20:55` |
| GPUs | 8x NVIDIA A100-PCIE-40GB (`01:00.0` to `E1:00.0`, PCIe Gen4 x16 `SYS`) |
| NVIDIA driver and CUDA | Driver `610.57.04`, CUDA `13.2` |
| Software stack | Python `3.12.13`, PyTorch `2.15.0+cu132`, `torchvision 0.29.2`, `networkx 3.7` |
| Run wall time across 14 runs | Median `446.0 s` (range `255.5` to `503.7 s`, 14 runs, 0 `NaN` runs) |
| Committed dataset | [report/data/phase1_system_metrics_50k.json](../data/phase1_system_metrics_50k.json) |

We report local compute timings from synchronized GPU wall clocks per round (median, min, and max over 25 rounds per run). We compute network transfer time from exact FP32 receive bytes per honest peer per round at `10 Gb/s` (datacenter), `1 Gb/s` (standard Ethernet, matching `octave`'s external link), and `100 Mb/s` (WAN or edge).

---

## Metrics we measure

- **Received traffic (MB per peer per round).** Total uncompressed FP32 payload downloaded by one honest peer from all of its neighbors in one communication round (`peer degree x 44.73 MB`).
- **Useful traffic (MB per peer per round).** Downloaded FP32 bytes from neighbors that receive positive trust weight and contribute to model aggregation.
- **False-positive waste (MB per peer per round).** Downloaded FP32 bytes from honest neighbors whose cosine score falls at or below the threshold ($\tau = 0.003$), so ReLU assigns zero weight and discards the payload.
- **Blocked-attacker waste (MB per peer per round).** Downloaded FP32 bytes from Byzantine neighbors that receive zero trust weight and are discarded after download.
- **Wasted share (%).** Percentage of all downloaded FP32 bytes (false-positive waste plus blocked-attacker waste) that receive zero trust weight.
- **Total wasted (GB over 25 rounds).** Total gigabytes of zero-weight FP32 payloads downloaded across all honest peers over the full 25-round training run.
- **Per-peer round time (seconds).** Median synchronized GPU compute time per honest peer (local training, validation gradient, and aggregation) plus FP32 download time at `10 Gb/s`, `1 Gb/s`, or `100 Mb/s`.
- **Simultaneous receive buffer (MiB).** Memory required to hold all incoming FP32 neighbor updates at once during batched aggregation (`peer degree x 42.65 MiB`).
- **Repeat downloads after first block.** Full FP32 payloads downloaded from a Byzantine neighbor in later rounds after that edge was already blocked once.
- **Consecutive rejection streak.** Longest run of back-to-back rounds in which a directed peer edge receives zero trust weight.

---

## Overview of the four measured gaps

| System gap | What Phase 1 TrustGraph-DFL does | Measured cost on 50k ResNet-18 |
|---|---|---|
| 1. Bandwidth wasted on discarded updates | Downloads every neighbor's full `44.73 MB` FP32 update before scoring and zeroing weights | `42.5%` to `67.6%` of received bytes discarded (`127.2` to `181.3 MB` per peer per round; `46.8` to `63.6 GB` over 25 rounds) |
| 2. Transfer cost of uncompressed FP32 updates | Transfers full FP32 vectors even for accepted neighbors | `2.15` to `2.43 s` transfer vs. `0.66 s` compute at `1 Gb/s` (`76.5%` to `79.0%` of round time; `97.0%` at `100 Mb/s`) |
| 3. Receive-side memory growth with degree | Holds all `d` incoming FP32 neighbor models in memory during batched aggregation | `85.3 MiB` (`2.0x`) on Ring to `341.2 MiB` (`8.0x`) and up to `469.2 MiB` (`11.0x`) on Random |
| 4. Repeated downloads from blocked attackers | Re-downloads full payloads from the same attacker edges every round while honest edges see false positives | Attacker edges blocked `15.6` to `25.0` of 25 rounds (`24.9` to `40.8 GB` repeat waste); `97.8%` to `100%` of honest edges hit at least one false positive |

---

## 1. Communication wasted on discarded updates

TrustGraph-DFL evaluates each neighbor update only after the receiver holds the complete FP32 model. Whenever cosine similarity falls at or below the threshold, the receiver assigns zero weight and discards the downloaded payload.

![Fig. 5 Batched FP32 Communication Waste and Round-Time Bottleneck](../figures/fig5-bandwidth-and-round-time-50k.png)

Across the four main Random ER ($p=0.3$) regimes in Figure 5a, honest peers discard `42.5%` to `67.6%` of all received FP32 traffic. Under 30% random noise, every attacker update receives zero weight, pushing wasted receive traffic to `181.3 MB` per peer per round (`67.6%`, `63.47 GB` discarded across honest peers over 25 rounds). Under 30% model poisoning, peers waste `149.3 MB` per round (`55.6%`, `52.24 GB` over 25 rounds). Even with zero attackers (`none`), false positives under default `sender` scoring cause peers to discard `127.2 MB` per round (`42.5%`, `63.60 GB` over 25 rounds).

| Regime on Random ER ($p=0.3$) | Mean / max degree | Received (MB/peer/round) | Useful (MB/peer/round) | False-positive waste (MB/peer/round) | Blocked-attacker waste (MB/peer/round) | Wasted share (%) | Total wasted (GB, 25 rounds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| No attack (0% Byzantine) | 6.7 / 11 | 299.7 | 172.5 | 127.2 | 0.0 | 42.45 | 63.60 |
| Label flip (30%, $E_b=2$) | 6.0 / 8 | 268.4 | 134.6 | 57.9 | 75.9 | 49.86 | 46.83 |
| Model poison (30%, $\lambda=10$) | 6.0 / 8 | 268.4 | 119.1 | 64.4 | 84.8 | 55.62 | 52.24 |
| Random noise (30%, $\sigma=0.5$) | 6.0 / 8 | 268.4 | 87.0 | 59.9 | 121.4 | 67.57 | 63.47 |

The waste persists across topologies and attacker fractions. Under 30% label flipping, wasted receive traffic is `35.29%` on Ring (`31.6 MB` per peer per round, `11.05 GB` over 25 rounds), `35.71%` on Small-World (`62.7 MB` per round, `21.96 GB` over 25 rounds), and `49.86%` on Random (`133.8 MB` per round, `46.83 GB` over 25 rounds). Sweeping label-flipping attackers from 10% to 50% on Random keeps wasted receive traffic between `42.98%` and `49.86%` (`130.3` to `142.1 MB` per peer per round).

| Sweep slice (label flip, $E_b=2$) | Mean / max degree | Received (MB/peer/round) | Useful (MB/peer/round) | False-positive waste (MB/peer/round) | Blocked-attacker waste (MB/peer/round) | Wasted share (%) | Total wasted (GB, 25 rounds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ring, 30% Byzantine | 2.0 / 2 | 89.5 | 57.9 | 21.6 | 10.0 | 35.29 | 11.05 |
| Small-World, 30% Byzantine | 3.9 / 7 | 175.7 | 113.0 | 36.0 | 26.7 | 35.71 | 21.96 |
| Random, 10% Byzantine | 6.8 / 11 | 303.1 | 172.8 | 114.9 | 15.4 | 42.98 | 58.64 |
| Random, 20% Byzantine | 6.6 / 11 | 296.3 | 162.1 | 91.5 | 42.7 | 45.28 | 53.67 |
| Random, 40% Byzantine | 6.2 / 11 | 275.8 | 142.2 | 59.5 | 74.1 | 48.43 | 40.08 |
| Random, 50% Byzantine | 6.8 / 11 | 304.1 | 162.1 | 52.8 | 89.3 | 46.71 | 35.51 |

In the Section 5.5 ablations (`euclidean` and `no_relu`), wasted traffic drops to `0.0%` only because every attacker update receives positive weight and collapses model accuracy.

---

## 2. Communication cost of uncompressed FP32 updates

Even accepted updates remain large in uncompressed FP32. On an NVIDIA A100-PCIE-40GB under 30% model poisoning on Random, local peer compute takes `0.660 s` per round in total. Local SGD takes a median of `0.602 s` (range `0.587` to `1.223 s`, 25 rounds), computing the validation gradient takes `0.055 s` (range `0.053` to `3.323 s`, where the maximum is round 1 cuDNN initialization), and cosine scoring plus aggregation takes `0.0032 s` (range `0.0029` to `0.134 s`). Across all 14 runs, median per-peer compute stays between `0.647 s` and `0.669 s`.

In Figure 5b, uncompressed FP32 transfer dominates total round time once bandwidth drops to `1 Gb/s` or below.

| Link bandwidth (30% model poison, Random) | Local train + val + agg (s) | Useful FP32 transfer (s) | Wasted FP32 transfer (s) | Total round time (s) | Communication share (%) |
|---|---:|---:|---:|---:|---:|
| `10 Gb/s` datacenter | 0.660 | 0.095 | 0.119 | 0.875 | 24.5 |
| `1 Gb/s` Ethernet | 0.660 | 0.953 | 1.194 | 2.807 | 76.5 |
| `100 Mb/s` WAN / edge | 0.660 | 9.528 | 11.941 | 22.129 | 97.0 |

Two gaps stand out in this table. First, at `1 Gb/s` Ethernet, communication takes `2.147 s` (`76.5%` of the round), which is `3.3x` slower than all local GPU compute combined. Second, eliminating discarded downloads alone is not enough, because even the `119.1 MB` of useful FP32 traffic takes `0.953 s` at `1 Gb/s` and `9.528 s` at `100 Mb/s`, still exceeding the `0.660 s` local GPU compute time. Across the other topologies under 30% label flipping at `1 Gb/s`, communication still accounts for `51.7%` of round time on Ring (`0.72 s` transfer vs. `0.67 s` compute) and `68.0%` on Small-World (`1.41 s` transfer vs. `0.66 s` compute).

---

## 3. Receive-side memory scaling with peer degree

Our Phase 1 TrustGraph-DFL implementation collects all neighbor updates before performing batched aggregation. Simultaneous FP32 receive buffers therefore grow linearly as $O(d)$ with peer degree $d$, where one ResNet-18 update occupies `42.65 MiB`.

![Fig. 6 Receive-Buffer Memory Growth and Repeated-Transfer Behavior](../figures/fig6-memory-and-rejection-streaks-50k.png)

In Figure 6a (30% label flipping), simultaneous receive buffers rise from `85.3 MiB` (`2.0x` a single update) on Ring to `167.6 MiB` mean and `298.6 MiB` max (`7.0x`) on Small-World, and to `255.9 MiB` mean and `341.2 MiB` max (`8.0x`) on Random. In the Random runs where higher-degree nodes are honest (`none`, and 10%, 20%, 40%, and 50% Byzantine), maximum honest degree reaches `11`, requiring `469.2 MiB` (`11.0x` a single update) for receive buffers alone.

| Topology and slice | Mean honest degree | Max honest degree | Single update (MiB) | Mean receive buffer (MiB) | Max receive buffer (MiB) | Max / single ratio |
|---|---:|---:|---:|---:|---:|---:|
| Ring (30% label flip) | 2.0 | 2 | 42.7 | 85.3 | 85.3 | 2.0x |
| Small-World (30% label flip) | 3.9 | 7 | 42.7 | 167.6 | 298.6 | 7.0x |
| Random (30% label flip, model poison, noise) | 6.0 | 8 | 42.7 | 255.9 | 341.2 | 8.0x |
| Random (no attack, 10%, 20%, 40%, 50% Byzantine) | 6.2 to 6.8 | 11 | 42.7 | 263.0 to 290.1 | 469.2 | 11.0x |

ResNet-18 fits inside a `40 GiB` A100 GPU (`7.06` to `8.08 GiB` peak allocated GPU memory in our simulator, which stages all 20 peer shards and models on one GPU). However, the measured `8x` to `11x` receive-buffer multiplier shows the scaling gap directly: holding `d` full payloads simultaneously grows with both model size and graph degree, whereas a streaming receiver needs only constant buffer space independent of `d`.

---

## 4. Repeated communication with untrustworthy neighbors

TrustGraph-DFL has no edge memory across rounds. After blocking a malicious neighbor in round $t$, the receiver still downloads that neighbor's full `44.73 MB` FP32 payload again in round $t+1$.

In Figure 6b, Byzantine edges generate sustained multi-round block streaks across all three attacks on Random ER ($p=0.3$). Under random noise, every one of the 38 directed attacker edges is blocked in all `25.0` rounds, causing `912` repeat downloads (`40.79 GB`) after each edge's first block. Under model poisoning, attacker edges are blocked in `17.5` of 25 rounds on average (`94.7%` hit at least 3 consecutive blocks, with `626` repeat downloads wasting `28.00 GB`). Under label flipping, attacker edges are blocked in `15.6` of 25 rounds (`73.7%` hit at least 3 consecutive blocks, with `556` repeat downloads wasting `24.87 GB`).

| Attack on Random (30% Byzantine, 25 rounds) | Attacker mean blocked rounds | Attacker mean max streak | Attacker edges with >=3 streak (%) | Attacker repeat downloads (GB wasted) | Honest edges with >=1 false positive (%) | Honest isolated single false positives | Honest edges with >=3 streak (%) | Honest edges with >=5 streak (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Label flip ($E_b=2$) | 15.6 / 25 | 7.3 | 73.7 | 556 (24.87 GB) | 97.8 | 282 | 34.8 | 17.4 |
| Model poison ($\lambda=10$) | 17.5 / 25 | 8.3 | 94.7 | 626 (28.00 GB) | 100.0 | 210 | 52.2 | 23.9 |
| Random noise ($\sigma=0.5$) | 25.0 / 25 | 25.0 | 100.0 | 912 (40.79 GB) | 100.0 | 282 | 43.5 | 21.7 |
| No attack (0% Byzantine) | 0.0 / 25 | 0.0 | 0.0 | 0 (0.00 GB) | 100.0 | 666 | 50.0 | 27.6 |

At the same time, honest non-IID edges do not have zero false positives. Across all four Random regimes, `97.8%` to `100.0%` of directed honest edges hit at least one false positive over 25 rounds, with `210` to `666` isolated single-round false positives per run. Even a stricter threshold of 3 consecutive false positives catches `34.8%` to `52.2%` of honest edges under default `sender` scoring (`17.4%` to `27.6%` reach 5 consecutive false positives). Permanently banning a neighbor after one false positive would disconnect almost every honest peer, while doing nothing lets chronic attackers waste `24.9` to `40.8 GB` in repeat downloads.

---

## Framing for Phase 2

These Phase 1 measurements confirm all four system gaps on ResNet-18 before scaling to larger models. In Phase 2, we will build PeerSieve to reduce each measured gap while preserving Byzantine resilience.

1. **Pre-transfer sketch screening for wasted downloads.** Because `42.5%` to `67.6%` of received FP32 bytes receive zero trust weight, Phase 2 will screen compact CountSketch summaries against the receiver's validation descent direction before requesting full payloads.
2. **Blockwise INT4 compression with error feedback for useful traffic.** Because even useful FP32 updates take `0.95 s` at `1 Gb/s` and `9.53 s` at `100 Mb/s` (exceeding `0.66 s` of A100 compute), Phase 2 will compress accepted payloads with blockwise 4-bit quantization and error feedback to shrink transfer time.
3. **Audited streaming aggregation for $O(d)$ receive memory.** Because batched receive buffers grow up to `8x` to `11x` a single update with peer degree, Phase 2 will verify and accumulate one neighbor payload at a time in constant receive buffer space, preventing rejected payloads from modifying the running aggregate.
4. **Per-edge probation with recovery for repeat transfers.** Because attacker edges trigger `556` to `912` repeat downloads (`24.9` to `40.8 GB`) while `97.8%` to `100%` of honest edges experience transient false positives, Phase 2 will place repeatedly failing edges on temporary probation rather than permanent bans and allow recovery after a passing sketch check.

For exact reproduction and summary export commands, see [experiments/phase1/REPRODUCE.md](../../experiments/phase1/REPRODUCE.md).
