# TrustGraph-DFL system cost and Phase 2 motivation

This report measures the communication, round-time, receive-memory, and repeat-transfer overhead of batched FP32 TrustGraph-DFL on our Phase 1 CIFAR-10 ResNet-18 setup. Accuracy, attack resilience, topology, ablation, and false positive rate (FPR) findings live in [reproduce_report.md](reproduce_report.md). Here we quantify the four system bottlenecks in our Phase 1 baseline so Phase 2 can reduce them directly.

---

## Hardware and measurement setup

We ran all 14 configurations of `paper_resnet18_50k` (seed 42) on `octave` using our Phase 1 simulator with synchronized GPU timers and FP32 transfer accounting. Every run trained `PretrainedResNet18CIFAR` (`11,181,642` FP32 parameters, `44.73 MB` or `42.65 MiB` per update) across 20 peers for 25 communication rounds with 0 `NaN` runs.

| Component | Configuration |
|---|---|
| GPUs | 8x NVIDIA A100-PCIE-40GB (`01:00.0` to `E1:00.0`, PCIe Gen4 x16 `SYS`) |
| External network (lab inventory) | `1 Gb/s` Ethernet without RDMA (nominal NIC line rate on `octave`) |
| NVIDIA driver and CUDA | Driver `610.57.04`, CUDA `13.2` |
| Software stack | Python `3.12.13`, PyTorch `2.15.0+cu132`, `torchvision 0.29.2`, `networkx 3.7` |
| Run wall time across 14 runs | Median `446.0 s` (range `255.5` to `503.7 s`, 14 runs, 0 `NaN` runs) |
| Committed dataset | [report/data/phase1_system_metrics_50k.json](../data/phase1_system_metrics_50k.json) |

Because our Phase 1 simulator runs all 20 peers on a single GPU without physical socket transfers, we measure GPU compute directly and estimate network transfer time analytically from FP32 byte counts.

- **Measured GPU compute.** Synchronized CUDA wall-clock times per benign peer per round across 25 rounds (median, min, and max) for local SGD training, validation gradient computation, and TrustGraph aggregation.
- **Estimated transfer time.** Lower-bound wire serialization time $\text{transfer time} = (\text{received bytes} \times 8) / \text{link bandwidth}$. We anchor this estimate at `1 Gb/s` (the nominal external Ethernet NIC rate recorded for our `octave` cluster) and bracket it with two standard reference tiers (`10 Gb/s` datacenter Ethernet and `100 Mb/s` WAN or edge). Because this formula uses nominal line rates and excludes TCP handshakes, packet headers, congestion, and host-to-GPU copies, physical multi-node transfers will take longer.

---

## Metrics we measure

- **Received traffic (MB per peer per round).** Total uncompressed FP32 payload downloaded by one benign peer from all of its neighbors in one communication round (`peer degree x 44.73 MB`).
- **Useful traffic (MB per peer per round).** Downloaded FP32 bytes from neighbors that receive positive trust weight and contribute to model aggregation.
- **False-positive waste (MB per peer per round).** Downloaded FP32 bytes from benign neighbors whose cosine score falls at or below the threshold ($\tau = 0.003$), so ReLU assigns zero weight and discards the payload.
- **Blocked-attacker waste (MB per peer per round).** Downloaded FP32 bytes from Byzantine neighbors that receive zero trust weight and are discarded after download.
- **Wasted share (%).** Percentage of all downloaded FP32 bytes (false-positive waste plus blocked-attacker waste) that receive zero trust weight.
- **Total wasted (GB over 25 rounds).** Total gigabytes of zero-weight FP32 payloads downloaded across all benign peers over the full 25-round training run.
- **Per-peer round time (seconds).** Measured median A100 compute time per benign peer plus estimated FP32 download time at `10 Gb/s`, `1 Gb/s`, or `100 Mb/s`.
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
| 4. Repeated downloads from blocked attackers | Re-downloads full payloads from the same attacker edges every round while benign edges see false positives | Attacker edges blocked `15.6` to `25.0` of 25 rounds (`24.9` to `40.8 GB` repeat waste); `97.8%` to `100%` of benign edges hit at least one false positive |

---

## 1. Communication waste and round-time bottleneck (Figure 5)

Figure 5 links byte volume directly to wall-clock round time. Panel (a) breaks down how many FP32 megabytes each benign peer downloads per round across the four Random ER ($p=0.3$) regimes. Panel (b) takes the `Model Poison` bar from Panel (a) (`268.4 MB` per peer per round) and converts its useful and wasted byte segments into transfer time alongside measured A100 compute across three network speeds.

![Fig. 5 Batched FP32 Communication Waste and Round-Time Bottleneck](../figures/fig5-bandwidth-and-round-time-50k.png)

### 1.1 Bandwidth wasted on discarded updates (Gap 1, Figure 5a)

TrustGraph-DFL scores a neighbor only after downloading its full `44.73 MB` FP32 update. Whenever cosine similarity falls at or below $\tau = 0.003$, ReLU assigns zero weight and discards the downloaded payload. Reading Figure 5a from left to right shows how much of each bar is blue (useful), pink (false-positive waste), and dark red (blocked-attacker waste).

1. **No Attack (0% Byzantine).** Benign peers average `6.7` neighbors and receive `299.7 MB` per round, but `127.2 MB` (`42.5%`) is pink false-positive waste (`63.60 GB` discarded over 25 rounds).
2. **Label Flip (30% Byzantine, $E_b=2$).** Benign peers average `6.0` neighbors and receive `268.4 MB` per round. Only `134.6 MB` (blue) is useful, while `57.9 MB` (pink) and `75.9 MB` (red) are discarded (`49.9%` wasted, `46.83 GB` over 25 rounds).
3. **Model Poison (30% Byzantine, $\lambda=10$).** Out of `268.4 MB` received per round, `119.1 MB` (blue) is useful and `149.3 MB` (`64.4 MB` pink plus `84.8 MB` red, `55.6%`) is wasted (`52.24 GB` over 25 rounds).
4. **Random Noise (30% Byzantine, $\sigma=0.5$).** TrustGraph-DFL blocks every noise attacker, so `121.4 MB` (red) plus `59.9 MB` (pink) goes to waste. Peers discard `181.3 MB` of `268.4 MB` per round (`67.6%` wasted, `63.47 GB` over 25 rounds).

| Regime on Random ER ($p=0.3$) | Mean / max degree | Received (MB/peer/round) | Useful (MB/peer/round) | False-positive waste (MB/peer/round) | Blocked-attacker waste (MB/peer/round) | Wasted share (%) | Total wasted (GB, 25 rounds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| No attack (0% Byzantine) | 6.7 / 11 | 299.7 | 172.5 | 127.2 | 0.0 | 42.45 | 63.60 |
| Label flip (30%, $E_b=2$) | 6.0 / 8 | 268.4 | 134.6 | 57.9 | 75.9 | 49.86 | 46.83 |
| Model poison (30%, $\lambda=10$) | 6.0 / 8 | 268.4 | 119.1 | 64.4 | 84.8 | 55.62 | 52.24 |
| Random noise (30%, $\sigma=0.5$) | 6.0 / 8 | 268.4 | 87.0 | 59.9 | 121.4 | 67.57 | 63.47 |

The same pattern holds across topologies and attacker fractions. Under 30% label flipping, wasted receive traffic is `35.29%` on Ring, `35.71%` on Small-World, and `49.86%` on Random. Across 10% to 50% label-flipping attackers on Random, waste stays between `42.98%` and `49.86%`. In the Section 5.5 ablations (`euclidean` and `no_relu`), wasted traffic drops to `0.0%` only because every attacker update receives positive weight and degrades accuracy.

| Sweep slice (label flip, $E_b=2$) | Mean / max degree | Received (MB/peer/round) | Useful (MB/peer/round) | False-positive waste (MB/peer/round) | Blocked-attacker waste (MB/peer/round) | Wasted share (%) | Total wasted (GB, 25 rounds) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ring, 30% Byzantine | 2.0 / 2 | 89.5 | 57.9 | 21.6 | 10.0 | 35.29 | 11.05 |
| Small-World, 30% Byzantine | 3.9 / 7 | 175.7 | 113.0 | 36.0 | 26.7 | 35.71 | 21.96 |
| Random, 10% Byzantine | 6.8 / 11 | 303.1 | 172.8 | 114.9 | 15.4 | 42.98 | 58.64 |
| Random, 20% Byzantine | 6.6 / 11 | 296.3 | 162.1 | 91.5 | 42.7 | 45.28 | 53.67 |
| Random, 40% Byzantine | 6.2 / 11 | 275.8 | 142.2 | 59.5 | 74.1 | 48.43 | 40.08 |
| Random, 50% Byzantine | 6.8 / 11 | 304.1 | 162.1 | 52.8 | 89.3 | 46.71 | 35.51 |

### 1.2 Converting FP32 bytes to round time (Gap 2, Figure 5b)

Figure 5b turns the `Model Poison` bar from Figure 5a into wall-clock seconds per benign peer per round by adding two parts.

1. **Gray bar (measured A100 compute, `0.660 s`).** On our A100-PCIE-40GB GPU during the 30% model-poisoning run, one benign peer takes a median of `0.602 s` for local SGD (range `0.587` to `1.223 s`, 25 rounds), `0.055 s` for the validation gradient (range `0.053` to `3.323 s`, peaking on round 1 cuDNN initialization), and `0.0032 s` for cosine scoring and aggregation (range `0.0029` to `0.134 s`). Across all 14 runs, total median compute stays between `0.647 s` and `0.669 s`.
2. **Blue and red bars (estimated FP32 transfer time from Panel a).** From the `Model Poison` bar in Figure 5a, each peer downloads `119.1 MB` of useful updates (blue) and `149.3 MB` of wasted updates (`64.4 MB` pink plus `84.8 MB` red, combined as the red bar in Panel b), totaling `268.4 MB` (`2.147` gigabits) per round. Dividing those bit counts by each link speed (`bytes x 8 / link_bps`) gives the stacked transfer times below.

| Link bandwidth (Model Poison, `268.4 MB` from Fig. 5a) | Measured A100 compute (s) | Est. useful transfer, `119.1 MB` (s) | Est. wasted transfer, `149.3 MB` (s) | Total round time (s) | Communication share (%) |
|---|---:|---:|---:|---:|---:|
| `10 Gb/s` (datacenter reference) | 0.660 | 0.095 | 0.119 | 0.875 | 24.5 |
| `1 Gb/s` Ethernet (our `octave` cluster NIC) | 0.660 | 0.953 | 1.194 | 2.807 | 76.5 |
| `100 Mb/s` (WAN or edge reference) | 0.660 | 9.528 | 11.941 | 22.129 | 97.0 |

Reading Figure 5b from left to right highlights two findings. At the `10 Gb/s` datacenter reference rate, compute dominates (`0.660 s` of `0.875 s`, `24.5%` communication). At our `octave` cluster's `1 Gb/s` Ethernet NIC rate, transfer time jumps to `2.147 s` (`76.5%` of the `2.807 s` round), which is `3.3x` slower than GPU compute. At the `100 Mb/s` WAN reference rate, transfer takes `21.469 s` (`97.0%` of the round). Eliminating the red wasted bar alone is not enough, because the blue useful bar alone (`0.953 s` at `1 Gb/s` and `9.528 s` at `100 Mb/s`) still exceeds all local A100 compute (`0.660 s`). Across the other topologies under 30% label flipping at our cluster's `1 Gb/s` rate, communication still takes `51.7%` of round time on Ring (`0.72 s` transfer vs. `0.67 s` compute) and `68.0%` on Small-World (`1.41 s` transfer vs. `0.66 s` compute).

---

## 2. Receive-buffer memory and repeated transfers (Figure 6)

Figure 6 measures the remaining two bottlenecks of batched, stateless TrustGraph-DFL. Panel (a) shows linear receive-buffer memory scaling across graph topologies, and Panel (b) compares multi-round rejection streaks across directed peer edges.

![Fig. 6 Receive-Buffer Memory Growth and Repeated-Transfer Behavior](../figures/fig6-memory-and-rejection-streaks-50k.png)

### 2.1 Receive-side memory scaling with peer degree (Gap 3, Figure 6a)

Batched TrustGraph-DFL holds every neighbor's `42.65 MiB` FP32 update in memory simultaneously before computing weights and averaging, so receive-buffer memory grows as $O(d)$ with peer degree $d$. Reading Figure 6a from left to right against the dashed `42.7 MiB` single-update reference line shows this linear growth under 30% label flipping.

1. **Ring (`deg = 2`).** Every peer has degree 2, requiring `85.3 MiB` (`2.0x` a single update) for both mean and max buffers.
2. **Small-World (`deg = 4` nominal).** Benign peers average `3.9` neighbors (`167.6 MiB` mean buffer), while the highest-degree benign peer has 7 neighbors (`298.6 MiB`, `7.0x`).
3. **Random (ER $p = 0.3$).** On the 30% Byzantine runs, benign peers average `6.0` neighbors (`255.9 MiB` mean) and reach degree 8 (`341.2 MiB`, `8.0x`). On the `none`, 10%, 20%, 40%, and 50% Byzantine Random runs, maximum benign degree reaches 11 (`469.2 MiB`, `11.0x`).

| Topology and slice | Mean benign degree | Max benign degree | Single update (MiB) | Mean receive buffer (MiB) | Max receive buffer (MiB) | Max / single ratio |
|---|---:|---:|---:|---:|---:|---:|
| Ring (30% label flip) | 2.0 | 2 | 42.7 | 85.3 | 85.3 | 2.0x |
| Small-World (30% label flip) | 3.9 | 7 | 42.7 | 167.6 | 298.6 | 7.0x |
| Random (30% label flip, model poison, noise) | 6.0 | 8 | 42.7 | 255.9 | 341.2 | 8.0x |
| Random (no attack, 10%, 20%, 40%, 50% Byzantine) | 6.2 to 6.8 | 11 | 42.7 | 263.0 to 290.1 | 469.2 | 11.0x |

ResNet-18 fits inside a `40 GiB` A100 GPU (`7.06` to `8.08 GiB` peak allocated GPU memory in our simulator, which stages all 20 peer shards and models on one GPU). Even so, the measured `8x` to `11x` multiplier exposes the scaling gap. Holding `d` full payloads simultaneously grows with both model size and graph degree, whereas a streaming receiver needs only constant buffer space independent of `d`.

### 2.2 Repeated downloads from blocked attackers vs. benign false positives (Gap 4, Figure 6b)

TrustGraph-DFL keeps no history per neighbor across rounds. Even after blocking an attacker in round $t$, the receiver downloads another full `44.73 MB` FP32 update from that same attacker in round $t+1$. Reading Figure 6b from left to right compares three shares of directed peer edges over 25 rounds on Random ER ($p=0.3$): attacker edges with `>=3` consecutive blocks (red), benign edges with `>=1` false positive (blue), and benign edges with `>=3` consecutive false positives (teal).

1. **Label Flip ($E_b=2$).** Attacker edges are blocked in `15.6` of 25 rounds on average, and `74%` (`73.7%`) hit at least 3 consecutive blocks (`556` repeat downloads wasting `24.87 GB`). Meanwhile, `98%` (`97.8%`) of benign edges hit at least one false positive, and `35%` (`34.8%`) hit a 3-round false-positive streak.
2. **Model Poison ($\lambda=10$).** Attacker edges are blocked in `17.5` of 25 rounds, and `95%` (`94.7%`) hit at least 3 consecutive blocks (`626` repeat downloads wasting `28.00 GB`). Every benign edge (`100%`) hits at least one false positive, and `52%` (`52.2%`) hit a 3-round streak.
3. **Random Noise ($\sigma=0.5$).** Every attacker edge (`100%`) is blocked in all `25.0` rounds (`912` repeat downloads wasting `40.79 GB`). Every benign edge (`100%`) hits at least one false positive, and `43%` (`43.5%`) hit a 3-round streak.

| Attack on Random (30% Byzantine, 25 rounds) | Attacker mean blocked rounds | Attacker mean max streak | Attacker edges with >=3 streak (%) | Attacker repeat downloads (GB wasted) | Benign edges with >=1 false positive (%) | Benign isolated single false positives | Benign edges with >=3 streak (%) | Benign edges with >=5 streak (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Label flip ($E_b=2$) | 15.6 / 25 | 7.3 | 73.7 | 556 (24.87 GB) | 97.8 | 282 | 34.8 | 17.4 |
| Model poison ($\lambda=10$) | 17.5 / 25 | 8.3 | 94.7 | 626 (28.00 GB) | 100.0 | 210 | 52.2 | 23.9 |
| Random noise ($\sigma=0.5$) | 25.0 / 25 | 25.0 | 100.0 | 912 (40.79 GB) | 100.0 | 282 | 43.5 | 21.7 |
| No attack (0% Byzantine) | 0.0 / 25 | 0.0 | 0.0 | 0 (0.00 GB) | 100.0 | 666 | 50.0 | 27.6 |

The contrast between the blue and red bars in Figure 6b defines the design constraint for Phase 2. Doing nothing lets chronic attackers force `556` to `912` repeat downloads (`24.9` to `40.8 GB`), while permanently banning a peer after a single false positive would cut `97.8%` to `100.0%` of benign edges (`210` to `666` isolated single-round false positives per run).

---

## 3. Framing for Phase 2

These Phase 1 measurements confirm all four system gaps on ResNet-18 before scaling to larger models. In Phase 2, we will build PeerSieve to reduce each measured gap while preserving Byzantine resilience.

1. **Pre-transfer sketch screening for Gap 1.** Because `42.5%` to `67.6%` of received FP32 bytes receive zero trust weight (Figure 5a), Phase 2 will screen compact CountSketch summaries against the receiver's validation descent direction before requesting full payloads.
2. **Blockwise INT4 compression with error feedback for Gap 2.** Because even useful FP32 updates take `0.953 s` at `1 Gb/s` and `9.528 s` at `100 Mb/s`, exceeding `0.660 s` of A100 compute (Figure 5b), Phase 2 will compress accepted payloads with blockwise 4-bit quantization and error feedback to shrink transfer time.
3. **Audited streaming aggregation for Gap 3.** Because batched receive buffers grow up to `8x` to `11x` a single update with peer degree (Figure 6a), Phase 2 will verify and accumulate one neighbor payload at a time in constant receive buffer space, preventing rejected payloads from modifying the running aggregate.
4. **Per-edge probation with recovery for Gap 4.** Because attacker edges trigger `556` to `912` repeat downloads (`24.9` to `40.8 GB`) while `97.8%` to `100%` of benign edges experience transient false positives (Figure 6b), Phase 2 will place repeatedly failing edges on temporary probation rather than permanent bans and allow recovery after a passing sketch check.

For exact reproduction and summary export commands, see [experiments/phase1/REPRODUCE.md](../../experiments/phase1/REPRODUCE.md).
