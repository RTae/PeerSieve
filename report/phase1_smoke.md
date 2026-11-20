# Phase 1 Local Smoke Test Report (TrustGraph-DFL CIFAR-10)

**Date:** 2026-11-20  
**Host:** Local Apple Silicon (`arm64`, CPU execution via `uv` / PyTorch 2.15.0)  
**Target paper:** Xu et al., *TrustGraph-DFL: Byzantine-Resilient Decentralized Federated Learning via Consistency-Weighted Neighborhood Aggregation* (CIBDA 2026, `TrustGraph-DFL.pdf`)

## 1. Unit Tests

All 9 unit and offline integration tests pass (`uv run pytest -v`, `22.78s`):

- Balanced Dirichlet ($\alpha_{\mathrm{dir}} = 0.5$) partitioning: exact 500 samples/node (450 train + 50 val), 1,000 samples/class, verified non-IID class entropy.
- Label-flipping (`0 <-> 1`), model poisoning ($\lambda = 10$), and Gaussian noise ($\sigma = 0.5$).
- Connected topologies (`ring`, `random` ER $p=0.3$, `small_world` WS $k=4, p=0.3$) and Ring degree-2 edge cases.
- All 5 aggregators (`fedavg`, `trimmed_mean`, `krum`, `balance`, `trustgraph`), including exact all-negative self-fallback ($\theta_i^{t+1} = \theta_i^{t+1/2}$) and gradient-sign inversion check.
- Regression check confirming `vector_to_model` copies parameters without aliasing `prev_models[i]`.

## 2. Real CIFAR-10 Smoke Suite

**Configuration:**
- Dataset: CIFAR-10 (10,000 subsampled train images across $n=20$ peers, 500/peer, $\alpha_{\mathrm{dir}}=0.5$, 10% local validation split; 10,000 clean test images).
- Model: 2-conv, 2-FC CNN (`SimpleCIFARCNN`, `1,070,794` parameters).
- Optimization: SGD ($\eta=0.01$, momentum $0.9$, weight decay $10^{-4}$, batch size $64$, 1 local epoch/round).
- Topology: Erdős–Rényi random graph ($p=0.3$, seed `42`), 30% Byzantine (`6/20` peers).
- Horizon: 5 rounds (`eval_every=0`, evaluating final round on all 14 benign peers + benign consensus model).
- Raw artifacts: `outputs/phase1_smoke/`.

| Topology | Attack | Byz% | Method | Seeds | Final Acc (%) | Cons Acc (%) | FPR (%) | Byz Accept (%) | Cos(Ben/Byz) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| random | label_flip | 30 | balance | 1 | 16.80 | 23.19 | 0.00 | 100.00 | n/a |
| random | label_flip | 30 | fedavg | 1 | 17.09 | 22.79 | 0.00 | 100.00 | n/a |
| random | label_flip | 30 | krum | 1 | 16.01 | 21.21 | 17.83 | 65.79 | n/a |
| random | label_flip | 30 | trimmed_mean | 1 | 16.40 | 19.87 | 44.59 | 48.57 | n/a |
| random | label_flip | 30 | trustgraph(descent_neg) | 1 | 14.04 | 17.59 | 45.65 | 45.26 | +0.027 / -0.030 |
| random | label_flip | 30 | trustgraph(paper_pos) | 1 | 17.02 | 21.87 | 54.35 | 54.21 | -0.012 / +0.055 |
| random | model_poison | 30 | balance | 1 | 9.24 | 4.24 | 0.00 | 100.00 | n/a |
| random | model_poison | 30 | fedavg | 1 | 10.00 | 10.00 | 0.00 | 100.00 | n/a |
| random | model_poison | 30 | krum | 1 | 9.55 | 10.00 | 21.30 | 66.84 | n/a |
| random | model_poison | 30 | trimmed_mean | 1 | 10.18 | 10.00 | 47.76 | 53.96 | n/a |
| random | model_poison | 30 | trustgraph(descent_neg) | 1 | 14.86 | 15.23 | 61.74 | 16.32 | -0.063 / -0.126 |
| random | model_poison | 30 | trustgraph(paper_pos) | 1 | 6.42 | 4.92 | 54.78 | 93.68 | -0.034 / +0.212 |

## 3. Diagnostic Findings Before Full Cluster Runs

1. **Preprint gradient sign bug confirmed on CIFAR-10:**  
   Under `model_poison` ($\lambda=10$), `trustgraph(paper_pos)` (literal $+\nabla F_i$ from Eq. 3 / Alg. 1) assigns positive cosine to Byzantine updates (`+0.212`) and negative cosine to honest updates (`-0.034`), accepting `93.68%` of Byzantine updates and collapsing below chance (`6.42%` mean / `4.92%` consensus). Flipping to the descent direction `trustgraph(descent_neg)` ($-\nabla F_i$) drops Byzantine acceptance to `16.32%` and is the only configuration that rises above chance (`14.86%` mean / `15.23%` consensus) in 5 rounds.

2. **Non-IID local validation split drives high honest rejection (FPR):**  
   A round-0 diagnostic on CIFAR-10 (`n=20`, `500 samples/node`, $\alpha_{\mathrm{dir}}=0.5$) measured cosine alignment between honest updates and $-\nabla F_i(\theta^0; \mathcal{D}_i^{\mathrm{val}})$:
   - `val_split=local` (literal paper: 10% held out from each peer's non-IID shard): self-cosine `+0.8214`, cross-peer honest cosine `-0.0062`, **53.33% of honest cross-peer pairs are already negative at round 0**.
   - `val_split=iid` (diagnostic: stratified balanced 50-sample validation set per peer): self-cosine `+0.1158`, cross-peer honest cosine **`+0.0877`**, negative cross-peer fraction drops to **23.33%**.
   - Under `alpha_dir=100.0` (near-IID): cross-peer honest cosine `+0.3289`, only **3.33%** negative.
   
   **Action for cluster run:** Keep `val_split=local` as the primary reproduction of the paper, and run a targeted `val_split=iid` diagnostic to isolate the effect of validation-set skew on FPR.

3. **BALANCE threshold calibration (`balance_gamma`) and local Byzantine majority:**  
   At round 0 on `SimpleCIFARCNN` ($\|\theta^0\|_2 = 11.05$), relative distances $\|\theta_j^{1/2}-\theta_i^{1/2}\|_2 / \|\theta_i^{1/2}\|_2$ are:
   - Benign neighbors: `0.056 – 0.073` (mean `0.065`)
   - `model_poison` ($\lambda = 10$): `0.256`
   - `random_noise` ($\sigma = 0.5$): `46.84`
   
   With `balance_gamma = 0.3`, early-round BALANCE accepted `model_poison` (`0.256 <= 0.30`). On the cluster, we should test both `balance_gamma = 0.3` and a calibrated `balance_gamma = 0.15` so BALANCE is evaluated fairly.  
   Also note that on Erdős–Rényi $G(20, 0.3)$ at 30% global Byzantine nodes (seed `42`), 5 of 14 benign peers have $\ge 50\%$ Byzantine neighbors (up to 5/8), which explains why colluding identical-vector `model_poison` defeats local Krum on those nodes.

4. **CPU timing baseline (motivation for cluster GPU run):**  
   On local CPU, 1 round of 20 peers takes `~5.0s` and 1 evaluation of all 14 benign models + consensus on 10,000 test images takes `~27s` (`~52s` per 5-round run). A 200-round run on CPU would take `~25–30 min` each; on an `octave` A100 GPU, 200 rounds should take `~1–2 min` per run.
