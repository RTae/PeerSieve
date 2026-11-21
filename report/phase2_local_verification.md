# Phase 2 Pre-GPU Verification Report — PeerSieve Core Mechanisms

**Date:** 2026-11-21 14:56  
**Git Commit (base):** `f7a0142`  
**Environment:** Python `3.12.12`, PyTorch `2.15.0`, `macOS-15.6.1-arm64-arm-64bit`  
**Scope:** All four pre-GPU verification checks from Proposal Table 2 (`src/sketch.py`, `src/compression.py`, `src/audit.py`, `src/quarantine.py`, `src/transport.py`).

---

## 1. CountSketch Screen & Tile-Size Invariance (`src/sketch.py`)

- **Model layout:** `8` LoRA layers, `28,928` trainable parameters, `16,384` buckets (`32 KiB` FP16).
- **Bit-exact tile-size invariance (`127`, `128`, `500`, `1000`, `4096`, `8192`, `65536`, full — dense & packed INT4):** **PASS**.
- **Attack-free accuracy across `40` backprop LoRA updates (starting tolerances $\tau_c = 0.05$, $\tau_n = 0.03$):**

| Metric | Median | 95th Percentile | Max | Starting Tolerance | Status |
|---|---:|---:|---:|---:|---|
| Cosine discrepancy $|\widehat c - c|$ | 0.00436 | 0.01831 | 0.02172 | 0.05000 | PASS |
| Norm relative discrepancy $|\|s\|_2/\|y\|_2 - 1|$ | 0.00331 | 0.00906 | 0.01398 | 0.03000 | PASS |

---

## 2. Blockwise INT4 Compression vs. Top-1% Sparsification (`src/compression.py`)

- **Multi-round residual conservation error** $\|\sum_{t=0}^{39} y^{(t)} + e^{(40)} - \sum \Delta w^{(t)}\|_2$: **`5.46e-06`**.
- **Wire compression vs. FP32 (`115,712` B):** Blockwise INT4 (`block_size=128`, FP16 scale) = **`14,948` B** (**`7.74×`** smaller).

| Compressor | Wire Bytes | Median $|\cos(y, g^{\mathrm{val}}) - \cos(\Delta w, g^{\mathrm{val}})|$ | Max Alignment Error | Alignment Sign Flips |
|---|---:|---:|---:|---:|
| **Blockwise INT4 + EF (`block_size=128`)** | 14,948 | **0.00096** | **0.00525** | **0/40** |
| Top-1% Sparse + EF | 2,312 | 0.19513 | 0.25299 | 0/40 |

---

## 3. Two-Pass Audited Streaming Aggregation (`src/audit.py`)

- **Zero corruption on failed/skipped payloads:** **PASS** (`running_sum` and `total_weight` bit-for-bit identical before and after every rejected payload).
- **Small honest update (`0.5× r_i`) left unclipped:** `clip_factor_applied = 1.000000` (unlike FLTrust, updates smaller than $r_i$ are not rescaled up).

### 3.1 Rejection & Fault-Classification Checks

| Test Case | Outcome | Reason | `running_sum` & `total_weight` Unchanged? |
|---|---|---|---|
| `tampered_payload_bytes` | `hard_fault` | `sketch_mismatch` | YES |
| `sender_id_spoofing` | `hard_fault` | `sender_mismatch` | YES |
| `reserved_nibble_15` | `hard_fault` | `invalid_nibble` | YES |
| `oversized_wire_payload` | `hard_fault` | `size_limit_exceeded` | YES |
| `sketch_kernel_collision_norm_inflation` | `soft_fault` | `cosine_discrepancy` | YES |
| `zero_payload` | `zero_skip` | `zero_payload` | YES |

### 3.2 Receiver-Norm Cap Under Positive Rescaling

| Multiplier | Receiver Norm $r_i$ | Decoded Norm $\|y_j\|_2$ | Clip Factor $a_{ij}$ | Clipped Contribution Norm | Unclipped Ablation Norm |
|---:|---:|---:|---:|---:|---:|
| `0.5×` | 1.94838 | 0.98178 | 1.00000 | **0.98178** | 0.98178 |
| `10×` | 1.94838 | 19.63731 | 0.09922 | **1.94838** | 19.63731 |
| `100×` | 1.94838 | 196.34622 | 0.00992 | **1.94838** | 196.34622 |

### 3.3 Receive-Side Buffer Memory Projection (Rank-128 LoRA, 40M Params, 6 Neighbors)

> **Note (Projection, not GPU measurement):** Counts receive-side tensor buffers only (excludes training weights/optimizer states and sender quantization residual).

- **Batched FP32 TrustGraph-DFL (`6` neighbor buffers + `val_dir`):** `1068.1 MiB`
- **PeerSieve Audited Streaming (`running_sum` + `val_dir` + `4 MiB` decode tile + `32 KiB` sketch + `1` staged INT4 payload):** `328.9 MiB` (**`30.8%`** of batched FP32; constant in neighbor degree).

---

## 4. Multi-Process `torch.distributed` (Gloo) Quarantine & Referral Verification (`src/quarantine.py`, `src/transport.py`)

- **Execution:** 4 separate OS processes over `torch.distributed` Gloo (`12` synchronized rounds, wall time `1.22s`).
- **Banned peer reinstatement check (`can_consider_referral(2)`):** `False` (permanently banned after round-0 `sketch_mismatch`).
- **Total received bytes at Rank 0 across 12 rounds:** `806,848` B (sketches + selective INT4 payloads), vs. `1,572,864` B for unscreened FP32 across 2 neighbors.

| Round | Sketches Rx | Payloads Req | Accepted | Hard Faults | Active | Probation | Banned | Referrals Admitted |
|---:|---:|---:|---:|---:|---|---|---|---|
| 0 | 3 | 2 | 1 | 1 | `[1, 3]` | `[]` | `[2]` | `[3]` |
| 1 | 2 | 1 | 1 | 0 | `[1, 3]` | `[]` | `[2]` | `[]` |
| 2 | 2 | 1 | 1 | 0 | `[1, 3]` | `[]` | `[2]` | `[]` |
| 3 | 2 | 1 | 1 | 0 | `[1, 3]` | `[]` | `[2]` | `[]` |
| 4 | 2 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 5 | 1 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 6 | 2 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 7 | 1 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 8 | 1 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 9 | 1 | 1 | 1 | 0 | `[1]` | `[3]` | `[2]` | `[]` |
| 10 | 2 | 1 | 1 | 0 | `[1, 3]` | `[]` | `[2]` | `[]` |
| 11 | 2 | 2 | 2 | 0 | `[1, 3]` | `[]` | `[2]` | `[]` |
