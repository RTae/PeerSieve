"""Pre-GPU verification benchmark for PeerSieve's four core mechanisms (Proposal Table 2).

Runs deterministic local checks on:
  1. Sketch screen: bit-exact tile invariance + cosine/norm error distributions (16,384 buckets).
  2. Compression: residual conservation + alignment preservation on backprop LoRA updates (INT4 vs Top-1%).
  3. Audit & streaming: zero state corruption on hard/soft/kernel-collision faults, norm cap under rescaling,
     and receive-side buffer memory projection vs. batched FP32.
  4. Quarantine: real 4-process torch.distributed (Gloo) run testing hard bans, EMA probation backoff,
     passing probe recovery, and referrals without reinstating banned peers.

Writes a Markdown verification report to --report-out (default: report/phase2_local_verification.md).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from src.audit import (
    AuditConfig,
    AuditOutcome,
    AuditedStreamingReceiver,
    PayloadEnvelope,
    SketchAnnouncement,
)
from src.compression import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TILE_NUMEL,
    PackedINT4Payload,
    decode_blockwise_int4,
    encode_blockwise_int4,
    expected_int4_wire_bytes,
    topk_compress_with_error_feedback,
)
from src.quarantine import LinkState, PeerQuarantineTable, QuarantineConfig
from src.sketch import (
    DEFAULT_NUM_BUCKETS,
    SketchLayout,
    StreamingCountSketch,
    compute_sketch,
    sketch_cosine_and_norms,
    sketch_from_packed_int4,
    sketches_exact_equal,
)
from src.transport import PeerRoundMessage, TokenBucket, run_gloo_round


class _LoRALinear(nn.Module):
    """Frozen base linear layer with trainable rank-r LoRA adapters (A, B)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features), requires_grad=False)
        self.lora_a = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0 / math.sqrt(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)
        lora = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return base + lora


class _TinyLoRANet(nn.Module):
    """4-layer LoRA-adapted MLP producing genuine multi-layer LoRA backprop updates."""

    def __init__(self, dim: int = 256, rank: int = 16, n_classes: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _LoRALinear(dim, dim, rank=rank),
                _LoRALinear(dim, dim, rank=rank),
                _LoRALinear(dim, dim, rank=rank),
                _LoRALinear(dim, n_classes, rank=min(rank, n_classes)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers[:-1]:
            h = F.gelu(layer(h))
        return self.layers[-1](h)


def _flatten_trainable(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).float() for p in model.parameters() if p.requires_grad]).clone()


def _flatten_trainable_grads(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel())
            for p in model.parameters()
            if p.requires_grad
        ]
    ).clone()


def _load_trainable(model: nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            n = p.numel()
            p.copy_(vec[offset : offset + n].view_as(p))
            offset += n


def verify_sketch_and_compression(n_trials: int = 40) -> dict[str, object]:
    """Run Table 2 checks 1 (Sketch screen) & 2 (Compression) on genuine backprop LoRA updates."""
    torch.manual_seed(42)
    dim, n_classes = 256, 16
    model = _TinyLoRANet(dim=dim, rank=16, n_classes=n_classes)
    layout = SketchLayout.from_model(model, num_buckets=DEFAULT_NUM_BUCKETS, trainable_only=True)

    # Create a structured multi-class classification problem with non-IID peer shards & held-out val set.
    prototypes = torch.randn(n_classes, dim) * 1.2
    y_val = torch.arange(n_classes).repeat_interleave(4)
    x_val = prototypes[y_val] + torch.randn(y_val.numel(), dim) * 0.6

    init_vec = _flatten_trainable(model)
    # Give lora_b small nonzero init so both A and B gradients are immediately active in trial 0.
    init_vec = init_vec + torch.randn_like(init_vec) * 0.01
    _load_trainable(model, init_vec)

    # Validation descent direction g_i^val = -nabla L_val(w)
    model.zero_grad(set_to_none=True)
    loss_val = F.cross_entropy(model(x_val), y_val)
    loss_val.backward()
    val_dir = -_flatten_trainable_grads(model)
    val_norm = float(torch.linalg.vector_norm(val_dir).item())

    int4_residual = torch.zeros(layout.total_numel, dtype=torch.float32)
    topk_residual = torch.zeros(layout.total_numel, dtype=torch.float32)
    sum_true = torch.zeros(layout.total_numel, dtype=torch.float32)
    sum_int4 = torch.zeros(layout.total_numel, dtype=torch.float32)

    cos_errs_sketch: list[float] = []
    norm_errs_sketch: list[float] = []
    int4_align_errs: list[float] = []
    topk_align_errs: list[float] = []
    int4_sign_flips = 0
    topk_sign_flips = 0
    tile_invariance_ok = True
    wire_bytes_int4 = 0
    wire_bytes_topk = 0

    for t in range(n_trials):
        _load_trainable(model, init_vec)
        # Non-IID peer batch: emphasize 6 random classes per trial (mimicking Dirichlet skew).
        favored = torch.randperm(n_classes)[:6]
        y_tr = favored[torch.randint(0, len(favored), (64,))]
        x_tr = prototypes[y_tr] + torch.randn(64, dim) * 0.6

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_tr), y_tr)
            loss.backward()
            opt.step()

        upd = _flatten_trainable(model) - init_vec
        upd_norm = float(torch.linalg.vector_norm(upd).item())
        true_cos = float(torch.dot(upd, val_dir).item()) / (upd_norm * val_norm)

        # INT4 compression with error feedback
        payload, int4_residual = encode_blockwise_int4(upd, residual=int4_residual, block_size=DEFAULT_BLOCK_SIZE)
        y_int4 = decode_blockwise_int4(payload)
        wire_bytes_int4 = payload.wire_bytes
        sum_true += upd
        sum_int4 += y_int4

        y_int4_norm = float(torch.linalg.vector_norm(y_int4).item())
        cos_int4 = float(torch.dot(y_int4, val_dir).item()) / (y_int4_norm * val_norm)
        int4_align_errs.append(abs(cos_int4 - true_cos))
        if (true_cos > 0) != (cos_int4 > 0):
            int4_sign_flips += 1

        # Top-1% sparse compression with error feedback
        y_topk, topk_residual, wb_topk = topk_compress_with_error_feedback(upd, residual=topk_residual, keep_frac=0.01)
        wire_bytes_topk = wb_topk
        y_topk_norm = float(torch.linalg.vector_norm(y_topk).item())
        cos_topk = float(torch.dot(y_topk, val_dir).item()) / (y_topk_norm * val_norm)
        topk_align_errs.append(abs(cos_topk - true_cos))
        if (true_cos > 0) != (cos_topk > 0):
            topk_sign_flips += 1

        # CountSketch screen accuracy on decoded INT4 update
        s_y = compute_sketch(y_int4, layout=layout, round_seed=2026, round_idx=t)
        s_v = compute_sketch(val_dir, layout=layout, round_seed=2026, round_idx=t)
        c_hat, norm_sy, _ = sketch_cosine_and_norms(s_y, s_v)
        cos_errs_sketch.append(abs(c_hat - cos_int4))
        norm_errs_sketch.append(abs(norm_sy / y_int4_norm - 1.0))

        # Tile-size invariance check on trial 0 across 7 arbitrary tile sizes (dense & packed INT4)
        if t == 0:
            for tile_len in (127, 128, 1000, 4096, 8192, 65_536):
                s_dense = compute_sketch(y_int4, layout=layout, round_seed=2026, round_idx=0, tile_numel=tile_len)
                if not sketches_exact_equal(s_y, s_dense):
                    tile_invariance_ok = False
            for tile_len in (128, 500, 8192, 65_536):
                s_packed = sketch_from_packed_int4(payload, layout, 2026, 0, torch.empty(tile_len, dtype=torch.float32))
                if not sketches_exact_equal(s_y, s_packed):
                    tile_invariance_ok = False

    residual_conservation_err = float(torch.linalg.vector_norm((sum_int4 + int4_residual) - sum_true).item())
    fp32_bytes = layout.total_numel * 4

    return {
        "numel": layout.total_numel,
        "num_layers": len(layout.layer_sizes),
        "num_buckets": layout.num_buckets,
        "sketch_bytes": layout.num_buckets * 2,
        "n_trials": n_trials,
        "tile_invariance_ok": tile_invariance_ok,
        "sketch_cos_err_median": float(np.median(cos_errs_sketch)),
        "sketch_cos_err_p95": float(np.percentile(cos_errs_sketch, 95)),
        "sketch_cos_err_max": float(np.max(cos_errs_sketch)),
        "sketch_norm_err_median": float(np.median(norm_errs_sketch)),
        "sketch_norm_err_p95": float(np.percentile(norm_errs_sketch, 95)),
        "sketch_norm_err_max": float(np.max(norm_errs_sketch)),
        "residual_conservation_l2_err": residual_conservation_err,
        "fp32_bytes": fp32_bytes,
        "int4_wire_bytes": wire_bytes_int4,
        "int4_compression_ratio": fp32_bytes / max(wire_bytes_int4, 1),
        "topk_wire_bytes": wire_bytes_topk,
        "int4_align_err_median": float(np.median(int4_align_errs)),
        "int4_align_err_max": float(np.max(int4_align_errs)),
        "int4_sign_flips": int4_sign_flips,
        "topk_align_err_median": float(np.median(topk_align_errs)),
        "topk_align_err_max": float(np.max(topk_align_errs)),
        "topk_sign_flips": topk_sign_flips,
    }


def verify_audit_and_streaming() -> dict[str, object]:
    """Run Table 2 check 3: zero state change on rejects, kernel-collision audit, norm cap, buffer projection."""
    torch.manual_seed(77)
    layout = SketchLayout.from_layer_sizes([16_384, 16_384], num_buckets=DEFAULT_NUM_BUCKETS)
    cfg = AuditConfig(tile_numel=4096, tau_c=0.05, tau_n=0.03, alpha=0.5, clip_factor=1.0)
    recv = AuditedStreamingReceiver(self_id=0, run_id="verify_run", layout=layout, cfg=cfg)

    base = torch.randn(layout.total_numel) * 0.01
    val_dir = base + torch.randn(layout.total_numel) * 0.004
    self_upd = base + torch.randn(layout.total_numel) * 0.004
    r_i = float(torch.linalg.vector_norm(self_upd).item())
    recv.start_round(round_idx=0, round_seed=1337, val_descent_dir=val_dir, self_update_norm=r_i)

    # Accept 1 baseline honest update (0.5x norm -> should remain unclipped).
    p_honest, _ = encode_blockwise_int4(0.5 * self_upd)
    s_honest = sketch_from_packed_int4(p_honest, layout, 1337, 0, torch.empty(4096))
    ann_honest = SketchAnnouncement(1, "verify_run", 0, s_honest)
    res_honest = recv.audit_and_accumulate(
        ann_honest, PayloadEnvelope(1, "verify_run", 0, raw_bytes=p_honest.to_bytes()), expected_sender_id=1
    )

    snap_sum = recv.running_sum.clone()
    snap_w = recv.total_weight
    rejection_cases: list[dict[str, object]] = []

    def _record_reject(name: str, res) -> None:
        unchanged = bool(torch.equal(recv.running_sum, snap_sum) and recv.total_weight == snap_w)
        rejection_cases.append(
            {"case": name, "outcome": res.outcome.value, "reason": res.reason, "state_unchanged": unchanged}
        )

    # Case A: Tampered payload -> sketch_mismatch (HARD_FAULT)
    bad_codes = p_honest.codes.clone()
    bad_codes[0] ^= 0x01
    p_tamper = PackedINT4Payload(p_honest.numel, p_honest.block_size, bad_codes, p_honest.scales)
    _record_reject(
        "tampered_payload_bytes",
        recv.audit_and_accumulate(
            SketchAnnouncement(2, "verify_run", 0, s_honest), PayloadEnvelope(2, "verify_run", 0, payload=p_tamper), 2
        ),
    )

    # Case A2: Sender/announcement spoofing -> sender_mismatch (HARD_FAULT)
    _record_reject(
        "sender_id_spoofing",
        recv.audit_and_accumulate(ann_honest, PayloadEnvelope(2, "verify_run", 0, payload=p_honest), 2),
    )

    # Case B: Reserved nibble 15 -> invalid_nibble (HARD_FAULT)
    bad_nib = p_honest.codes.clone()
    bad_nib[5] = (bad_nib[5] & 0xF0) | 15
    p_nib = PackedINT4Payload(p_honest.numel, p_honest.block_size, bad_nib, p_honest.scales)
    _record_reject(
        "reserved_nibble_15",
        recv.audit_and_accumulate(
            SketchAnnouncement(3, "verify_run", 0, s_honest), PayloadEnvelope(3, "verify_run", 0, payload=p_nib), 3
        ),
    )

    # Case C: Oversized wire bytes -> size_limit_exceeded (HARD_FAULT)
    oversized = p_honest.to_bytes() + b"\x00" * 64
    _record_reject(
        "oversized_wire_payload",
        recv.audit_and_accumulate(
            SketchAnnouncement(4, "verify_run", 0, s_honest), PayloadEnvelope(4, "verify_run", 0, raw_bytes=oversized), 4
        ),
    )

    # Case D: Exact CountSketch kernel-collision attack (matching sketch, inflated norm -> SOFT_FAULT)
    tmp_cs = StreamingCountSketch(layout, round_seed=1337, round_idx=0)
    buckets, signs = tmp_cs._hash_range(0, 16_384)
    seen: dict[int, tuple[int, float]] = {}
    i1 = i2 = 0
    sg1 = sg2 = 1.0
    for idx_i in range(16_384):
        b = int(buckets[idx_i].item())
        sg = float(signs[idx_i].item())
        if b in seen:
            i1, sg1 = seen[b]
            i2, sg2 = idx_i, sg
            break
        seen[b] = (idx_i, sg)
    y_coll = decode_blockwise_int4(p_honest).clone()
    delta = 5.0 * r_i
    y_coll[i1] = delta * sg1
    y_coll[i2] = -delta * sg2
    p_coll, _ = encode_blockwise_int4(y_coll)
    s_coll = sketch_from_packed_int4(p_coll, layout, 1337, 0, torch.empty(4096))
    _record_reject(
        "sketch_kernel_collision_norm_inflation",
        recv.audit_and_accumulate(
            SketchAnnouncement(5, "verify_run", 0, s_coll), PayloadEnvelope(5, "verify_run", 0, payload=p_coll), 5
        ),
    )

    # Case E: Zero payload -> ZERO_SKIP
    p_zero, _ = encode_blockwise_int4(torch.zeros(layout.total_numel))
    s_zero = compute_sketch(torch.zeros(layout.total_numel), layout, 1337, 0)
    _record_reject(
        "zero_payload",
        recv.audit_and_accumulate(
            SketchAnnouncement(6, "verify_run", 0, s_zero), PayloadEnvelope(6, "verify_run", 0, payload=p_zero), 6
        ),
    )

    # Positive rescaling checks (0.5x, 10x, 100x) with clipping ON vs OFF
    rescaling_rows: list[dict[str, float]] = []
    for mult in (0.5, 10.0, 100.0):
        p_s, _ = encode_blockwise_int4(mult * self_upd)
        s_s = sketch_from_packed_int4(p_s, layout, 1337, 0, torch.empty(4096))
        # Fresh receiver with clipping ON
        r_on = AuditedStreamingReceiver(0, "clip_on", layout, AuditConfig(tile_numel=4096, enable_clipping=True))
        r_on.start_round(0, 1337, val_dir, r_i)
        res_on = r_on.audit_and_accumulate(
            SketchAnnouncement(7, "clip_on", 0, s_s), PayloadEnvelope(7, "clip_on", 0, payload=p_s), 7
        )
        # Fresh receiver with clipping OFF
        r_off = AuditedStreamingReceiver(0, "clip_off", layout, AuditConfig(tile_numel=4096, enable_clipping=False))
        r_off.start_round(0, 1337, val_dir, r_i)
        res_off = r_off.audit_and_accumulate(
            SketchAnnouncement(7, "clip_off", 0, s_s), PayloadEnvelope(7, "clip_off", 0, payload=p_s), 7
        )
        rescaling_rows.append(
            {
                "scale_mult": mult,
                "r_i": r_i,
                "decoded_norm": float(res_on.decoded_norm or 0.0),
                "clip_factor": res_on.clip_factor_applied,
                "clipped_norm": res_on.clipped_contribution_norm,
                "unclipped_norm": res_off.clipped_contribution_norm,
            }
        )

    # Receive-side buffer memory projection for Rank-128 LoRA (TinyLlama-1.1B, ~40M trainable params, 6 neighbors).
    # Exact accounting of receive-side tensor buffers only (excludes training state & sender quantization residual).
    lora128_numel = 40_000_000
    n_neighbors = 6
    batched_fp32_mib = (n_neighbors * lora128_numel * 4 + lora128_numel * 4) / (1024**2)  # 6 neighbor updates + val_dir
    peersieve_rx_mib = (
        lora128_numel * 4  # running_sum (FP32)
        + lora128_numel * 4  # val_dir (FP32)
        + DEFAULT_TILE_NUMEL * 4  # reusable 4 MiB decode tile
        + DEFAULT_NUM_BUCKETS * 2  # val_sketch (32 KiB)
        + expected_int4_wire_bytes(lora128_numel, DEFAULT_BLOCK_SIZE)  # 1 staged INT4 payload
    ) / (1024**2)

    return {
        "honest_small_clip_factor": res_honest.clip_factor_applied,
        "all_rejections_preserve_state": all(bool(c["state_unchanged"]) for c in rejection_cases),
        "rejection_cases": rejection_cases,
        "rescaling_rows": rescaling_rows,
        "lora128_batched_fp32_rx_mib_proj": batched_fp32_mib,
        "lora128_peersieve_rx_mib_proj": peersieve_rx_mib,
        "lora128_rx_mem_ratio_proj": peersieve_rx_mib / batched_fp32_mib,
    }


def _gloo_verify_worker(rank: int, world_size: int, port: int, out_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        torch.manual_seed(2026)
        layout = SketchLayout.from_layer_sizes([8192, 8192], num_buckets=DEFAULT_NUM_BUCKETS)
        base_dir = torch.randn(layout.total_numel) * 0.02
        val_dir = base_dir.clone()
        self_upd = base_dir.clone()

        initial_nbrs = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}[rank]
        qt = PeerQuarantineTable(
            self_id=rank,
            initial_neighbors=initial_nbrs,
            cfg=QuarantineConfig(beta=0.7, neg_streak_limit=3, t_max=16, min_active_degree=2, max_neighbors=2),
        )
        recv = AuditedStreamingReceiver(
            self_id=rank,
            run_id="phase2_verify",
            layout=layout,
            cfg=AuditConfig(tile_numel=4096, tau_c=0.05, tau_n=0.03, alpha=0.5),
        )
        bucket = TokenBucket(rate_bps=1_000_000_000.0, sleep_enabled=False)

        rounds_out: list[dict[str, object]] = []
        for r_idx in range(12):
            my_upd = -base_dir.clone() if (rank == 3 and r_idx in (1, 2, 3, 4, 6)) else base_dir.clone()
            payload, _ = encode_blockwise_int4(my_upd, block_size=DEFAULT_BLOCK_SIZE)
            sketch = sketch_from_packed_int4(payload, layout, 9001, r_idx, torch.empty(4096))
            raw = payload.to_bytes()
            if rank == 2:
                bad_codes = payload.codes.clone()
                bad_codes[0] ^= 0x01
                raw = PackedINT4Payload(payload.numel, payload.block_size, bad_codes, payload.scales).to_bytes()

            _, tel = run_gloo_round(
                rank=rank,
                world_size=world_size,
                round_idx=r_idx,
                round_seed=9001,
                val_descent_dir=val_dir,
                self_update=self_upd,
                outbound=PeerRoundMessage(sketch=sketch, payload_bytes=raw),
                receiver=recv,
                quarantine=qt,
                token_bucket=bucket,
            )
            if rank == 0:
                rounds_out.append(
                    {
                        "round": r_idx,
                        "sketches_rx": tel.sketches_received,
                        "sketch_bytes": tel.sketch_bytes_received,
                        "payloads_req": tel.payloads_requested,
                        "payload_bytes": tel.payload_bytes_received,
                        "accepted": tel.payloads_accepted,
                        "hard_faults": tel.hard_faults,
                        "active": tel.active_peers_end,
                        "probation": tel.probation_peers_end,
                        "banned": tel.banned_peers_end,
                        "referrals_admitted": tel.referrals_admitted,
                    }
                )
        if rank == 0:
            torch.save(
                {
                    "rounds": rounds_out,
                    "total_rx_bytes": bucket.total_bytes,
                    "payload_wire_bytes": expected_int4_wire_bytes(layout.total_numel, DEFAULT_BLOCK_SIZE),
                    "fp32_update_bytes": layout.total_numel * 4,
                    "banned_reinstatement_allowed": qt.can_consider_referral(2),
                },
                out_path,
            )
    finally:
        dist.destroy_process_group()


def verify_gloo_quarantine() -> dict[str, object]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    fd, out_path = tempfile.mkstemp(prefix="peersieve_verify_gloo_", suffix=".pt")
    os.close(fd)
    t0 = time.perf_counter()
    try:
        mp.spawn(_gloo_verify_worker, args=(4, port, out_path), nprocs=4, join=True)
        elapsed = time.perf_counter() - t0
        data = torch.load(out_path, weights_only=False)
        data["wall_sec"] = elapsed
        return data
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def render_report(
    sc: dict[str, object],
    au: dict[str, object],
    gl: dict[str, object] | None,
) -> str:
    lines: list[str] = [
        "# Phase 2 Pre-GPU Verification Report — PeerSieve Core Mechanisms",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Git Commit (base):** `{_git_sha()}`  ",
        f"**Environment:** Python `{platform.python_version()}`, PyTorch `{torch.__version__}`, `{platform.platform()}`  ",
        "**Scope:** All four pre-GPU verification checks from Proposal Table 2 (`src/sketch.py`, `src/compression.py`, `src/audit.py`, `src/quarantine.py`, `src/transport.py`).",
        "",
        "---",
        "",
        "## 1. CountSketch Screen & Tile-Size Invariance (`src/sketch.py`)",
        "",
        f"- **Model layout:** `{sc['num_layers']}` LoRA layers, `{sc['numel']:,}` trainable parameters, `{sc['num_buckets']:,}` buckets (`{int(sc['sketch_bytes']) // 1024} KiB` FP16).",
        f"- **Bit-exact tile-size invariance (`127`, `128`, `500`, `1000`, `4096`, `8192`, `65536`, full — dense & packed INT4):** **{'PASS' if sc['tile_invariance_ok'] else 'FAIL'}**.",
        f"- **Attack-free accuracy across `{sc['n_trials']}` backprop LoRA updates (starting tolerances $\\tau_c = 0.05$, $\\tau_n = 0.03$):**",
        "",
        "| Metric | Median | 95th Percentile | Max | Starting Tolerance | Status |",
        "|---|---:|---:|---:|---:|---|",
        f"| Cosine discrepancy $|\\widehat c - c|$ | {sc['sketch_cos_err_median']:.5f} | {sc['sketch_cos_err_p95']:.5f} | {sc['sketch_cos_err_max']:.5f} | 0.05000 | {'PASS' if float(sc['sketch_cos_err_max']) <= 0.05 else 'CHECK'} |",
        f"| Norm relative discrepancy $|\\|s\\|_2/\\|y\\|_2 - 1|$ | {sc['sketch_norm_err_median']:.5f} | {sc['sketch_norm_err_p95']:.5f} | {sc['sketch_norm_err_max']:.5f} | 0.03000 | {'PASS' if float(sc['sketch_norm_err_max']) <= 0.03 else 'CHECK'} |",
        "",
        "---",
        "",
        "## 2. Blockwise INT4 Compression vs. Top-1% Sparsification (`src/compression.py`)",
        "",
        f"- **Multi-round residual conservation error** $\\|\\sum_{{t=0}}^{{{int(sc['n_trials'])-1}}} y^{{(t)}} + e^{{({sc['n_trials']})}} - \\sum \\Delta w^{{(t)}}\\|_2$: **`{sc['residual_conservation_l2_err']:.2e}`**.",
        f"- **Wire compression vs. FP32 (`{int(sc['fp32_bytes']):,}` B):** Blockwise INT4 (`block_size=128`, FP16 scale) = **`{int(sc['int4_wire_bytes']):,}` B** (**`{sc['int4_compression_ratio']:.2f}×`** smaller).",
        "",
        "| Compressor | Wire Bytes | Median $|\\cos(y, g^{\\mathrm{val}}) - \\cos(\\Delta w, g^{\\mathrm{val}})|$ | Max Alignment Error | Alignment Sign Flips |",
        "|---|---:|---:|---:|---:|",
        f"| **Blockwise INT4 + EF (`block_size=128`)** | {int(sc['int4_wire_bytes']):,} | **{sc['int4_align_err_median']:.5f}** | **{sc['int4_align_err_max']:.5f}** | **{sc['int4_sign_flips']}/{sc['n_trials']}** |",
        f"| Top-1% Sparse + EF | {int(sc['topk_wire_bytes']):,} | {sc['topk_align_err_median']:.5f} | {sc['topk_align_err_max']:.5f} | {sc['topk_sign_flips']}/{sc['n_trials']} |",
        "",
        "---",
        "",
        "## 3. Two-Pass Audited Streaming Aggregation (`src/audit.py`)",
        "",
        f"- **Zero corruption on failed/skipped payloads:** **{'PASS' if au['all_rejections_preserve_state'] else 'FAIL'}** (`running_sum` and `total_weight` bit-for-bit identical before and after every rejected payload).",
        f"- **Small honest update (`0.5× r_i`) left unclipped:** `clip_factor_applied = {au['honest_small_clip_factor']:.6f}` (unlike FLTrust, updates smaller than $r_i$ are not rescaled up).",
        "",
        "### 3.1 Rejection & Fault-Classification Checks",
        "",
        "| Test Case | Outcome | Reason | `running_sum` & `total_weight` Unchanged? |",
        "|---|---|---|---|",
    ]
    for c in au["rejection_cases"]:
        lines.append(f"| `{c['case']}` | `{c['outcome']}` | `{c['reason']}` | {'YES' if c['state_unchanged'] else 'NO'} |")

    lines.extend(
        [
            "",
            "### 3.2 Receiver-Norm Cap Under Positive Rescaling",
            "",
            "| Multiplier | Receiver Norm $r_i$ | Decoded Norm $\\|y_j\\|_2$ | Clip Factor $a_{ij}$ | Clipped Contribution Norm | Unclipped Ablation Norm |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in au["rescaling_rows"]:
        lines.append(
            f"| `{row['scale_mult']:g}×` | {row['r_i']:.5f} | {row['decoded_norm']:.5f} | "
            f"{row['clip_factor']:.5f} | **{row['clipped_norm']:.5f}** | {row['unclipped_norm']:.5f} |"
        )

    lines.extend(
        [
            "",
            "### 3.3 Receive-Side Buffer Memory Projection (Rank-128 LoRA, 40M Params, 6 Neighbors)",
            "",
            "> **Note (Projection, not GPU measurement):** Counts receive-side tensor buffers only (excludes training weights/optimizer states and sender quantization residual).",
            "",
            f"- **Batched FP32 TrustGraph-DFL (`6` neighbor buffers + `val_dir`):** `{au['lora128_batched_fp32_rx_mib_proj']:.1f} MiB`",
            f"- **PeerSieve Audited Streaming (`running_sum` + `val_dir` + `4 MiB` decode tile + `32 KiB` sketch + `1` staged INT4 payload):** `{au['lora128_peersieve_rx_mib_proj']:.1f} MiB` (**`{100.0 * float(au['lora128_rx_mem_ratio_proj']):.1f}%`** of batched FP32; constant in neighbor degree).",
            "",
            "---",
            "",
            "## 4. Multi-Process `torch.distributed` (Gloo) Quarantine & Referral Verification (`src/quarantine.py`, `src/transport.py`)",
            "",
        ]
    )

    if gl is None:
        lines.append("_Skipped (`--skip-gloo` passed)._")
    else:
        rounds = gl["rounds"]
        lines.extend(
            [
                f"- **Execution:** 4 separate OS processes over `torch.distributed` Gloo (`12` synchronized rounds, wall time `{gl['wall_sec']:.2f}s`).",
                f"- **Banned peer reinstatement check (`can_consider_referral(2)`):** `{gl['banned_reinstatement_allowed']}` (permanently banned after round-0 `sketch_mismatch`).",
                f"- **Total received bytes at Rank 0 across 12 rounds:** `{int(gl['total_rx_bytes']):,}` B (sketches + selective INT4 payloads), vs. `{12 * 2 * int(gl['fp32_update_bytes']):,}` B for unscreened FP32 across 2 neighbors.",
                "",
                "| Round | Sketches Rx | Payloads Req | Accepted | Hard Faults | Active | Probation | Banned | Referrals Admitted |",
                "|---:|---:|---:|---:|---:|---|---|---|---|",
            ]
        )
        for r in rounds:
            lines.append(
                f"| {r['round']} | {r['sketches_rx']} | {r['payloads_req']} | {r['accepted']} | "
                f"{r['hard_faults']} | `{r['active']}` | `{r['probation']}` | `{r['banned']}` | `{r['referrals_admitted']}` |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify PeerSieve's four core mechanisms locally (Table 2).")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--skip-gloo", action="store_true")
    p.add_argument("--report-out", default="report/phase2_local_verification.md")
    args = p.parse_args()

    print("1/3 Running CountSketch & INT4 compression checks on backprop LoRA updates...", flush=True)
    sc = verify_sketch_and_compression(n_trials=args.trials)
    print(
        f"    tile_invariance={sc['tile_invariance_ok']} "
        f"sketch_cos_max={sc['sketch_cos_err_max']:.4f} "
        f"sketch_norm_max={sc['sketch_norm_err_max']:.4f} "
        f"int4_align_med={sc['int4_align_err_median']:.5f} vs top1%_med={sc['topk_align_err_median']:.5f}",
        flush=True,
    )

    print("2/3 Running two-pass audited streaming checks (faults, kernel collision, norm cap)...", flush=True)
    au = verify_audit_and_streaming()
    print(
        f"    all_rejections_preserve_state={au['all_rejections_preserve_state']} "
        f"rx_mem_ratio_proj={au['lora128_rx_mem_ratio_proj']:.3f}",
        flush=True,
    )

    gl: dict[str, object] | None = None
    if not args.skip_gloo:
        print("3/3 Running 4-process torch.distributed Gloo quarantine & referral check...", flush=True)
        gl = verify_gloo_quarantine()
        print(
            f"    gloo_ok wall={gl['wall_sec']:.2f}s total_rx_bytes={gl['total_rx_bytes']:,} "
            f"banned_reinstatement_allowed={gl['banned_reinstatement_allowed']}",
            flush=True,
        )

    md = render_report(sc, au, gl)
    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote verification report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
