"""Phase 2 unit and multi-process Gloo verification tests for PeerSieve (Table 2 checks)."""

from __future__ import annotations

from datetime import timedelta
import math
import os
import socket
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.audit import (
    AuditConfig,
    AuditOutcome,
    AuditedStreamingReceiver,
    PayloadEnvelope,
    SketchAnnouncement,
)
from src.compression import (
    DEFAULT_BLOCK_SIZE,
    HardProtocolFault,
    PackedINT4Payload,
    decode_blockwise_int4,
    encode_blockwise_int4,
    expected_int4_wire_bytes,
    iter_decoded_tiles,
    topk_compress_with_error_feedback,
    validate_packed_int4,
)
from src.quarantine import LinkState, PeerQuarantineTable, QuarantineConfig, backoff_rounds
from src.sketch import (
    SketchLayout,
    StreamingCountSketch,
    allocate_layer_buckets,
    compute_sketch,
    sketch_cosine_and_norms,
    sketch_from_packed_int4,
    sketches_exact_equal,
    validate_sketch_tensor,
)
from src.transport import PeerRoundMessage, TokenBucket, run_gloo_round


# ---------------------------------------------------------------------------
# 1. Compression & Error Feedback Checks (Table 2: Compression)
# ---------------------------------------------------------------------------


def test_int4_roundtrip_wire_and_residual_conservation() -> None:
    torch.manual_seed(7)
    # Odd length non-multiple of block_size=128 exercises padding and tail nibble checks.
    numel = 1029
    residual = torch.zeros(numel, dtype=torch.float32)
    sum_true = torch.zeros(numel, dtype=torch.float32)
    sum_decoded = torch.zeros(numel, dtype=torch.float32)

    for step in range(5):
        # Mix normal blocks, tiny-magnitude blocks, and an all-zero block.
        upd = torch.randn(numel) * (0.02 if step % 2 == 0 else 0.005)
        upd[256:384] = 0.0
        payload, residual = encode_blockwise_int4(upd, residual=residual, block_size=DEFAULT_BLOCK_SIZE)

        raw = payload.to_bytes()
        assert len(raw) == expected_int4_wire_bytes(numel, DEFAULT_BLOCK_SIZE)
        reloaded = PackedINT4Payload.from_bytes(raw, max_bytes=len(raw))
        validate_packed_int4(reloaded, expected_numel=numel, expected_block_size=DEFAULT_BLOCK_SIZE)

        dec_full = decode_blockwise_int4(reloaded)
        # Verify tiled decode matches full decode bit-for-bit across aligned and unaligned tile sizes.
        for tile_len in (2, 16, 128, 250, 512, 2048):
            tile_buf = torch.empty(tile_len, dtype=torch.float32)
            parts = [view.clone() for _, _, view in iter_decoded_tiles(reloaded, tile_buf)]
            dec_tiled = torch.cat(parts)
            assert torch.equal(dec_tiled, dec_full), f"Mismatch for tile_len={tile_len}"

        sum_true += upd
        sum_decoded += dec_full

    # Conservation identity: sum(y_t) + e_T == sum(Delta w_t).
    assert torch.allclose(sum_decoded + residual, sum_true, atol=1e-5, rtol=1e-5)


def test_int4_validation_rejects_hard_protocol_faults() -> None:
    upd = torch.randn(257) * 0.1
    payload, _ = encode_blockwise_int4(upd, block_size=128)
    validate_packed_int4(payload, expected_numel=257, expected_block_size=128)

    # Reserved nibble 15.
    bad_codes = payload.codes.clone()
    bad_codes[0] = (bad_codes[0] & 0xF0) | 15
    with pytest.raises(HardProtocolFault, match="invalid_nibble"):
        validate_packed_int4(
            PackedINT4Payload(257, 128, bad_codes, payload.scales), expected_numel=257, expected_block_size=128
        )

    # Nonzero odd-length trailing padding nibble.
    bad_pad = payload.codes.clone()
    bad_pad[-1] = (bad_pad[-1] & 0x0F) | (3 << 4)
    with pytest.raises(HardProtocolFault, match="invalid_padding"):
        validate_packed_int4(
            PackedINT4Payload(257, 128, bad_pad, payload.scales), expected_numel=257, expected_block_size=128
        )

    # Nonfinite scale.
    bad_scale = payload.scales.clone()
    bad_scale[0] = torch.tensor(float("nan"), dtype=torch.float16)
    with pytest.raises(HardProtocolFault, match="nonfinite_scale"):
        validate_packed_int4(
            PackedINT4Payload(257, 128, payload.codes, bad_scale), expected_numel=257, expected_block_size=128
        )

    # Negative scale.
    neg_scale = payload.scales.clone()
    neg_scale[0] = torch.tensor(-0.5, dtype=torch.float16)
    with pytest.raises(HardProtocolFault, match="negative_scale"):
        validate_packed_int4(
            PackedINT4Payload(257, 128, payload.codes, neg_scale), expected_numel=257, expected_block_size=128
        )

    # Oversized wire buffer rejected before decoding.
    raw = payload.to_bytes()
    with pytest.raises(HardProtocolFault, match="size_limit_exceeded"):
        PackedINT4Payload.from_bytes(raw + b"\x00", max_bytes=len(raw))


def test_int4_preserves_alignment_better_than_top1pct_on_lora_like_updates() -> None:
    torch.manual_seed(11)
    # Construct correlated rank-16 LoRA-style updates (heavy-tailed + shared direction + dense signal).
    d = 32_768
    shared = torch.randn(d)
    shared = shared / torch.linalg.vector_norm(shared)
    val_dir = shared + 0.5 * (torch.randn(d) / math.sqrt(d))
    upd = 0.4 * shared + 0.6 * (torch.randn(d) / math.sqrt(d))

    true_cos = float(torch.dot(upd, val_dir) / (torch.linalg.vector_norm(upd) * torch.linalg.vector_norm(val_dir)))

    int4_payload, _ = encode_blockwise_int4(upd, block_size=128)
    y_int4 = decode_blockwise_int4(int4_payload)
    cos_int4 = float(torch.dot(y_int4, val_dir) / (torch.linalg.vector_norm(y_int4) * torch.linalg.vector_norm(val_dir)))

    y_topk, _, _ = topk_compress_with_error_feedback(upd, keep_frac=0.01)
    cos_topk = float(torch.dot(y_topk, val_dir) / (torch.linalg.vector_norm(y_topk) * torch.linalg.vector_norm(val_dir)))

    err_int4 = abs(cos_int4 - true_cos)
    err_topk = abs(cos_topk - true_cos)
    assert err_int4 < 0.01
    assert err_int4 < err_topk


# ---------------------------------------------------------------------------
# 2. CountSketch Screen & Tile-Invariance Checks (Table 2: Sketch screen)
# ---------------------------------------------------------------------------


def test_sketch_bit_exact_across_arbitrary_tile_sizes_and_packed_stream() -> None:
    torch.manual_seed(19)
    layer_sizes = [64, 4096, 128, 32_768, 1025, 16_384]
    layout = SketchLayout.from_layer_sizes(layer_sizes, num_buckets=16_384)
    assert sum(layout.bucket_counts) == 16_384
    assert all(c >= 1 for c in layout.bucket_counts)

    upd = torch.randn(layout.total_numel) * 0.05
    payload, _ = encode_blockwise_int4(upd, block_size=128)
    y_full = decode_blockwise_int4(payload)

    ref_sketch = compute_sketch(y_full, layout=layout, round_seed=12345, round_idx=7, tile_numel=None)
    validate_sketch_tensor(ref_sketch, expected_buckets=16_384)
    assert ref_sketch.element_size() * ref_sketch.numel() == 32_768  # Exact 32 KiB

    # Arbitrary dense tile sizes (including non-powers-of-two and prime chunks).
    for tile_numel in (127, 128, 1000, 4096, 8192, 20_000, 1_048_576):
        s_tiled = compute_sketch(y_full, layout=layout, round_seed=12345, round_idx=7, tile_numel=tile_numel)
        assert sketches_exact_equal(ref_sketch, s_tiled), f"Sketch differed for dense tile_numel={tile_numel}"

    # Streamed directly from PackedINT4Payload with varying decode-tile buffers.
    for tile_numel in (128, 500, 8192, 65_536):
        tile_buf = torch.empty(tile_numel, dtype=torch.float32)
        s_packed = sketch_from_packed_int4(payload, layout=layout, round_seed=12345, round_idx=7, tile_buf=tile_buf)
        assert sketches_exact_equal(ref_sketch, s_packed), f"Packed sketch differed for tile_numel={tile_numel}"


def test_sketch_cosine_and_norm_accuracy_within_audit_tolerances() -> None:
    torch.manual_seed(23)
    layout = SketchLayout.from_layer_sizes([16_384, 32_768, 16_384], num_buckets=16_384)
    shared = torch.randn(layout.total_numel) * 0.01
    val_dir = shared + torch.randn(layout.total_numel) * 0.01
    upd = shared + torch.randn(layout.total_numel) * 0.01

    payload, _ = encode_blockwise_int4(upd, block_size=128)
    y = decode_blockwise_int4(payload)

    s_y = compute_sketch(y, layout, round_seed=99, round_idx=3)
    s_v = compute_sketch(val_dir, layout, round_seed=99, round_idx=3)

    c_hat, norm_sy, _ = sketch_cosine_and_norms(s_y, s_v)
    norm_y = float(torch.linalg.vector_norm(y).item())
    norm_v = float(torch.linalg.vector_norm(val_dir).item())
    c_full = float(torch.dot(y, val_dir).item()) / (norm_y * norm_v)

    assert abs(c_hat - c_full) <= 0.05
    assert abs(norm_sy / norm_y - 1.0) <= 0.03


def test_sketch_rejects_negative_zero_and_nonfinite() -> None:
    neg_zero = torch.zeros(16, dtype=torch.float16)
    neg_zero[3] = torch.tensor(-0.0, dtype=torch.float16)
    with pytest.raises(HardProtocolFault, match="noncanonical_zero"):
        validate_sketch_tensor(neg_zero, expected_buckets=16)

    nan_s = torch.zeros(16, dtype=torch.float16)
    nan_s[0] = torch.tensor(float("nan"), dtype=torch.float16)
    with pytest.raises(HardProtocolFault, match="nonfinite_sketch"):
        validate_sketch_tensor(nan_s, expected_buckets=16)


# ---------------------------------------------------------------------------
# 3. Audited Streaming Aggregation Checks (Table 2: Audit and streaming)
# ---------------------------------------------------------------------------


def test_audit_rejected_payloads_leave_state_untouched_and_clip_caps_norm() -> None:
    torch.manual_seed(31)
    layout = SketchLayout.from_layer_sizes([8192, 8192], num_buckets=16_384)
    cfg = AuditConfig(tile_numel=2048, tau_c=0.05, tau_n=0.03, alpha=0.5, clip_factor=1.0)
    recv = AuditedStreamingReceiver(self_id=0, run_id="test_run", layout=layout, cfg=cfg)

    base = torch.randn(layout.total_numel) * 0.01
    val_dir = base + torch.randn(layout.total_numel) * 0.005
    self_upd = base + torch.randn(layout.total_numel) * 0.005
    r_i = float(torch.linalg.vector_norm(self_upd).item())
    recv.start_round(round_idx=1, round_seed=777, val_descent_dir=val_dir, self_update_norm=r_i)

    # 1) Accept an honest smaller update (norm < r_i) -> clip_factor_applied must stay 1.0.
    small_honest = 0.25 * self_upd
    p_small, _ = encode_blockwise_int4(small_honest)
    s_small = sketch_from_packed_int4(p_small, layout, 777, 1, torch.empty(1024))
    ann_small = SketchAnnouncement(sender_id=1, run_id="test_run", round_idx=1, sketch=s_small)
    scr_small = recv.screen_announcement(ann_small, expected_sender_id=1)
    assert scr_small.request_payload is True

    res_small = recv.audit_and_accumulate(
        ann_small, PayloadEnvelope(1, "test_run", 1, raw_bytes=p_small.to_bytes()), expected_sender_id=1
    )
    assert res_small.outcome == AuditOutcome.ACCEPTED
    assert math.isclose(res_small.clip_factor_applied, 1.0, rel_tol=1e-6)
    assert res_small.clipped_contribution_norm < r_i

    # Snapshot running_sum and total_weight before injecting failures.
    snap_sum = recv.running_sum.clone()
    snap_weight = recv.total_weight

    # 2) Hard fault: payload tampering causes sketch mismatch -> ZERO change to running_sum/total_weight.
    tampered_codes = p_small.codes.clone()
    tampered_codes[10] ^= 0x01
    p_tampered = PackedINT4Payload(p_small.numel, p_small.block_size, tampered_codes, p_small.scales)
    res_bad_sketch = recv.audit_and_accumulate(
        ann_small, PayloadEnvelope(1, "test_run", 1, payload=p_tampered), expected_sender_id=1
    )
    assert res_bad_sketch.outcome == AuditOutcome.HARD_FAULT
    assert res_bad_sketch.reason == "sketch_mismatch"
    assert torch.equal(recv.running_sum, snap_sum)
    assert recv.total_weight == snap_weight

    # 3) Soft fault: exact CountSketch kernel-collision perturbation leaves sketch IDENTICAL
    #    while inflating ||y||_2 -> caught by norm_discrepancy with ZERO change to running_sum!
    # Find two coordinates in layer 0 that hash to the same bucket.
    tmp_cs = StreamingCountSketch(layout, round_seed=777, round_idx=1)
    buckets, signs = tmp_cs._hash_range(0, 8192)
    seen: dict[int, tuple[int, float]] = {}
    pair: tuple[int, int, float, float] | None = None
    for idx_i in range(8192):
        b = int(buckets[idx_i].item())
        sg = float(signs[idx_i].item())
        if b in seen:
            prev_i, prev_sg = seen[b]
            pair = (prev_i, idx_i, prev_sg, sg)
            break
        seen[b] = (idx_i, sg)
    assert pair is not None
    i1, i2, sg1, sg2 = pair
    # Build payload with exact cancelling contribution in bucket b (huge norm, zero bucket change!).
    y_coll = decode_blockwise_int4(p_small).clone()
    delta = 5.0 * r_i
    y_coll[i1] = delta * sg1
    y_coll[i2] = -delta * sg2
    p_coll, _ = encode_blockwise_int4(y_coll)
    y_coll_dec = decode_blockwise_int4(p_coll)
    # Re-ensure exact cancellation on decoded values (same block scale or exact equal magnitudes -> exact 0 sum).
    s_coll = compute_sketch(y_coll_dec, layout, round_seed=777, round_idx=1)
    ann_coll = SketchAnnouncement(sender_id=2, run_id="test_run", round_idx=1, sketch=s_coll)
    res_coll = recv.audit_and_accumulate(
        ann_coll, PayloadEnvelope(2, "test_run", 1, payload=p_coll), expected_sender_id=2
    )
    assert res_coll.outcome == AuditOutcome.SOFT_FAULT
    assert res_coll.reason in {"norm_discrepancy", "cosine_discrepancy", "nonpositive_full_cosine"}
    assert torch.equal(recv.running_sum, snap_sum)
    assert recv.total_weight == snap_weight

    # 4) Zero update skips without fault and leaves running_sum unchanged.
    p_zero, _ = encode_blockwise_int4(torch.zeros(layout.total_numel))
    s_zero = compute_sketch(torch.zeros(layout.total_numel), layout, round_seed=777, round_idx=1)
    res_zero = recv.audit_and_accumulate(
        SketchAnnouncement(3, "test_run", 1, s_zero),
        PayloadEnvelope(3, "test_run", 1, payload=p_zero),
        expected_sender_id=3,
    )
    assert res_zero.outcome == AuditOutcome.ZERO_SKIP
    assert torch.equal(recv.running_sum, snap_sum)
    assert recv.total_weight == snap_weight

    # 5) Positive rescaling (50x): accepted contribution must respect norm cap <= r_i.
    rescaled = 50.0 * self_upd
    p_big, _ = encode_blockwise_int4(rescaled)
    s_big = sketch_from_packed_int4(p_big, layout, 777, 1, torch.empty(1024))
    ann_big = SketchAnnouncement(4, "test_run", 1, s_big)
    res_big = recv.audit_and_accumulate(
        ann_big, PayloadEnvelope(4, "test_run", 1, payload=p_big), expected_sender_id=4
    )
    assert res_big.outcome == AuditOutcome.ACCEPTED
    assert res_big.clip_factor_applied < 0.05
    assert res_big.clipped_contribution_norm <= r_i + 1e-4

    # Finalized weighted average norm is bounded by r_i.
    nbr_avg = recv.running_sum / recv.total_weight
    assert float(torch.linalg.vector_norm(nbr_avg).item()) <= r_i + 1e-4

    # Receive-side buffer bytes remain constant regardless of how many neighbors are processed.
    mem_bytes = recv.receive_side_tensor_bytes(p_big)
    batched_6_fp32 = 6 * layout.total_numel * 4
    assert mem_bytes < batched_6_fp32


# ---------------------------------------------------------------------------
# 4. Quarantine Unit Checks & Multi-Process Gloo Tests (Table 2: Quarantine)
# ---------------------------------------------------------------------------


def test_quarantine_ema_backoff_probation_recovery_and_permanent_ban() -> None:
    assert [backoff_rounds(k, t_max=16) for k in range(1, 7)] == [2, 4, 8, 16, 16, 16]

    qt = PeerQuarantineTable(self_id=0, initial_neighbors=[1, 2], cfg=QuarantineConfig(max_neighbors=2))
    neg_screen = lambda c=-0.2: type("S", (), {"request_payload": False, "sketch_cosine": c, "sketch_norm": 1.0, "outcome": AuditOutcome.SOFT_FAULT, "reason": "screen_nonpositive"})()
    pos_screen = lambda c=0.3: type("S", (), {"request_payload": True, "sketch_cosine": c, "sketch_norm": 1.0, "outcome": AuditOutcome.ACCEPTED, "reason": "screen_pass"})()

    # One negative sketch skips transfer but keeps peer 1 ACTIVE (no premature probation).
    assert qt.on_screen_result(1, 0, neg_screen(-0.1)) is False
    assert qt.get_record(1).state == LinkState.ACTIVE
    assert qt.on_screen_result(1, 1, neg_screen(-0.1)) is False
    assert qt.get_record(1).state == LinkState.ACTIVE
    # Third consecutive negative EMA round moves peer 1 to PROBATION with backoff 2^1 = 2 (next probe round 4).
    assert qt.on_screen_result(1, 2, neg_screen(-0.1)) is False
    rec1 = qt.get_record(1)
    assert rec1.state == LinkState.PROBATION
    assert rec1.next_probe_round == 4
    assert qt.should_accept_sketch(1, 3) is False
    assert qt.should_accept_sketch(1, 4) is True

    # Failed probe at round 4 extends probation with backoff 2^2 = 4 (next probe round 8).
    assert qt.on_screen_result(1, 4, neg_screen(-0.2)) is False
    assert rec1.state == LinkState.PROBATION
    assert rec1.next_probe_round == 8
    assert qt.should_accept_sketch(1, 7) is False
    assert qt.should_accept_sketch(1, 8) is True

    # Passing probe at round 8 restores peer 1 to ACTIVE without requesting payload on probe round.
    assert qt.on_screen_result(1, 8, pos_screen(0.4)) is False
    assert rec1.state == LinkState.ACTIVE
    assert rec1.recoveries == 1
    assert qt.on_screen_result(1, 9, pos_screen(0.4)) is True

    # Hard fault permanently bans peer 2; banned peer can never be reinstated via referral.
    qt.on_audit_result(2, 9, type("A", (), {"outcome": AuditOutcome.HARD_FAULT, "reason": "sketch_mismatch"})())
    assert qt.is_banned(2)
    assert qt.should_accept_sketch(2, 10) is False
    assert qt.can_consider_referral(2) is False
    assert qt.needs_referral() is True  # Only peer 1 active (< min_active_degree=2)
    assert qt.can_consider_referral(3) is True
    assert qt.admit_referral(3, 10, pos_screen(0.5)) is True
    assert sorted(qt.active_peers()) == [1, 3]
    assert qt.is_banned(2)  # Permanent ban still recorded after slot reuse


def _gloo_worker_quarantine_and_referrals(rank: int, world_size: int, port: int, out_path: str) -> None:
    """Multi-process Gloo test worker verifying bans, probation recovery, and referrals."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    try:
        torch.manual_seed(1000)
        layout = SketchLayout.from_layer_sizes([2048, 2048], num_buckets=1024)
        base_dir = torch.randn(layout.total_numel) * 0.05
        # Shared validation descent direction so cosine alignment is cleanly controllable.
        val_dir = base_dir.clone()
        self_upd = base_dir.clone()

        # Topology:
        #   Rank 0 starts with neighbors [1, 2] (max_neighbors=2, min_active_degree=2)
        #   Rank 1 is honest and knows [0, 3] (so Rank 1 can refer Rank 3 to Rank 0!)
        #   Rank 2 is Byzantine (sends tampered payload in round 0 -> HARD_FAULT -> permanent ban)
        #   Rank 3 is admitted by Rank 0 via referral in round 0 (ema_score ~ +1.0);
        #          rounds 1..4: sends -base_dir -> EMA crosses <0 at r=2, streak=3 at r=4 -> PROBATION (next probe r=6);
        #          round 6: probe fails -> extends PROBATION k=2, backoff=4 (next probe r=10);
        #          round 10: probe passes -> restored to ACTIVE;
        #          round 11: payload audited & accepted!
        initial_nbrs = {0: [1, 2], 1: [0, 3], 2: [0], 3: [1]}[rank]
        qt = PeerQuarantineTable(
            self_id=rank,
            initial_neighbors=initial_nbrs,
            cfg=QuarantineConfig(beta=0.7, neg_streak_limit=3, t_max=16, min_active_degree=2, max_neighbors=2),
        )
        recv = AuditedStreamingReceiver(
            self_id=rank,
            run_id="gloo_verify",
            layout=layout,
            cfg=AuditConfig(tile_numel=1024, tau_c=0.10, tau_n=0.10, alpha=0.5),
        )
        bucket = TokenBucket(rate_bps=1_000_000_000.0, sleep_enabled=False)

        records: list[dict[str, object]] = []
        for r_idx in range(12):
            round_seed = 4242
            if rank == 3 and r_idx in (1, 2, 3, 4, 6):
                my_upd = -base_dir.clone()
            else:
                my_upd = base_dir.clone()

            payload, _ = encode_blockwise_int4(my_upd, block_size=128)
            sketch = sketch_from_packed_int4(payload, layout, round_seed, r_idx, torch.empty(1024))
            raw_bytes = payload.to_bytes()

            if rank == 2:
                # Byzantine peer tampers payload bytes so rebuilt sketch fails exact match -> HARD_FAULT.
                bad_codes = payload.codes.clone()
                bad_codes[0] ^= 0x01
                bad_payload = PackedINT4Payload(payload.numel, payload.block_size, bad_codes, payload.scales)
                raw_bytes = bad_payload.to_bytes()

            msg = PeerRoundMessage(sketch=sketch, payload_bytes=raw_bytes)
            _, tel = run_gloo_round(
                rank=rank,
                world_size=world_size,
                round_idx=r_idx,
                round_seed=round_seed,
                val_descent_dir=val_dir,
                self_update=self_upd,
                outbound=msg,
                receiver=recv,
                quarantine=qt,
                token_bucket=bucket,
            )
            if rank == 0:
                records.append(
                    {
                        "round": r_idx,
                        "active": list(tel.active_peers_end),
                        "probation": list(tel.probation_peers_end),
                        "banned": list(tel.banned_peers_end),
                        "referrals_admitted": list(tel.referrals_admitted),
                        "payloads_accepted": tel.payloads_accepted,
                        "hard_faults": tel.hard_faults,
                        "sketches_received": tel.sketches_received,
                        "payloads_requested": tel.payloads_requested,
                        "audit_2": tel.audit_results[2].reason if 2 in tel.audit_results else None,
                        "accepted_3": (3 in tel.audit_results and tel.audit_results[3].outcome == AuditOutcome.ACCEPTED),
                    }
                )

        if rank == 0:
            torch.save(records, out_path)
    finally:
        dist.destroy_process_group()


def test_gloo_multiprocess_ban_probation_recovery_and_referral() -> None:
    if not dist.is_gloo_available():
        pytest.skip("Gloo backend unavailable")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    fd2, out_path = tempfile.mkstemp(prefix="peersieve_gloo_out_", suffix=".pt")
    os.close(fd2)

    try:
        mp.spawn(
            _gloo_worker_quarantine_and_referrals,
            args=(4, port, out_path),
            nprocs=4,
            join=True,
        )
        records = torch.load(out_path, weights_only=False)
        assert len(records) == 12

        # Round 0: Rank 2 commits sketch_mismatch -> permanently banned; active degree drops to 1 (<2),
        # so Rank 1 refers Rank 3 -> Rank 0 screens Rank 3's sketch over Gloo and admits Rank 3!
        r0 = records[0]
        assert r0["audit_2"] == "sketch_mismatch"
        assert r0["hard_faults"] == 1
        assert r0["banned"] == [2]
        assert r0["referrals_admitted"] == [3]
        assert r0["active"] == [1, 3]
        assert r0["payloads_accepted"] == 1  # Rank 1 accepted; Rank 2 rejected cleanly

        # Rounds 1..4: Rank 2 stays banned; Rank 3 EMA turns <0 at r=2, reaches 3-round negative streak at r=4
        # -> enters PROBATION with backoff 2^1 = 2 (next probe round 6).
        assert records[1]["banned"] == [2]
        assert records[1]["active"] == [1, 3]
        assert records[2]["active"] == [1, 3]
        assert records[3]["active"] == [1, 3]
        assert records[4]["probation"] == [3]
        assert records[4]["active"] == [1]

        # Round 5: Rank 3 is skipped completely (in probation; only Rank 1 screened).
        assert records[5]["probation"] == [3]
        assert records[5]["sketches_received"] == 1

        # Round 6: Scheduled probe of Rank 3 fails -> backoff extends to 2^2 = 4 (next probe round 10).
        assert records[6]["probation"] == [3]
        assert records[6]["sketches_received"] == 2  # Rank 1 + Rank 3 probe
        assert records[6]["payloads_requested"] == 1  # Only Rank 1 requested

        # Rounds 7..9: Rank 3 stays silent during extended backoff.
        for r_silent in (7, 8, 9):
            assert records[r_silent]["probation"] == [3]
            assert records[r_silent]["sketches_received"] == 1

        # Round 10: Scheduled probe of Rank 3 passes -> restored to ACTIVE!
        assert records[10]["active"] == [1, 3]
        assert records[10]["probation"] == []
        assert records[10]["banned"] == [2]

        # Round 11: Rank 3 is ACTIVE again and its INT4 payload passes full audit & is accepted!
        assert records[11]["active"] == [1, 3]
        assert records[11]["payloads_accepted"] == 2
        assert records[11]["accepted_3"] is True
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)
