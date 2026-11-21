"""Token-bucket receive limiter and multi-process torch.distributed Gloo protocol.

Implements PeerSieve's peer-to-peer exchange protocol across separate processes
using torch.distributed (Gloo for CPU/control-plane verification and cluster
coordination):
  Phase A: Selective sketch announcements (32 KiB FP16) + pre-transfer screening
  Phase B: Selective INT4 payload transfers (strictly capped at max_wire_bytes)
           + two-pass audited streaming aggregation
  Phase C: Referral requests when active degree < min_active_degree + candidate
           sketch vetting (never reinstating banned peers)
  Phase D: Blended update finalization & telemetry accounting
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import torch
import torch.distributed as dist

from src.audit import (
    AuditOutcome,
    AuditResult,
    AuditedStreamingReceiver,
    PayloadEnvelope,
    SketchAnnouncement,
)
from src.compression import HardProtocolFault, PackedINT4Payload, expected_int4_wire_bytes
from src.quarantine import LinkState, PeerQuarantineTable


class TokenBucket:
    """Receive-side token-bucket rate limiter (bytes/sec) for bandwidth emulation.

    Supports both real-time sleep pacing (sleep_enabled=True, used in cluster
    timing runs) and virtual-time accounting (sleep_enabled=False, used in fast
    deterministic unit tests).
    """

    def __init__(
        self,
        rate_bps: float | None = 1_000_000_000.0,  # 1 Gb/s default (None = uncapped)
        burst_bytes: float | None = None,
        sleep_enabled: bool = False,
    ) -> None:
        if rate_bps is not None and rate_bps <= 0.0:
            raise ValueError(f"rate_bps must be positive or None, got {rate_bps}")
        self.rate_bps = rate_bps
        self.bytes_per_sec = (rate_bps / 8.0) if rate_bps is not None else None
        self.capacity = (
            float(burst_bytes)
            if burst_bytes is not None
            else (self.bytes_per_sec * 0.01 if self.bytes_per_sec is not None else float("inf"))
        )
        self.tokens = self.capacity
        self.sleep_enabled = bool(sleep_enabled)
        self._last_time = time.monotonic()

        self.total_bytes: int = 0
        self.total_wait_sec: float = 0.0

    def reset_stats(self) -> None:
        self.total_bytes = 0
        self.total_wait_sec = 0.0
        self.tokens = self.capacity
        self._last_time = time.monotonic()

    def consume(self, n_bytes: int) -> float:
        """Account for n_bytes received and return wait time (seconds) induced by cap."""
        if n_bytes < 0:
            raise ValueError(f"n_bytes must be non-negative, got {n_bytes}")
        self.total_bytes += int(n_bytes)
        if self.bytes_per_sec is None or n_bytes == 0:
            return 0.0

        now = time.monotonic()
        elapsed = max(0.0, now - self._last_time)
        self._last_time = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.bytes_per_sec)

        deficit = float(n_bytes) - self.tokens
        if deficit <= 0.0:
            self.tokens -= float(n_bytes)
            return 0.0

        wait_sec = deficit / self.bytes_per_sec
        self.tokens = 0.0
        self.total_wait_sec += wait_sec
        if self.sleep_enabled and wait_sec > 0.0:
            time.sleep(wait_sec)
            self._last_time = time.monotonic()
        return wait_sec


@dataclass
class PeerRoundMessage:
    """Data a peer is prepared to transmit during a single Gloo protocol round."""

    sketch: torch.Tensor  # float16, shape (num_buckets,)
    payload_bytes: bytes  # serialized PackedINT4Payload (or adversarial corrupted bytes)
    # Optional custom sketch/payload overrides keyed by destination receiver rank (for Byzantine tests).
    per_receiver_sketch: dict[int, torch.Tensor] = field(default_factory=dict)
    per_receiver_payload_bytes: dict[int, bytes] = field(default_factory=dict)
    # Optional explicit advertised round/run/sender overrides for protocol-fault testing.
    advertised_sender_id: int | None = None
    advertised_round_idx: int | None = None
    advertised_run_id: str | None = None

    def sketch_for(self, receiver_rank: int) -> torch.Tensor:
        return self.per_receiver_sketch.get(receiver_rank, self.sketch)

    def payload_bytes_for(self, receiver_rank: int) -> bytes:
        return self.per_receiver_payload_bytes.get(receiver_rank, self.payload_bytes)


@dataclass
class PeerRoundTelemetry:
    round_idx: int
    self_id: int
    sketches_received: int = 0
    sketch_bytes_received: int = 0
    payloads_requested: int = 0
    payloads_received: int = 0
    payload_bytes_received: int = 0
    payloads_accepted: int = 0
    soft_faults: int = 0
    hard_faults: int = 0
    zero_skips: int = 0
    referrals_requested: int = 0
    referrals_admitted: list[int] = field(default_factory=list)
    active_peers_end: list[int] = field(default_factory=list)
    probation_peers_end: list[int] = field(default_factory=list)
    banned_peers_end: list[int] = field(default_factory=list)
    used_self_fallback: bool = False
    receive_wait_sec: float = 0.0
    audit_results: dict[int, AuditResult] = field(default_factory=dict)


def _all_gather_int64_row(row: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    gathered = [torch.empty_like(row) for _ in range(world_size)]
    dist.all_gather(gathered, row)
    return gathered


def _send_sketch(
    dst: int,
    sketch: torch.Tensor,
    adv_sender: int,
    adv_round: int,
    run_id_hash: int,
) -> None:
    meta = torch.tensor([adv_sender, adv_round, run_id_hash, int(sketch.numel())], dtype=torch.int64)
    dist.send(meta, dst=dst)
    go = torch.empty(1, dtype=torch.int64)
    dist.recv(go, src=dst)
    if int(go.item()) == 1:
        # Transmit raw int16 bit representation so canonical zero / NaN bits survive unmodified.
        bits = sketch.detach().to(device="cpu", dtype=torch.float16).contiguous().view(torch.int16)
        dist.send(bits, dst=dst)


def _recv_sketch(
    src: int,
    expected_buckets: int,
    expected_run_id: str,
    expected_run_id_hash: int,
    bucket: TokenBucket,
) -> SketchAnnouncement:
    meta = torch.empty(4, dtype=torch.int64)
    dist.recv(meta, src=src)
    adv_sender, adv_round, adv_run_hash, n_elems = (int(x) for x in meta.tolist())
    run_id = expected_run_id if adv_run_hash == expected_run_id_hash else f"mismatched:{adv_run_hash}"
    if n_elems != expected_buckets:
        dist.send(torch.tensor([0], dtype=torch.int64), dst=src)
        return SketchAnnouncement(
            sender_id=adv_sender,
            run_id=run_id,
            round_idx=adv_round,
            sketch=torch.empty(0, dtype=torch.float16),
        )
    dist.send(torch.tensor([1], dtype=torch.int64), dst=src)
    bits = torch.empty(n_elems, dtype=torch.int16)
    dist.recv(bits, src=src)
    bucket.consume(n_elems * 2)
    sketch = bits.view(torch.float16)
    return SketchAnnouncement(sender_id=adv_sender, run_id=run_id, round_idx=adv_round, sketch=sketch)


def _run_id_tag(run_id: str) -> int:
    """Stable positive 63-bit integer tag for compact run_id envelope checks."""
    h = 1469598103934665603
    for b in run_id.encode("utf-8"):
        h = ((h ^ b) * 1099511628211) & 0x7FFF_FFFF_FFFF_FFFF
    return h


def run_gloo_round(
    *,
    rank: int,
    world_size: int,
    round_idx: int,
    round_seed: int,
    val_descent_dir: torch.Tensor,
    self_update: torch.Tensor,
    outbound: PeerRoundMessage,
    receiver: AuditedStreamingReceiver,
    quarantine: PeerQuarantineTable,
    token_bucket: TokenBucket | None = None,
) -> tuple[torch.Tensor, PeerRoundTelemetry]:
    """Execute one full synchronized PeerSieve round over torch.distributed (Gloo)."""
    bucket = token_bucket if token_bucket is not None else TokenBucket(rate_bps=None)
    telemetry = PeerRoundTelemetry(round_idx=round_idx, self_id=rank)
    wait_before = bucket.total_wait_sec

    self_norm = float(torch.linalg.vector_norm(self_update.detach().float()).item())
    receiver.start_round(
        round_idx=round_idx,
        round_seed=round_seed,
        val_descent_dir=val_descent_dir,
        self_update_norm=self_norm,
    )

    my_run_tag = _run_id_tag(receiver.run_id)
    adv_sender = rank if outbound.advertised_sender_id is None else int(outbound.advertised_sender_id)
    adv_round = round_idx if outbound.advertised_round_idx is None else int(outbound.advertised_round_idx)
    adv_run_tag = my_run_tag if outbound.advertised_run_id is None else _run_id_tag(outbound.advertised_run_id)

    # ------------------------------------------------------------------
    # Phase A: Selective Sketch Announcement & Pre-Transfer Screening
    # ------------------------------------------------------------------
    want_sketch = torch.zeros(world_size, dtype=torch.int64)
    for pid in quarantine.peers_to_screen(round_idx):
        if 0 <= pid < world_size:
            want_sketch[pid] = 1
    sketch_matrix = _all_gather_int64_row(want_sketch, world_size)

    announcements: dict[int, SketchAnnouncement] = {}
    want_payload = torch.zeros(world_size, dtype=torch.int64)

    # Deterministic pair ordering (sender, receiver) guarantees deadlock-free P2P transfers.
    for s in range(world_size):
        for r in range(world_size):
            if s == r or int(sketch_matrix[r][s].item()) == 0:
                continue
            if rank == s:
                _send_sketch(r, outbound.sketch_for(r), adv_sender, adv_round, adv_run_tag)
            elif rank == r:
                ann = _recv_sketch(s, receiver.layout.num_buckets, receiver.run_id, my_run_tag, bucket)
                announcements[s] = ann
                telemetry.sketches_received += 1
                telemetry.sketch_bytes_received += int(ann.sketch.numel()) * 2

                scr = receiver.screen_announcement(ann, expected_sender_id=s)
                req = quarantine.on_screen_result(s, round_idx, scr)
                if scr.outcome == AuditOutcome.HARD_FAULT:
                    telemetry.hard_faults += 1
                elif scr.outcome == AuditOutcome.ZERO_SKIP:
                    telemetry.zero_skips += 1
                if req:
                    want_payload[s] = 1
                    telemetry.payloads_requested += 1

    # ------------------------------------------------------------------
    # Phase B: Selective Compressed Payload Transfer + Two-Pass Audit
    # ------------------------------------------------------------------
    payload_matrix = _all_gather_int64_row(want_payload, world_size)
    max_wire = receiver.max_wire_bytes

    for s in range(world_size):
        for r in range(world_size):
            if s == r or int(payload_matrix[r][s].item()) == 0:
                continue
            if rank == s:
                raw = outbound.payload_bytes_for(r)
                hdr = torch.tensor([adv_sender, adv_round, adv_run_tag, len(raw)], dtype=torch.int64)
                dist.send(hdr, dst=r)
                go = torch.empty(1, dtype=torch.int64)
                dist.recv(go, src=r)
                if int(go.item()) == 1:
                    buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                    dist.send(buf, dst=r)
            elif rank == r:
                hdr = torch.empty(4, dtype=torch.int64)
                dist.recv(hdr, src=s)
                p_sender, p_round, p_run_tag, p_len = (int(x) for x in hdr.tolist())
                # Enforce receiver size limit BEFORE allocating or receiving payload bytes.
                if p_len <= 0 or p_len > max_wire:
                    dist.send(torch.tensor([0], dtype=torch.int64), dst=s)
                    res = AuditResult(AuditOutcome.HARD_FAULT, "size_limit_exceeded")
                else:
                    dist.send(torch.tensor([1], dtype=torch.int64), dst=s)
                    buf = torch.empty(p_len, dtype=torch.uint8)
                    dist.recv(buf, src=s)
                    bucket.consume(p_len)
                    telemetry.payloads_received += 1
                    telemetry.payload_bytes_received += p_len

                    env = PayloadEnvelope(
                        sender_id=p_sender,
                        run_id=receiver.run_id if p_run_tag == my_run_tag else f"mismatched:{p_run_tag}",
                        round_idx=p_round,
                        raw_bytes=bytes(buf.numpy().tobytes()),
                    )
                    res = receiver.audit_and_accumulate(announcements[s], env, expected_sender_id=s)

                telemetry.audit_results[s] = res
                quarantine.on_audit_result(s, round_idx, res)
                if res.outcome == AuditOutcome.ACCEPTED:
                    telemetry.payloads_accepted += 1
                elif res.outcome == AuditOutcome.SOFT_FAULT:
                    telemetry.soft_faults += 1
                elif res.outcome == AuditOutcome.HARD_FAULT:
                    telemetry.hard_faults += 1
                elif res.outcome == AuditOutcome.ZERO_SKIP:
                    telemetry.zero_skips += 1

    # ------------------------------------------------------------------
    # Phase C: Referrals when Active Degree < min_active_degree
    # ------------------------------------------------------------------
    ref_req = torch.zeros(world_size, dtype=torch.int64)
    if quarantine.needs_referral():
        for helper in quarantine.active_peers():
            if 0 <= helper < world_size:
                ref_req[helper] = 1
                telemetry.referrals_requested += 1
    ref_matrix = _all_gather_int64_row(ref_req, world_size)

    # Active helpers recommend one of their own active neighbors (or -1 if none).
    proposed_candidates: list[int] = []
    for helper in range(world_size):
        for requester in range(world_size):
            if helper == requester or int(ref_matrix[requester][helper].item()) == 0:
                continue
            if rank == helper:
                cand = quarantine.recommend_neighbor(requester_id=requester)
                val = -1 if cand is None else int(cand)
                dist.send(torch.tensor([val], dtype=torch.int64), dst=requester)
            elif rank == requester:
                reply = torch.empty(1, dtype=torch.int64)
                dist.recv(reply, src=helper)
                cand = int(reply.item())
                if cand >= 0 and cand not in proposed_candidates and quarantine.can_consider_referral(cand):
                    proposed_candidates.append(cand)

    # Probe eligible unbanned referral candidates by requesting their authenticated sketch.
    probe_req = torch.zeros(world_size, dtype=torch.int64)
    for cand in proposed_candidates:
        if 0 <= cand < world_size:
            probe_req[cand] = 1
    probe_matrix = _all_gather_int64_row(probe_req, world_size)

    for cand in range(world_size):
        for requester in range(world_size):
            if cand == requester or int(probe_matrix[requester][cand].item()) == 0:
                continue
            if rank == cand:
                _send_sketch(requester, outbound.sketch_for(requester), adv_sender, adv_round, adv_run_tag)
            elif rank == requester:
                ann = _recv_sketch(cand, receiver.layout.num_buckets, receiver.run_id, my_run_tag, bucket)
                telemetry.sketches_received += 1
                telemetry.sketch_bytes_received += int(ann.sketch.numel()) * 2
                scr = receiver.screen_announcement(ann, expected_sender_id=cand)
                if quarantine.admit_referral(cand, round_idx, scr):
                    telemetry.referrals_admitted.append(cand)

    # ------------------------------------------------------------------
    # Phase D: Finalize Round Update
    # ------------------------------------------------------------------
    blended_update, used_fallback = receiver.finalize_update(self_update)
    telemetry.used_self_fallback = used_fallback
    telemetry.active_peers_end = quarantine.active_peers()
    telemetry.probation_peers_end = quarantine.probation_peers()
    telemetry.banned_peers_end = sorted(quarantine.banned_peers)
    telemetry.receive_wait_sec = bucket.total_wait_sec - wait_before
    return blended_update, telemetry
