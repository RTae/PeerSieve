"""Two-pass audited streaming aggregation for PeerSieve (Section 3.3, Eqs. 3 & 4).

Key invariants enforced:
1. Check before accumulate: Pass 1 validates metadata, wire bounds, INT4 packing,
   exact bit-level CountSketch reproduction, full cosine > tau_audit, cosine
   discrepancy <= tau_c, and relative norm discrepancy <= tau_n. Any failure
   discards the payload with ZERO change to running_sum or total_weight.
2. Buffer reuse: A single reusable decode_tile buffer (default 4 MiB FP32) and
   a single running_sum buffer are reused across all neighbors; receiver tensor
   memory does not scale with neighbor degree.
3. Receiver-norm cap: On Pass 2, each accepted payload is scaled by
   a_ij = min(1, clip_factor * r_i / ||y_j||_2) (leaving smaller updates intact)
   and weighted by its full cosine w_ij = max(0, c_ij).
4. Hard vs. soft faults: Protocol/encoding/sketch mismatches return HARD_FAULT
   (permanent ban), whereas valid-sketch alignment/norm failures return
   SOFT_FAULT (recoverable probation), and zero updates return ZERO_SKIP.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from src.compression import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_TILE_NUMEL,
    HardProtocolFault,
    PackedINT4Payload,
    expected_int4_wire_bytes,
    iter_decoded_tiles,
    validate_packed_int4,
)
from src.sketch import (
    SketchLayout,
    StreamingCountSketch,
    compute_sketch,
    sketch_cosine_and_norms,
    sketches_exact_equal,
    validate_sketch_tensor,
)


class AuditOutcome(str, Enum):
    ACCEPTED = "accepted"
    ZERO_SKIP = "zero_skip"
    SOFT_FAULT = "soft_fault"
    HARD_FAULT = "hard_fault"


@dataclass(frozen=True)
class AuditConfig:
    tau_screen: float = 0.0
    tau_audit: float = 0.0
    tau_c: float = 0.05
    tau_n: float = 0.03
    block_size: int = DEFAULT_BLOCK_SIZE
    tile_numel: int = DEFAULT_TILE_NUMEL
    alpha: float = 0.5
    clip_factor: float = 1.0  # Proposal Eq. 4: a_ij = min(1, r_i / ||y_j||_2)
    enable_audit: bool = True
    enable_clipping: bool = True


@dataclass(frozen=True)
class SketchAnnouncement:
    sender_id: int
    run_id: str
    round_idx: int
    sketch: torch.Tensor  # float16, shape (num_buckets,)


@dataclass(frozen=True)
class PayloadEnvelope:
    sender_id: int
    run_id: str
    round_idx: int
    payload: PackedINT4Payload | None = None
    raw_bytes: bytes | None = None


@dataclass(frozen=True)
class ScreenResult:
    request_payload: bool
    sketch_cosine: float
    sketch_norm: float
    outcome: AuditOutcome  # ACCEPTED (pass screen), ZERO_SKIP, SOFT_FAULT (nonpositive screen), HARD_FAULT
    reason: str


@dataclass(frozen=True)
class AuditResult:
    outcome: AuditOutcome
    reason: str
    sketch_cosine: float | None = None
    full_cosine: float | None = None
    sketch_norm: float | None = None
    decoded_norm: float | None = None
    cosine_err: float | None = None
    norm_rel_err: float | None = None
    clip_factor_applied: float = 0.0
    weight_applied: float = 0.0
    clipped_contribution_norm: float = 0.0


class AuditedStreamingReceiver:
    """Receiver-side reusable state for one or more PeerSieve aggregation rounds."""

    def __init__(
        self,
        self_id: int,
        run_id: str,
        layout: SketchLayout,
        cfg: AuditConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.self_id = int(self_id)
        self.run_id = str(run_id)
        self.layout = layout
        self.cfg = cfg if cfg is not None else AuditConfig()
        self.device = torch.device(device)

        tile_len = max(2, min(int(self.cfg.tile_numel), max(layout.total_numel, 2)))
        if tile_len % 2 == 1 and tile_len > 2:
            tile_len -= 1
        # Reusable buffers allocated once and shared across all neighbors & rounds.
        self.decode_tile = torch.empty(tile_len, dtype=torch.float32, device=self.device)
        self.running_sum = torch.zeros(layout.total_numel, dtype=torch.float32, device=self.device)
        self.max_wire_bytes = expected_int4_wire_bytes(layout.total_numel, self.cfg.block_size)

        # Per-round local reference state (initialized via start_round).
        self.round_idx: int = -1
        self.round_seed: int = 0
        self.total_weight: float = 0.0
        self.accepted_senders: list[int] = []
        self._val_dir: torch.Tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        self._val_norm: float = 0.0
        self._val_sketch: torch.Tensor = torch.empty(0, dtype=torch.float16, device=self.device)
        self._val_sketch_norm: float = 0.0
        self._self_update_norm: float = 0.0
        self._staged_payload_bytes: int = 0

    @torch.no_grad()
    def start_round(
        self,
        round_idx: int,
        round_seed: int,
        val_descent_dir: torch.Tensor,
        self_update_norm: float,
    ) -> torch.Tensor:
        """Reset round accumulators and compute the receiver's validation descent sketch.

        Note: val_descent_dir is g_i^val = -nabla L_i^val(w_i^(t)), aligned with
        after-minus-before SGD/AdamW updates (Delta w = w_after - w_before).
        """
        if val_descent_dir.ndim != 1 or val_descent_dir.numel() != self.layout.total_numel:
            raise ValueError(
                f"val_descent_dir length {val_descent_dir.numel()} != layout.total_numel {self.layout.total_numel}"
            )
        if self_update_norm < 0.0 or not math.isfinite(self_update_norm):
            raise ValueError(f"Invalid self_update_norm={self_update_norm}")

        self.round_idx = int(round_idx)
        self.round_seed = int(round_seed)
        self.running_sum.zero_()
        self.total_weight = 0.0
        self.accepted_senders.clear()
        self._staged_payload_bytes = 0

        self._val_dir = val_descent_dir.detach().to(device=self.device, dtype=torch.float32)
        self._val_norm = float(torch.linalg.vector_norm(self._val_dir).item())
        self._val_sketch = compute_sketch(
            self._val_dir,
            layout=self.layout,
            round_seed=self.round_seed,
            round_idx=self.round_idx,
            tile_numel=self.decode_tile.numel(),
        )
        self._val_sketch_norm = float(torch.linalg.vector_norm(self._val_sketch.float()).item())
        self._self_update_norm = float(self_update_norm)
        return self._val_sketch

    def receive_side_tensor_bytes(self, staged_payload: PackedINT4Payload | None = None) -> int:
        """Report total receive-side tensor buffer bytes (independent of neighbor count)."""
        staged = staged_payload.tensor_bytes if staged_payload is not None else self._staged_payload_bytes
        fixed = (
            self.decode_tile.numel() * self.decode_tile.element_size()
            + self.running_sum.numel() * self.running_sum.element_size()
            + self._val_dir.numel() * self._val_dir.element_size()
            + self._val_sketch.numel() * self._val_sketch.element_size()
        )
        return int(fixed + staged)

    @torch.no_grad()
    def screen_announcement(self, announcement: SketchAnnouncement, expected_sender_id: int) -> ScreenResult:
        """Pre-transfer sketch screen (Section 3.1, Eq. 1)."""
        if announcement.sender_id != expected_sender_id:
            return ScreenResult(False, 0.0, 0.0, AuditOutcome.HARD_FAULT, "sender_mismatch")
        if announcement.run_id != self.run_id:
            return ScreenResult(False, 0.0, 0.0, AuditOutcome.HARD_FAULT, "run_mismatch")
        if announcement.round_idx != self.round_idx:
            return ScreenResult(False, 0.0, 0.0, AuditOutcome.HARD_FAULT, "round_mismatch")

        try:
            validate_sketch_tensor(announcement.sketch, expected_buckets=self.layout.num_buckets)
        except HardProtocolFault as exc:
            return ScreenResult(False, 0.0, 0.0, AuditOutcome.HARD_FAULT, exc.reason)

        cos_hat, s_norm, v_norm = sketch_cosine_and_norms(announcement.sketch, self._val_sketch)
        if s_norm == 0.0 or v_norm == 0.0:
            return ScreenResult(False, 0.0, s_norm, AuditOutcome.ZERO_SKIP, "zero_sketch")
        if cos_hat > self.cfg.tau_screen:
            return ScreenResult(True, cos_hat, s_norm, AuditOutcome.ACCEPTED, "screen_pass")
        return ScreenResult(False, cos_hat, s_norm, AuditOutcome.SOFT_FAULT, "screen_nonpositive")

    @torch.no_grad()
    def audit_and_accumulate(
        self,
        announcement: SketchAnnouncement,
        envelope: PayloadEnvelope,
        expected_sender_id: int,
    ) -> AuditResult:
        """Execute Pass 1 (full audit) and, only on full pass, Pass 2 (clip, weight, accumulate)."""
        # Envelope authentication and replay binding.
        if announcement.sender_id != expected_sender_id or envelope.sender_id != expected_sender_id:
            return AuditResult(AuditOutcome.HARD_FAULT, "sender_mismatch")
        if announcement.run_id != self.run_id or envelope.run_id != self.run_id:
            return AuditResult(AuditOutcome.HARD_FAULT, "run_mismatch")
        if announcement.round_idx != self.round_idx or envelope.round_idx != self.round_idx:
            return AuditResult(AuditOutcome.HARD_FAULT, "round_mismatch")

        try:
            validate_sketch_tensor(announcement.sketch, expected_buckets=self.layout.num_buckets)
            if envelope.raw_bytes is not None:
                payload = PackedINT4Payload.from_bytes(envelope.raw_bytes, max_bytes=self.max_wire_bytes)
            elif envelope.payload is not None:
                payload = envelope.payload
            else:
                raise HardProtocolFault("missing_payload", "envelope contains neither payload nor raw_bytes")
            validate_packed_int4(
                payload,
                expected_numel=self.layout.total_numel,
                expected_block_size=self.cfg.block_size,
                max_wire_bytes=self.max_wire_bytes,
            )
        except HardProtocolFault as exc:
            return AuditResult(AuditOutcome.HARD_FAULT, exc.reason)

        # Stage only this single neighbor's compressed payload while auditing.
        self._staged_payload_bytes = payload.tensor_bytes
        try:
            sketch_cos, s_norm, _ = sketch_cosine_and_norms(announcement.sketch, self._val_sketch)

            # --- Pass 1: Stream payload into reusable decode_tile to rebuild sketch & measure exact dot/norm ---
            sketcher = (
                StreamingCountSketch(
                    layout=self.layout,
                    round_seed=self.round_seed,
                    round_idx=self.round_idx,
                    device=self.device,
                )
                if self.cfg.enable_audit
                else None
            )

            dot_val = 0.0
            norm_sq = 0.0
            for start, end, tile_view in iter_decoded_tiles(payload, self.decode_tile):
                if not bool(torch.isfinite(tile_view).all().item()):
                    return AuditResult(AuditOutcome.HARD_FAULT, "nonfinite_payload")
                if sketcher is not None:
                    sketcher.update(tile_view)
                # Accumulate scalar reductions in FP64 to prevent catastrophic cancellation.
                t64 = tile_view.double()
                v64 = self._val_dir[start:end].double()
                dot_val += float(torch.dot(t64, v64).item())
                norm_sq += float(torch.dot(t64, t64).item())

            y_norm = math.sqrt(max(norm_sq, 0.0))
            full_cos = (dot_val / (y_norm * self._val_norm)) if (y_norm > 0.0 and self._val_norm > 0.0) else 0.0

            if self.cfg.enable_audit:
                assert sketcher is not None
                try:
                    rebuilt_sketch = sketcher.finalize()
                except HardProtocolFault as exc:
                    return AuditResult(AuditOutcome.HARD_FAULT, exc.reason)

                # Hard fault: decoded payload must reproduce the announced FP16 sketch bit-for-bit.
                if not sketches_exact_equal(rebuilt_sketch, announcement.sketch):
                    return AuditResult(
                        AuditOutcome.HARD_FAULT,
                        "sketch_mismatch",
                        sketch_cosine=sketch_cos,
                        full_cosine=full_cos,
                        sketch_norm=s_norm,
                        decoded_norm=y_norm,
                    )

                # Zero payload skips cleanly without ban or probation.
                if y_norm == 0.0:
                    return AuditResult(
                        AuditOutcome.ZERO_SKIP,
                        "zero_payload",
                        sketch_cosine=sketch_cos,
                        full_cosine=0.0,
                        sketch_norm=s_norm,
                        decoded_norm=0.0,
                    )

                cos_err = abs(sketch_cos - full_cos)
                norm_rel_err = abs((s_norm / y_norm) - 1.0)

                if full_cos <= self.cfg.tau_audit:
                    return AuditResult(
                        AuditOutcome.SOFT_FAULT,
                        "nonpositive_full_cosine",
                        sketch_cosine=sketch_cos,
                        full_cosine=full_cos,
                        sketch_norm=s_norm,
                        decoded_norm=y_norm,
                        cosine_err=cos_err,
                        norm_rel_err=norm_rel_err,
                    )
                if cos_err > self.cfg.tau_c:
                    return AuditResult(
                        AuditOutcome.SOFT_FAULT,
                        "cosine_discrepancy",
                        sketch_cosine=sketch_cos,
                        full_cosine=full_cos,
                        sketch_norm=s_norm,
                        decoded_norm=y_norm,
                        cosine_err=cos_err,
                        norm_rel_err=norm_rel_err,
                    )
                if norm_rel_err > self.cfg.tau_n:
                    return AuditResult(
                        AuditOutcome.SOFT_FAULT,
                        "norm_discrepancy",
                        sketch_cosine=sketch_cos,
                        full_cosine=full_cos,
                        sketch_norm=s_norm,
                        decoded_norm=y_norm,
                        cosine_err=cos_err,
                        norm_rel_err=norm_rel_err,
                    )

                weight = max(0.0, full_cos)
            else:
                # No-audit ablation (proposal Section 4.2): use sketch weight + decoded-norm clip.
                if y_norm == 0.0:
                    return AuditResult(AuditOutcome.ZERO_SKIP, "zero_payload", decoded_norm=0.0)
                cos_err = abs(sketch_cos - full_cos)
                norm_rel_err = abs((s_norm / y_norm) - 1.0)
                weight = max(0.0, sketch_cos)
                if weight <= 0.0:
                    return AuditResult(
                        AuditOutcome.SOFT_FAULT,
                        "nonpositive_sketch_cosine",
                        sketch_cosine=sketch_cos,
                        full_cosine=full_cos,
                        sketch_norm=s_norm,
                        decoded_norm=y_norm,
                        cosine_err=cos_err,
                        norm_rel_err=norm_rel_err,
                    )

            # --- Pass 2: Clip by receiver's own update norm, weight, and accumulate into running_sum ---
            if self.cfg.enable_clipping:
                cap = self.cfg.clip_factor * self._self_update_norm
                clip_applied = min(1.0, cap / y_norm) if y_norm > 0.0 else 0.0
            else:
                clip_applied = 1.0

            coeff = weight * clip_applied
            if coeff > 0.0:
                for start, end, tile_view in iter_decoded_tiles(payload, self.decode_tile):
                    self.running_sum[start:end].add_(tile_view, alpha=coeff)
            self.total_weight += weight
            self.accepted_senders.append(expected_sender_id)

            return AuditResult(
                AuditOutcome.ACCEPTED,
                "accepted",
                sketch_cosine=sketch_cos,
                full_cosine=full_cos,
                sketch_norm=s_norm,
                decoded_norm=y_norm,
                cosine_err=cos_err,
                norm_rel_err=norm_rel_err,
                clip_factor_applied=clip_applied,
                weight_applied=weight,
                clipped_contribution_norm=clip_applied * y_norm,
            )
        finally:
            self._staged_payload_bytes = 0

    @torch.no_grad()
    def finalize_update(self, self_update: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Blend weighted accepted neighbor average with self_update (Section 3.3, Eq. 4).

        Returns (blended_update, used_self_fallback).
        """
        own = self_update.detach().to(device=self.device, dtype=torch.float32)
        if self.total_weight <= 0.0:
            return own.clone(), True
        nbr_avg = self.running_sum / float(self.total_weight)
        blended = self.cfg.alpha * own + (1.0 - self.cfg.alpha) * nbr_avg
        return blended, False
