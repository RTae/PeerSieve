"""Deterministic layer-proportional CountSketch for PeerSieve pre-transfer screening & audit.

Implements PeerSieve Sections 3.1 & 3.3 (Eq. 1 & 3):
- Divides num_buckets (default 16,384 -> 32 KiB in FP16) across model layers in
  proportion to parameter count.
- Assigns each coordinate a deterministic bucket and sign from the public round
  seed using overflow-free Carter-Wegman polynomial hashing modulo P = 2^31 - 1.
- Accumulates signed coordinates in a fixed canonical FP32 pairwise order so
  sender and receiver produce bit-identical FP16 sketches regardless of tile
  size or CPU/GPU execution backend.
- Enforces canonical +0.0 zero representation and rejects nonfinite sketches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Sequence

import torch
import torch.nn as nn

from src.compression import HardProtocolFault, PackedINT4Payload, iter_decoded_tiles

DEFAULT_NUM_BUCKETS: int = 16_384  # 32 KiB in float16
DEFAULT_CANONICAL_CHUNK: int = 8_192  # 32 KiB carry buffer for tile-invariant accumulation
_MERSENNE_P: int = 2_147_483_647  # 2^31 - 1 (products < 2^62 fit safely in signed int64)


def allocate_layer_buckets(layer_sizes: Sequence[int], num_buckets: int = DEFAULT_NUM_BUCKETS) -> tuple[int, ...]:
    """Allocate num_buckets across layers proportionally to parameter count (Hamilton method)."""
    if num_buckets <= 0:
        raise ValueError(f"num_buckets must be positive, got {num_buckets}")
    sizes = [int(s) for s in layer_sizes]
    if any(s < 0 for s in sizes):
        raise ValueError("layer_sizes must be non-negative")
    nonempty = [i for i, s in enumerate(sizes) if s > 0]
    if not nonempty:
        return tuple(0 for _ in sizes)
    if len(nonempty) > num_buckets:
        raise ValueError(
            f"Cannot assign distinct buckets to {len(nonempty)} non-empty layers with only {num_buckets} buckets"
        )

    counts = [0] * len(sizes)
    for i in nonempty:
        counts[i] = 1
    remaining = num_buckets - len(nonempty)
    if remaining == 0:
        return tuple(counts)

    total = sum(sizes[i] for i in nonempty)
    quotas = [(i, (sizes[i] * remaining) / total) for i in nonempty]
    floors = {i: int(q) for i, q in quotas}
    for i, f in floors.items():
        counts[i] += f
    leftover = remaining - sum(floors.values())
    if leftover > 0:
        # Largest-remainder tie-break: higher remainder first, then lower layer index.
        ranked = sorted(quotas, key=lambda item: (-(item[1] - floors[item[0]]), item[0]))
        for i, _ in ranked[:leftover]:
            counts[i] += 1
    assert sum(counts) == num_buckets
    return tuple(counts)


@dataclass(frozen=True)
class SketchLayout:
    """Immutable parameter-to-bucket layout shared by sender and receiver."""

    layer_sizes: tuple[int, ...]
    layer_starts: tuple[int, ...]
    layer_ends: tuple[int, ...]
    bucket_starts: tuple[int, ...]
    bucket_counts: tuple[int, ...]
    num_buckets: int
    total_numel: int

    @classmethod
    def from_layer_sizes(cls, layer_sizes: Sequence[int], num_buckets: int = DEFAULT_NUM_BUCKETS) -> SketchLayout:
        sizes = tuple(int(s) for s in layer_sizes)
        if any(s < 0 for s in sizes):
            raise ValueError("layer_sizes must be non-negative")
        b_counts = allocate_layer_buckets(sizes, num_buckets=num_buckets)
        l_starts: list[int] = []
        l_ends: list[int] = []
        b_starts: list[int] = []
        p_cur = 0
        b_cur = 0
        for s, bc in zip(sizes, b_counts):
            l_starts.append(p_cur)
            p_cur += s
            l_ends.append(p_cur)
            b_starts.append(b_cur)
            b_cur += bc
        return cls(
            layer_sizes=sizes,
            layer_starts=tuple(l_starts),
            layer_ends=tuple(l_ends),
            bucket_starts=tuple(b_starts),
            bucket_counts=b_counts,
            num_buckets=num_buckets,
            total_numel=p_cur,
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        num_buckets: int = DEFAULT_NUM_BUCKETS,
        trainable_only: bool = True,
    ) -> SketchLayout:
        sizes = [int(p.numel()) for p in model.parameters() if (p.requires_grad or not trainable_only)]
        return cls.from_layer_sizes(sizes, num_buckets=num_buckets)

    @classmethod
    def flat(cls, numel: int, num_buckets: int = DEFAULT_NUM_BUCKETS) -> SketchLayout:
        return cls.from_layer_sizes([int(numel)], num_buckets=num_buckets)


def _derive_round_hash_keys(round_seed: int, round_idx: int) -> tuple[int, ...]:
    """Derive 10 deterministic 31-bit hash coefficients from (round_seed, round_idx)."""
    msg = f"peersieve-countsketch-v1:{int(round_seed)}:{int(round_idx)}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=64).digest()
    raw = struct.unpack("<10I", digest[:40])
    # Keep coefficients in [1, P - 1] so linear/cubic terms are never degenerate.
    return tuple(1 + (val % (_MERSENNE_P - 1)) for val in raw)


def canonicalize_sketch_fp16(accum_fp32: torch.Tensor) -> torch.Tensor:
    """Round FP32 bucket sums to FP16, enforce +0.0 canonical zero, and reject nonfinite values."""
    if not bool(torch.isfinite(accum_fp32).all().item()):
        raise HardProtocolFault("nonfinite_sketch", "FP32 sketch accumulator contains NaN or Inf")
    sketch = accum_fp32.to(torch.float16)
    if not bool(torch.isfinite(sketch).all().item()):
        raise HardProtocolFault("nonfinite_sketch", "FP16 sketch overflowed to Inf or NaN")
    # Enforce single canonical representation (+0.0, bit pattern 0x0000) for all zeros.
    sketch = torch.where(sketch == 0, torch.zeros_like(sketch), sketch)
    return sketch.contiguous()


def validate_sketch_tensor(sketch: torch.Tensor, expected_buckets: int = DEFAULT_NUM_BUCKETS) -> None:
    """Validate an announced FP16 CountSketch tensor before screening or audit."""
    if sketch.dtype != torch.float16 or sketch.ndim != 1 or sketch.numel() != expected_buckets:
        raise HardProtocolFault(
            "invalid_sketch",
            f"expected ({expected_buckets},) float16 sketch, got {sketch.dtype} {tuple(sketch.shape)}",
        )
    if not bool(torch.isfinite(sketch).all().item()):
        raise HardProtocolFault("nonfinite_sketch", "announced sketch contains NaN or Inf")
    # Reject -0.0 (bit pattern -32768 == 0x8000 in int16 view).
    bits = sketch.contiguous().view(torch.int16)
    if bool((bits == -32768).any().item()):
        raise HardProtocolFault("noncanonical_zero", "announced sketch contains negative zero (-0.0)")


def sketches_exact_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact equality check between two FP16 sketch tensors."""
    if a.dtype != torch.float16 or b.dtype != torch.float16 or a.shape != b.shape:
        return False
    a_bits = a.detach().contiguous().view(torch.int16)
    b_bits = b.detach().to(device=a.device).contiguous().view(torch.int16)
    return bool(torch.equal(a_bits, b_bits))


@torch.no_grad()
def sketch_cosine_and_norms(s_a: torch.Tensor, s_b: torch.Tensor) -> tuple[float, float, float]:
    """Compute (cosine, ||s_a||_2, ||s_b||_2) in FP32 from two FP16 sketches."""
    a = s_a.detach().float()
    b = s_b.detach().to(device=a.device, dtype=torch.float32)
    norm_a = float(torch.linalg.vector_norm(a).item())
    norm_b = float(torch.linalg.vector_norm(b).item())
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0, norm_a, norm_b
    dot = float(torch.dot(a, b).item())
    return dot / (norm_a * norm_b), norm_a, norm_b


class StreamingCountSketch:
    """Tile-size-invariant, CPU/CUDA bit-exact streaming FP32 -> FP16 CountSketch builder."""

    def __init__(
        self,
        layout: SketchLayout,
        round_seed: int,
        round_idx: int,
        device: torch.device | str = "cpu",
        canonical_chunk: int = DEFAULT_CANONICAL_CHUNK,
    ) -> None:
        if canonical_chunk <= 0:
            raise ValueError(f"canonical_chunk must be positive, got {canonical_chunk}")
        self.layout = layout
        self.round_seed = int(round_seed)
        self.round_idx = int(round_idx)
        self.device = torch.device(device)
        self.canonical_chunk = int(canonical_chunk)

        self._keys = _derive_round_hash_keys(self.round_seed, self.round_idx)
        self._accum = torch.zeros(layout.num_buckets, dtype=torch.float32, device=self.device)
        self._carry = torch.empty(0, dtype=torch.float32, device=self.device)
        self._processed = 0  # Number of elements already accumulated into self._accum
        self._fed = 0  # Total elements fed via update() (including carry)
        self._nonfinite = False

        # Filter to non-empty layers for O(log L) coordinate-to-layer lookup.
        nonempty = [i for i, s in enumerate(layout.layer_sizes) if s > 0]
        self._layer_ends = torch.tensor([layout.layer_ends[i] for i in nonempty], dtype=torch.int64, device=self.device)
        self._layer_starts = torch.tensor(
            [layout.layer_starts[i] for i in nonempty], dtype=torch.int64, device=self.device
        )
        self._bucket_starts = torch.tensor(
            [layout.bucket_starts[i] for i in nonempty], dtype=torch.int64, device=self.device
        )
        self._bucket_counts = torch.tensor(
            [layout.bucket_counts[i] for i in nonempty], dtype=torch.int64, device=self.device
        )

    @torch.no_grad()
    def _hash_range(self, global_start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic (bucket_ids, signs) for global indices [global_start, global_start + length)."""
        g_idx = torch.arange(global_start, global_start + length, dtype=torch.int64, device=self.device)
        layer_pos = torch.searchsorted(self._layer_ends, g_idx, right=True)
        local_idx = g_idx - self._layer_starts[layer_pos]
        b_start = self._bucket_starts[layer_pos]
        b_count = self._bucket_counts[layer_pos]

        # Split global and local indices into 31-bit limbs for overflow-free arithmetic mod P = 2^31 - 1.
        lo = g_idx & 0x7FFF_FFFF
        hi = g_idx >> 31
        loc_lo = local_idx & 0x7FFF_FFFF
        (a1, b1, c1, d1, s3, s2, s1, s0, m1, m2) = self._keys

        # Bucket hash: 2-step polynomial mix over (global_idx, local_idx) mod P, mapped into layer's bucket slice.
        h_b = (a1 * lo + b1 * hi + c1 * loc_lo + d1) % _MERSENNE_P
        h_b = (m1 * h_b + m2) % _MERSENNE_P
        buckets = b_start + (h_b % b_count)

        # Sign hash: 4-wise independent degree-3 Carter-Wegman polynomial mod P.
        x = (lo + hi * 17) % _MERSENNE_P
        h_s = (s3 * x + s2) % _MERSENNE_P
        h_s = (h_s * x + s1) % _MERSENNE_P
        h_s = (h_s * x + s0) % _MERSENNE_P
        signs = torch.where((h_s & 1) == 0, 1.0, -1.0).to(dtype=torch.float32)
        return buckets, signs

    @torch.no_grad()
    def _accumulate_canonical_chunk(self, chunk: torch.Tensor, global_start: int) -> None:
        """Reduce one canonical chunk into self._accum in a fixed deterministic FP32 pairwise order."""
        length = int(chunk.numel())
        if length == 0:
            return
        if not bool(torch.isfinite(chunk).all().item()):
            self._nonfinite = True
            return

        buckets, signs = self._hash_range(global_start, length)
        signed_vals = chunk.float() * signs

        # Stable sort by bucket preserves ascending global coordinate order within each bucket.
        order = torch.argsort(buckets, stable=True)
        b_cur = buckets[order]
        v_cur = signed_vals[order]

        # Deterministic O(log max_run) segmented pairwise FP32 reduction: zero atomicAdd, O(chunk) memory.
        while b_cur.numel() > 0:
            n = int(b_cur.numel())
            is_start = torch.ones(n, dtype=torch.bool, device=self.device)
            if n > 1:
                is_start[1:] = b_cur[1:] != b_cur[:-1]
            has_next = torch.zeros(n, dtype=torch.bool, device=self.device)
            if n > 1:
                has_next[:-1] = b_cur[:-1] == b_cur[1:]

            # Flush singleton runs (unique buckets -> collision-free addition into self._accum).
            is_singleton = is_start & (~has_next)
            if bool(is_singleton.any().item()):
                self._accum[b_cur[is_singleton]] += v_cur[is_singleton]

            # Pairwise-sum even/odd elements inside every multi-element run.
            idx = torch.arange(n, dtype=torch.int64, device=self.device)
            run_starts = idx[is_start]
            run_id = torch.cumsum(is_start.to(torch.int64), dim=0) - 1
            pos = idx - run_starts[run_id]

            is_even = (pos & 1) == 0
            carry_mask = is_even & (~is_singleton)
            if not bool(carry_mask.any().item()):
                break

            pair_left = is_even & has_next
            pair_right = torch.zeros(n, dtype=torch.bool, device=self.device)
            if n > 1:
                pair_right[1:] = pair_left[:-1]

            v_next = v_cur.clone()
            if bool(pair_left.any().item()):
                v_next[pair_left] = v_cur[pair_left] + v_cur[pair_right]

            b_cur = b_cur[carry_mask]
            v_cur = v_next[carry_mask]

    @torch.no_grad()
    def update(self, tile: torch.Tensor) -> None:
        """Consume the next contiguous tile of coordinates in ascending order."""
        if tile.ndim != 1:
            raise ValueError(f"Expected 1D tile tensor, got shape {tuple(tile.shape)}")
        n = int(tile.numel())
        if n == 0:
            return
        if self._fed + n > self.layout.total_numel:
            raise ValueError(f"Fed {self._fed + n} elements, exceeding layout.total_numel={self.layout.total_numel}")
        self._fed += n

        t = tile.detach().to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(t).all().item()):
            self._nonfinite = True
            return

        if self._carry.numel() > 0:
            need = self.canonical_chunk - int(self._carry.numel())
            if n < need:
                self._carry = torch.cat([self._carry, t])
                return
            full = torch.cat([self._carry, t[:need]])
            self._accumulate_canonical_chunk(full, self._processed)
            self._processed += self.canonical_chunk
            self._carry = torch.empty(0, dtype=torch.float32, device=self.device)
            t = t[need:]

        offset = 0
        rem = int(t.numel())
        while offset + self.canonical_chunk <= rem:
            chunk = t[offset : offset + self.canonical_chunk]
            self._accumulate_canonical_chunk(chunk, self._processed)
            self._processed += self.canonical_chunk
            offset += self.canonical_chunk

        if offset < rem:
            self._carry = t[offset:].clone()

    @torch.no_grad()
    def finalize(self) -> torch.Tensor:
        """Flush any trailing carry buffer and return canonical FP16 sketch."""
        if self._nonfinite:
            raise HardProtocolFault("nonfinite_sketch", "nonfinite values encountered while building sketch")
        if self._fed != self.layout.total_numel:
            raise ValueError(f"Incomplete sketch: fed {self._fed} elements, expected {self.layout.total_numel}")
        if self._carry.numel() > 0:
            self._accumulate_canonical_chunk(self._carry, self._processed)
            self._processed += int(self._carry.numel())
            self._carry = torch.empty(0, dtype=torch.float32, device=self.device)
        if self._nonfinite:
            raise HardProtocolFault("nonfinite_sketch", "nonfinite values encountered in final sketch chunk")
        return canonicalize_sketch_fp16(self._accum)


@torch.no_grad()
def compute_sketch(
    vector: torch.Tensor,
    layout: SketchLayout,
    round_seed: int,
    round_idx: int,
    tile_numel: int | None = None,
) -> torch.Tensor:
    """Compute canonical FP16 CountSketch of a 1D tensor (optionally streamed in tiles)."""
    if vector.ndim != 1 or vector.numel() != layout.total_numel:
        raise ValueError(f"Expected 1D tensor of length {layout.total_numel}, got {tuple(vector.shape)}")
    builder = StreamingCountSketch(layout=layout, round_seed=round_seed, round_idx=round_idx, device=vector.device)
    step = layout.total_numel if (tile_numel is None or tile_numel <= 0) else int(tile_numel)
    for start in range(0, layout.total_numel, max(step, 1)):
        builder.update(vector[start : start + step])
    return builder.finalize()


@torch.no_grad()
def sketch_from_packed_int4(
    payload: PackedINT4Payload,
    layout: SketchLayout,
    round_seed: int,
    round_idx: int,
    tile_buf: torch.Tensor,
) -> torch.Tensor:
    """Build canonical FP16 CountSketch directly from a PackedINT4Payload using a reusable tile buffer."""
    if payload.numel != layout.total_numel:
        raise HardProtocolFault("length_mismatch", f"payload numel {payload.numel} != layout {layout.total_numel}")
    builder = StreamingCountSketch(layout=layout, round_seed=round_seed, round_idx=round_idx, device=tile_buf.device)
    for _, _, view in iter_decoded_tiles(payload, tile_buf):
        builder.update(view)
    return builder.finalize()
