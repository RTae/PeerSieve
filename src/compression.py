"""Blockwise INT4 update compression with error feedback and tiled decoding.

Implements PeerSieve Section 3.2 (Eq. 2):
  u_{j,b}^{(t)} = Delta w_{j,b}^{(t)} + e_{j,b}^{(t)}
  s_b = fp16(max |u_{j,b}^{(t)}| / 7)
  q_{j,b}^{(t)} = clip(round(u_{j,b}^{(t)} / s_b), -7, 7)  for s_b > 0
  y_{j,b}^{(t)} = s_b * q_{j,b}^{(t)}  (zero block if s_b == 0)
  e_{j,b}^{(t+1)} = u_{j,b}^{(t)} - y_{j,b}^{(t)}

Packed wire format stores two signed codes in [-7, 7] per uint8 byte using
unsigned offset nibbles (q + 7 in [0, 14]). Nibble value 15 is reserved and
rejected as a hard protocol fault during receiver validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterator

import torch

DEFAULT_BLOCK_SIZE: int = 128
DEFAULT_TILE_BYTES: int = 4 * 1024 * 1024  # 4 MiB reusable FP32 decode tile
DEFAULT_TILE_NUMEL: int = DEFAULT_TILE_BYTES // 4  # 1,048,576 FP32 elements
INT4_MAX_CODE: int = 7
NIBBLE_OFFSET: int = 7
RESERVED_NIBBLE: int = 15

_WIRE_MAGIC = b"PS4\x01"
_WIRE_HEADER = struct.Struct("<4sQIQQ")  # magic, numel, block_size, codes_len, scales_len
_FP16_MAX = float(torch.finfo(torch.float16).max)


class HardProtocolFault(ValueError):
    """Raised when a received payload violates wire format, bounds, or sketch invariants."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        msg = f"{reason}: {detail}" if detail else reason
        super().__init__(msg)


@dataclass(frozen=True)
class PackedINT4Payload:
    """Compact blockwise INT4 payload (codes + FP16 block scales)."""

    numel: int
    block_size: int
    codes: torch.Tensor  # uint8, shape (ceil(numel / 2),)
    scales: torch.Tensor  # float16, shape (ceil(numel / block_size),)

    @property
    def n_blocks(self) -> int:
        return math.ceil(self.numel / self.block_size) if self.numel > 0 else 0

    @property
    def tensor_bytes(self) -> int:
        """Byte size of codes and FP16 scales (excluding framing header)."""
        return int(self.codes.numel()) * self.codes.element_size() + int(self.scales.numel()) * self.scales.element_size()

    @property
    def wire_bytes(self) -> int:
        """Total serialized wire size including header."""
        return _WIRE_HEADER.size + self.tensor_bytes

    def to(self, device: torch.device | str) -> PackedINT4Payload:
        return PackedINT4Payload(
            numel=self.numel,
            block_size=self.block_size,
            codes=self.codes.to(device),
            scales=self.scales.to(device),
        )

    def to_bytes(self) -> bytes:
        codes_cpu = self.codes.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        scales_cpu = self.scales.detach().to(device="cpu", dtype=torch.float16).contiguous()
        codes_bytes = codes_cpu.numpy().tobytes()
        scales_bytes = scales_cpu.numpy().tobytes()
        header = _WIRE_HEADER.pack(_WIRE_MAGIC, self.numel, self.block_size, len(codes_bytes), len(scales_bytes))
        return header + codes_bytes + scales_bytes

    @classmethod
    def from_bytes(cls, raw: bytes | bytearray | memoryview, max_bytes: int | None = None) -> PackedINT4Payload:
        view = memoryview(raw)
        if max_bytes is not None and len(view) > max_bytes:
            raise HardProtocolFault("size_limit_exceeded", f"{len(view)} > {max_bytes}")
        if len(view) < _WIRE_HEADER.size:
            raise HardProtocolFault("invalid_header", "truncated wire header")
        magic, numel, block_size, codes_len, scales_len = _WIRE_HEADER.unpack_from(view, 0)
        if magic != _WIRE_MAGIC:
            raise HardProtocolFault("invalid_header", f"unexpected magic {magic!r}")
        if block_size <= 0:
            raise HardProtocolFault("invalid_header", f"block_size={block_size}")
        expected_total = _WIRE_HEADER.size + codes_len + scales_len
        if len(view) != expected_total:
            raise HardProtocolFault("length_mismatch", f"got {len(view)} bytes, header claims {expected_total}")
        if scales_len % 2 != 0:
            raise HardProtocolFault("length_mismatch", f"odd float16 byte length {scales_len}")

        offset = _WIRE_HEADER.size
        codes = torch.frombuffer(bytearray(view[offset : offset + codes_len]), dtype=torch.uint8).clone()
        offset += codes_len
        scales = torch.frombuffer(bytearray(view[offset : offset + scales_len]), dtype=torch.float16).clone()
        return cls(numel=int(numel), block_size=int(block_size), codes=codes, scales=scales)


def expected_int4_wire_bytes(numel: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    """Exact wire byte count for a valid PackedINT4Payload of length numel."""
    if numel < 0 or block_size <= 0:
        raise ValueError("numel must be >= 0 and block_size > 0")
    codes_len = (numel + 1) // 2
    n_blocks = math.ceil(numel / block_size) if numel > 0 else 0
    return _WIRE_HEADER.size + codes_len + 2 * n_blocks


def validate_packed_int4(
    payload: PackedINT4Payload,
    expected_numel: int,
    expected_block_size: int = DEFAULT_BLOCK_SIZE,
    max_wire_bytes: int | None = None,
) -> None:
    """Strictly validate packed INT4 payload structure and encodings before decoding."""
    if payload.numel != expected_numel:
        raise HardProtocolFault("length_mismatch", f"numel {payload.numel} != expected {expected_numel}")
    if payload.block_size != expected_block_size:
        raise HardProtocolFault("invalid_header", f"block_size {payload.block_size} != {expected_block_size}")
    if max_wire_bytes is not None and payload.wire_bytes > max_wire_bytes:
        raise HardProtocolFault("size_limit_exceeded", f"{payload.wire_bytes} > {max_wire_bytes}")

    expected_codes = (expected_numel + 1) // 2
    expected_blocks = math.ceil(expected_numel / expected_block_size) if expected_numel > 0 else 0

    if payload.codes.dtype != torch.uint8 or payload.codes.ndim != 1 or payload.codes.numel() != expected_codes:
        raise HardProtocolFault(
            "invalid_encoding",
            f"codes dtype={payload.codes.dtype}, shape={tuple(payload.codes.shape)}, expected ({expected_codes},) uint8",
        )
    if payload.scales.dtype != torch.float16 or payload.scales.ndim != 1 or payload.scales.numel() != expected_blocks:
        raise HardProtocolFault(
            "invalid_encoding",
            f"scales dtype={payload.scales.dtype}, shape={tuple(payload.scales.shape)}, expected ({expected_blocks},) fp16",
        )

    if expected_blocks > 0:
        if not bool(torch.isfinite(payload.scales).all().item()):
            raise HardProtocolFault("nonfinite_scale", "payload contains NaN or Inf FP16 block scales")
        if bool((payload.scales < 0).any().item()):
            raise HardProtocolFault("negative_scale", "payload contains negative FP16 block scales")

    if expected_codes > 0:
        low = payload.codes & 0x0F
        high = (payload.codes >> 4) & 0x0F
        if bool(((low == RESERVED_NIBBLE) | (high == RESERVED_NIBBLE)).any().item()):
            raise HardProtocolFault("invalid_nibble", "reserved nibble value 15 present in packed INT4 codes")
        if expected_numel % 2 == 1 and int(high[-1].item()) != 0:
            raise HardProtocolFault("invalid_padding", "nonzero padding nibble in odd-length INT4 payload")


@torch.no_grad()
def encode_blockwise_int4(
    update: torch.Tensor,
    residual: torch.Tensor | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[PackedINT4Payload, torch.Tensor]:
    """Compress a 1D FP32 update with blockwise INT4 and return (payload, next_residual)."""
    if update.ndim != 1:
        raise ValueError(f"Expected 1D update tensor, got shape {tuple(update.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not bool(torch.isfinite(update).all().item()):
        raise ValueError("Cannot compress update containing nonfinite values")

    numel = int(update.numel())
    device = update.device
    u = update.detach().float()
    if residual is not None:
        if residual.shape != update.shape:
            raise ValueError(f"residual shape {tuple(residual.shape)} != update shape {tuple(update.shape)}")
        if not bool(torch.isfinite(residual).all().item()):
            raise ValueError("Residual contains nonfinite values")
        u = u + residual.detach().to(device=device, dtype=torch.float32)

    if numel == 0:
        empty = PackedINT4Payload(
            numel=0,
            block_size=block_size,
            codes=torch.empty(0, dtype=torch.uint8, device=device),
            scales=torch.empty(0, dtype=torch.float16, device=device),
        )
        return empty, torch.empty(0, dtype=torch.float32, device=device)

    n_blocks = math.ceil(numel / block_size)
    padded_numel = n_blocks * block_size
    if padded_numel > numel:
        u_padded = torch.zeros(padded_numel, dtype=torch.float32, device=device)
        u_padded[:numel] = u
    else:
        u_padded = u

    blocks = u_padded.view(n_blocks, block_size)
    max_abs = blocks.abs().amax(dim=1)
    # Clamp scale to finite FP16 max so extreme values remain finite and error-feedback carries overflow.
    raw_scales = torch.clamp(max_abs / float(INT4_MAX_CODE), min=0.0, max=_FP16_MAX)
    scales_fp16 = raw_scales.to(torch.float16)
    scales_f32 = scales_fp16.float()

    nonzero = scales_f32 > 0.0
    safe_scales = torch.where(nonzero, scales_f32, torch.ones_like(scales_f32)).unsqueeze(1)
    q_blocks = torch.clamp(torch.round(blocks / safe_scales), -INT4_MAX_CODE, INT4_MAX_CODE).to(torch.int8)
    q_blocks = torch.where(nonzero.unsqueeze(1), q_blocks, torch.zeros_like(q_blocks))

    # Reconstruct exact decoded values y using rounded FP16 scales (sender & receiver match identically).
    y_padded = (q_blocks.float() * scales_f32.unsqueeze(1)).view(-1)
    y = y_padded[:numel]
    next_residual = u - y

    # Pack signed codes in [-7, 7] into uint8 bytes via unsigned offset q + 7 in [0, 14].
    q_flat = q_blocks.view(-1)[:numel]
    nibbles = (q_flat.to(torch.int16) + NIBBLE_OFFSET).to(torch.uint8)
    if numel % 2 == 1:
        nibbles = torch.cat([nibbles, torch.zeros(1, dtype=torch.uint8, device=device)])
    even = nibbles[0::2]
    odd = nibbles[1::2]
    codes = even | (odd << 4)

    payload = PackedINT4Payload(
        numel=numel,
        block_size=block_size,
        codes=codes.contiguous(),
        scales=scales_fp16.contiguous(),
    )
    return payload, next_residual


@torch.no_grad()
def decode_int4_slice_into(
    payload: PackedINT4Payload,
    start_elem: int,
    count: int,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """Decode elements [start_elem, start_elem + count) into out_buf[:count] in-place.

    Requires start_elem to be even (aligned to packed byte boundaries) and
    count <= out_buf.numel(). Returns a 1D FP32 view out_buf[:count].
    """
    if start_elem < 0 or count < 0 or start_elem + count > payload.numel:
        raise ValueError(f"Invalid slice [{start_elem}, {start_elem + count}) for numel={payload.numel}")
    if start_elem % 2 != 0:
        raise ValueError(f"start_elem must be even for byte-aligned nibble unpacking, got {start_elem}")
    if out_buf.dtype != torch.float32 or out_buf.ndim != 1 or out_buf.numel() < count:
        raise ValueError(f"out_buf must be 1D float32 with numel >= {count}, got {out_buf.dtype} {tuple(out_buf.shape)}")

    out = out_buf[:count]
    if count == 0:
        return out

    byte_start = start_elem // 2
    byte_count = (count + 1) // 2
    packed = payload.codes[byte_start : byte_start + byte_count].to(device=out.device)

    low = (packed & 0x0F).to(torch.int16) - NIBBLE_OFFSET
    high = ((packed >> 4) & 0x0F).to(torch.int16) - NIBBLE_OFFSET

    # Fill even/odd positions directly into out_buf without allocating a full-model vector.
    n_even = (count + 1) // 2
    n_odd = count // 2
    out[0::2] = low[:n_even].float()
    if n_odd > 0:
        out[1::2] = high[:n_odd].float()

    # Apply corresponding FP16 block scales (exact same rounded FP16 values as encoder).
    bs = payload.block_size
    first_block = start_elem // bs
    last_block = (start_elem + count - 1) // bs
    block_scales = payload.scales[first_block : last_block + 1].to(device=out.device, dtype=torch.float32)

    if start_elem % bs == 0 and (count % bs == 0):
        out.view(last_block - first_block + 1, bs).mul_(block_scales.unsqueeze(1))
    else:
        # Handle arbitrary even tile sizes cleanly without requiring block-size alignment.
        local_idx = torch.arange(start_elem, start_elem + count, device=out.device, dtype=torch.int64)
        blk_rel = (local_idx // bs) - first_block
        out.mul_(block_scales[blk_rel])
    return out


@torch.no_grad()
def iter_decoded_tiles(
    payload: PackedINT4Payload,
    tile_buf: torch.Tensor,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield (start, end, tile_view) streaming over payload using a single reusable buffer."""
    if tile_buf.dtype != torch.float32 or tile_buf.ndim != 1 or tile_buf.numel() < 2:
        raise ValueError("tile_buf must be a 1D float32 tensor with at least 2 elements")
    # Ensure step is even so every slice starts on a packed byte boundary.
    step = (int(tile_buf.numel()) // 2) * 2
    numel = payload.numel
    for start in range(0, numel, step):
        count = min(step, numel - start)
        view = decode_int4_slice_into(payload, start, count, tile_buf)
        yield start, start + count, view


@torch.no_grad()
def decode_blockwise_int4(payload: PackedINT4Payload, device: torch.device | str | None = None) -> torch.Tensor:
    """Decode an entire PackedINT4Payload into a newly allocated 1D FP32 tensor."""
    target_device = payload.codes.device if device is None else torch.device(device)
    out = torch.empty(payload.numel, dtype=torch.float32, device=target_device)
    if payload.numel > 0:
        decode_int4_slice_into(payload, 0, payload.numel, out)
    return out


@torch.no_grad()
def topk_compress_with_error_feedback(
    update: torch.Tensor,
    residual: torch.Tensor | None = None,
    keep_frac: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Top-k sparsification with error feedback (Phase 2 sparse baseline, Table 3).

    Returns (reconstructed_dense_update, next_residual, approx_wire_bytes) where
    wire bytes count k * (4-byte index + 4-byte fp32 value).
    """
    if update.ndim != 1:
        raise ValueError(f"Expected 1D update tensor, got shape {tuple(update.shape)}")
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")

    u = update.detach().float()
    if residual is not None:
        u = u + residual.detach().to(device=u.device, dtype=torch.float32)

    numel = int(u.numel())
    if numel == 0:
        return torch.empty_like(u), torch.empty_like(u), 0

    k = max(1, min(numel, int(round(numel * keep_frac))))
    _, idx = torch.topk(u.abs(), k=k, sorted=False)
    decoded = torch.zeros_like(u)
    decoded[idx] = u[idx]
    next_residual = u - decoded
    wire_bytes = k * 8  # int32 index + fp32 value
    return decoded, next_residual, wire_bytes
