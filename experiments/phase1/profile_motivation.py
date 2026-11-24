"""Stage 2 system-cost profiler for batched FP32 TrustGraph-DFL (motivation study).

Measures the four system bottlenecks motivating PeerSieve without implementing any
PeerSieve defense:
1. GPU receive-buffer memory scaling and OOM cliff across neighbor degrees (d=1..12).
2. Multi-GPU torch.distributed communication vs. compute/validation/scoring round time.
3. Exact per-edge wasted bandwidth and repeated Byzantine transfers over 25 training rounds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from src.aggregators import AggregatorConfig, aggregate_trustgraph
from src.attacks import AttackConfig, apply_round_attack, label_flip_mode_for_attack
from src.data import apply_label_flip
from src.models import (
    build_model,
    compute_validation_gradient,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)
from src.simulator import ExperimentConfig, _round_lr, _stage_device_data, prepare_dataset, seed_everything
from src.topology import build_topology, select_byzantine_nodes


# ---------------------------------------------------------------------------
# Pure-PyTorch TinyLlama-1.1B architecture + LoRA adapters (exact param match)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TinyLlamaConfig:
    """Architecture matching TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1,100,048,384 params)."""

    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 22
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    rms_norm_eps: float = 1e-5
    max_seq_len: int = 256
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class LoRALinear(nn.Module):
    """Frozen linear projection plus FP32 low-rank LoRA adapters A and B."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = (float(alpha) if alpha is not None else float(rank)) / float(rank)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scaling
        return out + delta.to(out.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class TinyLlamaAttention(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return self.o_proj(attn)


class TinyLlamaMLP(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLlamaBlock(nn.Module):
    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = TinyLlamaAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = TinyLlamaMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlamaModel(nn.Module):
    """Exact TinyLlama-1.1B causal LM architecture implemented in native PyTorch."""

    def __init__(self, cfg: TinyLlamaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([TinyLlamaBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("rope_sin", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seqlen = input_ids.shape[1]
        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[:, :, :seqlen, :].to(dtype=x.dtype)
        sin = self.rope_sin[:, :, :seqlen, :].to(dtype=x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))


def apply_lora_to_tinyllama(model: TinyLlamaModel, rank: int, base_dtype: torch.dtype | None = None) -> TinyLlamaModel:
    """Freeze TinyLlama base weights and attach FP32 LoRA adapters to all attention and MLP projections."""
    for p in model.parameters():
        p.requires_grad_(False)
    if base_dtype is not None:
        model.to(dtype=base_dtype)
    for block in model.layers:
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(block.self_attn, attr, LoRALinear(getattr(block.self_attn, attr), rank=rank))
        for attr in ("gate_proj", "up_proj", "down_proj"):
            setattr(block.mlp, attr, LoRALinear(getattr(block.mlp, attr), rank=rank))
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def trainable_to_fp32_vector(model: nn.Module) -> torch.Tensor:
    """Concatenate all trainable parameters into a contiguous 1D FP32 tensor."""
    parts = [p.detach().reshape(-1).float() for p in model.parameters() if p.requires_grad]
    return torch.cat(parts).clone()


def compute_lm_validation_gradient(
    model: nn.Module,
    val_ids: torch.Tensor,
    batch_size: int = 32,
    use_bf16_autocast: bool = False,
) -> torch.Tensor:
    """Compute exact mean causal-LM validation gradient over trainable parameters as 1D FP32."""
    model.zero_grad(set_to_none=True)
    model.eval()
    n_samples = int(val_ids.shape[0])
    device_type = val_ids.device.type
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = val_ids[start:end]
        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16_autocast):
            logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="mean")
        scaled = loss * ((end - start) / float(n_samples))
        scaled.backward()
    grads = [
        (p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel(), device=p.device))
        for p in model.parameters()
        if p.requires_grad
    ]
    vec = torch.cat(grads).clone()
    model.zero_grad(set_to_none=True)
    return vec


def build_profile_workload(
    workload: str,
    device: torch.device,
    llama_cfg: TinyLlamaConfig | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer, bool]:
    """Construct workload model, optimizer, and whether to use BF16 autocast."""
    use_cuda = device.type == "cuda"
    if workload == "resnet18":
        model = build_model("resnet18_scratch", num_classes=10).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
        return model, opt, False

    cfg = llama_cfg if llama_cfg is not None else TinyLlamaConfig()
    if workload == "tinyllama_lora_r32":
        model = TinyLlamaModel(cfg)
        apply_lora_to_tinyllama(model, rank=32, base_dtype=(torch.bfloat16 if use_cuda else None))
        model.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        return model, opt, use_cuda

    if workload == "tinyllama_lora_r128":
        model = TinyLlamaModel(cfg)
        apply_lora_to_tinyllama(model, rank=128, base_dtype=(torch.bfloat16 if use_cuda else None))
        model.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
        return model, opt, use_cuda

    if workload == "tinyllama_full":
        model = TinyLlamaModel(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        return model, opt, use_cuda

    raise ValueError(f"Unknown workload: {workload!r}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _batched_trustgraph_score_and_agg(
    self_update: torch.Tensor,
    val_grad: torch.Tensor,
    neighbor_updates: list[torch.Tensor],
    tau: float = 0.003,
) -> tuple[torch.Tensor, list[float]]:
    """Score and aggregate a batch of simultaneously buffered FP32 neighbor updates."""
    ref = -val_grad
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    scores: list[float] = []
    weights: list[float] = []
    for u in neighbor_updates:
        un = float(torch.linalg.vector_norm(u).item())
        s = float(torch.dot(u, ref).item() / (un * ref_norm)) if (un > 0.0 and ref_norm > 0.0) else 0.0
        scores.append(s)
        weights.append(max(0.0, s - tau))
    denom = sum(weights)
    agg = torch.zeros_like(self_update)
    if denom > 0.0:
        for u, w in zip(neighbor_updates, weights):
            if w > 0.0:
                agg.add_(u, alpha=w / (denom + 1e-8))
    agg.mul_(0.5).add_(self_update, alpha=0.5)
    return agg, scores


# ---------------------------------------------------------------------------
# Benchmark 1: Receive-side GPU memory scaling & OOM cliff
# ---------------------------------------------------------------------------


def run_memory_scaling(
    workloads: list[str],
    degrees: list[int],
    device: torch.device,
    llama_cfg: TinyLlamaConfig | None = None,
) -> dict[str, object]:
    """Measure peak GPU memory of batched FP32 TrustGraph-DFL as neighbor degree d grows."""
    gib = float(1024 ** 3)
    mb = 1e6
    results: dict[str, object] = {
        "device": str(device),
        "gpu_total_gib": (
            float(torch.cuda.get_device_properties(device).total_memory) / gib if device.type == "cuda" else 0.0
        ),
        "workloads": {},
    }

    for wl in workloads:
        _clear_cuda(device)
        torch.manual_seed(42)
        model, opt, use_bf16 = build_profile_workload(wl, device, llama_cfg=llama_cfg)
        total_p, train_p = count_parameters(model)
        payload_mb = (train_p * 4.0) / mb

        # Run 1 real training step + 1 real validation gradient pass to materialize optimizer states and val_grad.
        if wl == "resnet18":
            xb = torch.randn(64, 3, 32, 32, device=device)
            yb = torch.randint(0, 10, (64,), device=device)
            model.train()
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            val_grad = compute_validation_gradient(model, xb, yb, batch_size=64)
            del xb, yb
        else:
            cfg = llama_cfg if llama_cfg is not None else TinyLlamaConfig()
            xb = torch.randint(0, cfg.vocab_size, (16, cfg.max_seq_len), device=device)
            model.train()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(xb[:, :-1])
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            del logits, loss
            val_ids = torch.randint(0, cfg.vocab_size, (32, cfg.max_seq_len), device=device)
            val_grad = compute_lm_validation_gradient(model, val_ids, batch_size=16, use_bf16_autocast=use_bf16)
            del xb, val_ids

        self_update = trainable_to_fp32_vector(model)
        _sync(device)
        base_alloc_gib = float(torch.cuda.memory_allocated(device)) / gib if device.type == "cuda" else 0.0
        base_peak_gib = float(torch.cuda.max_memory_allocated(device)) / gib if device.type == "cuda" else 0.0

        degree_rows: list[dict[str, object]] = []
        for d in degrees:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            nbr_bufs: list[torch.Tensor] = []
            oom = False
            err_msg = None
            peak_alloc_gib = None
            peak_reserved_gib = None
            after_alloc_gib = None
            try:
                for _ in range(d):
                    nbr_bufs.append(torch.empty(train_p, dtype=torch.float32, device=device).fill_(1e-3))
                out, _ = _batched_trustgraph_score_and_agg(self_update, val_grad, nbr_bufs)
                _sync(device)
                if device.type == "cuda":
                    after_alloc_gib = float(torch.cuda.memory_allocated(device)) / gib
                    peak_alloc_gib = float(torch.cuda.max_memory_allocated(device)) / gib
                    peak_reserved_gib = float(torch.cuda.max_memory_reserved(device)) / gib
                del out
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower() or isinstance(exc, torch.OutOfMemoryError):
                    oom = True
                    err_msg = str(exc).splitlines()[0][:180]
                else:
                    raise
            finally:
                del nbr_bufs
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            degree_rows.append(
                {
                    "degree": d,
                    "recv_buffers_gib": round((d * train_p * 4.0) / gib, 4),
                    "recv_payload_mb": round(d * payload_mb, 2),
                    "oom": oom,
                    "error": err_msg,
                    "after_alloc_gib": round(after_alloc_gib, 4) if after_alloc_gib is not None else None,
                    "peak_alloc_gib": round(peak_alloc_gib, 4) if peak_alloc_gib is not None else None,
                    "peak_reserved_gib": round(peak_reserved_gib, 4) if peak_reserved_gib is not None else None,
                }
            )
            status = "OOM" if oom else f"peak={peak_alloc_gib:.2f} GiB" if peak_alloc_gib is not None else "ok"
            print(f"[mem] {wl:20s} d={d:2d}: recv_buf={d * train_p * 4.0 / gib:6.2f} GiB -> {status}", flush=True)

        results["workloads"][wl] = {
            "total_params": total_p,
            "trainable_params": train_p,
            "payload_fp32_mb": round(payload_mb, 3),
            "base_alloc_gib": round(base_alloc_gib, 4),
            "base_peak_gib": round(base_peak_gib, 4),
            "degrees": degree_rows,
        }
        del model, opt, val_grad, self_update
        _clear_cuda(device)

    return results


# ---------------------------------------------------------------------------
# Benchmark 2: 8-GPU torch.distributed communication & round-time breakdown
# ---------------------------------------------------------------------------


def build_k_regular_neighbors(world_size: int, degree: int) -> list[list[int]]:
    """Build deterministic symmetric k-regular neighbor lists on world_size peers."""
    if degree >= world_size:
        raise ValueError(f"degree={degree} must be < world_size={world_size}")
    if (world_size * degree) % 2 != 0:
        raise ValueError(f"world_size*degree must be even; got {world_size}*{degree}")
    offsets: list[int] = []
    half = degree // 2
    for step in range(1, half + 1):
        offsets.append(step)
        offsets.append(world_size - step)
    if degree % 2 == 1:
        offsets.append(world_size // 2)
    return [[(r + off) % world_size for off in offsets] for r in range(world_size)]


def _stats(vals: list[float]) -> dict[str, float]:
    return {
        "median": round(float(statistics.median(vals)), 5),
        "min": round(float(min(vals)), 5),
        "max": round(float(max(vals)), 5),
        "repeats": len(vals),
    }


def _paced_token_bucket_exchange(
    send_vec: torch.Tensor,
    neighbors: list[int],
    bw_bps: float,
    chunk_bytes: int = 4 * 1024 * 1024,
) -> float:
    """Measure a real chunked P2P transfer paced by a receive-side token bucket."""
    elems_per_chunk = max(1, chunk_bytes // 4)
    n_elems = int(send_vec.numel())
    rx_bufs = [torch.empty_like(send_vec) for _ in neighbors]
    t0 = time.perf_counter()
    rx_bytes_done = 0
    for start in range(0, n_elems, elems_per_chunk):
        end = min(start + elems_per_chunk, n_elems)
        chunk_send = send_vec[start:end].contiguous()
        ops = []
        for idx, nbr in enumerate(neighbors):
            ops.append(dist.P2POp(dist.isend, chunk_send, nbr))
            ops.append(dist.P2POp(dist.irecv, rx_bufs[idx][start:end], nbr))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        _sync(send_vec.device)
        rx_bytes_done += len(neighbors) * (end - start) * 4
        target_elapsed = (rx_bytes_done * 8.0) / bw_bps
        now = time.perf_counter() - t0
        if target_elapsed > now:
            time.sleep(target_elapsed - now)
    dist.barrier()
    return time.perf_counter() - t0


def run_distributed_profile(
    workloads: list[str],
    degree: int = 6,
    warmup: int = 1,
    repeats: int = 5,
    llama_cfg: TinyLlamaConfig | None = None,
) -> dict[str, object] | None:
    """Profile batched FP32 TrustGraph-DFL round time across 8 A100 GPUs via torch.distributed."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    all_nbrs = build_k_regular_neighbors(world_size, min(degree, world_size - 1))
    my_nbrs_full = all_nbrs[rank]

    out_workloads: dict[str, object] = {}
    for wl in workloads:
        _clear_cuda(device)
        torch.manual_seed(42 + rank)
        model, opt, use_bf16 = build_profile_workload(wl, device, llama_cfg=llama_cfg)
        total_p, train_p = count_parameters(model)
        payload_bytes = train_p * 4
        payload_mb = payload_bytes / 1e6

        # Full-parameter 1.1B OOMs on 40 GB A100 with 6 simultaneous FP32 receive buffers;
        # measure live 2-neighbor PCIe exchange on all 8 GPUs and scale to d=6 for network comparison.
        eff_degree = min(2, len(my_nbrs_full)) if wl == "tinyllama_full" else len(my_nbrs_full)
        my_nbrs = build_k_regular_neighbors(world_size, eff_degree)[rank]

        if wl == "resnet18":
            # Full 50k setup: 2,500 samples/node -> 2,250 train (~36 batches of 64) + 250 val.
            x_tr = torch.randn(2250, 3, 32, 32, device=device)
            y_tr = torch.randint(0, 10, (2250,), device=device)
            x_val = torch.randn(250, 3, 32, 32, device=device)
            y_val = torch.randint(0, 10, (250,), device=device)
        else:
            # Exact Proposal §5.2 setup: 4 AdamW steps, batch=16, seq=256; 64 val examples (2x32).
            cfg = llama_cfg if llama_cfg is not None else TinyLlamaConfig()
            x_tr = torch.randint(0, cfg.vocab_size, (4 * 16, cfg.max_seq_len), device=device)
            x_val = torch.randint(0, cfg.vocab_size, (64, cfg.max_seq_len), device=device)
            y_tr = y_val = None

        train_times: list[float] = []
        val_times: list[float] = []
        comm_times: list[float] = []
        agg_times: list[float] = []

        for step in range(warmup + repeats):
            dist.barrier()
            _sync(device)
            t0 = time.perf_counter()
            if wl == "resnet18":
                train_local_epoch(
                    model, x_tr, y_tr, lr=0.005, momentum=0.9, weight_decay=1e-4, batch_size=64, epochs=1, augment=True, grad_clip_norm=10.0
                )
            else:
                model.train()
                for s_idx in range(4):
                    xb = x_tr[s_idx * 16 : (s_idx + 1) * 16]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                        logits = model(xb[:, :-1])
                        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1))
                    loss.backward()
                    opt.step()
                opt.zero_grad(set_to_none=True)
            _sync(device)
            t_train = time.perf_counter() - t0

            dist.barrier()
            _sync(device)
            t0 = time.perf_counter()
            if wl == "resnet18":
                val_grad = compute_validation_gradient(model, x_val, y_val, batch_size=64)
            else:
                val_bs = 16 if wl == "tinyllama_full" else 32
                val_grad = compute_lm_validation_gradient(model, x_val, batch_size=val_bs, use_bf16_autocast=use_bf16)
            _sync(device)
            t_val = time.perf_counter() - t0

            if wl == "tinyllama_full" and device.type == "cuda":
                torch.cuda.empty_cache()
            send_vec = trainable_to_fp32_vector(model)
            recv_vecs = [torch.empty_like(send_vec) for _ in my_nbrs]
            dist.barrier()
            _sync(device)
            t0 = time.perf_counter()
            ops = []
            for idx, nbr in enumerate(my_nbrs):
                ops.append(dist.P2POp(dist.isend, send_vec, nbr))
                ops.append(dist.P2POp(dist.irecv, recv_vecs[idx], nbr))
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            _sync(device)
            t_comm = time.perf_counter() - t0

            dist.barrier()
            _sync(device)
            t0 = time.perf_counter()
            _out, _scores = _batched_trustgraph_score_and_agg(send_vec, val_grad, recv_vecs)
            _sync(device)
            t_agg = time.perf_counter() - t0

            del _out, recv_vecs, send_vec, val_grad

            # Use max across all ranks per round (synchronous DFL round barrier).
            t_tensor = torch.tensor([t_train, t_val, t_comm, t_agg], dtype=torch.float64, device=device)
            dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
            if step >= warmup:
                train_times.append(float(t_tensor[0].item()))
                val_times.append(float(t_tensor[1].item()))
                comm_times.append(float(t_tensor[2].item()))
                agg_times.append(float(t_tensor[3].item()))

        # Live paced 1 Gb/s token-bucket verification on resnet18 and tinyllama_lora_r32.
        paced_1gbps_sec = None
        if wl in {"resnet18", "tinyllama_lora_r32"}:
            send_vec = trainable_to_fp32_vector(model)
            dist.barrier()
            paced_1gbps_sec = round(_paced_token_bucket_exchange(send_vec, my_nbrs, bw_bps=1e9), 4)
            del send_vec

        target_d = len(my_nbrs_full)
        target_rx_bytes = target_d * payload_bytes
        target_rx_mb = target_rx_bytes / 1e6
        med_train = statistics.median(train_times)
        med_val = statistics.median(val_times)
        med_comm_eff = statistics.median(comm_times)
        med_agg_eff = statistics.median(agg_times)
        # Scale PCIe transfer and agg linearly if eff_degree < target_d (only for tinyllama_full due to OOM).
        scale = float(target_d) / float(eff_degree)
        med_comm_pcie_d6 = med_comm_eff * scale
        med_agg_d6 = med_agg_eff * scale
        pcie_eff_gbps = (eff_degree * payload_bytes * 8.0) / max(med_comm_eff, 1e-9) / 1e9

        bw_breakdown: dict[str, dict[str, float]] = {}
        for label, bw_bps in (("pcie_uncapped", None), ("10gbps", 10e9), ("1gbps", 1e9), ("100mbps", 100e6)):
            t_c = med_comm_pcie_d6 if bw_bps is None else max(med_comm_pcie_d6, (target_rx_bytes * 8.0) / bw_bps)
            t_tot = med_train + med_val + t_c + med_agg_d6
            bw_breakdown[label] = {
                "comm_sec": round(t_c, 4),
                "total_round_sec": round(t_tot, 4),
                "comm_share_pct": round(100.0 * t_c / max(t_tot, 1e-9), 2),
                "compute_plus_val_sec": round(med_train + med_val, 4),
                "score_agg_sec": round(med_agg_d6, 4),
            }

        if rank == 0:
            out_workloads[wl] = {
                "total_params": total_p,
                "trainable_params": train_p,
                "payload_fp32_mb": round(payload_mb, 3),
                "target_degree": target_d,
                "measured_degree": eff_degree,
                "rx_mb_per_round_d6": round(target_rx_mb, 2),
                "wasted_mb_at_33pct_byz": round(target_rx_mb * (2.0 / 6.0), 2),
                "wasted_mb_at_51pct_total_rej": round(target_rx_mb * 0.5119, 2),
                "pcie_effective_rx_gbps": round(pcie_eff_gbps, 3),
                "paced_1gbps_verified_sec": paced_1gbps_sec,
                "train_sec": _stats(train_times),
                "val_grad_sec": _stats(val_times),
                "comm_pcie_measured_sec": _stats(comm_times),
                "score_agg_measured_sec": _stats(agg_times),
                "round_time_by_link": bw_breakdown,
            }
            print(
                f"[dist] {wl:20s} d={eff_degree}/{target_d}: rx={target_rx_mb:7.1f} MB | "
                f"train={med_train:.3f}s val={med_val:.3f}s pcie={med_comm_pcie_d6:.3f}s "
                f"1Gb/s_comm={bw_breakdown['1gbps']['comm_sec']:.2f}s ({bw_breakdown['1gbps']['comm_share_pct']:.1f}% of round)",
                flush=True,
            )

        del model, opt, x_tr, x_val
        if y_tr is not None:
            del y_tr, y_val
        _clear_cuda(device)

    dist.barrier()
    dist.destroy_process_group()
    if rank != 0:
        return None
    return {
        "world_size": world_size,
        "target_degree": min(degree, world_size - 1),
        "backend": backend,
        "warmup": warmup,
        "repeats": repeats,
        "workloads": out_workloads,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Per-edge wasted bandwidth & repeated-rejection audit (25 rounds)
# ---------------------------------------------------------------------------


def _streak_stats(seq: list[bool]) -> tuple[int, int, int]:
    """Return (total_rejected, max_consecutive_rejected, repeat_rejections_after_first)."""
    total = sum(1 for x in seq if x)
    max_s = 0
    cur = 0
    for x in seq:
        if x:
            cur += 1
            if cur > max_s:
                max_s = cur
        else:
            cur = 0
    repeat_after_first = max(0, total - 1) if total > 0 else 0
    return total, max_s, repeat_after_first


def run_per_edge_rejection_audit(
    attack: str,
    dataset: str = "cifar10",
    model_name: str = "resnet18_pretrained",
    rounds: int = 25,
    samples_per_node: int = 2500,
    seed: int = 42,
    device_name: str = "auto",
    download_data: bool = False,
) -> dict[str, object]:
    """Run TrustGraph-DFL and record exact per-edge wasted bytes and repeat-rejection streaks."""
    cfg = ExperimentConfig(
        dataset=dataset,
        download_data=download_data,
        n_nodes=20,
        samples_per_node=samples_per_node,
        alpha_dir=0.5,
        val_split="iid",
        model_name=model_name,
        rounds=rounds,
        lr=0.005,
        lr_schedule="cosine",
        augment=True,
        grad_clip_norm=10.0,
        byz_local_epochs=(2 if attack == "label_flip_all" else 0),
        topology="random",
        er_p=0.3,
        byzantine_frac=0.3,
        attack=attack,
        aggregator="trustgraph",
        val_grad_sign="descent_neg",
        trustgraph_clip_norm=True,
        trustgraph_clip_mode="model",
        trustgraph_clip_factor=2.0,
        trustgraph_score_threshold=0.003,
        trustgraph_delta_mode="sender",
        seed=seed,
        device=device_name,
        eval_every=rounds,
        verbose=False,
    )
    agg_cfg = cfg.aggregator_config()
    atk_cfg = cfg.attack_config()
    seed_everything(cfg.seed)
    device = torch.device("cuda" if (device_name == "auto" and torch.cuda.is_available()) else ("cpu" if device_name == "auto" else device_name))

    data = prepare_dataset(cfg)
    graph = build_topology(cfg.topology, n_nodes=cfg.n_nodes, er_p=cfg.er_p, seed=cfg.seed)
    byz_frac_eff = 0.0 if cfg.attack == "none" else cfg.byzantine_frac
    byzantine_ids = select_byzantine_nodes(cfg.n_nodes, byz_frac_eff, seed=cfg.seed + 1000)
    benign_ids = [i for i in range(cfg.n_nodes) if i not in byzantine_ids]

    dev_data = _stage_device_data(data, cfg.attack, byzantine_ids, device)
    work_model = build_model(cfg.model_name, num_classes=data.num_classes).to(device)
    init_vec = model_to_vector(work_model)
    param_count = int(init_vec.numel())
    payload_mb = (param_count * 4.0) / 1e6
    node_models = [init_vec.clone() for _ in range(cfg.n_nodes)]

    train_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 2000)
    attack_gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 3000)

    # edge_rejected[(i, j)] = list of bools across rounds (True = assigned zero weight)
    edge_rejected: dict[tuple[int, int], list[bool]] = {}
    for i in benign_ids:
        for j in graph.neighbors(i):
            edge_rejected[(i, j)] = []

    t0 = time.perf_counter()
    for r in range(cfg.rounds):
        prev_models = [v.clone() for v in node_models]
        post_models: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
        updates: list[torch.Tensor] = [torch.empty(0, device=device) for _ in range(cfg.n_nodes)]
        val_grads: dict[int, torch.Tensor] = {}
        lr_now = _round_lr(cfg, r)

        for i in range(cfg.n_nodes):
            if i in byzantine_ids and cfg.attack in {"model_poison", "random_noise"}:
                continue
            vector_to_model(prev_models[i], work_model)
            if i not in byzantine_ids:
                val_grads[i] = compute_validation_gradient(work_model, dev_data.x_val[i], dev_data.y_val[i], batch_size=cfg.batch_size)
            n_ep = cfg.byz_local_epochs if (i in byzantine_ids and cfg.byz_local_epochs > 0) else cfg.local_epochs
            train_local_epoch(
                work_model, dev_data.x_train[i], dev_data.y_train[i],
                lr=lr_now, momentum=cfg.momentum, weight_decay=cfg.weight_decay,
                batch_size=cfg.batch_size, epochs=n_ep, augment=cfg.augment,
                grad_clip_norm=cfg.grad_clip_norm, generator=train_gen,
            )
            pv = model_to_vector(work_model)
            post_models[i] = pv
            updates[i] = pv - prev_models[i]

        apply_round_attack(
            attack_cfg=atk_cfg, byzantine_ids=byzantine_ids, benign_ids=benign_ids,
            prev_models=prev_models, post_models=post_models, updates=updates, generator=attack_gen,
        )

        next_models = [v.clone() for v in prev_models]
        for i in benign_ids:
            nbrs = list(graph.neighbors(i))
            res = aggregate_trustgraph(
                post_models[i], val_grads[i], nbrs,
                {j: post_models[j] for j in nbrs}, {j: updates[j] for j in nbrs},
                agg_cfg, self_prev=prev_models[i], self_update=updates[i],
            )
            next_models[i] = res.new_model
            for j in nbrs:
                edge_rejected[(i, j)].append(res.neighbor_weights[j] <= 0.0)

        if byzantine_ids:
            if cfg.attack in {"label_flip", "label_flip_all"}:
                for m in byzantine_ids:
                    nbrs = list(graph.neighbors(m))
                    next_models[m] = torch.stack([post_models[m]] + [post_models[j] for j in nbrs], dim=0).mean(dim=0)
            else:
                ben_mean = torch.stack([next_models[i] for i in benign_ids], dim=0).mean(dim=0)
                for m in byzantine_ids:
                    next_models[m] = ben_mean.clone()
        node_models = next_models

    wall_sec = time.perf_counter() - t0
    honest_edges = [seq for (_, j), seq in edge_rejected.items() if j not in byzantine_ids]
    byz_edges = [seq for (_, j), seq in edge_rejected.items() if j in byzantine_ids]

    total_rx = sum(len(seq) for seq in edge_rejected.values())
    honest_rej = sum(sum(1 for x in seq if x) for seq in honest_edges)
    byz_rej = sum(sum(1 for x in seq if x) for seq in byz_edges)
    total_rej = honest_rej + byz_rej

    byz_edge_stats = [_streak_stats(seq) for seq in byz_edges]
    honest_edge_stats = [_streak_stats(seq) for seq in honest_edges]

    n_benign = len(benign_ids)
    rx_mb_per_peer_round = (total_rx / max(n_benign * cfg.rounds, 1)) * payload_mb
    wasted_mb_per_peer_round = (total_rej / max(n_benign * cfg.rounds, 1)) * payload_mb

    out: dict[str, object] = {
        "attack": attack,
        "dataset": dataset,
        "model_name": model_name,
        "param_count": param_count,
        "payload_fp32_mb": round(payload_mb, 3),
        "rounds": rounds,
        "n_benign": n_benign,
        "n_byzantine": len(byzantine_ids),
        "n_honest_edges": len(honest_edges),
        "n_byz_edges": len(byz_edges),
        "total_transfers": total_rx,
        "total_rejected_transfers": total_rej,
        "honest_rejected_transfers": honest_rej,
        "byz_rejected_transfers": byz_rej,
        "wasted_transfer_pct": round(100.0 * total_rej / max(total_rx, 1), 2),
        "honest_fpr_pct": round(100.0 * honest_rej / max(len(honest_edges) * rounds, 1), 2),
        "byz_rejection_pct": round(100.0 * byz_rej / max(len(byz_edges) * rounds, 1), 2) if byz_edges else 0.0,
        "rx_mb_per_peer_round": round(rx_mb_per_peer_round, 2),
        "wasted_mb_per_peer_round": round(wasted_mb_per_peer_round, 2),
        "total_wasted_gb_all_benign_25r": round((total_rej * payload_mb) / 1000.0, 3),
        "byz_repeat_rejections": {
            "mean_rejected_rounds_of_total": round(statistics.mean([s[0] for s in byz_edge_stats]), 2) if byz_edge_stats else 0.0,
            "min_rejected_rounds": min([s[0] for s in byz_edge_stats]) if byz_edge_stats else 0,
            "max_rejected_rounds": max([s[0] for s in byz_edge_stats]) if byz_edge_stats else 0,
            "mean_max_consec_streak": round(statistics.mean([s[1] for s in byz_edge_stats]), 2) if byz_edge_stats else 0.0,
            "repeat_transfers_after_first_rej": sum(s[2] for s in byz_edge_stats),
            "frac_edges_with_streak_ge3_pct": round(100.0 * sum(1 for s in byz_edge_stats if s[1] >= 3) / max(len(byz_edge_stats), 1), 2) if byz_edge_stats else 0.0,
        },
        "honest_rejection_streaks": {
            "mean_rejected_rounds_of_total": round(statistics.mean([s[0] for s in honest_edge_stats]), 2) if honest_edge_stats else 0.0,
            "mean_max_consec_streak": round(statistics.mean([s[1] for s in honest_edge_stats]), 2) if honest_edge_stats else 0.0,
            "frac_edges_with_ge1_rej_pct": round(100.0 * sum(1 for s in honest_edge_stats if s[0] >= 1) / max(len(honest_edge_stats), 1), 2),
            "frac_edges_with_streak_ge3_pct": round(100.0 * sum(1 for s in honest_edge_stats if s[1] >= 3) / max(len(honest_edge_stats), 1), 2),
            "frac_edges_with_streak_ge5_pct": round(100.0 * sum(1 for s in honest_edge_stats if s[1] >= 5) / max(len(honest_edge_stats), 1), 2),
        },
        "wall_sec": round(wall_sec, 2),
    }
    print(
        f"[audit] {attack:15s}: wasted={out['wasted_transfer_pct']:5.2f}% "
        f"({out['wasted_mb_per_peer_round']:.1f}/{out['rx_mb_per_peer_round']:.1f} MB/peer/r) | "
        f"byz_rej={out['byz_rejection_pct']:5.1f}% | honest_ge1={out['honest_rejection_streaks']['frac_edges_with_ge1_rej_pct']:5.1f}% "
        f"streak>=3={out['honest_rejection_streaks']['frac_edges_with_streak_ge3_pct']:5.1f}% ({wall_sec:.1f}s)",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 batched FP32 TrustGraph-DFL system profiler.")
    p.add_argument("--mode", choices=["memory", "distributed", "per_edge", "merge"], required=True)
    p.add_argument("--out", required=True, help="Output JSON path (or input directory for merge).")
    p.add_argument(
        "--workloads",
        nargs="+",
        default=["resnet18", "tinyllama_lora_r32", "tinyllama_lora_r128", "tinyllama_full"],
    )
    p.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 4, 6, 8, 12])
    p.add_argument("--degree", type=int, default=6)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--attack", choices=["none", "label_flip_all", "model_poison", "random_noise"], default="model_poison")
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--model", default="resnet18_pretrained")
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--samples-per-node", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--merge-dir", default=None, help="Directory containing memory.json, distributed.json, per_edge_*.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if args.mode == "memory":
        dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
        res = run_memory_scaling(args.workloads, args.degrees, dev)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        return

    if args.mode == "distributed":
        res = run_distributed_profile(args.workloads, degree=args.degree, warmup=args.warmup, repeats=args.repeats)
        if res is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        return

    if args.mode == "per_edge":
        res = run_per_edge_rejection_audit(
            attack=args.attack,
            dataset=args.dataset,
            model_name=args.model,
            rounds=args.rounds,
            samples_per_node=args.samples_per_node,
            seed=args.seed,
            device_name=args.device,
            download_data=False,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        return

    if args.mode == "merge":
        src_dir = Path(args.merge_dir) if args.merge_dir else out_path.parent
        merged: dict[str, object] = {
            "description": "Stage 2 batched FP32 TrustGraph-DFL system bottleneck measurements on octave (8x A100-PCIE-40GB)",
            "memory_scaling": json.loads((src_dir / "memory.json").read_text(encoding="utf-8")),
            "distributed_profile": json.loads((src_dir / "distributed.json").read_text(encoding="utf-8")),
            "per_edge_audit": {},
        }
        for p in sorted(src_dir.glob("per_edge_*.json")):
            item = json.loads(p.read_text(encoding="utf-8"))
            merged["per_edge_audit"][str(item["attack"])] = item
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
        print(f"Merged Stage 2 profile -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
