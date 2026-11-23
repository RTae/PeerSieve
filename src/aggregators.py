"""Decentralized aggregation rules for Phase 1 TrustGraph-DFL reproduction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.models import ParamSlice


@dataclass
class AggregatorConfig:
    """Configuration for DFL aggregation and TrustGraph-DFL variants."""

    name: str = "trustgraph"  # fedavg | trustgraph | balance | krum | trimmed_mean
    alpha: float = 0.5  # self-weight for TrustGraph-DFL and BALANCE
    eps: float = 1e-8
    # TrustGraph-DFL specific
    val_grad_sign: str = "descent_neg"  # descent_neg (-nabla F) | paper_pos (+nabla F)
    trustgraph_use_relu: bool = True
    trustgraph_metric: str = "cosine"  # cosine | euclidean
    trustgraph_score_threshold: float = 0.0
    trustgraph_clip_norm: bool = False  # receiver-norm cap on accepted neighbor contributions
    trustgraph_clip_mode: str = "update"  # update (PeerSieve Eq. 4) | model (consensus-preserving)
    trustgraph_clip_factor: float = 2.0  # multiplier on r_i for model-space displacement cap
    trustgraph_delta_mode: str = "sender"  # sender (theta_j^{t+1/2}-theta_j^t) | receiver (theta_j^{t+1/2}-theta_i^t)
    trustgraph_score_scope: str = "global"  # global | layer_mean | backbone_only | head_only
    # BALANCE specific (Fang et al., CCS 2024)
    balance_gamma: float = 0.3
    balance_kappa: float = 1.0
    # Krum specific
    krum_mode: str = "multi"  # multi | single
    # Trimmed mean specific
    trim_ratio: float = 0.3

    def validate(self) -> None:
        valid_names = {"fedavg", "trustgraph", "balance", "krum", "trimmed_mean"}
        if self.name not in valid_names:
            raise ValueError(f"Unknown aggregator {self.name!r}; expected one of {sorted(valid_names)}.")
        if self.val_grad_sign not in {"descent_neg", "paper_pos"}:
            raise ValueError(f"Unknown val_grad_sign: {self.val_grad_sign!r}")
        if self.trustgraph_metric not in {"cosine", "euclidean"}:
            raise ValueError(f"Unknown trustgraph_metric: {self.trustgraph_metric!r}")
        if self.trustgraph_clip_mode not in {"update", "model"}:
            raise ValueError(f"Unknown trustgraph_clip_mode: {self.trustgraph_clip_mode!r}")
        if self.trustgraph_delta_mode not in {"sender", "receiver"}:
            raise ValueError(f"Unknown trustgraph_delta_mode: {self.trustgraph_delta_mode!r}")
        if self.trustgraph_score_scope not in {"global", "layer_mean", "backbone_only", "head_only"}:
            raise ValueError(f"Unknown trustgraph_score_scope: {self.trustgraph_score_scope!r}")
        if self.krum_mode not in {"multi", "single"}:
            raise ValueError(f"Unknown krum_mode: {self.krum_mode!r}")


@dataclass
class AggregationResult:
    """Output and per-neighbor diagnostics of one node's aggregation step."""

    new_model: torch.Tensor
    neighbor_weights: dict[int, float]
    neighbor_zero_weight_frac: dict[int, float]
    scores: dict[int, float] = field(default_factory=dict)
    self_fallback: bool = False


def _self_only_result(
    self_post: torch.Tensor,
    neighbor_ids: list[int],
    scores: dict[int, float] | None = None,
    self_fallback: bool = True,
) -> AggregationResult:
    return AggregationResult(
        new_model=self_post.clone(),
        neighbor_weights={j: 0.0 for j in neighbor_ids},
        neighbor_zero_weight_frac={j: 1.0 for j in neighbor_ids},
        scores=scores if scores is not None else {},
        self_fallback=self_fallback,
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def _is_classifier_head(name: str) -> bool:
    parts = name.split(".")
    return ("fc" in parts) or ("fc2" in parts)


def _is_semantic_head(name: str) -> bool:
    parts = name.split(".")
    return ("layer4" in parts) or ("fc" in parts) or ("fc1" in parts) or ("fc2" in parts)


def _scoped_cosine_similarity(
    a: torch.Tensor,
    b: torch.Tensor,
    scope: str,
    param_slices: Sequence[ParamSlice] | None,
) -> float:
    if scope == "global":
        return _cosine_similarity(a, b)
    if not param_slices:
        raise ValueError(f"param_slices is required when trustgraph_score_scope={scope!r}.")

    if scope == "layer_mean":
        selected = [s for s in param_slices if s.ndim >= 2]
        if not selected:
            selected = list(param_slices)
        dots = torch.stack([torch.dot(a[s.start : s.end], b[s.start : s.end]) for s in selected])
        nas = torch.stack([torch.linalg.vector_norm(a[s.start : s.end]) for s in selected])
        nbs = torch.stack([torch.linalg.vector_norm(b[s.start : s.end]) for s in selected])
        valid = (nas > 0.0) & (nbs > 0.0)
        if not bool(valid.any().item()):
            return 0.0
        return float((dots[valid] / (nas[valid] * nbs[valid])).mean().item())

    if scope == "backbone_only":
        selected = [s for s in param_slices if not _is_classifier_head(s.name)]
    elif scope == "head_only":
        selected = [s for s in param_slices if _is_semantic_head(s.name)]
    else:
        raise ValueError(f"Unknown trustgraph_score_scope: {scope!r}")

    if not selected:
        return _cosine_similarity(a, b)

    dot = torch.stack([torch.dot(a[s.start : s.end], b[s.start : s.end]) for s in selected]).sum()
    na2 = torch.stack([a[s.start : s.end].pow(2).sum() for s in selected]).sum()
    nb2 = torch.stack([b[s.start : s.end].pow(2).sum() for s in selected]).sum()
    denom = float((na2.sqrt() * nb2.sqrt()).item())
    if denom == 0.0:
        return 0.0
    return float(dot.item() / denom)


@torch.no_grad()
def aggregate_fedavg(
    self_post: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
) -> AggregationResult:
    """Unprotected decentralized FedAvg over N_i U {i}."""
    if not neighbor_ids:
        return _self_only_result(self_post, neighbor_ids, self_fallback=False)

    tensors = [self_post] + [neighbor_posts[j] for j in neighbor_ids]
    new_model = torch.stack(tensors, dim=0).mean(dim=0)
    w = 1.0 / len(tensors)
    return AggregationResult(
        new_model=new_model,
        neighbor_weights={j: w for j in neighbor_ids},
        neighbor_zero_weight_frac={j: 0.0 for j in neighbor_ids},
    )


@torch.no_grad()
def aggregate_trustgraph(
    self_post: torch.Tensor,
    val_grad: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
    neighbor_updates: dict[int, torch.Tensor],
    cfg: AggregatorConfig,
    self_prev: torch.Tensor | None = None,
    self_update: torch.Tensor | None = None,
    param_slices: Sequence[ParamSlice] | None = None,
) -> AggregationResult:
    """TrustGraph-DFL consistency-weighted neighborhood aggregation (Eq. 3-5).

    Supports:
    - val_grad_sign="paper_pos": literal Algorithm 1 / Eq. 3 using +nabla F_i(theta_i^t; D_i^val)
    - val_grad_sign="descent_neg": corrected descent direction -nabla F_i(theta_i^t; D_i^val)
    - trustgraph_clip_norm=True: PeerSieve Eq. (4) receiver-norm cap on neighbor updates
      a_ij = min(1, ||Delta_theta_i||_2 / ||Delta_theta_j||_2), blended in update space.
    """
    if not neighbor_ids:
        return _self_only_result(self_post, neighbor_ids)

    ref_vec = -val_grad if cfg.val_grad_sign == "descent_neg" else val_grad

    if cfg.trustgraph_delta_mode == "receiver" and self_prev is None:
        raise ValueError("self_prev is required when trustgraph_delta_mode='receiver'.")

    scores: dict[int, float] = {}
    raw_weights: dict[int, float] = {}
    for j in neighbor_ids:
        if cfg.trustgraph_metric == "cosine":
            cand_delta = (neighbor_posts[j] - self_prev) if cfg.trustgraph_delta_mode == "receiver" else neighbor_updates[j]
            s_ij = _scoped_cosine_similarity(cand_delta, ref_vec, cfg.trustgraph_score_scope, param_slices)
            scores[j] = s_ij
            if cfg.trustgraph_use_relu:
                raw_weights[j] = max(0.0, s_ij - cfg.trustgraph_score_threshold)
            else:
                # Ablation without ReLU: map [-1, 1] -> [0, 1] so negative scores retain weight.
                raw_weights[j] = 0.5 * (s_ij + 1.0)
        else:
            # Magnitude-based Euclidean distance ablation (Section 5.5).
            dist = float(torch.linalg.vector_norm(neighbor_posts[j] - self_post).item())
            s_ij = 1.0 / (1.0 + dist)
            scores[j] = s_ij
            raw_weights[j] = s_ij

    denom = sum(raw_weights.values())
    if denom <= 0.0:
        # Section 4.4: if all neighbors have non-positive consistency, rely solely on self.
        return _self_only_result(self_post, neighbor_ids, scores=scores, self_fallback=True)

    weights = {j: raw_weights[j] / (denom + cfg.eps) for j in neighbor_ids}
    if cfg.trustgraph_clip_norm:
        if self_prev is None or self_update is None:
            raise ValueError("self_prev and self_update are required when trustgraph_clip_norm=True.")
        r_i = float(torch.linalg.vector_norm(self_update).item())
        if cfg.trustgraph_clip_mode == "model":
            # Consensus-preserving model-space clip: cap displacement (theta_j^{t+1/2} - theta_i^t).
            cap = cfg.trustgraph_clip_factor * r_i
            nbr_avg = torch.zeros_like(self_post)
            for j in neighbor_ids:
                if weights[j] > 0.0:
                    disp = neighbor_posts[j] - self_prev
                    dn = float(torch.linalg.vector_norm(disp).item())
                    scale = min(1.0, cap / dn) if dn > 0.0 else 1.0
                    nbr_avg.add_(self_prev + scale * disp, alpha=weights[j])
            new_model = cfg.alpha * self_post + (1.0 - cfg.alpha) * nbr_avg
        else:
            agg_update = torch.zeros_like(self_update)
            for j in neighbor_ids:
                if weights[j] > 0.0:
                    u_j = neighbor_updates[j]
                    n_j = float(torch.linalg.vector_norm(u_j).item())
                    if n_j > 0.0:
                        a_ij = min(1.0, r_i / n_j)
                        agg_update.add_(u_j, alpha=weights[j] * a_ij)
            new_model = self_prev + cfg.alpha * self_update + (1.0 - cfg.alpha) * agg_update
    else:
        nbr_avg = torch.zeros_like(self_post)
        for j in neighbor_ids:
            if weights[j] > 0.0:
                nbr_avg.add_(neighbor_posts[j], alpha=weights[j])
        new_model = cfg.alpha * self_post + (1.0 - cfg.alpha) * nbr_avg

    return AggregationResult(
        new_model=new_model,
        neighbor_weights=weights,
        neighbor_zero_weight_frac={j: (1.0 if weights[j] <= 0.0 else 0.0) for j in neighbor_ids},
        scores=scores,
        self_fallback=False,
    )


@torch.no_grad()
def aggregate_balance(
    self_post: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
    round_idx: int,
    total_rounds: int,
    cfg: AggregatorConfig,
) -> AggregationResult:
    """BALANCE distance-threshold defense (Fang et al., CCS 2024)."""
    if not neighbor_ids:
        return _self_only_result(self_post, neighbor_ids)

    self_norm = float(torch.linalg.vector_norm(self_post).item())
    lam_t = round_idx / max(total_rounds, 1)
    threshold = cfg.balance_gamma * math.exp(-cfg.balance_kappa * lam_t) * self_norm

    accepted: list[int] = []
    dists: dict[int, float] = {}
    for j in neighbor_ids:
        dist = float(torch.linalg.vector_norm(neighbor_posts[j] - self_post).item())
        dists[j] = dist
        if dist <= threshold:
            accepted.append(j)

    if not accepted:
        return _self_only_result(self_post, neighbor_ids, scores=dists, self_fallback=True)

    w_acc = 1.0 / len(accepted)
    nbr_avg = torch.stack([neighbor_posts[j] for j in accepted], dim=0).mean(dim=0)
    new_model = cfg.alpha * self_post + (1.0 - cfg.alpha) * nbr_avg
    accepted_set = set(accepted)
    return AggregationResult(
        new_model=new_model,
        neighbor_weights={j: (w_acc if j in accepted_set else 0.0) for j in neighbor_ids},
        neighbor_zero_weight_frac={j: (0.0 if j in accepted_set else 1.0) for j in neighbor_ids},
        scores=dists,
        self_fallback=False,
    )


@torch.no_grad()
def aggregate_krum(
    self_id: int,
    self_post: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
    byzantine_frac: float,
    cfg: AggregatorConfig,
) -> AggregationResult:
    """Local neighborhood Krum / Multi-Krum over N_i U {i}."""
    if not neighbor_ids:
        return _self_only_result(self_post, neighbor_ids)

    candidates = [self_id] + list(neighbor_ids)
    cand_tensors = [self_post] + [neighbor_posts[j] for j in neighbor_ids]
    m = len(candidates)

    mat = torch.stack(cand_tensors, dim=0)
    dist_sq = torch.cdist(mat, mat, p=2).pow(2)

    f_loc = int(round(byzantine_frac * len(neighbor_ids)))
    f_loc = max(0, min(f_loc, max(0, m - 2)))
    # Standard Krum uses m - f - 2 nearest neighbors; clamp to [1, m - 1] for small degree (e.g., ring).
    k_nn = max(1, min(m - 1, m - f_loc - 2))

    krum_scores: list[float] = []
    for idx in range(m):
        row = dist_sq[idx]
        # Exclude distance to self (index idx) before selecting k_nn nearest peers.
        other = torch.cat([row[:idx], row[idx + 1 :]])
        nearest, _ = torch.topk(other, k=k_nn, largest=False)
        krum_scores.append(float(nearest.sum().item()))

    order = sorted(range(m), key=lambda idx: krum_scores[idx])
    if cfg.krum_mode == "single":
        selected_idx = [order[0]]
    else:
        q = max(1, m - f_loc)
        selected_idx = order[:q]

    selected_ids = {candidates[idx] for idx in selected_idx}
    new_model = torch.stack([cand_tensors[idx] for idx in selected_idx], dim=0).mean(dim=0)
    w_sel = 1.0 / len(selected_idx)

    return AggregationResult(
        new_model=new_model,
        neighbor_weights={j: (w_sel if j in selected_ids else 0.0) for j in neighbor_ids},
        neighbor_zero_weight_frac={j: (0.0 if j in selected_ids else 1.0) for j in neighbor_ids},
        scores={j: krum_scores[idx + 1] for idx, j in enumerate(neighbor_ids)},
        self_fallback=(selected_ids == {self_id}),
    )


@torch.no_grad()
def aggregate_trimmed_mean(
    self_post: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
    cfg: AggregatorConfig,
) -> AggregationResult:
    """Coordinate-wise trimmed mean over N_i U {i}."""
    if not neighbor_ids:
        return _self_only_result(self_post, neighbor_ids)

    cand_tensors = [self_post] + [neighbor_posts[j] for j in neighbor_ids]
    m = len(cand_tensors)
    dim = int(self_post.numel())
    k_trim = min(int(math.floor(cfg.trim_ratio * m)), (m - 1) // 2)

    mat = torch.stack(cand_tensors, dim=0)
    if k_trim == 0:
        new_model = mat.mean(dim=0)
        w = 1.0 / m
        return AggregationResult(
            new_model=new_model,
            neighbor_weights={j: w for j in neighbor_ids},
            neighbor_zero_weight_frac={j: 0.0 for j in neighbor_ids},
        )

    sorted_vals, sorted_idx = torch.sort(mat, dim=0)
    kept_vals = sorted_vals[k_trim : m - k_trim]
    kept_idx = sorted_idx[k_trim : m - k_trim]
    new_model = kept_vals.mean(dim=0)

    kept_counts = torch.bincount(kept_idx.reshape(-1), minlength=m).float()
    kept_frac = (kept_counts / dim).cpu().tolist()
    n_kept = float(m - 2 * k_trim)
    neighbor_weights = {j: kept_frac[idx + 1] / n_kept for idx, j in enumerate(neighbor_ids)}
    neighbor_zero_frac = {j: 1.0 - kept_frac[idx + 1] for idx, j in enumerate(neighbor_ids)}
    return AggregationResult(
        new_model=new_model,
        neighbor_weights=neighbor_weights,
        neighbor_zero_weight_frac=neighbor_zero_frac,
    )


def aggregate_node(
    self_id: int,
    self_post: torch.Tensor,
    val_grad: torch.Tensor,
    neighbor_ids: list[int],
    neighbor_posts: dict[int, torch.Tensor],
    neighbor_updates: dict[int, torch.Tensor],
    round_idx: int,
    total_rounds: int,
    byzantine_frac: float,
    cfg: AggregatorConfig,
    self_prev: torch.Tensor | None = None,
    self_update: torch.Tensor | None = None,
    param_slices: Sequence[ParamSlice] | None = None,
) -> AggregationResult:
    """Dispatch to the configured decentralized aggregation rule."""
    if cfg.name == "fedavg":
        return aggregate_fedavg(self_post, neighbor_ids, neighbor_posts)
    if cfg.name == "trustgraph":
        return aggregate_trustgraph(
            self_post,
            val_grad,
            neighbor_ids,
            neighbor_posts,
            neighbor_updates,
            cfg,
            self_prev=self_prev,
            self_update=self_update,
            param_slices=param_slices,
        )
    if cfg.name == "balance":
        return aggregate_balance(self_post, neighbor_ids, neighbor_posts, round_idx, total_rounds, cfg)
    if cfg.name == "krum":
        return aggregate_krum(self_id, self_post, neighbor_ids, neighbor_posts, byzantine_frac, cfg)
    if cfg.name == "trimmed_mean":
        return aggregate_trimmed_mean(self_post, neighbor_ids, neighbor_posts, cfg)
    raise ValueError(f"Unknown aggregator: {cfg.name!r}")
