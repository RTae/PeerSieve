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
    """Configuration for TrustGraph-DFL aggregation and variants."""

    name: str = "trustgraph"
    alpha: float = 0.5  # self-weight
    eps: float = 1e-8
    # TrustGraph-DFL specific
    val_grad_sign: str = "descent_neg"  # descent_neg (-nabla F) | paper_pos (+nabla F)
    trustgraph_use_relu: bool = True
    trustgraph_metric: str = "cosine"  # cosine | euclidean
    trustgraph_score_threshold: float = 0.0
    trustgraph_clip_norm: bool = False  # receiver-norm cap on accepted neighbor contributions
    trustgraph_clip_mode: str = "update"  # update (receiver-norm update cap) | model (consensus-preserving)
    trustgraph_clip_factor: float = 2.0  # multiplier on r_i for model-space displacement cap
    trustgraph_delta_mode: str = "sender"  # sender (theta_j^{t+1/2}-theta_j^t) | receiver (theta_j^{t+1/2}-theta_i^t)
    trustgraph_score_scope: str = "global"  # global | layer_mean | backbone_only | head_only
    trustgraph_ema_beta: float = 0.0  # 0.0 disables EMA; in (0, 1): R_ij^t = beta*R_ij^{t-1} + (1-beta)*s_ij^t

    def validate(self) -> None:
        if self.name != "trustgraph":
            raise ValueError(f"Unknown aggregator {self.name!r}; expected 'trustgraph'.")
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
        if not (0.0 <= self.trustgraph_ema_beta < 1.0):
            raise ValueError(f"trustgraph_ema_beta must be in [0, 1); got {self.trustgraph_ema_beta!r}")


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
    ema_scores: dict[int, float] | None = None,
) -> AggregationResult:
    """TrustGraph-DFL consistency-weighted neighborhood aggregation (Eq. 3-5).

    Supports:
    - val_grad_sign="paper_pos": literal Algorithm 1 / Eq. 3 using +nabla F_i(theta_i^t; D_i^val)
    - val_grad_sign="descent_neg": corrected descent direction -nabla F_i(theta_i^t; D_i^val)
    - trustgraph_clip_norm=True: receiver-norm cap on neighbor updates
      a_ij = min(1, ||Delta_theta_i||_2 / ||Delta_theta_j||_2), blended in update space.
    - trustgraph_ema_beta > 0: multi-round zero-initialized EMA on per-edge cosine scores
      R_ij^t = beta * R_ij^{t-1} + (1 - beta) * s_ij^t.
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
            if cfg.trustgraph_ema_beta > 0.0:
                prev_r = ema_scores.get(j, 0.0) if ema_scores is not None else 0.0
                eff_score = cfg.trustgraph_ema_beta * prev_r + (1.0 - cfg.trustgraph_ema_beta) * s_ij
            else:
                eff_score = s_ij
            scores[j] = eff_score
            if cfg.trustgraph_use_relu:
                raw_weights[j] = max(0.0, eff_score - cfg.trustgraph_score_threshold)
            else:
                # Ablation without ReLU: map [-1, 1] -> [0, 1] so negative scores retain weight.
                raw_weights[j] = 0.5 * (eff_score + 1.0)
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
    ema_scores: dict[int, float] | None = None,
) -> AggregationResult:
    """Dispatch to TrustGraph-DFL aggregation."""
    if cfg.name != "trustgraph":
        raise ValueError(f"Unknown aggregator: {cfg.name!r}; expected 'trustgraph'.")
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
        ema_scores=ema_scores,
    )
