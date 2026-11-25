"""Decentralized aggregation rules for Phase 1 TrustGraph-DFL reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class AggregatorConfig:
    """Configuration for TrustGraph-DFL aggregation and its ablations."""

    name: str = "trustgraph"
    alpha: float = 0.5  # Self-weight on the peer's own local model.
    eps: float = 1e-8
    val_grad_sign: str = "descent_neg"  # Either descent_neg (-nabla F) or paper_pos (+nabla F).
    trustgraph_use_relu: bool = True
    trustgraph_metric: str = "cosine"  # Either cosine or euclidean.
    trustgraph_score_threshold: float = 0.0
    trustgraph_clip_norm: bool = False  # Cap accepted neighbor contributions using the receiver norm.
    trustgraph_clip_mode: str = "update"  # Either update (update norm cap) or model (displacement cap).
    trustgraph_clip_factor: float = 2.0  # Multiplier on r_i for the model-space displacement cap.
    trustgraph_delta_mode: str = "sender"  # Either sender (theta_j^{t+1/2} - theta_j^t) or receiver (theta_j^{t+1/2} - theta_i^t).

    def validate(self) -> None:
        """Check that all aggregation configuration options are supported."""
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
    """Return an aggregation result that keeps self_post unchanged and zeroes all neighbors."""
    return AggregationResult(
        new_model=self_post.clone(),
        neighbor_weights={j: 0.0 for j in neighbor_ids},
        neighbor_zero_weight_frac={j: 1.0 for j in neighbor_ids},
        scores=scores if scores is not None else {},
        self_fallback=self_fallback,
    )


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return cosine similarity between two 1D tensors, or 0.0 if either norm is zero."""
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


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
) -> AggregationResult:
    """Compute TrustGraph-DFL consistency-weighted neighborhood aggregation (Eq. 3 to Eq. 5).

    Supports validation gradient sign selection (paper_pos for +nabla F_i and
    descent_neg for -nabla F_i), optional receiver-norm clipping, and
    receiver-anchored deltas (theta_j^{t+1/2} - theta_i^t).
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
            s_ij = _cosine_similarity(cand_delta, ref_vec)
            scores[j] = s_ij
            if cfg.trustgraph_use_relu:
                raw_weights[j] = max(0.0, s_ij - cfg.trustgraph_score_threshold)
            else:
                # Without ReLU, map cosine scores from [-1, 1] to [0, 1] so negative scores keep weight.
                raw_weights[j] = 0.5 * (s_ij + 1.0)
        else:
            # Euclidean distance ablation from Section 5.5.
            dist = float(torch.linalg.vector_norm(neighbor_posts[j] - self_post).item())
            s_ij = 1.0 / (1.0 + dist)
            scores[j] = s_ij
            raw_weights[j] = s_ij

    denom = sum(raw_weights.values())
    if denom <= 0.0:
        # Fall back to the local model when every neighbor receives zero weight (Section 4.4).
        return _self_only_result(self_post, neighbor_ids, scores=scores, self_fallback=True)

    weights = {j: raw_weights[j] / (denom + cfg.eps) for j in neighbor_ids}
    if cfg.trustgraph_clip_norm:
        if self_prev is None or self_update is None:
            raise ValueError("self_prev and self_update are required when trustgraph_clip_norm=True.")
        r_i = float(torch.linalg.vector_norm(self_update).item())
        if cfg.trustgraph_clip_mode == "model":
            # Cap each neighbor's displacement from the receiver's previous model.
            cap = cfg.trustgraph_clip_factor * r_i
            nbr_avg = torch.zeros_like(self_post)
            for j in neighbor_ids:
                if weights[j] > 0.0:
                    disp = neighbor_posts[j] - self_prev
                    dn = float(torch.linalg.vector_norm(disp).item())
                    scale = min(1.0, cap / dn) if dn > 0.0 else 1.0
                    nbr_avg.add_(self_prev, alpha=weights[j])
                    nbr_avg.add_(disp, alpha=weights[j] * scale)
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
