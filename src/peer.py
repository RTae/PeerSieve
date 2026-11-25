"""Decentralized peer runtime shared across local and multi-process DFL execution."""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn as nn

from src.aggregators import AggregationResult, AggregatorConfig, aggregate_trustgraph
from src.attacks import AttackConfig
from src.models import (
    compute_validation_gradient,
    evaluate_accuracy,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)


def sync_device(device: torch.device) -> None:
    """Block until pending CUDA kernels finish on device."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class PeerPayload:
    """Update message exchanged between neighboring peers in one DFL round."""

    sender_id: int
    post_model: torch.Tensor
    update: torch.Tensor

    def payload_bytes(self) -> int:
        """Return the byte size of one model parameter vector in this payload."""
        return int(self.post_model.numel()) * self.post_model.element_size()


@dataclass
class LocalStepResult:
    """Outputs and synchronized timings of one peer's local training step."""

    prev_model: torch.Tensor
    post_model: torch.Tensor
    update: torch.Tensor
    val_grad: torch.Tensor | None
    train_sec: float
    val_grad_sec: float


class PeerNode:
    """Self-contained decentralized FL peer owning local data, model weights, and neighbor list."""

    def __init__(
        self,
        peer_id: int,
        neighbors: list[int],
        is_byzantine: bool,
        model: nn.Module,
        init_vec: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Initialize a peer with its local shards, graph neighbors, and initial weights."""
        self.peer_id = peer_id
        self.neighbors = list(neighbors)
        self.is_byzantine = is_byzantine
        self.model = model
        self.model_vec = init_vec.clone().to(device)
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.device = device

    @property
    def param_count(self) -> int:
        """Return the total number of scalar parameters in this peer's model."""
        return int(self.model_vec.numel())

    def has_finite_weights(self) -> bool:
        """Return True if every model weight is finite (no NaN or Inf values)."""
        return bool(torch.isfinite(self.model_vec).all().item())

    def local_step(
        self,
        lr: float,
        momentum: float,
        weight_decay: float,
        batch_size: int,
        epochs: int,
        augment: bool,
        grad_clip_norm: float,
        val_grad_weight_decay: bool,
        train_gen: torch.Generator,
        skip_training: bool = False,
    ) -> LocalStepResult:
        """Compute pre-update validation gradient and run local SGD on this peer's shard."""
        prev_model = self.model_vec.clone()
        if skip_training:
            empty = torch.empty(0, device=self.device)
            return LocalStepResult(
                prev_model=prev_model,
                post_model=empty,
                update=empty,
                val_grad=None,
                train_sec=0.0,
                val_grad_sec=0.0,
            )

        vector_to_model(prev_model, self.model)
        val_grad: torch.Tensor | None = None
        val_sec = 0.0
        if not self.is_byzantine:
            sync_device(self.device)
            t_v0 = time.perf_counter()
            val_grad = compute_validation_gradient(
                self.model,
                self.x_val,
                self.y_val,
                batch_size=batch_size,
                weight_decay=(weight_decay if val_grad_weight_decay else 0.0),
            )
            sync_device(self.device)
            val_sec = time.perf_counter() - t_v0

        sync_device(self.device)
        t_tr0 = time.perf_counter()
        train_local_epoch(
            self.model,
            self.x_train,
            self.y_train,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            augment=augment,
            grad_clip_norm=grad_clip_norm,
            generator=train_gen,
        )
        sync_device(self.device)
        train_sec = time.perf_counter() - t_tr0

        post_vec = model_to_vector(self.model)
        update_vec = post_vec - prev_model
        return LocalStepResult(
            prev_model=prev_model,
            post_model=post_vec,
            update=update_vec,
            val_grad=val_grad,
            train_sec=train_sec,
            val_grad_sec=val_sec,
        )

    @torch.no_grad()
    def prepare_decentralized_payload(
        self,
        step: LocalStepResult,
        attack_cfg: AttackConfig,
        attack_gen: torch.Generator,
    ) -> PeerPayload:
        """Build this peer's outgoing payload using only local state (for distributed runs)."""
        if not self.is_byzantine or attack_cfg.name in {"none", "label_flip", "label_flip_all"}:
            return PeerPayload(sender_id=self.peer_id, post_model=step.post_model, update=step.update)

        if attack_cfg.name == "model_poison":
            mal_update = -attack_cfg.poison_lambda * step.update
            return PeerPayload(
                sender_id=self.peer_id,
                post_model=step.prev_model + mal_update,
                update=mal_update,
            )

        if attack_cfg.name == "random_noise":
            noise = torch.randn(
                step.prev_model.shape,
                generator=attack_gen,
                device="cpu",
                dtype=step.prev_model.dtype,
            ).to(self.device)
            noise.mul_(attack_cfg.noise_sigma)
            return PeerPayload(
                sender_id=self.peer_id,
                post_model=step.prev_model + noise,
                update=noise,
            )

        raise ValueError(f"Unhandled attack name: {attack_cfg.name!r}")

    @torch.no_grad()
    def aggregate_received(
        self,
        step: LocalStepResult,
        received: dict[int, PeerPayload],
        agg_cfg: AggregatorConfig,
        attack_name: str,
    ) -> tuple[AggregationResult | None, float]:
        """Run TrustGraph-DFL aggregation for benign peers or neighbor averaging for label-flip attackers."""
        sync_device(self.device)
        t0 = time.perf_counter()

        if not self.is_byzantine:
            if step.val_grad is None:
                raise RuntimeError(f"Benign peer {self.peer_id} is missing its validation gradient.")
            nbr_posts = {j: received[j].post_model for j in self.neighbors}
            nbr_updates = {j: received[j].update for j in self.neighbors}
            res = aggregate_trustgraph(
                self_post=step.post_model,
                val_grad=step.val_grad,
                neighbor_ids=self.neighbors,
                neighbor_posts=nbr_posts,
                neighbor_updates=nbr_updates,
                cfg=agg_cfg,
                self_prev=step.prev_model,
                self_update=step.update,
            )
            self.model_vec = res.new_model
            sync_device(self.device)
            return res, time.perf_counter() - t0

        if attack_name in {"label_flip", "label_flip_all"} and self.neighbors:
            tensors = [step.post_model] + [received[j].post_model for j in self.neighbors]
            self.model_vec = torch.stack(tensors, dim=0).mean(dim=0)
        elif self.neighbors:
            # Track nearby neighbors while excluding extreme poisoned payloads from peer attackers.
            ref_norm = max(float(torch.linalg.vector_norm(step.update).item()), 1e-6)
            max_dist = 4.0 * ref_norm
            inliers = [
                received[j].post_model
                for j in self.neighbors
                if bool(torch.isfinite(received[j].post_model).all().item())
                and float(torch.linalg.vector_norm(received[j].post_model - step.post_model).item()) <= max_dist
            ]
            if inliers:
                self.model_vec = torch.stack([step.post_model] + inliers, dim=0).mean(dim=0)
            else:
                self.model_vec = step.post_model.clone()
        else:
            self.model_vec = step.post_model.clone()
        sync_device(self.device)
        return None, time.perf_counter() - t0

    @torch.no_grad()
    def evaluate(self, x_test: torch.Tensor, y_test: torch.Tensor, eval_batch_size: int) -> tuple[float, float]:
        """Evaluate this peer's local model on the held-out test set."""
        vector_to_model(self.model_vec, self.model)
        return evaluate_accuracy(self.model, x_test, y_test, batch_size=eval_batch_size)
