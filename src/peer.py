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
    """Block until pending CUDA kernels finish on device.

    Args:
        device: Torch device to synchronize when using CUDA.

    Returns:
        None.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class PeerPayload:
    """Update message exchanged between neighboring peers in one DFL round."""

    sender_id: int
    post_model: torch.Tensor
    update: torch.Tensor

    def payload_bytes(self) -> int:
        """Return the byte size of one model parameter vector in this payload.

        Returns:
            Number of bytes occupied by post_model.
        """
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
        """Initialize a peer with its local shards, graph neighbors, and initial weights.

        Args:
            peer_id: Unique integer rank of this peer.
            neighbors: List of adjacent peer IDs in the communication graph.
            is_byzantine: Whether this peer behaves maliciously.
            model: PyTorch module instance used for training, validation, and evaluation.
            init_vec: Initial 1D parameter vector copied onto device.
            x_train: Local training images on device.
            y_train: Local training labels on device.
            x_val: Local validation images on device.
            y_val: Local validation labels on device.
            device: Compute device assigned to this peer.

        Returns:
            None.
        """
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
        """Return the total number of scalar parameters in this peer's model.

        Returns:
            Integer count of parameters in model_vec.
        """
        return int(self.model_vec.numel())

    def has_finite_weights(self) -> bool:
        """Check whether every weight in model_vec is finite.

        Returns:
            True if model_vec contains no NaN or Inf values, False otherwise.
        """
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
        """Compute the pre-update validation gradient and run local SGD on this peer's shard.

        Args:
            lr: Learning rate for this round.
            momentum: SGD momentum factor.
            weight_decay: L2 regularization weight decay.
            batch_size: Mini-batch size for validation gradient and SGD.
            epochs: Number of local training epochs.
            augment: Whether to apply CIFAR flip and crop augmentation.
            grad_clip_norm: Max gradient L2 norm (disabled when <= 0).
            val_grad_weight_decay: Whether to add L2 decay term to the validation gradient.
            train_gen: CPU random generator for shuffling and augmentation.
            skip_training: If True, skip SGD and return empty post_model/update placeholders.

        Returns:
            LocalStepResult containing pre/post weights, update vector, validation gradient, and timings.
        """
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
        """Build this peer's outgoing payload using only local state for distributed runs.

        Args:
            step: Output of local_step() for the current round.
            attack_cfg: Configured attack settings.
            attack_gen: CPU random generator for noise attacks.

        Returns:
            PeerPayload to transmit to graph neighbors.
        """
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
        """Aggregate received neighbor payloads and update model_vec in-place.

        Args:
            step: Output of local_step() for the current round.
            received: Map of neighbor ID to received PeerPayload.
            agg_cfg: TrustGraph aggregation configuration.
            attack_name: Active attack name used to pick Byzantine state update behavior.

        Returns:
            Tuple of (AggregationResult for benign peers or None for Byzantine peers, elapsed seconds).
        """
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
        """Evaluate this peer's local model on the held-out test set.

        Args:
            x_test: Test image tensor on device.
            y_test: Test label tensor on device.
            eval_batch_size: Mini-batch size for evaluation.

        Returns:
            Tuple of (accuracy in [0, 1], mean cross-entropy loss).
        """
        vector_to_model(self.model_vec, self.model)
        return evaluate_accuracy(self.model, x_test, y_test, batch_size=eval_batch_size)
