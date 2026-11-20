"""Byzantine attack implementations for Phase 1 TrustGraph-DFL reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AttackConfig:
    """Attack specification matching TrustGraph-DFL Section 3.2 & 5.1."""

    name: str = "none"  # none | label_flip | label_flip_all | model_poison | random_noise
    poison_lambda: float = 10.0
    noise_sigma: float = 0.5

    def validate(self) -> None:
        valid = {"none", "label_flip", "label_flip_all", "model_poison", "random_noise"}
        if self.name not in valid:
            raise ValueError(f"Unknown attack {self.name!r}; expected one of {sorted(valid)}.")


def is_data_poisoning_attack(attack_name: str) -> bool:
    return attack_name in {"label_flip", "label_flip_all"}


def label_flip_mode_for_attack(attack_name: str) -> str | None:
    if attack_name == "label_flip":
        return "pair_0_1"
    if attack_name == "label_flip_all":
        return "all_classes"
    return None


@torch.no_grad()
def apply_round_attack(
    attack_cfg: AttackConfig,
    byzantine_ids: set[int],
    benign_ids: list[int],
    prev_models: list[torch.Tensor],
    post_models: list[torch.Tensor],
    updates: list[torch.Tensor],
    generator: torch.Generator,
) -> None:
    """Mutate Byzantine peers' post_models and updates in-place for round-level attacks.

    For model_poison (Section 3.2):
        Delta_theta_mal = -lambda * mean(Delta_theta_benign), lambda = 10.
    For random_noise (Section 3.2 & 5.1):
        Delta_theta_mal ~ N(0, sigma^2 I), sigma = 0.5.

    In both cases, the broadcast model is anchored at mean(theta_benign^t) + Delta_theta_mal
    so that post_model and update are consistent and not distorted by stale malicious state.
    """
    if not byzantine_ids or attack_cfg.name in {"none", "label_flip", "label_flip_all"}:
        return

    if not benign_ids:
        raise ValueError("Cannot craft Byzantine payloads with zero benign peers.")

    benign_prev_mean = torch.stack([prev_models[i] for i in benign_ids], dim=0).mean(dim=0)

    if attack_cfg.name == "model_poison":
        benign_update_mean = torch.stack([updates[i] for i in benign_ids], dim=0).mean(dim=0)
        mal_update = -attack_cfg.poison_lambda * benign_update_mean
        mal_post = benign_prev_mean + mal_update
        for m in sorted(byzantine_ids):
            updates[m] = mal_update.clone()
            post_models[m] = mal_post.clone()
        return

    if attack_cfg.name == "random_noise":
        dim = benign_prev_mean.shape
        device = benign_prev_mean.device
        dtype = benign_prev_mean.dtype
        for m in sorted(byzantine_ids):
            noise = torch.randn(dim, generator=generator, device="cpu", dtype=dtype).to(device)
            noise.mul_(attack_cfg.noise_sigma)
            updates[m] = noise
            post_models[m] = benign_prev_mean + noise
        return

    raise ValueError(f"Unhandled attack name: {attack_cfg.name!r}")
