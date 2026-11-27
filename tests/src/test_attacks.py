from __future__ import annotations

import pytest
import torch

from src.attacks import (
    AttackConfig,
    apply_round_attack,
    is_data_poisoning_attack,
    label_flip_mode_for_attack,
)


def test_attack_config_validation_and_helpers() -> None:
    for name in ("none", "label_flip", "label_flip_all", "model_poison", "random_noise"):
        AttackConfig(name=name).validate()

    with pytest.raises(ValueError, match="Unknown attack"):
        AttackConfig(name="backdoor").validate()

    assert is_data_poisoning_attack("label_flip") is True
    assert is_data_poisoning_attack("label_flip_all") is True
    assert is_data_poisoning_attack("model_poison") is False
    assert is_data_poisoning_attack("none") is False

    assert label_flip_mode_for_attack("label_flip") == "pair_0_1"
    assert label_flip_mode_for_attack("label_flip_all") == "all_classes"
    assert label_flip_mode_for_attack("random_noise") is None


def test_apply_round_attack_model_poison() -> None:
    prev = [torch.ones(4) for _ in range(4)]
    updates = [
        torch.tensor([0.1, -0.1, 0.2, 0.0]),
        torch.tensor([0.3, -0.3, 0.0, 0.2]),
        torch.zeros(4),
        torch.zeros(4),
    ]
    posts = [prev[i] + updates[i] for i in range(4)]
    gen = torch.Generator().manual_seed(0)

    apply_round_attack(
        AttackConfig(name="model_poison", poison_lambda=10.0),
        byzantine_ids={2, 3},
        benign_ids=[0, 1],
        prev_models=prev,
        post_models=posts,
        updates=updates,
        generator=gen,
    )
    avg_ben = 0.5 * (updates[0] + updates[1])
    assert torch.allclose(updates[2], -10.0 * avg_ben)
    assert torch.allclose(updates[3], -10.0 * avg_ben)
    assert torch.allclose(posts[2], prev[0] + updates[2])


def test_apply_round_attack_random_noise_and_edge_cases() -> None:
    prev = [torch.full((64,), 2.0) for _ in range(3)]
    updates = [torch.zeros(64) for _ in range(3)]
    posts = [p.clone() for p in prev]

    apply_round_attack(
        AttackConfig(name="random_noise", noise_sigma=0.5),
        byzantine_ids={2},
        benign_ids=[0, 1],
        prev_models=prev,
        post_models=posts,
        updates=updates,
        generator=torch.Generator().manual_seed(7),
    )
    assert float(updates[2].norm().item()) > 0.0
    assert torch.allclose(posts[2], prev[0] + updates[2])
    assert torch.equal(updates[0], torch.zeros(64))

    # Data-poisoning and no-attack modes leave updates untouched.
    before = updates[2].clone()
    apply_round_attack(
        AttackConfig(name="label_flip"),
        byzantine_ids={2},
        benign_ids=[0, 1],
        prev_models=prev,
        post_models=posts,
        updates=updates,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(updates[2], before)

    with pytest.raises(ValueError, match="zero benign peers"):
        apply_round_attack(
            AttackConfig(name="model_poison"),
            byzantine_ids={0},
            benign_ids=[],
            prev_models=prev,
            post_models=posts,
            updates=updates,
            generator=torch.Generator().manual_seed(7),
        )
