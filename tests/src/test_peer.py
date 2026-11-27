from __future__ import annotations

import torch

from src.aggregators import AggregatorConfig
from src.attacks import AttackConfig
from src.models import build_model, model_to_vector
from src.peer import PeerNode, PeerPayload


def _make_peer(peer_id: int, neighbors: list[int], is_byzantine: bool) -> PeerNode:
    torch.manual_seed(peer_id)
    model = build_model("cnn2x2")
    init_vec = model_to_vector(model)
    x_tr = torch.randn(8, 3, 32, 32)
    y_tr = torch.randint(0, 10, (8,))
    x_va = torch.randn(4, 3, 32, 32)
    y_va = torch.randint(0, 10, (4,))
    return PeerNode(
        peer_id=peer_id,
        neighbors=neighbors,
        is_byzantine=is_byzantine,
        model=model,
        init_vec=init_vec,
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        device=torch.device("cpu"),
    )


def test_peer_payload_bytes_and_finite_check() -> None:
    vec = torch.zeros(16, dtype=torch.float32)
    payload = PeerPayload(sender_id=0, post_model=vec, update=vec)
    assert payload.payload_bytes() == 64

    peer = _make_peer(0, [1], is_byzantine=False)
    assert peer.param_count == int(peer.model_vec.numel())
    assert peer.has_finite_weights() is True
    peer.model_vec[0] = float("nan")
    assert peer.has_finite_weights() is False


def test_peer_local_step_and_decentralized_payloads() -> None:
    gen = torch.Generator().manual_seed(0)
    benign = _make_peer(0, [1], is_byzantine=False)
    step_b = benign.local_step(
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        batch_size=8,
        epochs=1,
        augment=False,
        grad_clip_norm=0.0,
        val_grad_weight_decay=False,
        train_gen=gen,
    )
    assert step_b.val_grad is not None
    assert step_b.val_grad.shape == benign.model_vec.shape
    assert torch.allclose(step_b.update, step_b.post_model - step_b.prev_model)

    byz = _make_peer(1, [0], is_byzantine=True)
    step_skip = byz.local_step(
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        batch_size=8,
        epochs=1,
        augment=False,
        grad_clip_norm=0.0,
        val_grad_weight_decay=False,
        train_gen=gen,
        skip_training=True,
    )
    assert step_skip.val_grad is None
    assert step_skip.post_model.numel() == 0

    step_m = byz.local_step(
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        batch_size=8,
        epochs=1,
        augment=False,
        grad_clip_norm=0.0,
        val_grad_weight_decay=False,
        train_gen=gen,
    )
    assert step_m.val_grad is None
    mp = byz.prepare_decentralized_payload(step_m, AttackConfig(name="model_poison", poison_lambda=5.0), gen)
    assert torch.allclose(mp.update, -5.0 * step_m.update)
    assert torch.allclose(mp.post_model, step_m.prev_model + mp.update)

    rn = byz.prepare_decentralized_payload(step_m, AttackConfig(name="random_noise", noise_sigma=0.25), gen)
    assert torch.allclose(rn.post_model, step_m.prev_model + rn.update)


def test_byzantine_peer_aggregation_drops_extreme_outliers() -> None:
    byz = _make_peer(1, [0, 2], is_byzantine=True)
    gen = torch.Generator().manual_seed(1)
    step = byz.local_step(
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        batch_size=8,
        epochs=1,
        augment=False,
        grad_clip_norm=0.0,
        val_grad_weight_decay=False,
        train_gen=gen,
    )
    near_model = step.post_model + 0.1 * step.update
    far_model = step.post_model + 1000.0 * step.update
    received = {
        0: PeerPayload(sender_id=0, post_model=near_model, update=near_model - step.prev_model),
        2: PeerPayload(sender_id=2, post_model=far_model, update=far_model - step.prev_model),
    }
    res, elapsed = byz.aggregate_received(step, received, AggregatorConfig(), attack_name="model_poison")
    assert res is None
    assert elapsed >= 0.0
    assert torch.allclose(byz.model_vec, 0.5 * (step.post_model + near_model), atol=1e-5)
