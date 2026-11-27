from __future__ import annotations

import math

import pytest
import torch

from src.models import (
    augment_cifar_batch,
    build_model,
    compute_validation_gradient,
    evaluate_accuracy,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)


def test_build_model_and_frozen_batchnorm() -> None:
    gn_model = build_model("cnn2x2_gn")
    xb = augment_cifar_batch(torch.randn(4, 3, 32, 32), generator=torch.Generator().manual_seed(0))
    assert xb.shape == (4, 3, 32, 32)
    assert gn_model(xb).shape == (4, 10)

    rn = build_model("resnet18_scratch")
    rn.train()
    for mod in rn.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            assert mod.training is False
    assert rn(xb).shape == (4, 10)

    with pytest.raises(ValueError, match="Unknown model name"):
        build_model("vgg99")


def test_validation_gradient_matches_exact_mean_across_batches() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    x = torch.randn(10, 3, 32, 32)
    y = torch.randint(0, 10, (10,))
    g_full = compute_validation_gradient(model, x, y, batch_size=64)
    g_batched = compute_validation_gradient(model, x, y, batch_size=4)
    assert torch.allclose(g_full, g_batched, atol=1e-5)
    assert model_to_vector(model).numel() > 100_000

    # Empty validation tensor yields a zero gradient vector.
    g_empty = compute_validation_gradient(model, torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
    assert torch.equal(g_empty, torch.zeros_like(g_full))


def test_validation_gradient_weight_decay_matches_l2_term() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    g0 = compute_validation_gradient(model, x, y, batch_size=8, weight_decay=0.0)
    g_wd = compute_validation_gradient(model, x, y, batch_size=8, weight_decay=1e-2)
    theta = model_to_vector(model)
    assert torch.allclose(g_wd - g0, 1e-2 * theta, atol=1e-6)


def test_vector_to_model_does_not_alias_and_checks_length() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    v0 = model_to_vector(model)
    v0_copy = v0.clone()
    vector_to_model(v0, model)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    train_local_epoch(model, x, y)
    v1 = model_to_vector(model)
    assert torch.equal(v0, v0_copy)
    assert float((v1 - v0).norm().item()) > 0.0

    with pytest.raises(ValueError, match="does not match"):
        vector_to_model(v0[:-1], model)
    with pytest.raises(ValueError, match="does not match"):
        vector_to_model(torch.cat([v0, torch.zeros(1)]), model)
    # Failed length checks must not partially overwrite model weights.
    assert torch.equal(model_to_vector(model), v1)


def test_local_grad_clip_bounds_step_magnitude() -> None:
    torch.manual_seed(0)
    m_unclipped = build_model("cnn2x2_gn")
    m_clipped = build_model("cnn2x2_gn")
    m_clipped.load_state_dict(m_unclipped.state_dict())
    before = model_to_vector(m_unclipped)

    x = torch.randn(16, 3, 32, 32) * 50.0
    y = torch.randint(0, 10, (16,))
    train_local_epoch(m_unclipped, x, y, lr=0.05, momentum=0.0, weight_decay=0.0, batch_size=16, epochs=1, grad_clip_norm=0.0)
    train_local_epoch(m_clipped, x, y, lr=0.05, momentum=0.0, weight_decay=0.0, batch_size=16, epochs=1, grad_clip_norm=1.0)

    upd_unclipped = float(torch.linalg.vector_norm(model_to_vector(m_unclipped) - before).item())
    upd_clipped = float(torch.linalg.vector_norm(model_to_vector(m_clipped) - before).item())
    assert upd_clipped <= 0.05 * 1.0 + 1e-4
    assert upd_unclipped > upd_clipped * 5.0


def test_evaluate_accuracy_returns_valid_range() -> None:
    torch.manual_seed(0)
    model = build_model("cnn2x2")
    x = torch.randn(12, 3, 32, 32)
    y = torch.randint(0, 10, (12,))
    acc, loss = evaluate_accuracy(model, x, y, batch_size=5)
    assert 0.0 <= acc <= 1.0
    assert math.isfinite(loss) and loss > 0.0
