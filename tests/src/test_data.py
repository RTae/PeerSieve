from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import (
    _stratified_class_totals,
    apply_label_flip,
    make_synthetic_cifar_tensors,
    partition_dirichlet_balanced,
)


def test_stratified_class_totals_exact_sum_and_balance() -> None:
    totals = _stratified_class_totals(n_total=23, num_classes=10)
    assert int(totals.sum()) == 23
    assert int(totals.max() - totals.min()) <= 1


def test_balanced_dirichlet_partition_exact_counts_and_non_iid() -> None:
    x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(n_train=10000, n_test=200, seed=1)
    part = partition_dirichlet_balanced(
        x_tr, y_tr, x_te, y_te, n_nodes=20, samples_per_node=500, alpha_dir=0.5, val_frac=0.1, seed=42
    )
    assert len(part.nodes) == 20
    entropies = []
    for i, nd in enumerate(part.nodes):
        assert nd.x_train.shape == (450, 3, 32, 32)
        assert nd.x_val.shape == (50, 3, 32, 32)
        assert sum(part.class_counts[i]) == 500
        probs = np.array(part.class_counts[i], dtype=np.float64) / 500.0
        probs = probs[probs > 0]
        entropies.append(-float(np.sum(probs * np.log(probs))))

    # Uniform 10-class entropy is ln(10) ~= 2.3026; Dirichlet(0.5) must be lower.
    assert np.mean(entropies) < 1.95
    total_per_class = np.sum(np.array(part.class_counts), axis=0)
    assert np.all(total_per_class == 1000)


def test_iid_val_split_is_balanced_per_node() -> None:
    x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(n_train=10000, n_test=200, seed=2)
    part = partition_dirichlet_balanced(
        x_tr,
        y_tr,
        x_te,
        y_te,
        n_nodes=20,
        samples_per_node=500,
        alpha_dir=0.5,
        val_frac=0.1,
        val_split="iid",
        seed=42,
    )
    for nd in part.nodes:
        assert nd.x_train.shape == (450, 3, 32, 32)
        assert nd.x_val.shape == (50, 3, 32, 32)
        counts = torch.bincount(nd.y_val, minlength=10)
        assert torch.all(counts == 5)


def test_partition_invalid_val_split_raises() -> None:
    x_tr, y_tr, x_te, y_te = make_synthetic_cifar_tensors(n_train=400, n_test=50, seed=3)
    with pytest.raises(ValueError, match="Unknown val_split"):
        partition_dirichlet_balanced(x_tr, y_tr, x_te, y_te, n_nodes=4, samples_per_node=40, val_split="random")


def test_label_flip_modes_and_invalid_mode() -> None:
    y = torch.tensor([0, 1, 2, 3, 0, 1, 9])
    assert torch.equal(apply_label_flip(y, mode="pair_0_1"), torch.tensor([1, 0, 2, 3, 1, 0, 9]))
    assert torch.equal(apply_label_flip(y, mode="all_classes", num_classes=10), torch.tensor([9, 8, 7, 6, 9, 8, 0]))
    with pytest.raises(ValueError, match="Unknown label flip mode"):
        apply_label_flip(y, mode="rotate")
