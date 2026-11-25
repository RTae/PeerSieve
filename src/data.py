from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class NodeDataset:
    """Local train and validation tensors owned by a single peer."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor


@dataclass
class PartitionedDataset:
    """Per-peer shards, shared held-out test set, and per-node class counts."""

    nodes: list[NodeDataset]
    x_test: torch.Tensor
    y_test: torch.Tensor
    class_counts: list[list[int]]
    num_classes: int = 10


def _normalize_cifar_uint8(images_nhwc: np.ndarray) -> torch.Tensor:
    """Convert uint8 NHWC CIFAR images to normalized FP32 NCHW tensors.

    Args:
        images_nhwc: Uint8 NumPy array of shape (N, 32, 32, 3).

    Returns:
        Contiguous FP32 tensor of shape (N, 3, 32, 32) normalized by CIFAR-10 mean and std.
    """
    arr = images_nhwc.astype(np.float32) / 255.0
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(np.ascontiguousarray(arr))


def _load_cifar10_raw_batches(
    batches_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load CIFAR-10 python batches directly from an extracted directory.

    Args:
        batches_dir: Path to the extracted cifar-10-batches-py directory.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) normalized tensors.
    """
    train_images = []
    train_labels: list[int] = []
    for batch_idx in range(1, 6):
        with open(batches_dir / f"data_batch_{batch_idx}", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_images.append(data)
        train_labels.extend(batch[b"labels"])

    with open(batches_dir / "test_batch", "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
    test_images = test_batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = test_batch[b"labels"]

    x_train = _normalize_cifar_uint8(np.concatenate(train_images, axis=0))
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = _normalize_cifar_uint8(test_images)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    return x_train, y_train, x_test, y_test


def load_cifar10_tensors(
    data_dir: str | Path = "data",
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load full CIFAR-10 train (50k) and test (10k) sets as normalized FP32 tensors.

    Args:
        data_dir: Directory containing or receiving cifar-10-batches-py.
        download: Whether to download CIFAR-10 via torchvision if missing locally.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) tensors.
    """
    root = Path(data_dir)
    batches_dir = root / "cifar-10-batches-py"
    if batches_dir.is_dir():
        return _load_cifar10_raw_batches(batches_dir)

    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to download CIFAR-10 (or place extracted "
            f"cifar-10-batches-py under {root})."
        ) from exc

    train_ds = CIFAR10(root=str(root), train=True, download=download)
    test_ds = CIFAR10(root=str(root), train=False, download=download)
    x_train = _normalize_cifar_uint8(train_ds.data)
    y_train = torch.tensor(train_ds.targets, dtype=torch.long)
    x_test = _normalize_cifar_uint8(test_ds.data)
    y_test = torch.tensor(test_ds.targets, dtype=torch.long)
    return x_train, y_train, x_test, y_test


def make_synthetic_cifar_tensors(
    n_train: int = 10000,
    n_test: int = 1000,
    num_classes: int = 10,
    noise_std: float = 0.85,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a learnable synthetic 10-class 3x32x32 image dataset for offline tests.

    Args:
        n_train: Number of training images to generate.
        n_test: Number of test images to generate.
        num_classes: Number of distinct class prototypes.
        noise_std: Standard deviation of additive Gaussian pixel noise.
        seed: Random seed for reproducible generation.

    Returns:
        Tuple of (x_train, y_train, x_test, y_test) tensors.
    """
    rng = np.random.default_rng(seed)
    prototypes = np.zeros((num_classes, 3, 32, 32), dtype=np.float32)
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, 32), np.linspace(-1.0, 1.0, 32), indexing="ij")
    for c in range(num_classes):
        angle = 2.0 * np.pi * c / num_classes
        freq = 1.5 + 0.35 * (c % 4)
        wave = np.sin(freq * (np.cos(angle) * xx + np.sin(angle) * yy))
        cx = 0.45 * np.cos(angle)
        cy = 0.45 * np.sin(angle)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.18)
        prototypes[c, c % 3] = 0.9 * blob + 0.45 * wave
        prototypes[c, (c + 1) % 3] = -0.5 * blob + 0.35 * np.cos(freq * xx)
        prototypes[c, (c + 2) % 3] = 0.4 * np.sin(freq * yy)

    def _sample(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n synthetic images and balanced class labels with Gaussian noise.

        Args:
            n: Number of examples to sample.

        Returns:
            Tuple of (images, labels) tensors.
        """
        y = np.arange(n, dtype=np.int64) % num_classes
        rng.shuffle(y)
        x = prototypes[y] + rng.normal(0.0, noise_std, size=(n, 3, 32, 32)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

    x_train, y_train = _sample(n_train)
    x_test, y_test = _sample(n_test)
    return x_train, y_train, x_test, y_test


def _stratified_class_totals(n_total: int, num_classes: int) -> np.ndarray:
    """Split n_total evenly across classes so all totals sum to n_total.

    Args:
        n_total: Total number of samples to allocate.
        num_classes: Number of classes.

    Returns:
        1D int64 array of length num_classes with per-class sample counts.
    """
    totals = np.full(num_classes, n_total // num_classes, dtype=np.int64)
    totals[: n_total % num_classes] += 1
    return totals


def _sample_class_pools(
    y_np: np.ndarray,
    class_totals: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Randomly draw shuffled index pools of the requested size for each class.

    Args:
        y_np: 1D NumPy array of integer training labels.
        class_totals: Required number of samples for each class.
        rng: NumPy random generator.

    Returns:
        List of shuffled index arrays, one per class.
    """
    pools: list[np.ndarray] = []
    for c, need in enumerate(class_totals):
        idx_c = np.where(y_np == c)[0]
        if len(idx_c) < need:
            raise ValueError(f"Class {c} only has {len(idx_c)} examples, need {int(need)}.")
        chosen = rng.choice(idx_c, size=int(need), replace=False)
        rng.shuffle(chosen)
        pools.append(chosen)
    return pools


def _balanced_dirichlet_counts(
    class_totals: np.ndarray,
    n_nodes: int,
    samples_per_node: int,
    alpha_dir: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute an integer (num_classes, n_nodes) count matrix with exact row and column sums.

    Args:
        class_totals: Exact total samples available per class.
        n_nodes: Number of peer nodes.
        samples_per_node: Exact number of samples assigned to each node.
        alpha_dir: Dirichlet concentration parameter controlling non-IID skew.
        rng: NumPy random generator.

    Returns:
        Int64 array of shape (num_classes, n_nodes) matching exact class and node totals.
    """
    num_classes = len(class_totals)
    total_samples = int(class_totals.sum())
    if total_samples != n_nodes * samples_per_node:
        raise ValueError(
            f"class_totals sum ({total_samples}) != n_nodes * samples_per_node "
            f"({n_nodes * samples_per_node})"
        )

    weights = rng.gamma(shape=alpha_dir, scale=1.0, size=(num_classes, n_nodes))
    weights = np.maximum(weights, 1e-12)
    row_targets = class_totals.astype(np.float64)
    col_targets = np.full(n_nodes, float(samples_per_node), dtype=np.float64)

    for _ in range(200):
        weights *= (row_targets / weights.sum(axis=1))[:, None]
        weights *= (col_targets / weights.sum(axis=0))[None, :]

    counts = np.floor(weights).astype(np.int64)
    remainders = weights - counts
    row_def = class_totals - counts.sum(axis=1)
    col_def = np.full(n_nodes, samples_per_node, dtype=np.int64) - counts.sum(axis=0)

    order = np.argsort(-remainders.ravel())
    for flat_idx in order:
        c = int(flat_idx // n_nodes)
        i = int(flat_idx % n_nodes)
        if row_def[c] > 0 and col_def[i] > 0:
            counts[c, i] += 1
            row_def[c] -= 1
            col_def[i] -= 1

    while row_def.sum() > 0:
        c_need = int(np.argmax(row_def))
        i_need = int(np.argmax(col_def))
        counts[c_need, i_need] += 1
        row_def[c_need] -= 1
        col_def[i_need] -= 1

    assert np.array_equal(counts.sum(axis=1), class_totals)
    assert np.all(counts.sum(axis=0) == samples_per_node)
    return counts


def _assign_from_pools(
    class_pools: list[np.ndarray],
    counts: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Slice disjoint per-node index arrays according to a class-by-node count matrix.

    Args:
        class_pools: Disjoint index arrays per class.
        counts: Int64 matrix of shape (num_classes, n_nodes).
        rng: NumPy random generator used to shuffle each node's indices.

    Returns:
        List of shuffled 1D index arrays, one per peer.
    """
    num_classes, n_nodes = counts.shape
    offsets = np.zeros(num_classes, dtype=np.int64)
    per_node: list[np.ndarray] = []
    for i in range(n_nodes):
        parts = []
        for c in range(num_classes):
            k = int(counts[c, i])
            if k > 0:
                start = int(offsets[c])
                parts.append(class_pools[c][start : start + k])
                offsets[c] += k
        idx = np.concatenate(parts)
        rng.shuffle(idx)
        per_node.append(idx)
    return per_node


def partition_dirichlet_balanced(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    n_nodes: int = 20,
    samples_per_node: int = 500,
    alpha_dir: float = 0.5,
    val_frac: float = 0.1,
    val_split: str = "local",
    num_classes: int = 10,
    seed: int = 42,
) -> PartitionedDataset:
    """Split training data across peers with balanced Dirichlet non-IID skew and validation sets.

    Args:
        x_train: Full training image tensor.
        y_train: Full training label tensor.
        x_test: Held-out test image tensor.
        y_test: Held-out test label tensor.
        n_nodes: Number of decentralized peers.
        samples_per_node: Total samples (train plus val) assigned to each peer.
        alpha_dir: Dirichlet concentration parameter.
        val_frac: Fraction of each peer's samples held out for validation.
        val_split: Either 'local' (non-IID shard holdout) or 'iid' (class-balanced validation).
        num_classes: Number of label classes.
        seed: Random seed for reproducible partitioning.

    Returns:
        PartitionedDataset holding per-node train/val shards, test tensors, and class counts.
    """
    if val_split not in {"local", "iid"}:
        raise ValueError(f"Unknown val_split {val_split!r}; expected 'local' or 'iid'.")

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(samples_per_node * val_frac))) if val_frac > 0 else 0
    n_train_per_node = samples_per_node - n_val
    y_np = y_train.cpu().numpy()

    if val_split == "local":
        class_totals = _stratified_class_totals(n_nodes * samples_per_node, num_classes)
        pools = _sample_class_pools(y_np, class_totals, rng)
        counts = _balanced_dirichlet_counts(class_totals, n_nodes, samples_per_node, alpha_dir, rng)
        node_indices = _assign_from_pools(pools, counts, rng)
        train_indices = [idx[n_val:] for idx in node_indices]
        val_indices = [idx[:n_val] for idx in node_indices]
    else:
        val_totals = _stratified_class_totals(n_nodes * n_val, num_classes)
        tr_totals = _stratified_class_totals(n_nodes * n_train_per_node, num_classes)
        combined_pools = _sample_class_pools(y_np, val_totals + tr_totals, rng)
        val_pools = [pool[: int(v)] for pool, v in zip(combined_pools, val_totals)]
        tr_pools = [pool[int(v) :] for pool, v in zip(combined_pools, val_totals)]

        counts = _balanced_dirichlet_counts(tr_totals, n_nodes, n_train_per_node, alpha_dir, rng)
        train_indices = _assign_from_pools(tr_pools, counts, rng)

        val_per_node: list[list[int]] = [[] for _ in range(n_nodes)]
        for cursor, idx_val in enumerate(np.concatenate(val_pools)):
            val_per_node[cursor % n_nodes].append(int(idx_val))
        val_indices = []
        for lst in val_per_node:
            arr = np.array(lst, dtype=np.int64)
            rng.shuffle(arr)
            val_indices.append(arr)

    nodes: list[NodeDataset] = []
    node_class_counts: list[list[int]] = []
    for i in range(n_nodes):
        tr_idx = torch.from_numpy(train_indices[i])
        va_idx = torch.from_numpy(val_indices[i])
        nodes.append(
            NodeDataset(
                x_train=x_train[tr_idx].clone(),
                y_train=y_train[tr_idx].clone(),
                x_val=x_train[va_idx].clone(),
                y_val=y_train[va_idx].clone(),
            )
        )
        node_class_counts.append([int(counts[c, i]) for c in range(num_classes)])

    return PartitionedDataset(
        nodes=nodes,
        x_test=x_test.clone(),
        y_test=y_test.clone(),
        class_counts=node_class_counts,
        num_classes=num_classes,
    )


def apply_label_flip(y: torch.Tensor, mode: str = "pair_0_1", num_classes: int = 10) -> torch.Tensor:
    """Return a label-flipped copy of target labels for data-poisoning attacks.

    Args:
        y: 1D integer label tensor.
        mode: Either 'pair_0_1' (swap classes 0 and 1) or 'all_classes' (map c -> C-1-c).
        num_classes: Total number of classes.

    Returns:
        New integer label tensor with flipped labels.
    """
    flipped = y.clone()
    if mode == "pair_0_1":
        # Literal TrustGraph-DFL Section 5.1: "Label-flipping between classes 0 and 1".
        mask0 = y == 0
        mask1 = y == 1
        flipped[mask0] = 1
        flipped[mask1] = 0
        return flipped
    if mode == "all_classes":
        return (num_classes - 1) - flipped
    raise ValueError(f"Unknown label flip mode: {mode!r}")
