"""Models and parameter-vector utilities for Phase 1 CIFAR-10 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


class SimpleCIFARCNN(nn.Module):
    """Two-convolution, two-FC CNN for 32x32 CIFAR-10 images.

    Matches the TrustGraph-DFL description ("two convolutional layers followed
    by two fully connected layers") without BatchNorm so model-parameter
    averaging in DFL has no running-stat ambiguity.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LeNetCIFARCNN(nn.Module):
    """Lighter 2-conv, 2-FC LeNet-style CNN for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def build_model(name: str = "cnn2x2", num_classes: int = 10) -> nn.Module:
    if name == "cnn2x2":
        return SimpleCIFARCNN(num_classes=num_classes)
    if name == "lenet2x2":
        return LeNetCIFARCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name!r}")


@torch.no_grad()
def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Return a detached, contiguous FP32 copy of all model parameters."""
    return parameters_to_vector(model.parameters()).detach().clone().float()


@torch.no_grad()
def vector_to_model(vec: torch.Tensor, model: nn.Module) -> None:
    """Copy a 1D parameter vector into a model in-place without aliasing vec."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.copy_(vec[offset : offset + numel].view_as(param).to(device=param.device, dtype=param.dtype))
        offset += numel
    if offset != vec.numel():
        raise ValueError(f"Vector length {vec.numel()} does not match model parameter count {offset}.")


def compute_validation_gradient(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute exact mean loss gradient nabla F(theta; D_val) as a 1D FP32 tensor."""
    model.zero_grad(set_to_none=True)
    n_samples = int(x_val.shape[0])
    if n_samples == 0:
        return torch.zeros_like(model_to_vector(model))

    model.eval()
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        logits = model(x_val[start:end])
        loss = F.cross_entropy(logits, y_val[start:end], reduction="sum") / n_samples
        loss.backward()

    grads = [
        p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel(), device=p.device)
        for p in model.parameters()
    ]
    grad_vec = torch.cat(grads).clone()
    model.zero_grad(set_to_none=True)
    return grad_vec


def train_local_epoch(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 1,
    generator: torch.Generator | None = None,
) -> None:
    """Train model in-place using SGD on preloaded tensors."""
    n_samples = int(x_train.shape[0])
    if n_samples == 0:
        return

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_samples, generator=generator, device="cpu")
        if x_train.device.type != "cpu":
            perm = perm.to(x_train.device)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Evaluate classification accuracy in [0, 1] and mean cross-entropy loss."""
    model.eval()
    n_samples = int(x_test.shape[0])
    correct = 0
    total_loss = 0.0
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        logits = model(x_test[start:end])
        loss = F.cross_entropy(logits, y_test[start:end], reduction="sum")
        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += int((preds == y_test[start:end]).sum().item())
    return correct / max(n_samples, 1), total_loss / max(n_samples, 1)
