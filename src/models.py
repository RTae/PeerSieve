"""Models, augmentation, and parameter-vector utilities for Phase 1 CIFAR-10 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


class SimpleCIFARCNN(nn.Module):
    """Two-convolution, two-FC CNN for 32x32 CIFAR-10 images.

    Matches the literal TrustGraph-DFL description ("two convolutional layers
    followed by two fully connected layers") without BatchNorm.
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


class GroupNormCIFARCNN(nn.Module):
    """Stabilized 2-conv, 2-FC CNN with GroupNorm and Dropout for CIFAR-10.

    GroupNorm has no running-stat buffers, avoiding BatchNorm drift under
    decentralized parameter averaging while reducing local overfitting.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.drop(torch.flatten(x, 1))
        x = self.drop(F.relu(self.fc1(x)))
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


class PretrainedResNet18CIFAR(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted for 32x32 CIFAR-10 in DFL.

    Upsamples batches on-the-fly to 128x128 so pretrained stem filters remain
    valid, and keeps BatchNorm running stats frozen in eval mode so non-IID
    and Byzantine peers never corrupt shared running buffers.
    """

    def __init__(self, num_classes: int = 10, image_size: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
        self.image_size = image_size
        self._freeze_bn_stats()

    def _freeze_bn_stats(self) -> None:
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

    def train(self, mode: bool = True) -> PretrainedResNet18CIFAR:
        super().train(mode)
        self._freeze_bn_stats()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(name: str = "cnn2x2", num_classes: int = 10) -> nn.Module:
    if name == "cnn2x2":
        return SimpleCIFARCNN(num_classes=num_classes)
    if name == "cnn2x2_gn":
        return GroupNormCIFARCNN(num_classes=num_classes)
    if name == "lenet2x2":
        return LeNetCIFARCNN(num_classes=num_classes)
    if name == "resnet18_pretrained":
        return PretrainedResNet18CIFAR(num_classes=num_classes, pretrained=True)
    if name == "resnet18_scratch":
        return PretrainedResNet18CIFAR(num_classes=num_classes, pretrained=False)
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


def augment_cifar_batch(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply vectorized RandomHorizontalFlip + RandomCrop(32, padding=4) to a (B, C, 32, 32) batch."""
    bsz, _, h, w = x.shape
    out = x.clone()
    flip_mask = torch.rand(bsz, generator=generator, device="cpu") < 0.5
    if bool(flip_mask.any()):
        flip_idx = flip_mask.to(x.device)
        out[flip_idx] = torch.flip(out[flip_idx], dims=[3])

    pad = 4
    padded = F.pad(out, (pad, pad, pad, pad), mode="reflect")
    dy = torch.randint(0, 2 * pad + 1, (bsz,), generator=generator, device="cpu").tolist()
    dx = torch.randint(0, 2 * pad + 1, (bsz,), generator=generator, device="cpu").tolist()
    cropped = torch.stack([padded[b, :, dy[b] : dy[b] + h, dx[b] : dx[b] + w] for b in range(bsz)], dim=0)
    return cropped


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
    augment: bool = False,
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
            xb = x_train[idx]
            if augment:
                xb = augment_cifar_batch(xb, generator=generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
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
