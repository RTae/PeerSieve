from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector


class SimpleCIFARCNN(nn.Module):
    """Two-convolution CNN with two linear layers for 32x32 CIFAR-10 images without BatchNorm."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize convolutional and linear layers for the 2-conv CIFAR CNN.

        Args:
            num_classes: Number of output classes.

        Returns:
            None.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of 32x32 RGB images.

        Args:
            x: Input image batch tensor of shape (B, 3, 32, 32).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class GroupNormCIFARCNN(nn.Module):
    """Two-convolution CNN with GroupNorm, Dropout, and two linear layers for CIFAR-10."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        """Initialize convolutional, GroupNorm, dropout, and linear layers.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability applied before linear layers.

        Returns:
            None.
        """
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
        """Compute class logits for a batch of 32x32 RGB images.

        Args:
            x: Input image batch tensor of shape (B, 3, 32, 32).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.drop(torch.flatten(x, 1))
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


class PretrainedResNet18CIFAR(nn.Module):
    """ImageNet-pretrained ResNet-18 adapted for 32x32 CIFAR-10 with frozen BatchNorm stats."""

    def __init__(self, num_classes: int = 10, image_size: int = 128, pretrained: bool = True) -> None:
        """Build a ResNet-18 backbone with a num_classes head and frozen BatchNorm stats.

        Args:
            num_classes: Number of output classes.
            image_size: Target spatial resolution for bilinear upsampling.
            pretrained: Whether to load default ImageNet weights.

        Returns:
            None.
        """
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone
        self.image_size = image_size
        self._freeze_bn_stats()

    def _freeze_bn_stats(self) -> None:
        """Set all BatchNorm2d layers to eval mode to freeze running statistics.

        Returns:
            None.
        """
        for mod in self.backbone.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.eval()

    def train(self, mode: bool = True) -> PretrainedResNet18CIFAR:
        """Toggle training mode while keeping BatchNorm running stats frozen.

        Args:
            mode: True for training mode, False for evaluation mode.

        Returns:
            This module instance.
        """
        super().train(mode)
        self._freeze_bn_stats()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize inputs to image_size if needed and return ResNet-18 class logits.

        Args:
            x: Input image batch tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return self.backbone(x)


def build_model(name: str = "cnn2x2", num_classes: int = 10) -> nn.Module:
    """Instantiate one of the supported CIFAR-10 model architectures by name.

    Args:
        name: Architecture key ('cnn2x2', 'cnn2x2_gn', 'resnet18_pretrained', or 'resnet18_scratch').
        num_classes: Number of output classes.

    Returns:
        Initialized PyTorch nn.Module on CPU.
    """
    if name == "cnn2x2":
        return SimpleCIFARCNN(num_classes=num_classes)
    if name == "cnn2x2_gn":
        return GroupNormCIFARCNN(num_classes=num_classes)
    if name == "resnet18_pretrained":
        return PretrainedResNet18CIFAR(num_classes=num_classes, pretrained=True)
    if name == "resnet18_scratch":
        return PretrainedResNet18CIFAR(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model name: {name!r}")


@torch.no_grad()
def model_to_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a detached, contiguous 1D FP32 copy.

    Args:
        model: PyTorch module to flatten.

    Returns:
        1D FP32 tensor copy of all model parameters.
    """
    return parameters_to_vector(model.parameters()).detach().clone().float()


@torch.no_grad()
def vector_to_model(vec: torch.Tensor, model: nn.Module) -> None:
    """Copy a 1D parameter vector into a model in-place without aliasing vec.

    Args:
        vec: 1D tensor whose length matches the model's parameter count.
        model: Target PyTorch module updated in-place.

    Returns:
        None.
    """
    total_numel = sum(param.numel() for param in model.parameters())
    if total_numel != vec.numel():
        raise ValueError(f"Vector length {vec.numel()} does not match model parameter count {total_numel}.")
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.copy_(vec[offset : offset + numel].view_as(param).to(device=param.device, dtype=param.dtype))
        offset += numel


def augment_cifar_batch(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply RandomHorizontalFlip and RandomCrop(32, padding=4) to a CIFAR image batch.

    Args:
        x: Input image batch tensor of shape (B, C, 32, 32).
        generator: Optional CPU random generator for deterministic augmentation.

    Returns:
        Augmented tensor of the same shape and device as x.
    """
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
    weight_decay: float = 0.0,
) -> torch.Tensor:
    """Compute the exact mean validation loss gradient nabla F(theta; D_val) as a 1D FP32 tensor.

    Args:
        model: PyTorch model evaluated in eval mode.
        x_val: Validation image tensor.
        y_val: Validation label tensor.
        batch_size: Mini-batch size used to accumulate exact mean gradients.
        weight_decay: Optional L2 penalty coefficient added as weight_decay * theta.

    Returns:
        1D FP32 gradient vector matching model_to_vector(model).shape.
    """
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

    grads = []
    for p in model.parameters():
        g = p.grad.detach().reshape(-1).float() if p.grad is not None else torch.zeros(p.numel(), device=p.device)
        if weight_decay > 0.0:
            g = g + weight_decay * p.detach().reshape(-1).float()
        grads.append(g)
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
    grad_clip_norm: float = 0.0,
    generator: torch.Generator | None = None,
) -> None:
    """Train model in-place using SGD on preloaded local tensors.

    Args:
        model: PyTorch model updated in-place.
        x_train: Local training images.
        y_train: Local training labels.
        lr: SGD learning rate.
        momentum: SGD momentum factor.
        weight_decay: L2 weight decay factor.
        batch_size: Mini-batch size.
        epochs: Number of local passes over x_train.
        augment: Whether to apply random CIFAR flip and crop augmentation.
        grad_clip_norm: Max gradient L2 norm (disabled when <= 0).
        generator: Optional CPU random generator for shuffling and augmentation.

    Returns:
        None.
    """
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
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 512,
) -> tuple[float, float]:
    """Evaluate classification accuracy and mean cross-entropy loss on a test tensor set.

    Args:
        model: PyTorch model to evaluate in eval mode.
        x_test: Test image tensor.
        y_test: Test label tensor.
        batch_size: Evaluation mini-batch size.

    Returns:
        Tuple of (accuracy in [0, 1], mean cross-entropy loss).
    """
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
