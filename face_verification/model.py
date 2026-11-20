"""Identity encoders: a compact CNN from scratch, or a fine-tuned ImageNet ResNet-18."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Sequential):
    """Reduce resolution while learning local face features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class FaceEncoder(nn.Module):
    """Encode RGB face batches as L2-normalized verification embeddings."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer.")
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            _ConvBlock(3, 16),
            _ConvBlock(16, 32),
            _ConvBlock(32, 64),
            _ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(128 * 2 * 2, embedding_dim)

    def forward_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized features for the identity classification head."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected an image batch with shape (N, 3, H, W).")
        return self.projection(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one unit-length embedding per input face."""
        return F.normalize(self.forward_raw(images), p=2, dim=1, eps=1e-12)


class ResNet18FaceEncoder(nn.Module):
    """ImageNet ResNet-18 adapted to small face crops and fine-tuned for embeddings.

    The max-pool is removed so a 64 px face still leaves a 4x4 map after layer4.
    The stem (conv1/bn1) parameters stay frozen; every residual stage and the new
    projection are trained, so the network is fine-tuned rather than used as-is.
    """

    FROZEN = ("conv1", "bn1")

    def __init__(self, embedding_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        if not isinstance(embedding_dim, int) or embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer.")
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        for name in self.FROZEN:
            for param in getattr(net, name).parameters():
                param.requires_grad = False
        self.embedding_dim = embedding_dim
        self.features = net
        self.projection = nn.Linear(512, embedding_dim)
        # Shared preprocessing yields [-1, 1]; ImageNet weights expect mean/std inputs.
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)

    def forward_raw(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected an image batch with shape (N, 3, H, W).")
        return self.projection(self.features(((images + 1) / 2 - self.mean) / self.std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_raw(images), p=2, dim=1, eps=1e-12)


BACKBONES = ("compact", "resnet18")


def build_encoder(backbone: str = "compact", embedding_dim: int = 128, pretrained: bool = False) -> nn.Module:
    if backbone == "compact":
        return FaceEncoder(embedding_dim=embedding_dim)
    if backbone == "resnet18":
        return ResNet18FaceEncoder(embedding_dim=embedding_dim, pretrained=pretrained)
    raise ValueError(f"Unsupported identity backbone: {backbone!r}")


class FaceClassifier(nn.Module):
    """Train an encoder through identity classification using raw logits."""

    def __init__(self, num_classes: int, embedding_dim: int = 128,
                 backbone: str = "compact", pretrained: bool = False) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        self.encoder = build_encoder(backbone, embedding_dim, pretrained)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits suitable for torch.nn.CrossEntropyLoss."""
        return self.classifier(self.encoder.forward_raw(images))
