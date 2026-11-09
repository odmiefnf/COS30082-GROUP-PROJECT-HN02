"""Compact CNNs trained from scratch for identity learning and verification."""

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


class FaceClassifier(nn.Module):
    """Train an encoder through identity classification using raw logits."""

    def __init__(self, num_classes: int, embedding_dim: int = 128) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        self.encoder = FaceEncoder(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits suitable for torch.nn.CrossEntropyLoss."""
        return self.classifier(self.encoder.forward_raw(images))
