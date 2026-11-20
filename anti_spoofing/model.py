"""Liveness models returning canonical two-class logits ``[spoof, live]``."""

from __future__ import annotations

import torch
from torch import nn

from face_verification.model import FaceEncoder


class LivenessCNN(nn.Module):
    """Compact two-class CNN initialized from scratch for liveness detection."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError("feature_dim must be a positive integer.")
        self.feature_dim = feature_dim
        self.backbone = FaceEncoder(embedding_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits for RGB batches normalized to [-1, 1]."""
        return self.classifier(self.backbone.forward_raw(images))


class MobileNetV3Liveness(nn.Module):
    """Partial fine-tuning MobileNetV3-Large returning canonical ``[spoof, live]`` logits."""

    def __init__(self, pretrained: bool = False, freeze_blocks: int = 4) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = mobilenet_v3_large(weights=weights)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
        if freeze_blocks > 0:
            for param in net.features[:freeze_blocks].parameters():
                param.requires_grad = False
        self.net = net

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def build_liveness_model(arch: str = "liveness_cnn", *, feature_dim: int = 128,
                         pretrained: bool = False, freeze_blocks: int = 4) -> nn.Module:
    if arch == "liveness_cnn":
        return LivenessCNN(feature_dim=feature_dim)
    if arch == "mobilenetv3_large":
        return MobileNetV3Liveness(pretrained=pretrained, freeze_blocks=freeze_blocks)
    raise ValueError(f"Unsupported liveness architecture: {arch!r}")
