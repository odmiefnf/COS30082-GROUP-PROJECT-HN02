"""A compact two-class CNN initialized from scratch for liveness detection."""

from __future__ import annotations

import torch
from torch import nn

from face_verification.model import FaceEncoder


class LivenessCNN(nn.Module):
    """Return logits in the order ``[spoof, live]``.

    Only the Section 2 architecture is reused. No face-verification or pretrained
    weights are loaded; all features and the new binary head are trained afresh.
    """

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
