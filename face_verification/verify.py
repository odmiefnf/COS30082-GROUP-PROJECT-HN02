"""Score one face-image trial with a trained encoder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from PIL import Image
import numpy as np

from .data import load_face_image, preprocess_face_image
from .evaluate import load_encoder, score_embeddings


class FaceVerifier:
    """Reusable pair scorer; a decision is produced only with an explicit threshold."""

    def __init__(
        self,
        checkpoint: str | Path,
        metric: str = "cosine",
        threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be cosine or euclidean.")
        if threshold is not None:
            if not math.isfinite(threshold):
                raise ValueError("threshold must be finite.")
            if metric == "cosine" and not -1 <= threshold <= 1:
                raise ValueError("A cosine threshold must lie in [-1, 1].")
            if metric == "euclidean" and not 0 <= threshold <= 2:
                raise ValueError("A unit-embedding Euclidean threshold must lie in [0, 2].")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use device=cpu or auto.")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.encoder, self.config = load_encoder(self.checkpoint, device=self.device)
        self.metric = metric
        self.threshold = threshold

    def verify(self, image1: str | Path, image2: str | Path) -> dict:
        """Return both metrics and, if configured, a thresholded same-person decision."""
        left, right = (Path(path).expanduser().resolve() for path in (image1, image2))
        batch = torch.stack([
            load_face_image(left, self.config["image_size"]),
            load_face_image(right, self.config["image_size"]),
        ]).to(self.device)
        return self._score_batch(batch, str(left), str(right))

    def verify_images(self, image1: Image.Image | np.ndarray, image2: Image.Image | np.ndarray) -> dict:
        """Verify oriented RGB face crops shared with the liveness module."""
        batch = torch.stack([
            preprocess_face_image(image1, self.config["image_size"]),
            preprocess_face_image(image2, self.config["image_size"]),
        ]).to(self.device)
        return self._score_batch(batch, "<in-memory face crop>", "<in-memory face crop>")

    def _score_batch(self, batch: torch.Tensor, image1: str, image2: str) -> dict:
        """Keep path and camera-crop inference on exactly the same scoring code."""
        with torch.inference_mode():
            embeddings = self.encoder(batch).cpu().numpy()
        cosine, distance = score_embeddings(embeddings[0], embeddings[1])
        value = cosine if self.metric == "cosine" else distance
        score = cosine if self.metric == "cosine" else -distance
        accepted = None
        if self.threshold is not None:
            accepted = value >= self.threshold if self.metric == "cosine" else value <= self.threshold
        return {
            "image1": image1, "image2": image2,
            "checkpoint": str(self.checkpoint), "method": self.config["method"],
            "cosine_similarity": float(cosine), "euclidean_distance": float(distance),
            "decision_metric": self.metric, "higher_is_same_score": float(score),
            "threshold": self.threshold, "same_person": bool(accepted) if accepted is not None else None,
            "accept_rule": "cosine >= threshold" if self.metric == "cosine" else "distance <= threshold",
            "note": "A threshold must be calibrated for this encoder and data; scores alone are not probabilities.",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image1", type=Path, required=True)
    parser.add_argument("--image2", type=Path, required=True)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--threshold", type=float, help="Raw metric threshold calibrated on validation pairs.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.num_threads < 1:
            raise ValueError("--num-threads must be positive.")
        if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
            torch.set_num_threads(args.num_threads)
        verifier = FaceVerifier(args.checkpoint, args.metric, args.threshold, args.device)
        print(json.dumps(verifier.verify(args.image1, args.image2), indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Verification error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
