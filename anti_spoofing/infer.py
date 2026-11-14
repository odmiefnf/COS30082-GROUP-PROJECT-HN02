"""Score liveness on a stored image or an RGB face crop."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from .data import crop_liveness_image, preprocess_liveness_image, read_celeba_bbox
from .evaluate import load_liveness_model


class LivenessDetector:
    """Use trained logits and a validation-calibrated threshold to gate identity checks."""

    def __init__(
        self, checkpoint: str | Path, threshold: float | None = None, device: str = "cpu"
    ) -> None:
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Use device=cpu or auto.")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.model, self.config = load_liveness_model(self.checkpoint, device=self.device)
        calibrated = (self.config.get("threshold_source") == "validation"
                      and self.config.get("threshold_calibrated") is True)
        self.threshold = (self.config.get("threshold") if calibrated else None) if threshold is None else threshold
        if self.threshold is not None:
            if isinstance(self.threshold, bool):
                raise ValueError("The live-score threshold must be a number, not a boolean.")
            self.threshold = float(self.threshold)
            if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
                raise ValueError("The live-score threshold must be finite and in [0, 1].")
        self.threshold_source = (
            self.config.get("threshold_source") if calibrated else "uncalibrated"
        ) if threshold is None else "explicit_override"

    def prepare_face(self, image, *, bbox=None, is_face_crop: bool = False) -> Image.Image:
        """Return an oriented RGB crop that can be shared with identity verification.

        CelebA-Spoof sidecars refer to stored pixel orientation, so this path
        intentionally does not apply EXIF rotations. For camera inputs, callers
        pass pixel xywh or explicitly state that the frame is already a crop.
        """
        if bbox is not None and is_face_crop:
            raise ValueError("Use a bounding box or an already-cropped face, not both.")
        if isinstance(image, (str, Path)):
            path = Path(image).expanduser().resolve()
            with Image.open(path) as source:
                rgb = source.convert("RGB")
            if bbox is None and not is_face_crop and self.config["crop_mode"] == "bbox":
                sidecar = path.with_name(path.stem + "_BB.txt")
                bbox = read_celeba_bbox(sidecar, image_width=rgb.width, image_height=rgb.height)
        else:
            if bbox is None and not is_face_crop and self.config["crop_mode"] == "bbox":
                raise ValueError("A camera/image-array input needs a pixel bbox or is_face_crop=True.")
            rgb = image
        return crop_liveness_image(rgb, bbox=bbox, context=self.config.get("context_crop", False) and not is_face_crop)

    def predict_crop(self, face_crop: Image.Image | np.ndarray) -> dict:
        """Predict an explicitly prepared face crop; score is softmax P(live)."""
        tensor = preprocess_liveness_image(face_crop, image_size=self.config["image_size"])
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
            if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                raise ValueError("Liveness model returned invalid class logits.")
            live_score = float(torch.softmax(logits.to(torch.float64), dim=1)[0, 1].cpu())
        is_live = None if self.threshold is None else live_score >= self.threshold
        return {
            "live_score": live_score, "spoof_score": 1.0 - live_score,
            "is_live": bool(is_live) if is_live is not None else None,
            "status": "uncalibrated" if is_live is None else "live" if is_live else "spoof",
            "threshold": self.threshold, "threshold_source": self.threshold_source,
            "accept_rule": "P(live) >= threshold", "live_label": 1, "spoof_label": 0,
            "checkpoint": str(self.checkpoint),
            "checkpoint_smoke_test": bool(self.config.get("training", {}).get("smoke_test", False)),
        }

    def predict(self, image, *, bbox=None, is_face_crop: bool = False) -> dict:
        return self.predict_crop(self.prepare_face(image, bbox=bbox, is_face_crop=is_face_crop))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, help="Explicit override of the validation threshold.")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--face-crop", action="store_true", help="Image is already an oriented face crop.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.num_threads < 1:
            raise ValueError("--num-threads must be positive.")
        if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
            torch.set_num_threads(args.num_threads)
        detector = LivenessDetector(args.checkpoint, args.threshold, args.device)
        result = detector.predict(args.image, bbox=args.bbox, is_face_crop=args.face_crop)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Liveness error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
