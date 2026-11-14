"""Section 3 integration: liveness gates a face-verification trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps
import torch

from attendance_app.service import uses_pad_context
from face_verification.verify import FaceVerifier


class AttendanceVerificationPipeline:
    """Run liveness first, then verify the same face crop against a reference.

    The result is a gate for the later attendance application. This module does
    not create attendance records, enroll people, or run the webcam interface.
    """

    def __init__(self, liveness_detector, identity_verifier: FaceVerifier, emotion_recognizer=None) -> None:
        self.liveness_detector = liveness_detector
        self.identity_verifier = identity_verifier
        self.emotion_recognizer = emotion_recognizer

    def verify(
        self, probe_image, reference_image, *, bbox=None, is_face_crop: bool = False
    ) -> dict:
        if uses_pad_context(self.liveness_detector):
            from anti_spoofing.data import crop_liveness_image

            if bbox is None or is_face_crop:
                raise ValueError("A context checkpoint requires the full probe image and its face box.")
            if isinstance(probe_image, (str, Path)):
                with Image.open(probe_image) as source:
                    probe_image = source.convert("RGB")
            face_crop = crop_liveness_image(probe_image, bbox=bbox)
            pad_crop = crop_liveness_image(probe_image, bbox=bbox, context=True)
        else:
            face_crop = self.liveness_detector.prepare_face(probe_image, bbox=bbox, is_face_crop=is_face_crop)
            pad_crop = face_crop
        liveness = self.liveness_detector.predict_crop(pad_crop)
        result = {"accepted": False, "status": None, "liveness": liveness, "identity": None}
        if self.emotion_recognizer is not None:
            result["emotion"] = None
        if self.emotion_recognizer is not None:
            try:
                result["emotion"] = self.emotion_recognizer.predict_crop(face_crop)
            except (OSError, ValueError, RuntimeError) as exc:
                result["emotion_error"] = str(exc)
        if liveness.get("is_live") is not True:
            result["status"] = ("liveness_threshold_missing" if liveness.get("is_live") is None
                                else "spoof_rejected")
            return result
        if isinstance(reference_image, (str, Path)):
            with Image.open(reference_image) as source:
                reference = ImageOps.exif_transpose(source).convert("RGB")
        else:
            reference = reference_image
        identity = self.identity_verifier.verify_images(face_crop, reference)
        result["identity"] = identity
        if identity.get("same_person") is True:
            result.update(accepted=True, status="live_identity_verified")
        elif identity.get("same_person") is None:
            result["status"] = "identity_threshold_missing"
        else:
            result["status"] = "identity_not_verified"
        return result


def main(argv: list[str] | None = None) -> int:
    from anti_spoofing.infer import LivenessDetector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liveness-checkpoint", type=Path, required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--emotion-checkpoint", type=Path,
                        help="Optional FER expression classifier; never changes attendance acceptance.")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Registered face-crop image.")
    parser.add_argument("--identity-metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--identity-threshold", type=float, required=True,
                        help="Raw metric threshold calibrated for the identity encoder.")
    parser.add_argument("--liveness-threshold", type=float,
                        help="Optional explicit override of the checkpoint's validation threshold.")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Probe face bounding box in original image pixels.")
    parser.add_argument("--face-crop", action="store_true", help="Probe is already an oriented face crop.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.num_threads < 1:
            raise ValueError("--num-threads must be positive.")
        if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
            torch.set_num_threads(args.num_threads)
        detector = LivenessDetector(args.liveness_checkpoint, args.liveness_threshold, args.device)
        verifier = FaceVerifier(args.identity_checkpoint, args.identity_metric,
                                args.identity_threshold, args.device)
        emotion = None
        if args.emotion_checkpoint is not None:
            from emotion_recognition.infer import EmotionRecognizer
            emotion = EmotionRecognizer(args.emotion_checkpoint, args.device)
        pipeline = AttendanceVerificationPipeline(detector, verifier, emotion)
        result = pipeline.verify(args.probe, args.reference, bbox=args.bbox, is_face_crop=args.face_crop)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Attendance verification error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
