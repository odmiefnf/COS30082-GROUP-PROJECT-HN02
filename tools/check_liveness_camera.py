"""Compare two calibrated liveness models on shared webcam crops without saving images."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from anti_spoofing.infer import LivenessDetector
from attendance_app.camera import FaceLocator, FrameProcessor
from face_verification.image_cache import sha256


class CompareGate:
    def __init__(self, models):
        self.models = models
        self.scores = {name: [] for name in models}
        self.passed = Counter()

    def reset_tracking(self):
        pass

    def analyze(self, crop):
        for name, model in self.models.items():
            result = model.predict_crop(crop)
            self.scores[name].append(result["live_score"])
            self.passed[name] += int(result["is_live"] is True)
        return {"status": "scored"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--condition", choices=("glasses", "no-glasses", "screen-replay", "print", "unspecified"), required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.seconds <= 0:
        parser.error("Positive frame count and duration required.")
    torch.set_num_threads(4)
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    models = {name: LivenessDetector(path, device="cpu") for name, path in paths.items()}
    gate = CompareGate(models)
    processor = FrameProcessor(FaceLocator(), gate)
    camera = cv2.VideoCapture(args.camera)
    counts, frames = Counter(), 0
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            raise RuntimeError("Cannot open the webcam for the liveness comparison.")
        for _ in range(8):
            camera.read()
        started = time.monotonic()
        while frames < args.frames and time.monotonic()-started < args.seconds:
            ok, bgr = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            result = processor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            counts[result["status"]] += 1
            frames += 1
        elapsed = time.monotonic()-started
    finally:
        camera.release()
    report = {
        "scope": "One short, self-described webcam condition; no independent identity/attack truth. Correlated frames are not an accuracy benchmark.",
        "condition_supplied_by_operator": args.condition, "frame_count": frames,
        "duration_seconds": elapsed, "frame_statuses": dict(counts),
        "same_original_rgb_crop_for_both_models": True,
        "images_or_video_saved": False, "profiles_or_attendance_written": False,
        "thresholds_retuned_on_camera": False,
        "models": {},
    }
    for name, model in models.items():
        scores = gate.scores[name]
        report["models"][name] = {
            "checkpoint": str(paths[name].resolve()), "sha256": sha256(paths[name]),
            "threshold": model.threshold, "scored_crops": len(scores),
            "passed": gate.passed[name], "not_passed": len(scores)-gate.passed[name],
            "mean_live_score": float(np.mean(scores)) if scores else None,
            "median_live_score": float(np.median(scores)) if scores else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
