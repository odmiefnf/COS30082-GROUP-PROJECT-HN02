"""Compare old/new face capture on the same live frames; do not save images."""
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


class PreviousLocator:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, rgb):
        scale = min(1., 640 / rgb.shape[1])
        small = cv2.resize(rgb, (round(rgb.shape[1]*scale), round(rgb.shape[0]*scale))) if scale < 1 else rgb
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        boxes = self.cascade.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.1,
                                              minNeighbors=6, minSize=(60, 60))
        return [tuple(round(int(v)/scale) for v in box) for box in boxes]


class GateOnly:
    def __init__(self, liveness):
        self.liveness = liveness

    def reset_tracking(self):
        pass

    def analyze(self, crop):
        live = self.liveness.predict_crop(crop)
        return {"status": "live" if live["is_live"] is True else "spoof", "liveness": live}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 1 or args.seconds <= 0:
        parser.error("Positive frame count and duration required.")
    project = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    checkpoint = project / "results/liveness_lcc/best_model.pt"
    live = LivenessDetector(checkpoint, device="cpu")
    gate = GateOnly(live)
    processors = {"previous": FrameProcessor(PreviousLocator(), gate), "updated": FrameProcessor(FaceLocator(), gate)}
    counts = {name: Counter() for name in processors}
    timings = {name: [] for name in processors}
    camera = cv2.VideoCapture(args.camera)
    frame_count = 0
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            raise RuntimeError("Cannot open the camera for the comparison.")
        # Discard camera auto-exposure startup frames; no pixels leave RAM.
        for _ in range(8):
            camera.read()
        started = time.monotonic()
        while frame_count < args.frames and time.monotonic()-started < args.seconds:
            ok, bgr = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for name, processor in processors.items():
                before = time.perf_counter()
                result = processor.process(rgb)
                timings[name].append((time.perf_counter()-before)*1000)
                counts[name][result["status"]] += 1
            frame_count += 1
        elapsed = time.monotonic()-started
    finally:
        camera.release()
    report = {
        "scope": "One short convenience webcam trial, same frames for both detectors; not an accuracy or anti-spoof benchmark.",
        "camera_index": args.camera, "frame_count": frame_count, "duration_seconds": elapsed,
        "images_or_video_saved": False, "profiles_or_attendance_written": False,
        "liveness_checkpoint_sha256": sha256(checkpoint), "liveness_threshold": live.threshold,
        "model_or_threshold_changed": False,
        "results": {name: {"statuses": dict(counts[name]), "mean_processing_ms": float(np.mean(values)),
                            "p95_processing_ms": float(np.percentile(values, 95))}
                    for name, values in timings.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
