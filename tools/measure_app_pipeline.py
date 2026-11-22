"""Measure detector robustness (Haar vs YuNet) and per-stage app latency; write one JSON report.

Detector check: genuine LCC-FASD evaluation frames, unmodified and rotated, counting frames
where exactly one face is found. Latency: median over repeated calls on one 960x720 frame.
These are engineering measurements, not accuracy or end-to-end webcam FPS claims.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import statistics
import time

import cv2
import numpy as np
from PIL import Image
import torch

from anti_spoofing.data import crop_liveness_image
from anti_spoofing.infer import LivenessDetector
from attendance_app.camera import FaceLocator
from attendance_app.gui import DEFAULT_LIVENESS_CHECKPOINT, PROJECT
from attendance_app.recognition import FaceRecognizer
from emotion_recognition.infer import EmotionRecognizer


def rotate(rgb, degrees):
    h, w = rgb.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1)
    return cv2.warpAffine(rgb, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def median_ms(call, repeats=30, warmup=5):
    for _ in range(warmup):
        call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(times), 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emotion-checkpoint", type=Path, default=PROJECT / "results/emotion_fer_mnv3/best_model.pt")
    args = parser.parse_args(argv)
    files = sorted(p for p in (PROJECT / "data/LCC_FASD/LCC_FASD_evaluation/real").glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    random.Random(1).shuffle(files)
    files = files[:args.frames]
    detectors = {"haar": FaceLocator("haar"), "yunet": FaceLocator("yunet")}
    robustness = {}
    for name, degrees in (("original", 0), ("rotated_20", 20), ("rotated_35", 35)):
        found = {key: 0 for key in detectors}
        for path in files:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            rgb = rotate(rgb, degrees) if degrees else rgb
            for key, locator in detectors.items():
                found[key] += len(locator.detect(rgb)) == 1
        robustness[name] = {key: f"{value}/{len(files)}" for key, value in found.items()}

    device = "auto"
    frame = cv2.resize(np.asarray(Image.open(files[0]).convert("RGB")), (960, 720))
    locator = detectors["yunet"]
    box = locator.detect(frame)[0]
    crop = crop_liveness_image(frame, bbox=box)
    wide = crop_liveness_image(frame, box, crop_scale=1.4)
    recognizer = FaceRecognizer(PROJECT / "results/identity_app_config.json", device)
    liveness = LivenessDetector(DEFAULT_LIVENESS_CHECKPOINT, device=device)
    expression = EmotionRecognizer(args.emotion_checkpoint, device)
    latency = {"yunet_detect_960x720_cpu": median_ms(lambda: locator.detect(frame)),
               "haar_detect_960x720_cpu": median_ms(lambda: detectors["haar"].detect(frame)),
               "liveness": median_ms(lambda: liveness.predict_crop(wide)),
               "identity_embed": median_ms(lambda: recognizer.embed(crop)),
               "expression": median_ms(lambda: expression.predict_crop(crop))}
    report = {
        "measured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Engineering check. Detector counts use genuine LCC-FASD evaluation frames (one face each); "
                 "latency is a median of single calls, excluding capture, drawing and streaming.",
        "detector_one_face_counts": robustness,
        "latency_ms": latency,
        "model_device": str(recognizer.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "opencv": cv2.__version__, "python": platform.python_version(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
