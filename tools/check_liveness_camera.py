"""Compare two calibrated liveness models on shared webcam crops without saving images."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from anti_spoofing.infer import LivenessDetector
from attendance_app.camera import FaceLocator, FrameProcessor
from attendance_app.evaluation import PRESENTATIONS, validate_presentation
from face_verification.image_cache import sha256


class CompareGate:
    def __init__(self, models):
        self.models = models
        self.context = {name: getattr(model, "config", {}).get("context_crop", False) is True
                        for name, model in models.items()}
        self.scores = {name: [] for name in models}
        self.passed = Counter()

    def reset_tracking(self, reason=None):
        pass

    def analyze(self, crop, *, context_crop=None):
        for name, model in self.models.items():
            if self.context[name] and context_crop is None:
                raise ValueError("Context model requires a 2x crop from the camera frame.")
            result = model.predict_crop(context_crop if self.context[name] else crop)
            self.scores[name].append(result["live_score"])
            self.passed[name] += int(result["is_live"] is True)
        return {"status": "scored"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--condition", choices=("glasses", "no-glasses", "screen-replay", "print", "unspecified"), required=True)
    parser.add_argument("--presentation-type", choices=PRESENTATIONS,
                        help="Required for attacks: phone-still-photo, recorded-video or printed-photo.")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="Show framing for 10 seconds before the measured trial.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cancel-file", type=Path, help="Stop cooperatively when this marker is created by the model-test dialog.")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= 120
            or not 1 <= args.frames <= 5000 or not 0 <= args.camera <= 99):
        parser.error("Use camera 0–99, duration in (0,120] and 1–5000 measured frames.")
    try:
        validate_presentation(args.condition, args.presentation_type)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("Choose a new .json output file; existing evidence is never overwritten.")
    if args.cancel_file is not None and args.cancel_file.exists():
        print("Comparison cancelled before loading models or opening the camera.", flush=True)
        return
    torch.set_num_threads(4)
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    models = {name: LivenessDetector(path, device="cpu") for name, path in paths.items()}
    for name, model in models.items():
        if model.threshold is None or model.threshold_source != "validation":
            raise ValueError(f"{name} requires a validation-calibrated liveness threshold.")
        training = model.config.get("training", {})
        if training.get("smoke_test") is not False or training.get("initialization") != "random_from_scratch":
            raise ValueError(f"{name} requires completed from-scratch training.")
    gate = CompareGate(models)
    locator = FaceLocator()
    processor = (FrameProcessor(locator, gate, pad_context=True) if any(gate.context.values())
                 else FrameProcessor(locator, gate))
    if args.cancel_file is not None and args.cancel_file.exists():
        print("Comparison cancelled before opening the camera.", flush=True)
        return
    started_utc = datetime.now(timezone.utc).isoformat()
    camera = cv2.VideoCapture(args.camera)
    counts, frames = Counter(), 0
    window, window_open = "PAD model comparison", False
    setup_seconds = 10. if args.preview else 0.
    setup_duration, cancelled_during = 0., None
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not camera.isOpened():
            raise RuntimeError("Cannot open the webcam for the liveness comparison.")
        for _ in range(8):
            camera.read()
        if args.preview:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_open = True
            cv2.resizeWindow(window, 960, 720)
            setup_started = time.monotonic()
            while time.monotonic()-setup_started < setup_seconds:
                if args.cancel_file is not None and args.cancel_file.exists():
                    cancelled_during = "setup"
                    break
                ok, bgr = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames during framing.")
                display = bgr.copy()
                remaining = max(0., setup_seconds-(time.monotonic()-setup_started))
                cv2.putText(display, f"Position your face - test starts in {remaining:.0f} s", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                cv2.putText(display, "Unscored setup | Q / Esc to cancel", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    cancelled_during = "setup"
                    break
            setup_duration = time.monotonic()-setup_started
        started = time.monotonic()
        while cancelled_during is None and frames < args.frames and time.monotonic()-started < args.seconds:
            if args.cancel_file is not None and args.cancel_file.exists():
                cancelled_during = "measurement"
                break
            ok, bgr = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames.")
            result = processor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            counts[result["status"]] += 1
            frames += 1
            if args.preview:
                display = bgr.copy()
                for x, y, width, height in result.get("boxes", []):
                    cv2.rectangle(display, (x, y), (x+width, y+height), (0, 255, 255), 2)
                remaining = max(0., args.seconds-(time.monotonic()-started))
                cv2.putText(display, result["status"].replace("_", " "), (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                cv2.putText(display, f"Scored: {len(gate.scores['baseline'])} | {remaining:.1f} s left | Q / Esc to stop",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    cancelled_during = "measurement"
        elapsed = 0. if cancelled_during == "setup" else time.monotonic()-started
    finally:
        camera.release()
        if window_open:
            cv2.destroyWindow(window)
    report = {
        "scope": "One short, self-described webcam condition; no independent identity/attack truth. Correlated frames are not an accuracy benchmark.",
        "condition_supplied_by_operator": args.condition, "frame_count": frames,
        "presentation_type_supplied_by_operator": args.presentation_type,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed, "frame_statuses": dict(counts),
        "preview_enabled": args.preview, "setup_seconds": setup_seconds,
        "setup_duration_seconds": setup_duration, "setup_frames_scored": False,
        "operator_cancelled": cancelled_during is not None, "cancelled_during": cancelled_during,
        "same_original_rgb_frame_for_both_models": True,
        "same_original_rgb_crop_for_both_models": len(set(gate.context.values())) == 1,
        "images_or_video_saved": False, "profiles_or_attendance_written": False,
        "thresholds_retuned_on_camera": False,
        "source_code_sha256": {
            "comparison": sha256(Path(__file__)),
            "frame_processor": sha256(Path(__file__).resolve().parents[1]/"attendance_app/camera.py")},
        "models": {},
    }
    for name, model in models.items():
        scores = gate.scores[name]
        report["models"][name] = {
            "checkpoint": str(paths[name].resolve()), "sha256": sha256(paths[name]),
            "threshold": model.threshold, "threshold_source": model.threshold_source,
            "crop_policy": "context-2x" if gate.context[name] else "tight",
            "scored_crops": len(scores),
            "passed": gate.passed[name], "not_passed": len(scores)-gate.passed[name],
            "mean_live_score": float(np.mean(scores)) if scores else None,
            "median_live_score": float(np.median(scores)) if scores else None,
        }
    serialized = json.dumps(report, indent=2, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(serialized+"\n")
    print(serialized, flush=True)


if __name__ == "__main__":
    main()
