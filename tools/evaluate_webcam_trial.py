"""Run one explicitly labelled webcam trial without changing the live gallery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

from attendance_app.evaluation import (CONDITIONS, PRESENTATIONS, TrialAccumulator, collect_frames,
    gallery_snapshot, report_header, validate_ground_truth, validate_presentation)

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--expected-profile")
    parser.add_argument("--presentation-type", choices=PRESENTATIONS,
                        help="Required for attacks: phone-still-photo, recorded-video or printed-photo; never inferred")
    parser.add_argument("--environment", help="Operator-supplied lighting/camera setup description")
    parser.add_argument("--notes", help="Operator-supplied notes; no conditions are inferred from camera images")
    parser.add_argument("--distance-cm", type=float, help="Operator-measured camera distance, if known")
    parser.add_argument("--database", type=Path, default=ROOT/"data/attendance/profiles.sqlite3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=15.)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= 120
            or not 1 <= args.max_frames <= 5000 or not 0 <= args.camera <= 99):
        parser.error("Use camera 0–99, duration in (0,120] and 1–5000 measured frames.")
    try:
        validate_presentation(args.condition, args.presentation_type)
    except ValueError as exc:
        parser.error(str(exc))
    if args.distance_cm is not None and (not math.isfinite(args.distance_cm) or not 0 < args.distance_cm <= 500):
        parser.error("A supplied camera distance must be finite and in (0,500] cm.")
    for text in (args.environment, args.notes):
        if text is not None and (len(text) > 1000 or any(ord(c) < 32 for c in text)):
            parser.error("Environment/notes must contain at most 1000 printable characters.")
    output = args.output.resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("Choose a new .json output file; existing evidence is never overwritten.")
    report = report_header(args.condition, args.expected_profile)
    report["operator_ground_truth"].update(presentation_type=args.presentation_type,
        environment=args.environment, notes=args.notes, distance_cm=args.distance_cm)
    accumulator, store, session = None, None, "evaluation-"+uuid.uuid4().hex
    elapsed, stage = 0., "Model setup"
    try:
        import cv2
        import torch
        from anti_spoofing.infer import LivenessDetector
        from emotion_recognition.infer import EmotionRecognizer
        from face_verification.image_cache import sha256
        from attendance_app.camera import FaceLocator, FrameProcessor
        from attendance_app.recognition import FaceRecognizer
        from attendance_app.service import AttendanceService
        torch.set_num_threads(4)
        recognizer = FaceRecognizer(ROOT/"results/identity_app_config.json", "cpu")
        liveness = LivenessDetector(ROOT/"results/liveness_camera_matched/best_model.pt", device="cpu")
        emotion = EmotionRecognizer(ROOT/"results/emotion_fer/best_model.pt", "cpu")
        if liveness.threshold is None:
            raise ValueError("A calibrated liveness threshold is required.")
        for name, metadata in (("Liveness", liveness.config), ("Expression", emotion.payload)):
            training = metadata.get("training", {})
            if training.get("smoke_test") is not False or training.get("initialization") != "random_from_scratch":
                raise ValueError(f"{name} requires completed from-scratch training.")
        report["models"] = {
            "identity": {"sha256": recognizer.fingerprint, "threshold": recognizer.threshold, "ambiguity_margin": recognizer.margin},
            "liveness": {"sha256": sha256(liveness.checkpoint), "threshold": liveness.threshold,
                         "threshold_source": liveness.threshold_source},
            "expression": {"sha256": sha256(emotion.checkpoint)},
            "detector": "Current application Haar locator", "device": "cpu", "threads": 4}
        stage = "Gallery snapshot and ground truth"
        with gallery_snapshot(args.database, recognizer.fingerprint, recognizer.dimension, ROOT/"tmp") as store:
            expected_id = validate_ground_truth(args.condition, args.expected_profile, store.profiles())
            accumulator = TrialAccumulator(args.condition, expected_id)
            service = AttendanceService(store, recognizer, liveness, emotion, session)
            processor = FrameProcessor(FaceLocator(), service)
            report["gallery_profile_count"] = len(service.profiles)
            report["gallery_appearance_counts"] = [len(person["appearances"]) for person in service.profiles]
            report["capture_request"] = {"camera": args.camera, "seconds": args.seconds, "max_frames": args.max_frames,
                                         "warmup_frames_excluded": 8, "requested_resolution": [960, 720]}
            stage = "Camera capture / application pipeline"
            started = time.monotonic()
            try:
                elapsed = collect_frames(cv2.VideoCapture, processor, accumulator, camera=args.camera,
                                         seconds=args.seconds, max_frames=args.max_frames)
                report["status"] = "completed"
            finally:
                if not elapsed:
                    elapsed = time.monotonic()-started
                report.update(accumulator.summary(elapsed, store.attendance(session)))
    except Exception as exc:
        report["error"] = {"stage": stage, "message": str(exc)}
        report["status"] = "not_completed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "counts": report.get("counts"), "error": report.get("error")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
