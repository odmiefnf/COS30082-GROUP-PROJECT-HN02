"""Time the real camera pipeline using a disposable gallery, without saving pixels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

from attendance_app.evaluation import gallery_snapshot


ROOT = Path(__file__).resolve().parents[1]


def timing_summary(values):
    """Milliseconds; an unvisited stage has no misleading zero mean."""
    if not values:
        return {"calls": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    import numpy as np
    a = np.asarray(values, dtype=float) * 1000
    return {"calls": len(values), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}


def time_method(owner, name, samples, key):
    original = getattr(owner, name)

    def measured(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            samples[key].append(time.perf_counter() - started)

    setattr(owner, name, measured)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--mode", choices=("synchronous", "latest"), default="synchronous")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--opencv-threads", type=int)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--database", type=Path, default=ROOT/"data/attendance/profiles.sqlite3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= 120
            or not 1 <= args.max_frames <= 5000 or not 1 <= args.num_threads <= 64
            or not 0 <= args.camera <= 99 or args.opencv_threads is not None and not 1 <= args.opencv_threads <= 64):
        parser.error("Use duration (0,120], 1–5000 frames, 1–64 threads and camera 0–99.")
    output = args.output.resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("Choose a new JSON output file; evidence cannot be overwritten.")
    report = {"schema_version": 1, "status": "not_completed", "mode": args.mode,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Performance timings only; no accuracy or liveness efficacy conclusion",
              "production_database_read_only": True, "images_or_video_saved": False,
              "camera_embeddings_in_report": False, "thresholds_retuned": False,
              "profiles_or_attendance_written_to_production": False,
              "capture_request": {"camera": args.camera, "seconds": args.seconds,
                                  "resolution": [960, 720], "warmup_frames_excluded": 8}}
    samples, statuses, sizes = defaultdict(list), Counter(), Counter()
    capture_observations = []
    camera = None
    stage = "Model setup"
    try:
        import cv2
        import torch
        from anti_spoofing.infer import LivenessDetector
        from emotion_recognition.infer import EmotionRecognizer
        from attendance_app.camera import FaceLocator, FrameProcessor
        from attendance_app.recognition import FaceRecognizer
        from attendance_app.service import AttendanceService
        torch.set_num_threads(args.num_threads)
        if args.opencv_threads is not None:
            cv2.setNumThreads(args.opencv_threads)
        setup = time.perf_counter()
        recognizer = FaceRecognizer(ROOT/"results/identity_app_config.json", args.device)
        liveness = LivenessDetector(ROOT/"results/liveness_camera_matched/best_model.pt", device=args.device)
        emotion = EmotionRecognizer(ROOT/"results/emotion_fer/best_model.pt", args.device)
        if liveness.threshold is None:
            raise ValueError("A validation-calibrated liveness threshold is required.")
        for name, metadata in (("Liveness", liveness.config), ("Expression", emotion.payload)):
            training = metadata.get("training", {})
            if training.get("smoke_test") is not False or training.get("initialization") != "random_from_scratch":
                raise ValueError(f"{name} requires a completed from-scratch checkpoint.")
        report["runtime"] = {"device": str(recognizer.device), "torch": torch.__version__,
                             "torch_threads": torch.get_num_threads(), "opencv": cv2.__version__,
                             "opencv_threads": cv2.getNumThreads(),
                             "gpu": torch.cuda.get_device_name(0) if recognizer.device.type == "cuda" else None}
        report["setup_seconds"] = time.perf_counter()-setup
        report["model_configuration"] = {"identity_sha256": recognizer.fingerprint,
            "identity_threshold": recognizer.threshold, "identity_margin": recognizer.margin,
            "liveness_threshold": liveness.threshold, "liveness_threshold_source": liveness.threshold_source}
        with gallery_snapshot(args.database, recognizer.fingerprint, recognizer.dimension, ROOT/"tmp") as store:
            service = AttendanceService(store, recognizer, liveness, emotion, "performance-"+uuid.uuid4().hex)
            locator = FaceLocator()
            processor = FrameProcessor(locator, service)
            for owner, method, key in ((locator, "detect", "face_detector"),
                    (service, "analyze", "service_analyze"), (liveness, "predict_crop", "liveness_model"),
                    (emotion, "predict_crop", "expression_model"), (recognizer, "embed", "identity_embedding"),
                    (recognizer, "match", "identity_match")):
                time_method(owner, method, samples, key)
            report["gallery_profile_count"] = len(service.profiles)
            stage = "Camera open"
            opened = time.perf_counter()
            last_sequence, processed_sequences, skipped_sequences, stale_samples = 0, set(), 0, 0
            if args.mode == "latest":
                from attendance_app.video_stream import LatestCamera

                def on_preview(sample):
                    # Only scalar metadata; pixels stay exclusively in the
                    # stream's replaceable RAM slot and current inference.
                    capture_observations.append((sample.sequence, sample.captured_at,
                                                 time.monotonic()-sample.captured_at))

                camera = LatestCamera(args.camera, on_preview=on_preview).open()
                report["camera_open_ms"] = (time.perf_counter()-opened)*1000
                while last_sequence < 8:
                    sample = camera.read_latest(last_sequence, timeout=.1)
                    if sample is not None:
                        last_sequence = sample.sequence
                report["camera_metadata"] = camera.metadata
            else:
                camera = cv2.VideoCapture(args.camera)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not camera.isOpened():
                    raise RuntimeError("The selected webcam could not be opened.")
                report["camera_open_ms"] = (time.perf_counter()-opened)*1000
                report["camera_backend"] = camera.getBackendName()
                for _ in range(8):
                    ok, _ = camera.read()
                    if not ok:
                        raise RuntimeError("Camera warmup failed.")
            print("CAPTURE_STARTED: performance timings only; no images or production attendance saved.", flush=True)
            stage = "Camera capture and pipeline"
            started = time.perf_counter()
            started_monotonic = time.monotonic()
            while sum(statuses.values()) < args.max_frames and time.perf_counter()-started < args.seconds:
                iteration = time.perf_counter()
                before = time.perf_counter()
                observed_at = None
                if args.mode == "latest":
                    sample = camera.read_latest(last_sequence, timeout=.03)
                    samples["latest_sample_wait"].append(time.perf_counter()-before)
                    if sample is None:
                        continue
                    skipped_sequences += max(0, sample.sequence-last_sequence-1)
                    last_sequence = sample.sequence
                    if time.monotonic()-sample.captured_at > 1:
                        stale_samples += 1
                        service.reset_tracking("The camera observation is stale. Try again.")
                        continue
                    if sample.sequence in processed_sequences:
                        raise RuntimeError("Duplicate frame sequence would repeat an observation.")
                    processed_sequences.add(sample.sequence)
                    rgb, observed_at = sample.rgb, sample.captured_at
                    samples["observation_age_at_inference_start"].append(time.monotonic()-observed_at)
                else:
                    ok, bgr = camera.read()
                    samples["camera_read"].append(time.perf_counter()-before)
                    if not ok:
                        raise RuntimeError("The camera stopped returning frames.")
                    before = time.perf_counter()
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    samples["rgb_conversion"].append(time.perf_counter()-before)
                sizes[(rgb.shape[1], rgb.shape[0])] += 1
                before = time.perf_counter()
                result = processor.process(rgb) if observed_at is None else processor.process(rgb, observed_at=observed_at)
                samples["frame_processing"].append(time.perf_counter()-before)
                if observed_at is not None:
                    samples["observation_age_at_inference_completion"].append(time.monotonic()-observed_at)
                before = time.perf_counter()
                store.attendance(service.session)
                samples["attendance_query"].append(time.perf_counter()-before)
                statuses[result["status"]] += 1
                samples["loop_total"].append(time.perf_counter()-iteration)
            elapsed = time.perf_counter()-started
            ended_monotonic = time.monotonic()
            if args.mode == "latest":
                camera.release()
                camera = None
                captured = [item for item in capture_observations if started_monotonic <= item[1] <= ended_monotonic]
                for previous, current in zip(captured, captured[1:]):
                    samples["capture_interval"].append(current[1]-previous[1])
                samples["capture_callback_age"].extend(item[2] for item in captured)
                report["latest_capture"] = {"frames_captured_in_window": len(captured),
                    "capture_fps": len(captured)/elapsed, "analysis_unique_sequences": len(processed_sequences),
                    "analysis_skipped_capture_sequences": skipped_sequences,
                    "stale_samples_not_analyzed": stale_samples, "duplicate_sequences_analyzed": 0,
                    "note": "Capture FPS is the source/preview availability rate, not measured Tk display FPS. Read/RGB timing inside LatestCamera is not instrumented."}
            if not statuses:
                raise RuntimeError("No measured frames were processed.")
            report.update(status="completed", duration_seconds=elapsed, processed_frames=sum(statuses.values()),
                          processing_fps=sum(statuses.values())/elapsed,
                          simulated_attendance_records=len(store.attendance(service.session)))
    except Exception as exc:
        report["error"] = {"stage": stage, "message": str(exc)}
    finally:
        if camera is not None:
            camera.release()
    report["timings"] = {key: timing_summary(values) for key, values in samples.items()}
    report["frame_statuses"] = dict(statuses)
    report["frame_sizes"] = [{"width": w, "height": h, "frames": count} for (w, h), count in sizes.items()]
    report["timing_note"] = "Nested stages overlap; sums of stage timings are not a total. Cold first inference is included. GUI drawing is not measured."
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "processing_fps": report.get("processing_fps"),
                      "timings": report["timings"], "error": report.get("error")}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
