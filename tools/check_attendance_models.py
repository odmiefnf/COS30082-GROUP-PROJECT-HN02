"""Offline integration of all three real models; never starts a camera or production store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

from PIL import Image, ImageOps
import torch

from anti_spoofing.infer import LivenessDetector
from emotion_recognition.infer import EmotionRecognizer
from attendance_app.recognition import FaceRecognizer
from attendance_app.service import AttendanceService, uses_pad_context
from attendance_app.store import ProfileStore
from face_verification.data import _discover_images
from face_verification.image_cache import sha256


def check(project, output, liveness_checkpoint=None):
    project = Path(project).resolve()
    torch.set_num_threads(4)
    config = project / "results/identity_app_config.json"
    recognizer = FaceRecognizer(config)
    liveness_path = Path(liveness_checkpoint).resolve() if liveness_checkpoint else project / "results/liveness_lcc/best_model.pt"
    emotion_path = project / "results/emotion_fer/best_model.pt"
    live = LivenessDetector(liveness_path)
    if uses_pad_context(live):
        raise ValueError("This check uses already-cropped face photos. Use tools.evaluate_webcam_trial for context PAD.")
    emotion = EmotionRecognizer(emotion_path)
    _, images = _discover_images(project / "data/face_recognition/classification_data/train_data")
    subjects = list(images)[:4]
    temporary_root = project / "tmp"
    temporary_root.mkdir(exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        store = ProfileStore(Path(temporary) / "offline_test.sqlite3", recognizer.fingerprint, recognizer.dimension)
        try:
            # Ground-truth dataset labels supply test-only reference profiles.
            # This bypasses camera enrollment solely to test real-model matching,
            # and must never be called a successful real-world registration trial.
            for subject in subjects[:2]:
                embeddings = []
                for path in images[subject][:3]:
                    with Image.open(path) as source:
                        embeddings.append(recognizer.embed(ImageOps.exif_transpose(source).convert("RGB")))
                store.enroll(f"Dataset {subject}", embeddings)
            service = AttendanceService(store, recognizer, live, emotion, "offline-software-check")
            for subject in subjects:
                for path in images[subject][3:9]:
                    with Image.open(path) as source:
                        crop = ImageOps.exif_transpose(source).convert("RGB")
                    started = time.perf_counter()
                    result = service.analyze(crop)
                    elapsed = time.perf_counter() - started
                    trials.append({"path": str(path), "dataset_subject": subject,
                                   "subject_enrolled_in_test_store": subject in subjects[:2],
                                   "elapsed_ms": elapsed*1000, "result": result})
            report = {"camera_activated": False, "production_store_used": False,
                      "fixture_registration": "Ground-truth dataset ID references, not a webcam enrollment test.",
                      "sample_source": "24 train-split photos; software integration only, not a held-out accuracy benchmark.",
                      "identity_checkpoint_sha256": recognizer.fingerprint,
                      "liveness_checkpoint_sha256": sha256(liveness_path),
                      "emotion_checkpoint_sha256": sha256(emotion_path),
                      "trial_count": len(trials),
                      "live_gate_passed": sum(t["result"]["liveness"]["is_live"] is True for t in trials),
                      "identity_and_expression_executed": sum(t["result"]["identity"] is not None and t["result"]["expression"] is not None for t in trials),
                      "unregistered_probe_matched_gallery": sum(not t["subject_enrolled_in_test_store"]
                          and (t["result"]["identity"] or {}).get("status") == "known" for t in trials),
                      "temporary_attendance_record_count": len(store.attendance(service.session)),
                      "trials": trials}
        finally:
            store.close()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/attendance_integration.json"))
    parser.add_argument("--liveness-checkpoint", type=Path,
                        help="Optional current app model; default preserves the original baseline integration protocol.")
    args = parser.parse_args(argv)
    try:
        check(args.project, args.output, args.liveness_checkpoint)
    except ValueError as exc:
        parser.exit(1, f"Attendance check failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
