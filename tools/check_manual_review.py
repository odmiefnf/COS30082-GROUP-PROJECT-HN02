"""Check human-review wiring with real CNNs and a scripted dataset-label oracle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageOps
import torch

from anti_spoofing.infer import LivenessDetector
from emotion_recognition.infer import EmotionRecognizer
from attendance_app.gui import CameraWorker
from attendance_app.recognition import FaceRecognizer
from attendance_app.service import AttendanceService
from attendance_app.store import ProfileStore
from face_verification.data import IMAGE_EXTENSIONS
from face_verification.image_cache import sha256


def read_crop(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=project/"results/manual_review_integration.json")
    output = parser.parse_args().output
    torch.set_num_threads(4)
    config = project/"results/identity_app_config.json"
    live_path, emotion_path = (project/f"results/{name}/best_model.pt" for name in ("liveness_lcc", "emotion_fer"))
    recognizer = FaceRecognizer(config)
    live, emotion = LivenessDetector(live_path), EmotionRecognizer(emotion_path)
    model_paths = [recognizer.checkpoint, live_path, emotion_path]
    before = {str(path): sha256(path) for path in model_paths}
    baseline = json.loads((project/"results/attendance_integration.json").read_text(encoding="utf-8"))
    case = next(t for t in baseline["trials"] if t["subject_enrolled_in_test_store"]
                and (t["result"]["identity"] or {}).get("status") == "unknown")
    registered = {}
    for trial in baseline["trials"]:
        if trial["subject_enrolled_in_test_store"]:
            registered.setdefault(trial["dataset_subject"], Path(trial["path"]).parent)
    temporary_root = project/"tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="manual_review_real_models_", dir=temporary_root) as temporary:
        store = ProfileStore(Path(temporary)/"integration.sqlite3", recognizer.fingerprint, recognizer.dimension)
        try:
            ids = {}
            for subject, directory in registered.items():
                paths = sorted((p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name)
                ids[subject] = store.enroll(f"Dataset {subject}", [recognizer.embed(read_crop(p)) for p in paths[:3]])
            service = AttendanceService(store, recognizer, live, emotion, "software-fixture")
            crop = read_crop(case["path"])
            original = service.analyze(crop, now=10.)
            assert original["status"] == "unknown" and original["liveness"]["is_live"] is True
            assert not store.attendance("software-fixture")
            request = service.request_manual_review(now=10.1)
            service.analyze(crop, now=10.2)
            decision = service.confirm_manual_review(review_id=request["id"], person_id=ids[case["dataset_subject"]],
                reviewer="Software fixture operator", reason="Dataset label oracle; software test, not a human decision.",
                confirmed=True, now=10.3)
            record = store.attendance("software-fixture")[0]
            assert decision["recorded"] and record["source"] == "manual_review"
            assert record["cosine"] < recognizer.threshold
            after_prediction = recognizer.match(recognizer.embed(crop), store.profiles())
            assert after_prediction["status"] == "unknown"
            blocked_case = next(t for t in baseline["trials"] if t["result"]["liveness"]["is_live"] is False)
            blocked_result = service.analyze(read_crop(blocked_case["path"]), now=11.)
            try:
                service.request_manual_review(now=11.1)
            except ValueError:
                blocked = True
            else:
                blocked = False
            assert blocked and blocked_result["liveness"]["is_live"] is False
        finally:
            store.close()
        args = SimpleNamespace(num_threads=4, identity_config=config, device="cpu", liveness_checkpoint=live_path,
            emotion_checkpoint=emotion_path, database=Path(temporary)/"startup.sqlite3", detector="haar", roi=None,
            session="software-startup", camera=0)
        commands, updates = queue.Queue(), queue.Queue()
        commands.put(("quit", None))
        with patch("cv2.VideoCapture", side_effect=AssertionError("Camera must remain off in this check")):
            CameraWorker(args, commands, updates).run()
        events = [updates.get_nowait()["event"] for _ in range(updates.qsize())]
        assert "ready" in events and "fatal" not in events, events
    assert before == {str(path): sha256(path) for path in model_paths}
    report = {"scope": "Software integration on a train-split photo; reviewer is scripted from the dataset label, not a human trial.",
        "camera_activated": False, "production_database_used": False, "temporary_database_removed": True,
        "real_model_startup": "ready_with_camera_off", "checkpoint_hashes_unchanged": before,
        "selected_dataset_subject": case["dataset_subject"], "model_status_before_review": original["status"],
        "model_status_after_review": after_prediction["status"], "automatic_threshold": recognizer.threshold,
        "stored_cosine": record["cosine"], "saved_source": record["source"], "saved_reviewer": record["reviewer"],
        "saved_reason": record["reason"], "live_gate_failure_blocked_review": blocked,
        "no_model_accuracy_improvement_claimed": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
