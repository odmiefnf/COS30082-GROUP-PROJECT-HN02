"""Attendance behavior checks; generated arrays are software fixtures, not real people."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from attendance_app.store import ProfileStore, clean_text
from attendance_app.recognition import FaceRecognizer
from attendance_app.service import AttendanceService
from attendance_app.camera import FrameProcessor, FaceLocator
from attendance_app.gui import CameraWorker
from face_verification.image_cache import sha256
from face_verification.model import FaceEncoder


class Fixture(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temporary.name)
        self.store = ProfileStore(self.root / "profiles.sqlite3", "a" * 64, 3)
        self.recognizer = Mock(dimension=3, threshold=.8)
        self.recognizer.embed.return_value = np.array([1., 0., 0.], dtype=np.float32)
        self.recognizer.match.return_value = {"status": "unknown", "person_id": None, "name": None, "cosine": None}
        self.live = Mock()
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .95}
        self.emotion = Mock()
        self.emotion.predict_crop.return_value = {"label": "angry"}
        self.service = AttendanceService(self.store, self.recognizer, self.live, self.emotion, "COS30082-01")

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def enroll(self, name="Kiệt / S123"):
        return self.store.enroll(name, [[1., 0., 0.]] * 3)

    def test_unicode_persistence_and_encoder_isolation(self):
        person_id = self.enroll()
        with self.assertRaises(ValueError):
            self.enroll("KIỆT / S123")
        self.assertEqual(len(self.store.profiles()), 1)
        self.assertEqual(self.store.profiles()[0]["sample_count"], 3)
        with self.assertRaisesRegex(ValueError, "another encoder"):
            ProfileStore(self.store.path, "b" * 64, 3)
        other = ProfileStore(self.store.path, "a" * 64, 3)
        self.assertEqual(other.profiles()[0]["id"], person_id)
        self.assertEqual(other.profiles()[0]["name"], "Kiệt / S123")
        other.close()
        self.assertEqual([p.name for p in self.root.iterdir()], ["profiles.sqlite3"])

    def test_invalid_registration_is_atomic(self):
        for values in ([[1., 0., 0.]] * 2, [[0., 0., 0.]] * 3, [[np.nan, 0., 0.]] * 3, [[1., 0.]] * 3):
            with self.assertRaises(ValueError):
                self.store.enroll("Sample", values)
        self.assertFalse(self.store.profiles())
        for value in ("", "   ", "x\ny", "x"*81, 10):
            with self.assertRaises(ValueError):
                clean_text(value)

    def test_spoof_and_unknown_threshold_skip_identity_and_registration(self):
        self.service.begin_registration("Live user")
        for live_value in (False, None):
            self.live.predict_crop.return_value = {"is_live": live_value}
            crop = Image.new("RGB", (100, 100))
            self.emotion.predict_crop.reset_mock()
            result = self.service.analyze(crop)
            self.assertFalse(result["attendance_recorded"])
            self.recognizer.embed.assert_not_called()
            self.emotion.predict_crop.assert_called_once_with(crop)
            self.assertEqual(result["expression"], {"label": "angry"})
            self.assertIsNone(result["identity"])
            self.assertEqual(result["status"], "spoof" if live_value is False else "liveness_unavailable")
        self.assertFalse(self.store.profiles())

    def test_expression_failure_cannot_bypass_a_failed_liveness_gate(self):
        self.service.begin_registration("Live user")
        self.live.predict_crop.return_value = {"is_live": False}
        self.emotion.predict_crop.side_effect = RuntimeError("FER unavailable")
        result = self.service.analyze(Image.new("RGB", (100, 100)))
        self.assertEqual(result["status"], "spoof")
        self.assertIn("expression_error", result)
        self.assertIsNone(result["expression"])
        self.recognizer.embed.assert_not_called()
        self.assertFalse(self.store.profiles())
        self.assertFalse(self.store.attendance(self.service.session))

    def test_slow_expression_on_rejected_face_is_not_published_as_fresh(self):
        self.live.predict_crop.return_value = {"is_live": False}
        with patch("attendance_app.service.time.monotonic", return_value=102.):
            result = self.service.analyze(Image.new("RGB", (100, 100)), now=100., max_observation_age=1.)
        self.assertEqual(result["status"], "observation_stale")
        self.assertIsNone(result["expression"])
        self.recognizer.embed.assert_not_called()
        self.assertFalse(self.store.profiles())
        self.assertFalse(self.store.attendance(self.service.session))

    def test_registration_requires_three_distinct_live_views_and_resets_on_spoof(self):
        self.service.begin_registration("Kiệt")
        for now in (0., .5, 1.):
            result = self.service.analyze(Image.new("RGB", (100, 100), (1, 2, 3)), now)
        self.assertEqual(result["registration_samples"], 1)
        self.live.predict_crop.return_value = {"is_live": False}
        self.service.analyze(Image.new("RGB", (100, 100)), 1.1)
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .95}
        for index in range(3):
            crop = Image.new("RGB", (100, 100), (index+20, 2, 3))
            result = self.service.analyze(crop, 2.+index*.5)
        self.assertTrue(result["registered"])
        self.assertEqual(len(self.store.profiles()), 1)
        self.assertFalse(self.store.attendance("COS30082-01"))
        self.assertIs(self.live.predict_crop.call_args.args[0], crop)
        self.assertIs(self.emotion.predict_crop.call_args.args[0], crop)
        self.assertIs(self.recognizer.embed.call_args.args[0], crop)

    def test_switching_enrollment_person_or_recognized_person_cannot_enroll(self):
        self.service.begin_registration("New")
        self.service.analyze(Image.new("RGB", (100, 100)), 0.)
        self.recognizer.embed.return_value = np.array([0., 1., 0.], dtype=np.float32)
        self.assertEqual(self.service.analyze(Image.new("RGB", (100, 100), "red"), .5)["status"], "registration_person_changed")
        self.recognizer.match.return_value = {"status": "known"}
        self.assertEqual(self.service.analyze(Image.new("RGB", (100, 100)), 1.)["status"], "registration_existing_or_ambiguous")
        self.assertFalse(self.store.profiles())
        self.service.cancel_registration()
        self.assertIsNone(self.service.pending_name)

    def test_attendance_needs_three_frames_and_is_unique_per_session(self):
        person_id = self.enroll()
        self.recognizer.match.return_value = {"status": "known", "person_id": person_id, "name": "Kiệt", "cosine": .99}
        self.service.profiles = self.store.profiles()
        crop = Image.new("RGB", (100, 100))
        self.assertFalse(self.service.analyze(crop, 0.)["attendance_recorded"])
        self.assertFalse(self.service.analyze(crop, .2)["attendance_recorded"])
        self.emotion.predict_crop.side_effect = RuntimeError("FER unavailable")
        result = self.service.analyze(crop, .4)
        self.assertTrue(result["attendance_recorded"])
        self.assertIn("expression_error", result)
        self.assertFalse(self.service.analyze(crop, .6)["attendance_recorded"])
        self.assertEqual(len(self.store.attendance("COS30082-01")), 1)
        self.service.session = "COS30082-02"
        self.service.reset_tracking()
        for index in range(3):
            result = self.service.analyze(crop, 1.+index*.2)
        self.assertTrue(result["attendance_recorded"])
        self.assertEqual(len(self.store.attendance("COS30082-02")), 1)

    def test_gap_or_spoof_breaks_recognition_streak(self):
        person_id = self.enroll()
        self.recognizer.match.return_value = {"status": "known", "person_id": person_id, "name": "Kiệt", "cosine": .99}
        crop = Image.new("RGB", (100, 100))
        self.service.analyze(crop, 0.)
        self.service.analyze(crop, .2)
        self.assertEqual(self.service.analyze(crop, 2.)["stable_frames"], 1)
        self.live.predict_crop.return_value = {"is_live": False}
        self.service.analyze(crop, 2.1)
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .95}
        self.assertEqual(self.service.analyze(crop, 2.2)["stable_frames"], 1)
        self.assertFalse(self.store.attendance("COS30082-01"))

    def test_no_face_multiple_faces_or_quality_failure_skip_all_models(self):
        locator = Mock()
        processor = FrameProcessor(locator, self.service)
        for boxes, expected in (([], "no_face"), ([(0, 0, 100, 100)]*2, "multiple_faces"),
                                 ([(0, 0, 40, 40)], "face_too_small_or_outside"),
                                 ([(0, 0, 100, 100)], "image_quality")):
            locator.detect.return_value = boxes
            self.assertEqual(processor.process(np.zeros((150, 150, 3), dtype=np.uint8))["status"], expected)
        self.live.predict_crop.assert_not_called()

    def test_valid_crop_reaches_real_gate_and_rgb_is_preserved(self):
        frame = np.random.default_rng(42).integers(0, 256, (160, 200, 3), dtype=np.uint8)
        locator = Mock(); locator.detect.return_value = [(30, 20, 100, 100)]
        result = FrameProcessor(locator, self.service).process(frame)
        self.assertEqual(result["status"], "unknown")
        np.testing.assert_array_equal(np.asarray(self.live.predict_crop.call_args.args[0]), frame[20:120, 30:130])
        self.assertFalse(self.store.attendance("COS30082-01"))

    def test_context_checkpoint_requires_and_uses_context_only_for_pad(self):
        self.live.config = {"context_crop": True}
        service = AttendanceService(self.store, self.recognizer, self.live, self.emotion, "context-fixture")
        with self.assertRaisesRegex(ValueError, "requires a context crop"):
            service.analyze(Image.new("RGB", (100, 100)))
        self.live.predict_crop.assert_not_called()
        frame = np.random.default_rng(42).integers(0, 256, (160, 200, 3), dtype=np.uint8)
        locator = Mock()
        locator.detect.return_value = [(30, 20, 100, 100)]
        result = FrameProcessor(locator, service, pad_context=True).process(frame)
        self.assertEqual(result["status"], "unknown")
        np.testing.assert_array_equal(np.asarray(self.live.predict_crop.call_args.args[0]), frame[:, :180])
        tight = frame[20:120, 30:130]
        np.testing.assert_array_equal(np.asarray(self.recognizer.embed.call_args.args[0]), tight)
        np.testing.assert_array_equal(np.asarray(self.emotion.predict_crop.call_args.args[0]), tight)
        np.testing.assert_array_equal(np.asarray(service.review.latest.crop), tight)

    def test_camera_read_failure_and_stop_release_device_without_using_a_real_camera(self):
        for fail_read in (True, False):
            with self.subTest(fail_read=fail_read):
                args = SimpleNamespace(num_threads=2, identity_config=self.root/"unused.json", device="cpu",
                    liveness_checkpoint=self.root/"unused_live.pt", emotion_checkpoint=self.root/"unused_fer.pt",
                    database=self.root/f"worker-{fail_read}.sqlite3", detector="haar", roi=None, session="test", camera=0)
                recognizer = Mock(fingerprint="a"*64, dimension=3, threshold=.8)
                trained = {"training": {"smoke_test": False, "initialization": "random_from_scratch"}}
                liveness = Mock(threshold=.5, config=trained)
                camera = Mock()
                camera.isOpened.return_value = True
                camera.read.return_value = (False, None) if fail_read else (True, np.zeros((100, 100, 3), dtype=np.uint8))
                locator = Mock(); locator.detect.return_value = []
                commands, updates = queue.Queue(), queue.Queue()
                commands.put(("start", "test"))
                if not fail_read:
                    commands.put(("stop", None))
                commands.put(("quit", None))
                with patch("attendance_app.recognition.FaceRecognizer", return_value=recognizer), \
                     patch("anti_spoofing.infer.LivenessDetector", return_value=liveness), \
                     patch("emotion_recognition.infer.EmotionRecognizer", return_value=Mock(payload=trained)), \
                     patch("attendance_app.camera.FaceLocator", return_value=locator), \
                     patch("cv2.VideoCapture", return_value=camera):
                    CameraWorker(args, commands, updates).run()
                camera.release.assert_called_once()
                events = [updates.get_nowait()["event"] for _ in range(updates.qsize())]
                self.assertIn("camera_error" if fail_read else "stopped", events)
                self.assertNotIn("fatal", events)


class RecognitionChecks(unittest.TestCase):
    def test_open_set_unknown_ambiguity_and_single_profile(self):
        recognizer = FaceRecognizer.__new__(FaceRecognizer)
        recognizer.dimension, recognizer.threshold, recognizer.margin = 3, .8, .05
        self.assertEqual(recognizer.match([1, 0, 0], [])["status"], "unknown")
        a = {"id": "a", "name": "An", "embedding": [1, 0, 0]}
        self.assertEqual(recognizer.match([1, 0, 0], [a])["status"], "known")
        self.assertEqual(recognizer.match([0, 1, 0], [a])["status"], "unknown")
        b = {"id": "b", "name": "Bình", "embedding": [.99, .01, 0]}
        self.assertEqual(recognizer.match([1, 0, 0], [a, b])["status"], "ambiguous")
        with self.assertRaises(ValueError):
            recognizer.match([float("nan"), 0, 0], [a])

    def test_checkpoint_fingerprint_preprocessing_and_smoke_test_guard(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            checkpoint = root / "test.pt"
            payload = {"model_state": FaceEncoder(8).state_dict(),
                       "config": {"image_size": 32, "embedding_dim": 8, "method": "classification"},
                       "training": {"smoke_test": True, "initialization": "random_from_scratch"}}
            torch.save(payload, checkpoint)
            path = root / "config.json"
            settings = {"checkpoint": "test.pt", "checkpoint_sha256": sha256(checkpoint), "metric": "cosine",
                        "threshold_source": "verification_validation", "threshold": .9}
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, "completed"):
                FaceRecognizer(path)
            # Fixture exercises the runtime guard; never delivered as a trained face model.
            payload["training"]["smoke_test"] = False
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                FaceRecognizer(path)
            settings["checkpoint_sha256"] = sha256(checkpoint)
            path.write_text(json.dumps(settings))
            recognizer = FaceRecognizer(path)
            crop = Image.new("RGB", (64, 64), (10, 90, 160))
            np.testing.assert_array_equal(recognizer.embed(crop), recognizer.embed(np.asarray(crop)))
            self.assertAlmostEqual(float(np.linalg.norm(recognizer.embed(crop))), 1., places=6)

    def test_camera_control_events_are_not_dropped_by_video_frames(self):
        updates = queue.Queue()
        worker = CameraWorker(Mock(), queue.Queue(), updates)
        worker.publish(event="started")
        for sequence in range(100):
            worker.publish(event="frame", sequence=sequence)
        worker.publish(event="camera_error", message="Disconnected")
        events = [updates.get_nowait()["event"] for _ in range(updates.qsize())]
        self.assertEqual(events, ["started", "camera_error"])
        self.assertEqual(worker.analyses.qsize(), 1)
        self.assertEqual(worker.analyses.get_nowait()["sequence"], 99)

    def test_haar_loads_and_empty_frame_has_no_faces(self):
        self.assertEqual(FaceLocator().detect(np.zeros((200, 200, 3), dtype=np.uint8)), [])


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
