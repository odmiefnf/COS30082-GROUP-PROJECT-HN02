"""Capture state regressions with vector fixtures; no camera or face benchmark."""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

from attendance_app.camera import FrameProcessor
from attendance_app.gui import CameraWorker
from attendance_app.recognition import FaceRecognizer
from attendance_app.service import AttendanceService
from attendance_app.store import ProfileStore


FRAME_STATE_FIELDS = {
    "manual_review_pending", "manual_review_valid", "appearance_dialog_pending",
    "appearance_request_valid", "appearance_request_reason", "appearance_pending",
    "registration_samples",
}


class CaptureReliabilityChecks(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="capture_reliability_", dir=temporary_root)
        self.root = Path(self.temporary.name)
        self.store = ProfileStore(self.root / "profiles.sqlite3", "a" * 64, 3)
        self.person = self.store.enroll("Fixture profile", [[1., 0., 0.]] * 3)
        self.recognizer = FaceRecognizer.__new__(FaceRecognizer)
        self.recognizer.dimension, self.recognizer.threshold, self.recognizer.margin = 3, .8, .05
        self.recognizer.embed = Mock(return_value=np.array([0., 1., 0.], dtype=np.float32))
        self.liveness = Mock()
        self.liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
        self.service = self.make_service()

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def make_service(self, session="capture-fixture"):
        self.recognizer.embed.return_value = np.array([0., 1., 0.], dtype=np.float32)
        self.liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
        return AttendanceService(self.store, self.recognizer, self.liveness, Mock(), session)

    @staticmethod
    def crop(color):
        return Image.new("RGB", (100, 100), (color, 50, 80))

    def open_request(self, service=None):
        service = self.service if service is None else service
        service.analyze(self.crop(1), now=10.)
        return service.request_appearance(now=10.1)

    def begin_capture(self, request, service=None, now=10.2):
        service = self.service if service is None else service
        service.begin_appearance(self.person, "Without glasses", True,
                                 request_id=request["id"], now=now)

    def invalid_frame(self, service, mode, now):
        if mode == "spoof":
            self.liveness.predict_crop.return_value = {"is_live": False, "live_score": .01}
            return service.analyze(self.crop(9), now=now)
        boxes = {"no_face": [], "multiple_faces": [(0, 0, 100, 100), (40, 0, 100, 100)],
                 "image_quality": [(0, 0, 100, 100)]}[mode]
        locator = Mock()
        locator.detect.return_value = boxes
        with patch("attendance_app.service.time.monotonic", return_value=now):
            return FrameProcessor(locator, service).process(np.zeros((150, 150, 3), dtype=np.uint8))

    def assert_state_fields(self, result):
        self.assertTrue(FRAME_STATE_FIELDS.issubset(result), FRAME_STATE_FIELDS - result.keys())

    def test_unknown_person_swap_in_open_dialog_cannot_attach_or_revive_ticket(self):
        request = self.open_request()
        self.recognizer.embed.return_value = np.array([0., 0., 1.], dtype=np.float32)
        swapped = self.service.analyze(self.crop(2), now=10.2)
        self.assertEqual(swapped["status"], "unknown")
        with self.assertRaises(ValueError):
            self.begin_capture(request, now=10.3)
        self.recognizer.embed.return_value = np.array([0., 1., 0.], dtype=np.float32)
        self.service.analyze(self.crop(3), now=10.4)
        with self.assertRaises(ValueError):
            self.begin_capture(request, now=10.5)
        state = self.service.frame_state(now=10.5)
        self.assertTrue(state["appearance_dialog_pending"])
        self.assertFalse(state["appearance_request_valid"])
        self.assertTrue(state["appearance_request_reason"])
        self.assertIsNone(self.service.pending_appearance)
        self.assertEqual(self.store.profiles()[0]["sample_count"], 3)

    def test_request_expiry_session_change_gap_and_staleness_do_not_revive(self):
        for mode in ("expired", "session", "gap", "stale"):
            with self.subTest(mode=mode):
                service = self.make_service("request-" + mode)
                request = self.open_request(service)
                if mode == "expired":
                    now = 41.
                    service.frame_state(now=now)
                elif mode == "session":
                    now = 10.2
                    service.session = "different-session"
                    service.analyze(self.crop(2), now=now)
                    service.session = "request-" + mode
                elif mode == "gap":
                    now = 12.
                    service.analyze(self.crop(2), now=now)
                else:
                    now = 11.2
                with self.assertRaises(ValueError):
                    self.begin_capture(request, service, now=now + .01)
                service.analyze(self.crop(3), now=now + .1)
                with self.assertRaises(ValueError):
                    self.begin_capture(request, service, now=now + .2)
                self.assertTrue(service.frame_state(now=now + .2)["appearance_dialog_pending"])
                self.assertEqual(self.store.profiles()[0]["sample_count"], 3)
                service.cancel_appearance_request(request["id"])

    def test_invalid_frame_paths_permanently_invalidate_appearance_request(self):
        for mode in ("no_face", "multiple_faces", "image_quality", "spoof"):
            with self.subTest(mode=mode):
                service = self.make_service("appearance-invalid-" + mode)
                request = self.open_request(service)
                result = self.invalid_frame(service, mode, now=10.2)
                self.assert_state_fields(result)
                self.assertTrue(result["appearance_dialog_pending"])
                self.assertFalse(result["appearance_request_valid"])
                self.assertTrue(result["appearance_request_reason"])
                self.assertEqual(result["registration_samples"], 0)
                self.liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
                service.analyze(self.crop(3), now=10.3)
                with self.assertRaises(ValueError):
                    self.begin_capture(request, service, now=10.4)
                self.assertEqual(self.store.profiles()[0]["sample_count"], 3)
                service.cancel_appearance_request(request["id"])

    def test_request_token_is_required_consumed_once_and_stale_cancel_is_ignored(self):
        request = self.open_request()
        with self.assertRaises(ValueError):
            self.service.begin_appearance(self.person, "Without glasses", True,
                                          request_id="wrong-request-token", now=10.2)
        self.begin_capture(request, now=10.3)
        for index in range(3):
            result = self.service.analyze(self.crop(20 + index), now=10.4 + .5 * index)
            self.assertFalse(result["attendance_recorded"])
        self.assertTrue(result["appearance_added"])
        self.assertEqual(self.store.profiles()[0]["sample_count"], 6)
        # Completion clears the old observation; provide a fresh face so replay
        # rejection exercises the consumed token rather than missing evidence.
        self.service.analyze(self.crop(24), now=11.45)
        with self.assertRaises(ValueError):
            self.service.begin_appearance(self.person, "Replayed", True,
                                          request_id=request["id"], now=11.5)
        current = self.service.request_appearance(now=11.5)
        self.assertNotEqual(current["id"], request["id"])
        self.service.cancel_appearance_request(request["id"])
        self.assertTrue(self.service.frame_state(now=11.6)["appearance_dialog_pending"])
        self.service.cancel_appearance_request(current["id"])
        self.assertFalse(self.service.frame_state(now=11.6)["appearance_dialog_pending"])

    def test_active_capture_expires_on_spoof_and_no_face_without_writing(self):
        for mode in ("spoof", "no_face"):
            with self.subTest(mode=mode):
                service = self.make_service("expired-capture-" + mode)
                request = self.open_request(service)
                self.begin_capture(request, service)
                service.analyze(self.crop(2), now=10.3)
                self.assertEqual(len(service.enrollment_samples), 1)
                result = self.invalid_frame(service, mode, now=41.)
                self.assert_state_fields(result)
                self.assertFalse(result["appearance_pending"])
                self.assertEqual(result["registration_samples"], 0)
                self.assertIsNone(service.pending_appearance)
                self.liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
                for index in range(3):
                    later = service.analyze(self.crop(30 + index), now=41.1 + .5 * index)
                    self.assertFalse(later.get("appearance_added", False))
                self.assertEqual(self.store.profiles()[0]["sample_count"], 3)

    def test_invalid_frame_paths_report_invalid_manual_review_while_holding(self):
        for mode in ("no_face", "multiple_faces", "image_quality", "spoof"):
            with self.subTest(mode=mode):
                service = self.make_service("review-invalid-" + mode)
                service.analyze(self.crop(1), now=10.)
                request = service.request_manual_review(now=10.1)
                result = self.invalid_frame(service, mode, now=10.2)
                self.assert_state_fields(result)
                self.assertTrue(result["manual_review_pending"])
                self.assertFalse(result["manual_review_valid"])
                self.liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
                self.recognizer.embed.return_value = np.array([1., 0., 0.], dtype=np.float32)
                for index in range(4):
                    held = service.analyze(self.crop(20 + index), now=10.3 + .1 * index)
                    self.assertFalse(held["attendance_recorded"])
                with self.assertRaises(ValueError):
                    service.confirm_manual_review(review_id=request["id"], person_id=self.person,
                        reviewer="Fixture operator", reason="Software check", confirmed=True, now=10.7)
                self.assertFalse(self.store.attendance(service.session))
                service.cancel_manual_review(request["id"])

    def test_invalid_appearance_dialog_holds_until_matching_explicit_cancel(self):
        request = self.open_request()
        self.service.reset_tracking()
        self.recognizer.embed.return_value = np.array([1., 0., 0.], dtype=np.float32)
        for index in range(4):
            result = self.service.analyze(self.crop(20 + index), now=10.2 + .1 * index)
            self.assertFalse(result["attendance_recorded"])
        self.service.cancel_appearance_request("unrelated-token")
        self.assertTrue(self.service.frame_state(now=10.6)["appearance_dialog_pending"])
        with self.assertRaises(ValueError):
            self.service.begin_registration("Another fixture")
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.6)
        self.assertFalse(self.store.attendance(self.service.session))
        self.service.cancel_appearance_request(request["id"])
        self.assertFalse(self.service.frame_state(now=10.7)["appearance_dialog_pending"])
        recorded = []
        for index in range(3):
            recorded.append(self.service.analyze(self.crop(30 + index), now=10.8 + .1 * index)["attendance_recorded"])
        self.assertEqual(recorded, [False, False, True])
        self.assertEqual(len(self.store.attendance(self.service.session)), 1)

    def test_opencv_open_or_property_error_releases_and_allows_retry(self):
        for failure in ("constructor", "property"):
            with self.subTest(failure=failure):
                args = SimpleNamespace(num_threads=2, identity_config=self.root / "unused.json", device="cpu",
                    liveness_checkpoint=self.root / "unused-live.pt", emotion_checkpoint=self.root / "unused-fer.pt",
                    database=self.root / (failure + ".sqlite3"), detector="haar", roi=None,
                    session="worker-fixture", camera=0)
                recognizer = Mock(fingerprint="a" * 64, dimension=3, threshold=.8)
                trained = {"training": {"smoke_test": False, "initialization": "random_from_scratch"}}
                liveness = Mock(threshold=.5, config=trained)
                successful_camera = Mock()
                successful_camera.isOpened.return_value = True
                successful_camera.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
                failed_camera = Mock()
                failed_camera.set.side_effect = cv2.error("Fixture camera property failure")
                first = cv2.error("Fixture camera constructor failure") if failure == "constructor" else failed_camera
                locator = Mock()
                locator.detect.return_value = []
                commands, updates = queue.Queue(), queue.Queue()
                for command in (("start", "worker-fixture"), ("start", "worker-fixture"), ("stop", None), ("quit", None)):
                    commands.put(command)
                with patch("attendance_app.recognition.FaceRecognizer", return_value=recognizer), \
                     patch("anti_spoofing.infer.LivenessDetector", return_value=liveness), \
                     patch("emotion_recognition.infer.EmotionRecognizer", return_value=Mock(payload=trained)), \
                     patch("attendance_app.camera.FaceLocator", return_value=locator), \
                     patch("cv2.VideoCapture", side_effect=[first, successful_camera]) as open_camera:
                    CameraWorker(args, commands, updates).run()
                events = [updates.get_nowait()["event"] for _ in range(updates.qsize())]
                self.assertEqual(open_camera.call_count, 2)
                self.assertIn("camera_error", events)
                self.assertIn("started", events)
                self.assertIn("stopped", events)
                self.assertNotIn("fatal", events)
                self.assertLess(events.index("camera_error"), events.index("started"))
                successful_camera.release.assert_called_once()
                if failure == "property":
                    failed_camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
