"""Camera ownership, setup fault attribution and guidance; no physical camera."""
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from attendance_app.camera import FrameProcessor
from attendance_app.device_check import camera_index, probe_camera
from attendance_app.diagnostics import frame_details, frame_guidance
from attendance_app.gui import CameraWorker


class CameraProbeChecks(unittest.TestCase):
    def test_index_rejects_invalid_input_without_opening_device(self):
        for value in (None, True, False, -1, 100, 1.5, "1.0", "invalid", float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera_index(value)
        self.assertEqual(camera_index("2"), 2)
        with patch("cv2.VideoCapture") as opened:
            self.assertFalse(probe_camera(-1)["ok"])
            opened.assert_not_called()

    def test_probe_reads_only_selected_device_releases_and_returns_no_pixels(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
        with patch("cv2.VideoCapture", return_value=camera) as opened:
            report = probe_camera(3)
        opened.assert_called_once_with(3)
        camera.read.assert_called_once()
        camera.release.assert_called_once()
        self.assertTrue(report["ok"])
        self.assertEqual((report["width"], report["height"]), (64, 48))
        self.assertFalse(any(isinstance(v, np.ndarray) for v in report.values()))

    def test_open_read_and_invalid_frame_failures_release_for_retry(self):
        for opened, read in ((False, (False, None)), (True, (False, None)),
                             (True, (True, np.zeros((10, 10), dtype=np.uint8))),
                             (True, (True, np.zeros((10, 10, 3), dtype=np.float32)))):
            camera = Mock()
            camera.isOpened.return_value, camera.read.return_value = opened, read
            with patch("cv2.VideoCapture", return_value=camera):
                self.assertFalse(probe_camera(0)["ok"])
            camera.release.assert_called_once()
        with patch("cv2.VideoCapture", side_effect=cv2.error("Fixture constructor failure")):
            self.assertFalse(probe_camera(0)["ok"])

    def test_opencv_read_and_release_exceptions_are_reported(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = cv2.error("Fixture read failure")
        with patch("cv2.VideoCapture", return_value=camera):
            self.assertFalse(probe_camera(0)["ok"])
        camera.release.assert_called_once()
        camera.read.side_effect = None
        camera.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
        camera.release.side_effect = cv2.error("Fixture release failure")
        with patch("cv2.VideoCapture", return_value=camera):
            report = probe_camera(0)
        self.assertFalse(report["ok"])
        self.assertIn("release failed", report["message"])


class FrameGuidanceChecks(unittest.TestCase):
    def test_quality_causes_are_separate_and_do_not_run_identity_or_pad(self):
        locator, service = Mock(), Mock()
        locator.detect.return_value = [(0, 0, 100, 100)]
        service.frame_state.return_value = {}
        checker = (np.indices((100, 100)).sum(axis=0) % 2).astype(np.uint8)
        for gray, reason, hint in ((checker*30, "dark", "front"),
                                   (225+checker*30, "bright", "glare"),
                                   (np.full((100, 100), 100, np.uint8), "blur", "focus")):
            result = FrameProcessor(locator, service).process(np.repeat(gray[:, :, None], 3, axis=2))
            self.assertEqual(result["status"], "image_quality")
            self.assertIn(reason, result["quality_reasons"])
            self.assertIn(hint, frame_guidance(result))
            self.assertIn("Brightness", frame_details(result))
        service.analyze.assert_not_called()

    def test_good_quality_telemetry_preserves_same_rgb_crop(self):
        locator, service = Mock(), Mock()
        locator.detect.return_value = [(0, 0, 100, 100)]
        service.analyze.return_value = {"status": "unknown", "identity": None}
        rgb = np.random.default_rng(5).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = FrameProcessor(locator, service).process(rgb)
        self.assertEqual(result["quality_reasons"], [])
        np.testing.assert_array_equal(np.asarray(service.analyze.call_args.args[0]), rgb)
        self.assertIn("same profile", frame_guidance(result))

    def test_stage_specific_guidance_does_not_call_failed_pad_recognition(self):
        self.assertIn("Expression error: fixture", frame_details({"status": "spoof", "expression_error": "fixture"}))
        self.assertIn("liveness did not pass", frame_guidance({"status": "spoof"}))
        self.assertIn("whole face", frame_guidance({"status": "no_face"}))
        self.assertIn("only one", frame_guidance({"status": "multiple_faces"}))
        self.assertIn("Move closer", frame_guidance({"status": "face_too_small_or_outside"}))
        self.assertIn("already stored", frame_guidance({"status": "attendance_confirmed"}))


class WorkerDiagnosticChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.args = SimpleNamespace(num_threads=2, identity_config=self.root/"identity.json", device="cpu",
            liveness_checkpoint=self.root/"live.pt", emotion_checkpoint=self.root/"fer.pt",
            database=self.root/"fixture.sqlite3", detector="haar", roi=None, session="fixture", camera=0)
        trained = {"training": {"smoke_test": False, "initialization": "random_from_scratch"}}
        self.recognizer = Mock(fingerprint="a"*64, dimension=3, threshold=.8)
        self.live = Mock(threshold=.5, config=trained)
        self.emotion = Mock(payload=trained)
        self.locator = Mock()
        self.locator.detect.return_value = []

    def tearDown(self):
        self.temp.cleanup()

    def run_worker(self, commands, failure=None, cameras=None):
        from contextlib import ExitStack
        requests, updates = queue.Queue(), queue.Queue()
        for item in commands:
            requests.put(item)
        paths = {"Identity": "attendance_app.recognition.FaceRecognizer", "Liveness": "anti_spoofing.infer.LivenessDetector",
                 "Expression": "emotion_recognition.infer.EmotionRecognizer", "Detector": "attendance_app.camera.FaceLocator"}
        models = {"Identity": self.recognizer, "Liveness": self.live, "Expression": self.emotion, "Detector": self.locator}
        with ExitStack() as stack:
            for key, path in paths.items():
                stack.enter_context(patch(path, side_effect=EOFError("Fixture corrupt checkpoint") if key == failure else None,
                                          return_value=models[key]))
            if failure == "Profiles":
                stack.enter_context(patch("attendance_app.store.ProfileStore", side_effect=PermissionError("Fixture access denied")))
            opened = stack.enter_context(patch("cv2.VideoCapture", side_effect=cameras))
            CameraWorker(self.args, requests, updates).run()
        return list(updates.queue), opened

    def test_setup_fault_names_exact_component_and_stops_before_camera(self):
        for component in ("Identity", "Liveness", "Expression", "Detector", "Profiles"):
            with self.subTest(component=component):
                events, opened = self.run_worker([], failure=component)
                fatal = next(e for e in events if e["event"] == "fatal")
                self.assertEqual(fatal["component"], component)
                self.assertEqual([e["component"] for e in events if e["event"] == "component" and e["status"] == "Error"], [component])
                self.assertNotIn("ready", [e["event"] for e in events])
                opened.assert_not_called()
                self.assertFalse(self.args.database.exists())

    def test_uncalibrated_pad_is_attributed_to_liveness(self):
        self.live.threshold = None
        events, opened = self.run_worker([])
        fatal = next(e for e in events if e["event"] == "fatal")
        self.assertEqual(fatal["component"], "Liveness")
        self.assertIn("calibrated", fatal["message"])
        opened.assert_not_called()

    def test_selected_index_changes_only_across_stopped_camera_runs(self):
        cameras = [Mock(), Mock()]
        for camera in cameras:
            camera.isOpened.return_value = True
            camera.read.return_value = (True, np.zeros((100, 100, 3), np.uint8))
        events, opened = self.run_worker([("start", {"session": "first", "camera": 2}), ("stop", None),
            ("start", {"session": "second", "camera": 4}), ("quit", None)], cameras=cameras)
        self.assertEqual([call.args[0] for call in opened.call_args_list], [2, 4])
        self.assertEqual([e["camera"] for e in events if e["event"] == "started"], [2, 4])
        self.assertNotIn("fatal", [e["event"] for e in events])
        for camera in cameras:
            camera.release.assert_called_once()

    def test_bad_index_does_not_open_and_worker_can_retry(self):
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((100, 100, 3), np.uint8))
        events, opened = self.run_worker([("start", {"session": "fixture", "camera": True}),
            ("start", {"session": "fixture", "camera": 1}), ("quit", None)], cameras=[camera])
        opened.assert_called_once_with(1)
        self.assertIn("camera_error", [e["event"] for e in events])
        self.assertIn("started", [e["event"] for e in events])
        self.assertNotIn("fatal", [e["event"] for e in events])


if __name__ == "__main__":
    unittest.main()
