"""Worker/video scheduling regressions without webcam, models or real records."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from attendance_app.camera import FrameProcessor
from attendance_app.gui import AttendanceWindow, CameraWorker
from attendance_app.service import AttendanceService
from attendance_app.video_stream import CameraSample


class FastFakeCapture:
    def __init__(self):
        self.frames = 0
        self.releases = 0

    def isOpened(self):
        return True

    def set(self, key, value):
        return True

    def get(self, key):
        return 30.

    def read(self):
        self.frames += 1
        return True, np.full((100, 100, 3), self.frames % 255, np.uint8)

    def release(self):
        self.releases += 1


class CameraSchedulingChecks(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(num_threads=1, identity_config=Path("unused-identity.json"),
            device="cpu", liveness_checkpoint=Path("unused-live.pt"),
            emotion_checkpoint=Path("unused-expression.pt"), database=Path("unused.sqlite3"),
            detector="haar", roi=None, session="scheduling-fixture", camera=0)
        self.commands, self.updates = queue.Queue(), queue.Queue()
        self.previews, self.analyses = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.worker = CameraWorker(self.args, self.commands, self.updates, self.previews, self.analyses)
        self.store = Mock()
        self.store.attendance.return_value = []
        self.service = Mock(profiles=[], pending_name=None, pending_appearance=None,
                            session=self.args.session, review_open=False, appearance_dialog_open=False)
        self.processor = Mock()
        self.processor.process.return_value = {"status": "unknown", "boxes": [],
            "liveness": {"is_live": True}, "identity": {"status": "unknown"},
            "attendance_recorded": False}

    @contextmanager
    def dependencies(self):
        trained = {"training": {"smoke_test": False, "initialization": "random_from_scratch"}}
        with ExitStack() as stack:
            for name, value in (
                ("torch.set_num_threads", None),
                ("attendance_app.recognition.FaceRecognizer", Mock(fingerprint="a"*64, dimension=3)),
                ("anti_spoofing.infer.LivenessDetector", Mock(threshold=.5, config=trained)),
                ("emotion_recognition.infer.EmotionRecognizer", Mock(payload=trained)),
                ("attendance_app.camera.FaceLocator", Mock()),
                ("attendance_app.store.ProfileStore", self.store),
                ("attendance_app.service.AttendanceService", self.service),
                ("attendance_app.reference_import.ReferenceImporter", Mock()),
                ("attendance_app.camera.FrameProcessor", self.processor),
            ):
                stack.enter_context(patch(name, return_value=value))
            yield stack

    def assert_no_fatal(self):
        failures = [item for item in list(self.updates.queue) if item["event"] == "fatal"]
        self.assertEqual(failures, [])

    def test_preview_advances_during_slow_inference_without_new_analysis(self):
        capture = FastFakeCapture()
        inference_started, finish_inference = threading.Event(), threading.Event()

        def process(rgb, *, observed_at):
            inference_started.set()
            if not finish_inference.wait(3):
                raise RuntimeError("Fixture inference was not released")
            return {"status": "unknown", "boxes": [], "liveness": {"is_live": True}}

        self.processor.process.side_effect = process
        self.commands.put(("start", self.args.session))
        with self.dependencies() as stack:
            stack.enter_context(patch("attendance_app.video_stream.cv2.VideoCapture", return_value=capture))
            self.worker.start()
            try:
                self.assertTrue(inference_started.wait(3))
                deadline = time.monotonic() + 2
                preview = None
                while time.monotonic() < deadline:
                    try:
                        preview = self.previews.get(timeout=.05)
                    except queue.Empty:
                        continue
                    if preview["sample"].sequence >= 4:
                        break
                self.assertIsNotNone(preview)
                self.assertGreaterEqual(preview["sample"].sequence, 4)
                self.assertEqual(self.processor.process.call_count, 1)
                self.assertTrue(self.analyses.empty())
                # Preview captures cannot invoke service recognition or count
                # as attendance observations while processing is blocked.
                self.service.analyze.assert_not_called()
            finally:
                self.commands.put(("quit", None))
                finish_inference.set()
                self.worker.join(3)
            self.assertFalse(self.worker.is_alive())
        self.assertEqual(capture.releases, 1)
        self.assert_no_fatal()

    def test_one_camera_sample_is_processed_once_and_keeps_capture_timestamp(self):
        captured = time.monotonic()
        sample = CameraSample(7, captured, np.zeros((100, 100, 3), np.uint8))
        camera = Mock()
        reads = []

        def read_latest(after_sequence, timeout):
            reads.append(after_sequence)
            if len(reads) >= 3:
                self.commands.put(("quit", None))
            return sample if after_sequence < sample.sequence else None

        camera.read_latest.side_effect = read_latest
        self.commands.put(("start", self.args.session))
        with self.dependencies() as stack:
            stack.enter_context(patch("attendance_app.gui.LatestCamera", return_value=camera))
            self.worker.run()
        self.assertEqual(reads, [0, 7, 7])
        self.processor.process.assert_called_once()
        self.assertIs(self.processor.process.call_args.args[0], sample.rgb)
        self.assertEqual(self.processor.process.call_args.kwargs, {"observed_at": captured})
        camera.release.assert_called_once()
        self.assert_no_fatal()

    def test_slow_analysis_cannot_refresh_the_registration_freshness_clock(self):
        sample = CameraSample(1, 100., np.zeros((100, 100, 3), np.uint8))
        camera = Mock()
        reads = 0

        def read_latest(after_sequence, timeout):
            nonlocal reads
            reads += 1
            if reads == 1:
                self.commands.put(("register", "Fixture person"))
                return sample
            self.commands.put(("quit", None))
            return None

        camera.read_latest.side_effect = read_latest
        self.commands.put(("start", self.args.session))
        with self.dependencies() as stack:
            stack.enter_context(patch("attendance_app.gui.LatestCamera", return_value=camera))
            # Fresh when selected; old when the next command is handled. The
            # inference result must retain capture time rather than completion.
            stack.enter_context(patch("attendance_app.gui.time.monotonic", side_effect=[100.2, 102.]))
            self.worker.run()
        self.processor.process.assert_called_once()
        self.service.begin_registration.assert_not_called()
        notices = [event["message"] for event in self.updates.queue if event["event"] == "notice"]
        self.assertTrue(any("current, unrecognized face" in message for message in notices))
        self.assert_no_fatal()

    def test_stale_camera_sample_is_rejected_before_processing(self):
        camera = Mock()

        def read_latest(after_sequence, timeout):
            self.commands.put(("quit", None))
            return CameraSample(1, time.monotonic()-2, np.zeros((100, 100, 3), np.uint8))

        camera.read_latest.side_effect = read_latest
        self.commands.put(("start", self.args.session))
        with self.dependencies() as stack:
            stack.enter_context(patch("attendance_app.gui.LatestCamera", return_value=camera))
            self.worker.run()
        self.processor.process.assert_not_called()
        self.service.reset_tracking.assert_called_once()
        self.assertIn("stale", self.service.reset_tracking.call_args.args[0])
        self.assert_no_fatal()

    def test_video_replacement_does_not_drop_ordered_control_events(self):
        self.worker.publish(event="started", stream_id=1)
        for sequence in range(100):
            self.worker.preview(CameraSample(sequence, 1., None), 1)
            self.worker.publish(event="frame", sequence=sequence)
            if sequence == 50:
                self.worker.publish(event="notice", message="Fixture control")
        self.worker.publish(event="stopped")
        self.assertEqual([item["event"] for item in self.updates.queue], ["started", "notice", "stopped"])
        self.assertEqual(self.previews.get_nowait()["sample"].sequence, 99)
        self.assertEqual(self.analyses.get_nowait()["sequence"], 99)

    def test_stop_clears_pending_pixels_and_resets_analysis_generation(self):
        self.worker.service = self.service
        camera = self.worker.camera = Mock()
        self.worker.latest, self.worker.latest_time = {"status": "unknown"}, 10.
        self.worker.last_sequence = 15
        old_generation = self.worker.stream_id
        self.worker.preview(CameraSample(15, 10., None), old_generation)
        self.worker.publish(event="frame", sequence=15)
        self.worker.stop_camera()
        self.assertGreater(self.worker.stream_id, old_generation)
        self.assertTrue(self.previews.empty())
        self.assertTrue(self.analyses.empty())
        self.assertIsNone(self.worker.latest)
        self.assertEqual(self.worker.last_sequence, 0)
        self.assertIsNone(self.worker.camera)
        camera.release.assert_called_once()
        self.service.cancel_registration.assert_called_once()

    def test_stopped_or_restarted_ui_ignores_old_analysis_and_preview(self):
        # These must return before rendering pixels or changing identity fields.
        for running, old_stream in ((False, 2), (True, 1)):
            with self.subTest(running=running, old_stream=old_stream):
                marker = object()
                ui = SimpleNamespace(running=running, stream_id=2, last_analysis=marker)
                AttendanceWindow.render_frame(ui, {"stream_id": old_stream})
                AttendanceWindow.render_preview(ui, {"stream_id": old_stream})
                self.assertIs(ui.last_analysis, marker)
        marker = object()
        ui = SimpleNamespace(running=True, stream_id=2, last_analysis=marker)
        AttendanceWindow.render_frame(ui, {"stream_id": 2, "captured_at": time.monotonic()-2})
        AttendanceWindow.render_preview(ui, {"stream_id": 2,
            "sample": CameraSample(1, time.monotonic()-2, None)})
        self.assertIs(ui.last_analysis, marker)

    def test_frame_processor_forwards_observation_time_to_service(self):
        locator = Mock()
        locator.detect.return_value = [(0, 0, 100, 100)]
        service = Mock()
        service.analyze.return_value = {"status": "unknown"}
        checkerboard = (np.indices((100, 100)).sum(axis=0) % 2 * 150 + 50).astype(np.uint8)
        rgb = np.repeat(checkerboard[:, :, None], 3, axis=2)
        result = FrameProcessor(locator, service).process(rgb, observed_at=123.25)
        self.assertEqual(result["status"], "unknown")
        service.analyze.assert_called_once()
        self.assertEqual(service.analyze.call_args.kwargs, {"now": 123.25, "max_observation_age": 1.})

    def test_slow_model_cannot_commit_enrollment_or_attendance_from_aged_frame(self):
        vector = np.array([1., 0., 0.], dtype=np.float32)
        crop = Image.new("RGB", (100, 100), (80, 100, 120))
        for mode in ("registration", "automatic_attendance"):
            with self.subTest(mode=mode):
                store = Mock()
                store.profiles.return_value = []
                recognizer = Mock(dimension=3, threshold=.8)
                recognizer.embed.return_value = vector
                recognizer.match.return_value = ({"status": "unknown"} if mode == "registration"
                    else {"status": "known", "person_id": "fixture-person", "cosine": .95})
                liveness = Mock()
                liveness.predict_crop.return_value = {"is_live": True, "live_score": .9}
                service = AttendanceService(store, recognizer, liveness, Mock(), "fixture")
                if mode == "registration":
                    service.begin_registration("Fixture person")
                    service.enrollment_samples = [vector.copy(), vector.copy()]
                    service.enrollment_hashes = {"previous-a", "previous-b"}
                else:
                    service.streak, service.last_person = 2, "fixture-person"
                with patch("attendance_app.service.time.monotonic", return_value=102.):
                    result = service.analyze(crop, now=100., max_observation_age=1.)
                self.assertEqual(result["status"], "observation_stale")
                self.assertFalse(result["attendance_recorded"])
                store.enroll.assert_not_called()
                store.record_attendance.assert_not_called()
                self.assertIsNone(service.review.latest)
                self.assertEqual(service.enrollment_samples, [])
                self.assertEqual(service.streak, 0)


if __name__ == "__main__":
    unittest.main()
