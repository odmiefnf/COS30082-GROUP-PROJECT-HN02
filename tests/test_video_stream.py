"""Capture scheduling checks; fake frames never access a real webcam."""
from __future__ import annotations

import threading
import time
import queue
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from attendance_app.video_stream import LatestCamera, put_latest


class FakeCamera:
    def __init__(self, fail_after=None, invalid=None):
        self.fail_after, self.invalid = fail_after, invalid
        self.read_count = 0
        self.calls = []
        self.releases = 0

    def note(self, operation):
        self.calls.append((operation, threading.get_ident()))

    def isOpened(self):
        self.note("isOpened")
        return True

    def set(self, key, value):
        self.note("set")
        return True

    def get(self, key):
        self.note("get")
        return 30.

    def read(self):
        self.note("read")
        self.read_count += 1
        if self.fail_after is not None and self.read_count > self.fail_after:
            return False, None
        if self.invalid is not None:
            return True, self.invalid
        # BGR ordering makes RGB conversion observable.
        frame = np.full((12, 20, 3), self.read_count % 255, dtype=np.uint8)
        frame[0, 0] = [1, 2, 3]
        return True, frame

    def release(self):
        self.note("release")
        self.releases += 1


class LatestCameraChecks(unittest.TestCase):
    def test_latest_queue_retains_newest_and_never_requires_consumer_progress(self):
        slot = queue.Queue(maxsize=1)
        for sequence in range(100):
            put_latest(slot, sequence)
        self.assertEqual(slot.qsize(), 1)
        self.assertEqual(slot.get_nowait(), 99)
        put_latest(slot, 100)
        self.assertEqual(slot.get_nowait(), 100)
        with self.assertRaises(ValueError):
            put_latest(queue.Queue(), "control events require ordered delivery")

    def test_slow_consumer_gets_latest_unique_sequence_and_rgb(self):
        camera = FakeCamera()
        third = threading.Event()
        observed = []

        def preview(sample):
            observed.append(sample.sequence)
            if sample.sequence >= 3:
                third.set()

        with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
            stream = LatestCamera(0, preview).open()
            try:
                self.assertTrue(third.wait(2))
                sample = stream.read_latest(timeout=0)
                self.assertGreaterEqual(sample.sequence, 3)
                np.testing.assert_array_equal(sample.rgb[0, 0], [3, 2, 1])
                self.assertFalse(sample.rgb.flags.writeable)
                self.assertLessEqual(sample.captured_at, time.monotonic())
                following = stream.read_latest(sample.sequence, timeout=0)
                if following is not None:
                    self.assertGreater(following.sequence, sample.sequence)
                self.assertEqual(stream.metadata["width"], 20)
                self.assertEqual(stream.metadata["height"], 12)
                self.assertEqual(stream.metadata["requested_width"], 960)
                self.assertEqual(stream.metadata["requested_height"], 720)
            finally:
                stream.close()
        self.assertEqual(camera.releases, 1)
        self.assertEqual(observed, sorted(set(observed)))

    def test_capture_handle_is_owned_by_one_background_thread_and_close_is_idempotent(self):
        camera = FakeCamera()
        got_frame = threading.Event()
        constructor_threads = []

        def construct(index):
            constructor_threads.append(threading.get_ident())
            return camera

        with patch("attendance_app.video_stream.cv2.VideoCapture", side_effect=construct):
            stream = LatestCamera(0, lambda sample: got_frame.set()).open()
            self.assertTrue(got_frame.wait(2))
            self.assertTrue(stream.isOpened())
            stream.release()
            count = camera.read_count
            stream.close()
            self.assertFalse(stream.isOpened())
            self.assertIsNone(stream.read_latest(timeout=0))
            self.assertEqual(camera.read_count, count)
            with self.assertRaises(RuntimeError):
                stream.open()
        owners = {owner for _, owner in camera.calls} | set(constructor_threads)
        self.assertEqual(len(owners), 1)
        self.assertNotIn(threading.get_ident(), owners)
        self.assertEqual(camera.releases, 1)

    def test_open_and_configuration_errors_release_before_retry(self):
        for mode in ("constructor", "not_open", "property"):
            with self.subTest(mode=mode):
                camera = Mock()
                camera.isOpened.return_value = mode != "not_open"
                if mode == "property":
                    camera.set.side_effect = cv2.error("fixture property error")
                outcome = cv2.error("fixture open error") if mode == "constructor" else camera
                with patch("attendance_app.video_stream.cv2.VideoCapture", side_effect=[outcome]):
                    stream = LatestCamera(0)
                    with self.assertRaisesRegex(RuntimeError, "Camera open"):
                        stream.open()
                    self.assertFalse(stream.isOpened())
                    stream.close()
                if mode != "constructor":
                    camera.release.assert_called_once()

    def test_final_unread_sample_survives_later_read_error_then_error_is_raised(self):
        camera = FakeCamera(fail_after=1)
        with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
            stream = LatestCamera(0).open()
            try:
                # Join only for this finite fake source, proving failure cleanup.
                stream._thread.join(2)
                self.assertFalse(stream._thread.is_alive())
                sample = stream.read_latest(timeout=0)
                self.assertEqual(sample.sequence, 1)
                with self.assertRaisesRegex(RuntimeError, "Camera read"):
                    stream.read_latest(sample.sequence, timeout=0)
                self.assertEqual(camera.releases, 1)
            finally:
                stream.close()

    def test_invalid_camera_pixels_fail_without_emitting_preview(self):
        for invalid in (np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 3)),
                        np.zeros((0, 4, 3), dtype=np.uint8)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                camera = FakeCamera(invalid=invalid)
                preview = Mock()
                with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
                    stream = LatestCamera(0, preview).open()
                    try:
                        with self.assertRaisesRegex(RuntimeError, "valid color frames"):
                            stream.read_latest(timeout=2)
                        stream._thread.join(2)
                        self.assertEqual(camera.releases, 1)
                        preview.assert_not_called()
                    finally:
                        stream.close()

    def test_close_wakes_waiting_consumer_without_returning_old_pixels(self):
        camera = FakeCamera()
        arrived = threading.Event()
        with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
            stream = LatestCamera(0, lambda sample: arrived.set()).open()
            self.assertTrue(arrived.wait(2))
            # An impossible future sequence waits until the stream is closed.
            result = []
            waiter = threading.Thread(target=lambda: result.append(stream.read_latest(10**9, timeout=5)))
            waiter.start()
            stream.close()
            waiter.join(2)
            self.assertFalse(waiter.is_alive())
            self.assertEqual(result, [None])

    def test_capture_pacing_prevents_a_fast_driver_from_busy_spinning(self):
        camera = FakeCamera()
        fifth = threading.Event()
        timestamps = []

        def preview(sample):
            timestamps.append(sample.captured_at)
            if len(timestamps) >= 5:
                fifth.set()

        with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
            stream = LatestCamera(0, preview, fps=30).open()
            try:
                self.assertTrue(fifth.wait(2))
            finally:
                stream.close()
        self.assertGreaterEqual(timestamps[4] - timestamps[0], .12)

    def test_validation_and_read_before_open(self):
        for fields in ({"index": True}, {"index": -1}, {"index": 100},
                       {"index": 0, "width": 0}, {"index": 0, "height": 1.5},
                       {"index": 0, "fps": float("nan")}, {"index": 0, "fps": 0}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                LatestCamera(**fields)
        stream = LatestCamera(0)
        with self.assertRaisesRegex(RuntimeError, "Open the camera"):
            stream.read_latest()
        for fields in ({"after_sequence": -1}, {"after_sequence": True},
                       {"timeout": float("inf")}, {"timeout": -1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                stream.read_latest(**fields)
        stream.close()

    def test_release_failure_is_reported_without_releasing_handle_twice(self):
        camera = FakeCamera()
        camera.release = Mock(side_effect=cv2.error("fixture release failure"))
        arrived = threading.Event()
        with patch("attendance_app.video_stream.cv2.VideoCapture", return_value=camera):
            stream = LatestCamera(0, lambda sample: arrived.set()).open()
            self.assertTrue(arrived.wait(2))
            with self.assertRaisesRegex(RuntimeError, "Camera release"):
                stream.close()
            stream.close()
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
