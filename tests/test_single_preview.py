"""Single-camera presentation checks using synthetic RGB, without Tk or camera."""
from copy import copy
import unittest
from unittest.mock import Mock, patch

import numpy as np

from attendance_app.gui import AttendanceWindow
from attendance_app.video_stream import CameraSample


class SinglePreviewChecks(unittest.TestCase):
    def setUp(self):
        self.live_color = np.array([30, 60, 90], dtype=np.uint8)
        self.analysis_color = np.array([220, 40, 60], dtype=np.uint8)
        self.live = np.broadcast_to(self.live_color, (120, 200, 3)).copy()
        self.analyzed = np.broadcast_to(self.analysis_color, self.live.shape).copy()
        self.analysis = {"stream_id": 1, "sequence": 10, "captured_at": 100.,
                         "rgb": self.analyzed,
                         "result": {"boxes": [[40, 30, 50, 50]], "status": "known",
                                    "identity": {"status": "known", "person_id": "fixture-person", "name": "Fixture"},
                                    "liveness": {"is_live": True}}}
        self.ui = AttendanceWindow.__new__(AttendanceWindow)
        self.ui.running, self.ui.stream_id = True, 1
        self.ui.last_analysis = self.analysis
        self.ui.canvas = Mock()
        self.ui.canvas.winfo_width.return_value = 200
        self.ui.canvas.winfo_height.return_value = 120
        self.ui.canvas.create_image.return_value = 7
        self.ui.canvas_image = None
        self.ui.photo = None
        self.ui.preview_times = []
        self.ui.analysis_fps = 20.
        self.ui.speed = Mock()
        self.ui.identity_text, self.ui.live_text, self.ui.expression_text = Mock(), Mock(), Mock()

    def render(self, sequence=11, captured_at=100.05, now=100.1):
        rendered = []

        def photo(image):
            rendered.append(np.asarray(image).copy())
            return image

        with patch("attendance_app.gui.time.monotonic", return_value=now), \
             patch("attendance_app.gui.ImageTk.PhotoImage", side_effect=photo):
            self.ui.render_preview({"stream_id": 1,
                "sample": CameraSample(sequence, captured_at, self.live)})
        self.assertEqual(len(rendered), 1)
        return rendered[0]

    def test_one_live_image_has_recognition_outline_and_preserves_result_cards(self):
        first = self.render()
        colors = {tuple(pixel) for pixel in first.reshape(-1, 3)}
        self.assertEqual(colors, {tuple(self.live_color), (74, 222, 128)})
        self.assertFalse(np.any(np.all(first == self.analysis_color, axis=2)),
                         "A separate analyzed image must not be pasted into the live image")
        self.assertEqual(first.shape, self.live.shape)
        np.testing.assert_array_equal(self.live, np.broadcast_to(self.live_color, self.live.shape))
        np.testing.assert_array_equal(self.analyzed, np.broadcast_to(self.analysis_color, self.analyzed.shape))
        # Repeated refresh updates one canvas image, rather than creating
        # additional views, and cannot replace identity/PAD/expression results.
        self.render(sequence=12, captured_at=100.1, now=100.15)
        self.ui.canvas.create_image.assert_called_once()
        self.ui.canvas.itemconfigure.assert_called_once()
        for value in (self.ui.identity_text, self.ui.live_text, self.ui.expression_text):
            value.set.assert_not_called()

    def test_outdated_or_incompatible_face_observations_do_not_overlay_new_pixels(self):
        cases = [
            ({"captured_at": 99.8}, 11, 100.05),
            ({"stream_id": 2}, 11, 100.05),
            ({"rgb": np.zeros((80, 120, 3), dtype=np.uint8)}, 11, 100.05),
            ({}, 9, 100.05),
            ({}, 11, 99.95),
        ]
        for changes, sequence, captured_at in cases:
            with self.subTest(changes=tuple(changes), sequence=sequence, captured_at=captured_at):
                self.ui.last_analysis = {**self.analysis, **changes}
                actual = self.render(sequence=sequence, captured_at=captured_at)
                np.testing.assert_array_equal(actual, self.live)

    def test_latest_face_loss_removes_the_guide_on_the_same_single_view(self):
        before = self.render()
        self.assertFalse(np.array_equal(before, self.live))
        lost = copy(self.analysis)
        lost.update(sequence=11, captured_at=100.08, result={"status": "no_face", "boxes": []})
        self.ui.last_analysis = lost
        after = self.render(sequence=12, captured_at=100.1, now=100.12)
        np.testing.assert_array_equal(after, self.live)
        self.ui.canvas.create_image.assert_called_once()
        self.ui.canvas.itemconfigure.assert_called_once()

    def test_outline_is_yellow_until_identity_is_known_and_liveness_passes(self):
        known = {"status": "known", "person_id": "fixture-person", "name": "Fixture"}
        cases = [
            ("unknown", {"status": "unknown", "person_id": None}, True, (251, 191, 36)),
            ("ambiguous", {"status": "ambiguous", "person_id": None}, True, (251, 191, 36)),
            ("spoof", known, False, (251, 191, 36)),
            ("liveness_unavailable", known, None, (251, 191, 36)),
            ("known", known, True, (74, 222, 128)),
            ("attendance_confirmed", known, True, (74, 222, 128)),
        ]
        for status, identity, is_live, color in cases:
            with self.subTest(status=status):
                self.ui.last_analysis = {**self.analysis, "result": {
                    "status": status, "identity": identity, "liveness": {"is_live": is_live},
                    "boxes": [[40, 30, 50, 50]]}}
                actual = self.render()
                self.assertEqual({tuple(p) for p in actual.reshape(-1, 3)},
                                 {tuple(self.live_color), color})


if __name__ == "__main__":
    unittest.main()
