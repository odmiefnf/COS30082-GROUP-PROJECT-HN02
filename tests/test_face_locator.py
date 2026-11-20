"""Detection regressions; synthetic frames do not measure face-model accuracy."""
from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from attendance_app.camera import FaceLocator, FrameProcessor


class FaceLocatorChecks(unittest.TestCase):
    def locator(self, primary, alternate):
        locator = FaceLocator("haar")
        locator.cascade = Mock()
        locator.cascade.detectMultiScale.return_value = primary
        locator.alternate = Mock()
        locator.alternate.detectMultiScale.return_value = alternate
        return locator

    def test_two_cascades_one_face_send_one_original_rgb_crop(self):
        locator = self.locator([(40, 30, 120, 120)], [(43, 33, 116, 116)])
        service = Mock()
        service.frame_state.return_value = {}
        service.analyze.return_value = {"status": "unknown"}
        rgb = np.random.default_rng(5).integers(0, 256, (220, 280, 3), dtype=np.uint8)
        result = FrameProcessor(locator, service).process(rgb)
        self.assertEqual(result["boxes"], [[40, 30, 120, 120]])
        self.assertEqual(result["status"], "unknown")
        service.analyze.assert_called_once()
        np.testing.assert_array_equal(np.asarray(service.analyze.call_args.args[0]), rgb[30:150, 40:160])

    def test_alternate_recovers_a_current_frame_face_when_primary_misses(self):
        locator = self.locator([], [(40, 30, 120, 120)])
        rgb = np.zeros((220, 280, 3), dtype=np.uint8)
        self.assertEqual(locator.detect(rgb), [(40, 30, 120, 120)])
        locator.alternate.detectMultiScale.return_value = []
        self.assertEqual(locator.detect(rgb), [])

    def test_partly_overlapping_people_are_not_merged_and_block_models(self):
        locator = self.locator([(30, 40, 120, 120)], [(110, 40, 120, 120), (32, 42, 118, 118)])
        service = Mock()
        service.frame_state.return_value = {}
        rgb = np.zeros((220, 280, 3), dtype=np.uint8)
        result = FrameProcessor(locator, service).process(rgb)
        self.assertEqual(result["status"], "multiple_faces")
        self.assertEqual(len(result["boxes"]), 2)
        service.analyze.assert_not_called()
        service.reset_tracking.assert_called_once()

    def test_nested_inner_cascade_box_on_same_face_is_deduplicated(self):
        locator = self.locator([(30, 30, 180, 180)], [(60, 60, 120, 120)])
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(locator.detect(rgb), [(30, 30, 180, 180)])

    def test_missing_detection_resets_tracking_instead_of_reusing_old_box(self):
        locator = self.locator([(40, 30, 120, 120)], [])
        service = Mock()
        service.frame_state.return_value = {}
        service.analyze.return_value = {"status": "unknown"}
        rgb = np.random.default_rng(6).integers(0, 256, (220, 280, 3), dtype=np.uint8)
        processor = FrameProcessor(locator, service)
        self.assertEqual(processor.process(rgb)["status"], "unknown")
        locator.cascade.detectMultiScale.return_value = []
        result = processor.process(rgb)
        self.assertEqual(result["status"], "no_face")
        self.assertEqual(result["boxes"], [])
        service.analyze.assert_called_once()
        service.reset_tracking.assert_called_once()

    def test_resize_returns_native_coordinates_and_uses_native_face_minimum(self):
        locator = self.locator([(100, 50, 100, 100)], [])
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertEqual(locator.detect(rgb), [(160, 80, 160, 160)])
        image = locator.cascade.detectMultiScale.call_args.args[0]
        self.assertEqual(image.shape, (450, 800))
        self.assertEqual(locator.cascade.detectMultiScale.call_args.kwargs["minSize"], (50, 50))


class YuNetLocatorChecks(unittest.TestCase):
    def test_real_model_loads_and_finds_no_face_in_a_blank_frame(self):
        self.assertEqual(FaceLocator().detect(np.zeros((480, 640, 3), dtype=np.uint8)), [])

    def test_boxes_become_native_haar_sized_squares_and_small_faces_are_dropped(self):
        locator = FaceLocator()
        locator.net = Mock()
        # x, y, w, h on the 800-wide detector image; 14 trailing landmark/score values.
        locator.net.detect.return_value = (1, np.array([[100, 50, 80, 100] + [0] * 11,
                                                        [500, 50, 30, 40] + [0] * 11], dtype=np.float32))
        boxes = locator.detect(np.zeros((720, 1280, 3), dtype=np.uint8))
        locator.net.setInputSize.assert_called_once_with((800, 450))
        side = round(.91 * 160)  # longest side 100 detector px -> 160 native px
        self.assertEqual(boxes, [(round(224 - side/2), round(160 - side/2), side, side)])


if __name__ == "__main__":
    unittest.main()
