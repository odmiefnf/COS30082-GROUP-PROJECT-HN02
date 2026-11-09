"""Train augmentation contracts; synthetic arrays do not measure PAD accuracy."""
from __future__ import annotations

import math
import unittest

import torch

from anti_spoofing.augmentation import augment_camera_batch


class CameraAugmentationChecks(unittest.TestCase):
    def test_cached_source_is_unchanged_and_seed_reproduces_output(self):
        source = torch.linspace(-1., 1., 4*3*32*48).reshape(4, 3, 32, 48)
        saved = source.clone()
        torch.manual_seed(42)
        first = augment_camera_batch(source)
        torch.testing.assert_close(source, saved, rtol=0, atol=0)
        self.assertNotEqual(first.data_ptr(), source.data_ptr())
        torch.manual_seed(42)
        second = augment_camera_batch(source)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.manual_seed(43)
        self.assertFalse(torch.equal(first, augment_camera_batch(source)))

    def test_shape_dtype_device_range_and_noncontiguous_input(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = torch.zeros((3, 3, 32, 80), dtype=dtype)[..., ::2]
                self.assertFalse(source.is_contiguous())
                result = augment_camera_batch(source)
                self.assertEqual(result.shape, source.shape)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.device, source.device)
                self.assertTrue(bool(torch.isfinite(result).all()))
                self.assertTrue(bool(((result >= -1) & (result <= 1)).all()))
                self.assertEqual(int(torch.count_nonzero(source)), 0)
        tiny = torch.zeros((1, 3, 1, 1))
        self.assertEqual(augment_camera_batch(tiny).shape, tiny.shape)

    def test_upper_face_patches_are_small_and_never_cover_full_width(self):
        torch.manual_seed(7)
        height, width = 32, 48
        output = augment_camera_batch(torch.zeros((64, 3, height, width)))
        patches = 0
        for sample in output:
            values, counts = torch.unique(sample[0], return_counts=True)
            self.assertLessEqual(len(values), 2)
            if len(values) == 1:
                continue
            patches += 1
            background = values[counts.argmax()]
            points = (sample[0] != background).nonzero()
            self.assertGreaterEqual(len(points), math.ceil(.01*height*width))
            self.assertLessEqual(len(points), math.floor(.06*height*width))
            self.assertLess(int(points[:, 1].max()-points[:, 1].min()+1), width/2)
            self.assertLess(int(points[:, 0].max()), .6*height)
        self.assertGreater(patches, 0)
        self.assertLess(patches, len(output))

    def test_invalid_pixels_or_layout_are_rejected(self):
        invalid = [None, torch.zeros((2, 3, 32, 32), dtype=torch.uint8),
                   torch.zeros((3, 32, 32)), torch.zeros((2, 1, 32, 32)),
                   torch.zeros((0, 3, 32, 32)), torch.full((1, 3, 16, 16), float("nan")),
                   torch.full((1, 3, 16, 16), float("inf")), torch.full((1, 3, 16, 16), 1.1)]
        for value in invalid:
            with self.subTest(value_type=type(value)), self.assertRaises(ValueError):
                augment_camera_batch(value)


if __name__ == "__main__":
    unittest.main()
