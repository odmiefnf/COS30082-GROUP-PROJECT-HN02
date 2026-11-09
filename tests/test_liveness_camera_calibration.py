"""Validation operating-point selection must preserve the spoof bound."""
import unittest

from tools.calibrate_liveness_camera import select_at_apcer


class CameraCalibrationChecks(unittest.TestCase):
    def test_maximizes_live_acceptance_without_exceeding_spoof_bound(self):
        threshold, stats = select_at_apcer([0, 0, 0, 1, 1, 1], [.1, .3, .9, .2, .6, .8], 1/3)
        self.assertAlmostEqual(threshold, .6)
        self.assertAlmostEqual(stats["APCER"], 1/3)
        self.assertAlmostEqual(stats["BPCER"], 1/3)

    def test_tied_scores_cannot_split_live_and_spoof_or_violate_bound(self):
        _, stats = select_at_apcer([0, 0, 1, 1], [.5, .5, .5, .9], 0.)
        self.assertEqual(stats["APCER"], 0.)
        self.assertEqual(stats["BPCER"], .5)
        for bound in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                select_at_apcer([0, 1], [.1, .9], bound)


if __name__ == "__main__":
    unittest.main()
