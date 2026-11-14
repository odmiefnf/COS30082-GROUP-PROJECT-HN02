"""Validation operating-point selection must preserve the spoof bound."""
import unittest

from tools.calibrate_liveness_camera import select_at_apcer, select_at_bpcer


class CameraCalibrationChecks(unittest.TestCase):
    def test_finite_upper_endpoint_is_considered_without_rejecting_score_one(self):
        for selector, bound in ((select_at_apcer, 0.), (select_at_bpcer, 1.)):
            threshold, stats = selector([0, 1], [.9, .2], bound)
            self.assertEqual(threshold, 1.)
            self.assertEqual(stats["APCER"], 0.)
            self.assertEqual(stats["BPCER"], 1.)
        with self.assertRaises(ValueError):
            select_at_apcer([0, 1], [1., 1.], 0.)
        _, stats = select_at_bpcer([0, 1], [1., 1.], 1.)
        self.assertEqual(stats["APCER"], 1.)
        self.assertEqual(stats["BPCER"], 0.)

    def test_minimizes_spoof_acceptance_with_fixed_genuine_rejection(self):
        threshold, stats = select_at_bpcer([0, 0, 0, 1, 1, 1], [.1, .3, .9, .2, .6, .8], 1/3)
        self.assertAlmostEqual(threshold, .6)
        self.assertAlmostEqual(stats["APCER"], 1/3)
        self.assertAlmostEqual(stats["BPCER"], 1/3)
        # The all-live operating point remains feasible at zero rejection.
        _, stats = select_at_bpcer([0, 1], [1., 0.], 0.)
        self.assertEqual(stats["BPCER"], 0.)
        self.assertEqual(stats["APCER"], 1.)

    def test_bpcer_bound_respects_ties_and_rejects_invalid_input(self):
        _, stats = select_at_bpcer([0, 0, 1, 1], [.5, .5, .5, .9], 0.)
        self.assertEqual(stats["APCER"], 1.)
        self.assertEqual(stats["BPCER"], 0.)
        for bound in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                select_at_bpcer([0, 1], [.1, .9], bound)

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
