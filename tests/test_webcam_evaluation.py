"""Evaluation ground truth, denominators and isolated recording; no webcam use."""
import hashlib
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from attendance_app.evaluation import (TrialAccumulator, collect_frames, gallery_snapshot,
                                     report_header, validate_ground_truth, validate_presentation)
from attendance_app.store import ProfileStore


def result(status="unknown", *, identity=None, live=None, quality=True, attendance=False):
    value = {"status": status, "boxes": [(0, 0, 100, 100)], "identity": identity,
             "liveness": live, "attendance_recorded": attendance}
    if quality:
        value.update(sharpness=100., brightness=100., quality_reasons=[])
    return value


class GroundTruthChecks(unittest.TestCase):
    def test_attack_presentation_is_declared_not_inferred(self):
        for value in ("phone-still-photo", "recorded-video"):
            self.assertEqual(validate_presentation("screen-replay", value), value)
        self.assertEqual(validate_presentation("print", "printed-photo"), "printed-photo")
        self.assertIsNone(validate_presentation("glasses", None))
        for condition, value in (("print", None), ("screen-replay", None),
                                 ("screen-replay", "printed-photo"), ("glasses", "recorded-video")):
            with self.assertRaises(ValueError):
                validate_presentation(condition, value)

    def test_exact_operator_profile_and_unknown_conditions_are_enforced(self):
        profiles = [{"id": "a", "name": "Fixture A"}]
        for condition in ("glasses", "no-glasses", "screen-replay", "print"):
            self.assertEqual(validate_ground_truth(condition, "fixture a", profiles), "a")
            for name in (None, "", "Someone else"):
                with self.assertRaises(ValueError):
                    validate_ground_truth(condition, name, profiles)
        self.assertIsNone(validate_ground_truth("unknown", None, profiles))
        with self.assertRaises(ValueError):
            validate_ground_truth("unknown", "Fixture A", profiles)
        with self.assertRaises(ValueError):
            validate_ground_truth("invented", "Fixture A", profiles)
        with self.assertRaises(ValueError):
            validate_ground_truth("glasses", "Fixture A", profiles*2)
        with self.assertRaises(ValueError):
            validate_ground_truth("unknown", None, [])


class TrialCountChecks(unittest.TestCase):
    def test_nonfinite_or_missing_pad_scores_never_enter_scored_denominator(self):
        for score in (float("nan"), float("inf"), -.1, 1.1, True, "0.9", None):
            stats = TrialAccumulator("glasses", "a")
            with self.assertRaises(ValueError):
                stats.observe(result(live={"is_live": True, "live_score": score}), .1)
            self.assertEqual(stats.counts["all_frames"], 0)
            self.assertEqual(stats.counts["pad_scored"], 0)
        stats = TrialAccumulator("glasses", "a")
        stats.observe(result("liveness_unavailable", live={"is_live": None, "live_score": None}), .1)
        self.assertEqual(stats.counts["pad_unavailable"], 1)
        self.assertEqual(stats.counts["pad_scored"], 0)
        self.assertIsNone(stats.summary(.1)["rates"]["pad_pass_per_scored_crop"])

    def test_denominators_include_rejections_before_cnn_and_null_empty_rates(self):
        stats = TrialAccumulator("glasses", "a")
        empty = stats.summary(0.)
        self.assertIsNone(empty["rates"]["pad_pass_per_scored_crop"])
        self.assertIsNone(empty["rates"]["correct_identity_per_all_frames"])
        stats.observe({"status": "no_face", "boxes": []}, .1)
        stats.observe(result("image_quality", quality=False), .2)
        stats.observe(result("spoof", live={"is_live": False, "live_score": .1}), .3)
        stats.observe(result(identity={"status": "unknown", "person_id": None},
                             live={"is_live": True, "live_score": .9}), .4)
        stats.observe(result("known", identity={"status": "known", "person_id": "a"},
                             live={"is_live": True, "live_score": .95}), .5)
        report = stats.summary(.5)
        self.assertEqual(report["counts"]["all_frames"], 5)
        self.assertEqual(report["counts"]["exactly_one_face"], 4)
        self.assertEqual(report["counts"]["quality_pass"], 3)
        self.assertEqual(report["counts"]["pad_scored"], 3)
        self.assertEqual(report["counts"]["identity_scored"], 2)
        self.assertEqual(report["rates"]["pad_pass_per_scored_crop"], 2/3)
        self.assertEqual(report["rates"]["correct_identity_per_scored_probe"], 1/2)
        self.assertEqual(report["rates"]["correct_identity_per_all_frames"], 1/5)

    def test_wrong_known_match_and_simulated_attendance_are_distinct(self):
        stats = TrialAccumulator("no-glasses", "a")
        for person_id, elapsed in (("b", .5), ("a", .8)):
            stats.observe(result("known", identity={"status": "known", "person_id": person_id}, attendance=True), elapsed)
        report = stats.summary(1., [{"person_id": "a"}, {"person_id": "b"}])
        self.assertEqual(report["counts"]["identity_correct"], 1)
        self.assertEqual(report["counts"]["identity_wrong"], 1)
        self.assertEqual(report["simulated_attendance"]["record_count"], 2)
        self.assertEqual(report["simulated_attendance"]["correct_record_count"], 1)
        self.assertEqual(report["simulated_attendance"]["other_record_count"], 1)
        self.assertEqual(report["simulated_attendance"]["first_record_seconds"], .5)

    def test_attacks_and_unknown_never_report_correct_identity_accuracy(self):
        for condition in ("unknown", "print", "screen-replay"):
            stats = TrialAccumulator(condition, None if condition == "unknown" else "a")
            stats.observe(result("known", identity={"status": "known", "person_id": "a"}), .1)
            report = stats.summary(.1, [{"person_id": "a"}])
            self.assertIsNone(report["counts"]["identity_correct"])
            self.assertIsNone(report["counts"]["identity_wrong"])
            self.assertIsNone(report["rates"]["correct_identity_per_all_frames"])
            if condition == "unknown":
                self.assertEqual(report["counts"]["unknown_matched"], 1)
                self.assertEqual(report["simulated_attendance"]["unknown_accept_record_count"], 1)
                self.assertIsNone(report["counts"]["attack_any_matched"])
            else:
                self.assertEqual(report["counts"]["attack_target_matched"], 1)
                self.assertEqual(report["counts"]["attack_any_matched"], 1)
                self.assertEqual(report["simulated_attendance"]["attack_accept_record_count"], 1)

    def test_unavailable_pad_and_ambiguous_identity_are_counted_separately(self):
        stats = TrialAccumulator("glasses", "a")
        stats.observe(result("liveness_unavailable", live={"is_live": None, "live_score": .9}), .1)
        stats.observe(result("ambiguous", identity={"status": "ambiguous", "person_id": None},
                             live={"is_live": True, "live_score": .9}), .2)
        report = stats.summary(.2)
        self.assertEqual(report["counts"]["pad_unavailable"], 1)
        self.assertEqual(report["counts"]["pad_not_passed"], 0)
        self.assertEqual(report["counts"]["identity_ambiguous"], 1)
        self.assertEqual(report["counts"]["identity_wrong"], 0)


class SnapshotChecks(unittest.TestCase):
    def test_production_gallery_bytes_and_rows_unchanged_after_scratch_attendance(self):
        root = Path(__file__).resolve().parents[1]/"tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            base = Path(temporary).resolve()
            source = base/"production.sqlite3"
            store = ProfileStore(source, "a"*64, 3)
            person = store.enroll("Fixture A", [[1, 0, 0]]*3)
            store.close()
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            snapshots = base/"scratch"
            with gallery_snapshot(source, "a"*64, 3, snapshots) as copied:
                copied.record_attendance("simulated", person, .95, .95)
                self.assertEqual(len(copied.attendance("simulated")), 1)
            self.assertEqual(list(snapshots.iterdir()), [])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            with closing(sqlite3.connect(source)) as checked:
                self.assertEqual(checked.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 0)
                self.assertEqual(checked.execute("SELECT COUNT(*) FROM profiles").fetchone()[0], 1)

    def test_missing_source_is_not_created_and_encoder_failure_cleans_scratch(self):
        root = Path(__file__).resolve().parents[1]/"tmp"
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            base = Path(temporary).resolve()
            missing = base/"missing.sqlite3"
            with self.assertRaises(ValueError), gallery_snapshot(missing, "a"*64, 3, base/"scratch"):
                pass
            self.assertFalse(missing.exists())
            store = ProfileStore(base/"source.sqlite3", "a"*64, 3)
            store.close()
            with self.assertRaises(ValueError), gallery_snapshot(base/"source.sqlite3", "b"*64, 3, base/"scratch"):
                pass
            self.assertEqual(list((base/"scratch").iterdir()), [])


class CollectorChecks(unittest.TestCase):
    def test_only_selected_camera_is_used_warmup_excluded_and_released(self):
        camera, processor = Mock(), Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        processor.process.return_value = {"status": "no_face", "boxes": []}
        factory = Mock(return_value=camera)
        stats = TrialAccumulator("unknown", None)
        collect_frames(factory, processor, stats, camera=3, seconds=15, max_frames=2, warmup_frames=2)
        factory.assert_called_once_with(3)
        self.assertEqual(camera.read.call_count, 4)
        self.assertEqual(processor.process.call_count, 2)
        self.assertEqual(stats.counts["all_frames"], 2)
        self.assertEqual(stats.summary(.1)["frame_sizes"], [{"width": 64, "height": 48, "frames": 2}])
        camera.release.assert_called_once()

    def test_read_or_processor_failure_releases_device_and_preserves_partial_counts(self):
        for failure in ("read", "processor"):
            camera, processor = Mock(), Mock()
            camera.isOpened.return_value = True
            frame = np.zeros((48, 64, 3), np.uint8)
            camera.read.side_effect = [(True, frame), (False, None)] if failure == "read" else None
            camera.read.return_value = (True, frame)
            processor.process.side_effect = ([{"status": "no_face", "boxes": []}, RuntimeError("fixture model failure")]
                                             if failure == "processor" else None)
            processor.process.return_value = {"status": "no_face", "boxes": []}
            stats = TrialAccumulator("unknown", None)
            with self.assertRaises(RuntimeError):
                collect_frames(Mock(return_value=camera), processor, stats, max_frames=2, warmup_frames=0)
            self.assertEqual(stats.counts["all_frames"], 1)
            camera.release.assert_called_once()

    def test_failed_open_or_invalid_request_never_claims_a_completed_trial(self):
        camera = Mock()
        camera.isOpened.return_value = False
        stats = TrialAccumulator("unknown", None)
        with self.assertRaises(RuntimeError):
            collect_frames(Mock(return_value=camera), Mock(), stats, warmup_frames=0)
        camera.release.assert_called_once()
        self.assertEqual(stats.counts["all_frames"], 0)
        self.assertEqual(report_header("unknown", None)["status"], "not_completed")
        factory = Mock()
        for seconds in (0, float("nan"), 121):
            with self.assertRaises(ValueError):
                collect_frames(factory, Mock(), stats, seconds=seconds)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
