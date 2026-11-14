"""Check 1:N trial separation and threshold behavior without a webcam."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from attendance_app.recognition import FaceRecognizer
from tools.evaluate_identity_open_set import RankedProbe, audited_test_pool, make_trials, outcomes, select_threshold


class OpenSetChecks(unittest.TestCase):
    def test_excludes_cross_split_and_test_duplicates_then_separates_people_and_photos(self):
        rows = []

        def add(split, identity, photo, pixel):
            rows.append((f"classification_data/{split}/{identity}/{photo}.png",
                         split, identity, "unused", pixel))

        add("train_data", "train", 0, "also_in_train")
        add("val_data", "train", 0, "val_only")
        add("test_data", "shared", 0, "also_in_train")
        add("test_data", "shared", 1, "ok")
        add("test_data", "copied", 0, "same_test_pixels")
        add("test_data", "copied", 1, "same_test_pixels")
        for identity in range(6):
            for photo in range(2):
                add("test_data", str(identity), photo, f"{identity}-{photo}")
        audit = {"images_decoded": len(rows), "classification": {
            split: {"images": sum(row[1] == split for row in rows)}
            for split in ("train_data", "val_data", "test_data")
        }}
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            index = Path(temporary) / "images.csv"
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("archive_path", "split", "identity", "file_sha256", "pixel_sha256"))
                writer.writerows(rows)
            pool, counts = audited_test_pool(index, audit)
        self.assertEqual(set(pool), {str(value) for value in range(6)})
        self.assertEqual(counts["exact_pixel_duplicate_test_images_excluded"], 3)
        trials = make_trials(pool, 2, 1, 42)
        self.assertEqual(trials, make_trials(pool, 2, 1, 42))
        cal_gallery, cal_unknown = trials["calibration"]
        test_gallery, test_unknown = trials["test"]
        self.assertFalse((set(cal_gallery) | set(cal_unknown)) & (set(test_gallery) | set(test_unknown)))
        for gallery, unknown in trials.values():
            self.assertFalse(gallery.keys() & unknown.keys())
            self.assertTrue(all(ref.pixel_sha256 != probe.pixel_sha256 for ref, probe in gallery.values()))

    def test_outcomes_match_real_gallery_rules(self):
        recognizer = FaceRecognizer.__new__(FaceRecognizer)
        recognizer.dimension = 2
        profiles = [{"id": name, "name": name, "embedding": vector}
                    for name, vector in (("a", [1., 0.]), ("b", [0., 1.]))]
        expected_vectors = [("a", [1., 0.]), ("a", [0., 1.]), ("a", [-1., -1.]),
                            (None, [1., 0.]), (None, [-1., -1.]), ("a", [1., 1.])]
        expected_statuses = ["known", "known", "unknown", "known", "unknown", "ambiguous"]
        ranked = []
        for (expected, vector), status in zip(expected_vectors, expected_statuses):
            recognizer.threshold = .6
            recognizer.margin = .05
            self.assertEqual(recognizer.match(np.array(vector), profiles)["status"], status)
            recognizer.threshold = -1.
            recognizer.margin = 0.
            match = recognizer.match(np.array(vector), profiles)
            ranked.append(RankedProbe(expected, match["person_id"], match["cosine"], match["gap"]))
        report = outcomes(ranked, .6, .05)
        for key in ("known_correct", "known_misidentified", "known_unknown",
                    "unknown_false_identifications", "unknown_unknown", "known_ambiguous"):
            self.assertEqual(report[key], 1)

    def test_threshold_calibration_keeps_tied_unknown_rejected_and_can_reject_all(self):
        probes = [
            RankedProbe("a", "a", .95, .1), RankedProbe("b", "b", .9, .1),
            RankedProbe(None, "a", .9, .1), RankedProbe(None, "a", .99, .01),
        ]
        self.assertEqual(select_threshold(probes, .05, 0.), .95)
        self.assertEqual(select_threshold(probes, .05, .5), .9)
        self.assertEqual(select_threshold(probes[1:], .05, 0.), 1.)


if __name__ == "__main__":
    unittest.main()
