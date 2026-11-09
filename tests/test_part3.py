"""Section 3 software checks using generated geometric pictures, not real faces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image, ImageDraw
import torch

from anti_spoofing.data import (
    LivenessDataset, crop_liveness_image, load_liveness_image, load_manifest,
    preprocess_liveness_image, read_celeba_bbox, validate_manifest_splits,
)
from anti_spoofing.metrics import calibrate_threshold, liveness_metrics
from anti_spoofing.model import LivenessCNN
from anti_spoofing.prepare import parse_official_labels, prepare_manifests
from attendance_pipeline import AttendanceVerificationPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_celeba_fixture(root: Path) -> None:
    """Mirror official path/label/bbox formats with unique synthetic image bytes."""
    metas = root / "metas/intra_test"
    metas.mkdir(parents=True)
    for split, subjects in (("train", range(1, 7)), ("test", range(101, 104))):
        annotations = {}
        for subject in subjects:
            for source_label, kind in ((0, "live"), (1, "spoof")):
                for index in range(2):
                    relative = Path("Data") / split / str(subject) / kind / f"{index}.png"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    color = ((subject * 29 + index * 37) % 230 + 10,
                             190 if kind == "live" else 30,
                             (subject * 11 + index * 19 + source_label * 41) % 250)
                    image = Image.new("RGB", (64, 48), color)
                    draw = ImageDraw.Draw(image)
                    if kind == "live":
                        draw.ellipse((18 + index, 13, 43, 35), fill=(subject % 255, 240, 180))
                    else:
                        for x in range(17, 48, 5):
                            draw.line((x, 12, x, 36), fill=(200, subject % 255, index * 90))
                    image.save(path)
                    path.with_name(path.stem + "_BB.txt").write_text("56 56 112 112 0.99\n")
                    attributes = [0] * 44
                    attributes[40] = 0 if kind == "live" else 1
                    attributes[43] = source_label
                    annotations[relative.as_posix()] = attributes
        (metas / f"{split}_label.json").write_text(json.dumps(annotations), encoding="utf-8")


class LivenessMetricChecks(unittest.TestCase):
    def test_attack_and_bona_fide_error_rates_have_the_right_direction(self):
        stats = liveness_metrics([0, 0, 1, 1], [0.9, 0.1, 0.8, 0.2], 0.5)
        for name in ("APCER", "BPCER", "ACER", "accuracy", "AUC"):
            self.assertAlmostEqual(stats[name], 0.5)
        self.assertEqual(stats["confusion_matrix"], [[1, 1], [1, 1]])
        tied = liveness_metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual((tied["APCER"], tied["BPCER"]), (1.0, 0.0))

    def test_calibration_is_finite_and_describes_validation_scope(self):
        calibration = calibrate_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(calibration["threshold_source"], "validation")
        self.assertTrue(calibration["threshold_calibrated"])
        self.assertAlmostEqual(calibration["validation_metrics"]["ACER"], 0.0)
        self.assertGreaterEqual(calibration["threshold"], 0.0)
        self.assertLessEqual(calibration["threshold"], 1.0)

    def test_invalid_or_one_class_scores_fail(self):
        for labels, scores in (([0, 0], [0.1, 0.2]), ([0, 1], [-0.1, 0.4]),
                               ([0, 1], [0.2, float("nan")]), ([0, 2], [0.1, 0.9])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                liveness_metrics(labels, scores, 0.5)


class CelebADataChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="section3_synthetic_")
        self.root = Path(self.temporary.name) / "CelebA_Spoof"
        make_celeba_fixture(self.root)
        self.manifests = Path(self.temporary.name) / "manifests"

    def tearDown(self):
        self.temporary.cleanup()

    def test_official_label_inversion_and_subject_disjoint_splits(self):
        prepare_manifests(self.root, self.manifests, val_fraction=0.34, seed=42)
        summary = validate_manifest_splits(
            self.manifests / "train.csv", self.manifests / "val.csv", self.manifests / "test.csv"
        )
        self.assertTrue(summary["disjoint_subjects"])
        self.assertTrue(summary["disjoint_paths"])
        subjects = []
        for split in ("train", "val", "test"):
            records = load_manifest(self.manifests / f"{split}.csv")
            self.assertEqual({row["label"] for row in records}, {0, 1})
            self.assertTrue(all(row["label"] == 1 - row["source_label"] for row in records))
            subjects.append({row["subject_id"] for row in records})
        self.assertFalse(subjects[0] & subjects[1] or subjects[0] & subjects[2] or subjects[1] & subjects[2])

    def test_bbox_scales_each_axis_and_matches_pixel_crop(self):
        sidecar = Path(self.temporary.name) / "bbox.txt"
        sidecar.write_text("56 56 112 112 1\n")
        box = read_celeba_bbox(sidecar, 448, 224)
        self.assertEqual(box, (112.0, 56.0, 224.0, 112.0))
        image = np.zeros((224, 448, 3), dtype=np.uint8)
        image[56:168, 112:336] = (0, 255, 0)
        crop = crop_liveness_image(image, bbox=box)
        self.assertEqual(crop.size, (224, 112))
        self.assertTrue(np.all(np.asarray(crop) == (0, 255, 0)))

    def test_missing_or_invalid_bbox_does_not_silently_use_full_image(self):
        path = self.root / "Data/train/1/live/0.png"
        sidecar = path.with_name(path.stem + "_BB.txt")
        sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            load_liveness_image(path)
        self.assertEqual(tuple(load_liveness_image(path, 32, crop_mode="full").shape), (3, 32, 32))
        with self.assertRaises(ValueError):
            crop_liveness_image(Image.new("RGB", (32, 32)), (100, 100, 5, 5))

    def test_duplicate_images_and_label_directory_conflicts_are_rejected(self):
        labels = self.root / "metas/intra_test/train_label.json"
        raw = json.loads(labels.read_text())
        first = next(iter(raw))
        raw[first][43] = 1
        labels.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            parse_official_labels(labels, self.root)
        make_path = self.manifests
        make_path.mkdir()
        image = self.root / "Data/test/101/live/0.png"
        with (make_path / "duplicates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_path", "label", "subject_id", "source_label"))
            writer.writeheader()
            record = {"image_path": str(image), "label": 1, "subject_id": "101", "source_label": 0}
            writer.writerow(record)
            writer.writerow(record)
        with self.assertRaises(ValueError):
            load_manifest(make_path / "duplicates.csv")


class IntegrationGateChecks(unittest.TestCase):
    def test_spoof_and_uncalibrated_probes_never_call_identity(self):
        for live_value in (False, None):
            detector = Mock()
            detector.prepare_face.return_value = Image.new("RGB", (32, 32))
            detector.predict_crop.return_value = {"is_live": live_value}
            verifier = Mock()
            result = AttendanceVerificationPipeline(detector, verifier).verify("probe", "reference")
            self.assertFalse(result["accepted"])
            verifier.verify_images.assert_not_called()

    def test_live_probe_needs_identity_match_and_shares_exact_crop(self):
        for identity_value in (True, False, None):
            face = Image.new("RGB", (32, 32), "green")
            detector = Mock()
            detector.prepare_face.return_value = face
            detector.predict_crop.return_value = {"is_live": True}
            verifier = Mock()
            verifier.verify_images.return_value = {"same_person": identity_value}
            reference = Image.new("RGB", (32, 32))
            result = AttendanceVerificationPipeline(detector, verifier).verify("probe", reference)
            self.assertEqual(result["accepted"], identity_value is True)
            detector.predict_crop.assert_called_once_with(face)
            verifier.verify_images.assert_called_once_with(face, reference)


class LivenessInferenceChecks(unittest.TestCase):
    def test_inference_uses_same_crop_and_does_not_guess_uncalibrated_threshold(self):
        from anti_spoofing.infer import LivenessDetector
        with tempfile.TemporaryDirectory(prefix="section3_inference_") as directory:
            root = Path(directory)
            image = root / "frame.png"
            Image.new("RGB", (64, 48), "green").save(image)
            image.with_name("frame_BB.txt").write_text("56 56 112 112 1\n")
            config = {"feature_dim": 16, "image_size": 32, "crop_mode": "bbox",
                      "live_label": 1, "spoof_label": 0, "threshold": 0.5,
                      "threshold_source": "validation", "threshold_calibrated": True}
            payload = {"model_state": LivenessCNN(16).state_dict(), "config": config, "training": {}}
            checkpoint = root / "calibrated.pt"
            torch.save(payload, checkpoint)
            detector = LivenessDetector(checkpoint)
            crop = detector.prepare_face(image)
            self.assertEqual(crop.size, (32, 24))
            from_path, from_crop = detector.predict(image), detector.predict_crop(crop)
            self.assertAlmostEqual(from_path["live_score"], from_crop["live_score"], places=12)
            self.assertEqual(from_path["is_live"], from_crop["is_live"])
            config["threshold_source"], config["threshold_calibrated"] = None, False
            torch.save(payload, root / "uncalibrated.pt")
            self.assertIsNone(LivenessDetector(root / "uncalibrated.pt").predict(image)["is_live"])
            self.assertTrue(LivenessDetector(root / "uncalibrated.pt", threshold=0).predict(image)["is_live"])
            with self.assertRaises(ValueError):
                detector.prepare_face(np.zeros((32, 32, 3), dtype=np.uint8))


class Section3EndToEndChecks(unittest.TestCase):
    def run_cli(self, *arguments, expected=0):
        completed = subprocess.run([sys.executable, "-m", *arguments], cwd=PROJECT_ROOT,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, expected, completed.stdout + "\n" + completed.stderr)
        return completed

    def test_prepare_train_calibrate_and_test_without_retuning_or_split_leakage(self):
        with tempfile.TemporaryDirectory(prefix="section3_synthetic_e2e_") as directory:
            root = Path(directory)
            data, manifests, output, evaluation = (root / name for name in ("data", "manifests", "training", "evaluation"))
            make_celeba_fixture(data)
            self.run_cli("anti_spoofing.prepare", "--data-root", str(data), "--output-dir", str(manifests),
                         "--val-fraction", "0.34", "--seed", "42")
            self.run_cli("anti_spoofing.train", "--train-manifest", str(manifests / "train.csv"),
                         "--val-manifest", str(manifests / "val.csv"), "--output-dir", str(output),
                         "--epochs", "1", "--batch-size", "4", "--image-size", "32",
                         "--feature-dim", "16", "--max-batches", "1", "--seed", "7", "--device", "cpu")
            checkpoint = output / "best_model.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertTrue(saved["config"]["threshold_calibrated"])
            self.assertEqual(saved["config"]["threshold_source"], "validation")
            self.assertTrue(saved["training"]["smoke_test"])
            self.assertNotIn("test", saved["training"]["provenance"])
            torch.manual_seed(7)
            fresh = LivenessCNN(16).state_dict()
            self.assertTrue(any(not torch.equal(value, fresh[name])
                                for name, value in saved["model_state"].items() if value.is_floating_point()))
            self.run_cli("anti_spoofing.evaluate", "--checkpoint", str(checkpoint),
                         "--manifest", str(manifests / "test.csv"), "--output-dir", str(evaluation), "--device", "cpu")
            report = json.loads((evaluation / "metrics.json").read_text())
            serialized = json.dumps(report)
            self.assertIn("held-out", serialized)
            self.assertIn("APCER", serialized)
            self.assertTrue(list(evaluation.glob("*.svg")))
            after = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["config"]["threshold"], after["config"]["threshold"])
            rejected = self.run_cli("anti_spoofing.evaluate", "--checkpoint", str(checkpoint),
                                    "--manifest", str(manifests / "val.csv"),
                                    "--output-dir", str(root / "forbidden"), "--device", "cpu", expected=2)
            self.assertIn("overlap", rejected.stderr.casefold())
            self.run_cli("anti_spoofing.infer", "--checkpoint", str(checkpoint), "--image",
                         str(data / "Data/test/101/live/0.png"), "--device", "cpu")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
