"""Section 2 checks. All generated pictures are synthetic shapes, not faces."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from PIL import Image, ImageDraw
import torch

from face_verification.data import FaceImageDataset, TripletFaceDataset, load_face_image
from face_verification.metrics import compute_auc, compute_roc
from face_verification.model import FaceEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_synthetic_dataset(root: Path) -> None:
    """Make a tiny integration fixture; verification IDs never occur in training."""
    colors = ((180, 50, 40), (40, 170, 60), (40, 70, 190))
    for split, image_count in (("train_data", 3), ("val_data", 1), ("test_data", 1)):
        for person, color in enumerate(colors):
            folder = root / "classification_data" / split / f"train_person_{person}"
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(image_count):
                picture = Image.new("RGB", (32, 32), color)
                ImageDraw.Draw(picture).rectangle((index + 3, 5, index + 11, 25), fill="white")
                picture.save(folder / f"{index}.png")
    for person, color in enumerate((colors[0], colors[2])):
        folder = root / "verification_data" / f"unseen_person_{person}"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(2):
            picture = Image.new("RGB", (32, 32), color)
            ImageDraw.Draw(picture).ellipse((4 + index, 6, 19 + index, 23), fill="white")
            picture.save(folder / f"{index}.png")
    pairs = [
        ("unseen_person_0/0.png", "unseen_person_0/1.png", 1),
        ("unseen_person_1/0.png", "unseen_person_1/1.png", 1),
        ("unseen_person_0/0.png", "unseen_person_1/0.png", 0),
        ("unseen_person_0/1.png", "unseen_person_1/1.png", 0),
    ]
    (root / "verification_pairs_val.txt").write_text(
        "\n".join(f"verification_data/{left} verification_data/{right} {label}"
                  for left, right, label in pairs) + "\n", encoding="utf-8"
    )


class MetricChecks(unittest.TestCase):
    def test_known_auc_and_score_direction(self):
        self.assertAlmostEqual(compute_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertAlmostEqual(compute_auc([0, 1], [0.2, 0.9]), 1.0)
        self.assertAlmostEqual(compute_auc([0, 1], [0.9, 0.2]), 0.0)

    def test_ties_match_independent_pairwise_probability(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        scores = np.array([0.0, 0.0, 0.5, 0.5, 0.8, 0.8, 0.3, 1.0])
        differences = scores[labels == 1, None] - scores[labels == 0][None, :]
        expected = np.mean((differences > 0) + 0.5 * (differences == 0))
        self.assertAlmostEqual(compute_auc(labels, scores), float(expected))
        self.assertAlmostEqual(compute_auc([0, 1, 0, 1], [1.0] * 4), 0.5)

    def test_roc_has_correct_endpoints_and_monotone_rates(self):
        fpr, tpr, thresholds = compute_roc([0, 1, 1, 0], [0.1, 0.8, 0.8, 0.3])
        self.assertEqual((fpr[0], tpr[0]), (0.0, 0.0))
        self.assertEqual((fpr[-1], tpr[-1]), (1.0, 1.0))
        self.assertTrue(np.all(np.diff(fpr) >= 0))
        self.assertTrue(np.all(np.diff(tpr) >= 0))
        self.assertEqual(len(thresholds), len(fpr))

    def test_invalid_evaluations_fail(self):
        for labels, scores in (([], []), ([1, 1], [0.1, 0.2]),
                               ([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan]),
                               ([0, 1], [0.1])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                compute_auc(labels, scores)

    def test_normalized_cosine_and_negative_distance_rank_the_same(self):
        rng = np.random.default_rng(21)
        left, right = rng.normal(size=(2, 40, 16))
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        cosine = np.sum(left * right, axis=1)
        distance = np.linalg.norm(left - right, axis=1)
        labels = np.tile([0, 1], 20)
        np.testing.assert_allclose(distance ** 2, 2 - 2 * cosine, atol=1e-12)
        self.assertAlmostEqual(compute_auc(labels, cosine), compute_auc(labels, -distance))


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="section2_synthetic_")
        self.root = Path(self.temporary.name)
        make_synthetic_dataset(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_image_conversion_handles_grayscale_and_range(self):
        path = self.root / "gray.png"
        Image.new("L", (48, 48), 255).save(path)
        tensor = load_face_image(path, image_size=32)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(torch.all(tensor == 1))

    def test_validation_preserves_training_labels_and_reports_unknown_ids(self):
        train = FaceImageDataset(self.root / "classification_data/train_data")
        val_path = self.root / "only_validation"
        for name in ("train_person_2", "unknown_person"):
            folder = val_path / name
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), "white").save(folder / "0.png")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = FaceImageDataset(val_path, class_to_idx=train.class_to_idx)
        self.assertEqual(len(val), 1)
        self.assertEqual(val.samples[0][1], train.class_to_idx["train_person_2"])
        self.assertIn("unknown_person", val.unknown_classes)
        self.assertTrue(caught)

    def test_triplets_use_distinct_same_id_positive_and_other_id_negative(self):
        dataset = TripletFaceDataset(self.root / "classification_data/train_data", seed=19)
        with patch("face_verification.data.load_face_image", side_effect=lambda path, image_size: path):
            for epoch in (0, 1):
                dataset.set_epoch(epoch)
                for index in range(len(dataset)):
                    anchor, positive, negative = dataset[index]
                    self.assertNotEqual(anchor, positive)
                    self.assertEqual(anchor.parent, positive.parent)
                    self.assertNotEqual(anchor.parent, negative.parent)
                    self.assertEqual((anchor, positive, negative), dataset[index])

    def test_pair_paths_headers_and_invalid_labels(self):
        from face_verification.evaluate import read_verification_pairs
        pairs_path = self.root / "verification_pairs_val.txt"
        pairs = read_verification_pairs(pairs_path, self.root)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(a.is_file() and b.is_file() for a, b, _ in pairs))
        original = pairs_path.read_text(encoding="utf-8")
        pairs_path.write_text("image1 image2 label\n" + original, encoding="utf-8")
        self.assertEqual(len(read_verification_pairs(pairs_path, self.root)), 4)
        pairs_path.write_text(original.rsplit(" ", 1)[0] + " 2\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_verification_pairs(pairs_path, self.root)

    def test_encoder_outputs_unit_vectors_and_reload_is_identical(self):
        torch.manual_seed(19)
        encoder = FaceEncoder(embedding_dim=16).eval()
        images = torch.stack([
            load_face_image(self.root / "verification_data/unseen_person_0/0.png", 32),
            load_face_image(self.root / "verification_data/unseen_person_1/0.png", 32),
        ])
        with torch.inference_mode():
            first = encoder(images)
        self.assertEqual(tuple(first.shape), (2, 16))
        torch.testing.assert_close(torch.linalg.vector_norm(first, dim=1), torch.ones(2))
        checkpoint = self.root / "encoder.pt"
        torch.save(encoder.state_dict(), checkpoint)
        loaded = FaceEncoder(embedding_dim=16).eval()
        loaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with torch.inference_mode():
            torch.testing.assert_close(first, loaded(images), rtol=0, atol=0)

    def test_single_trial_uses_calibrated_direction_and_no_guessed_threshold(self):
        from face_verification.verify import FaceVerifier
        checkpoint = self.root / "pair_encoder.pt"
        encoder = FaceEncoder(embedding_dim=16)
        torch.save({"model_state": encoder.state_dict(),
                    "config": {"embedding_dim": 16, "image_size": 32, "method": "test_fixture"}},
                   checkpoint)
        image = self.root / "verification_data/unseen_person_0/0.png"
        scorer = FaceVerifier(checkpoint)
        scores = scorer.verify(image, image)
        self.assertAlmostEqual(scores["cosine_similarity"], 1.0, places=6)
        self.assertAlmostEqual(scores["euclidean_distance"], 0.0, places=6)
        self.assertIsNone(scores["same_person"])
        scorer.threshold = 0.99
        self.assertTrue(scorer.verify(image, image)["same_person"])
        scorer.metric, scorer.threshold = "euclidean", 0.01
        self.assertTrue(scorer.verify(image, image)["same_person"])

    def test_data_checker_reports_missing_inputs_without_making_data(self):
        from face_verification.check_data import inspect_data
        missing = self.root / "does_not_exist"
        report = inspect_data(missing)
        self.assertEqual(report["status"], "not_ready")
        self.assertTrue(report["errors"])
        self.assertFalse(missing.exists())


class EndToEndChecks(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            [sys.executable, "-m", *arguments], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + "\n" + completed.stderr)
        return completed

    def test_both_training_methods_save_weights_and_evaluate_unseen_pairs(self):
        with tempfile.TemporaryDirectory(prefix="section2_synthetic_e2e_") as directory:
            root = Path(directory)
            data_root = root / "synthetic_data"
            make_synthetic_dataset(data_root)
            checkpoints = []
            for method in ("classification", "triplet"):
                output = root / method
                optional = ["--evaluate-test"] if method == "classification" else []
                self.run_cli(
                    "face_verification.train", "--method", method,
                    "--data-root", str(data_root), "--output-dir", str(output),
                    "--epochs", "1", "--batch-size", "3", "--image-size", "32",
                    "--embedding-dim", "16", "--seed", "7", "--device", "cpu",
                    "--max-batches", "1", *optional,
                )
                checkpoint = output / "best_model.pt"
                self.assertTrue(checkpoint.is_file())
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(saved["config"]["method"], method)
                self.assertEqual(saved["config"]["image_size"], 32)
                self.assertEqual(saved["config"]["embedding_dim"], 16)
                self.assertFalse(saved["training"]["augmentation"]["train"]["enabled"])
                self.assertFalse(saved["training"]["test_data_used_for_selection"])
                if method == "classification":
                    test_report = json.loads((output / "classification_test.json").read_text())
                    self.assertEqual(test_report["status"], "evaluated")
                    self.assertEqual(test_report["metrics"]["samples"], 3)
                    self.assertFalse(test_report["used_for_selection"])
                    self.assertIn("classifier_state", saved)
                torch.manual_seed(7)
                fresh = FaceEncoder(embedding_dim=16).state_dict()
                self.assertTrue(any(
                    not torch.equal(value, fresh[key])
                    for key, value in saved["model_state"].items()
                    if value.is_floating_point()
                ), "A training step must change encoder weights.")
                checkpoints.append(checkpoint)
            comparison = root / "evaluation"
            self.run_cli(
                "face_verification.evaluate", "--checkpoint", str(checkpoints[0]),
                "--checkpoint", str(checkpoints[1]), "--data-root", str(data_root),
                "--output-dir", str(comparison), "--device", "cpu", "--batch-size", "4",
            )
            with (comparison / "comparison.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4, "Two models times two metrics must be reported.")
            for row in rows:
                self.assertTrue(0 <= float(row["auc"]) <= 1)
            self.assertTrue(list(comparison.rglob("*.svg")))
            self.assertTrue((comparison / "comparison.json").is_file())
            self.run_cli("face_verification.check_data", "--data-root", str(data_root))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
