"""LCC-FASD adapter checks with synthetic pictures, never real face data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from anti_spoofing.data import LivenessDataset, load_manifest, validate_manifest_splits
from anti_spoofing.prepare_lcc_fasd import (
    extract_subject_id, prepare_manifests, read_split_records, resolve_data_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SPLITS = {
    "train": "LCC_FASD_training",
    "val": "LCC_FASD_development",
    "test": "LCC_FASD_evaluation",
}
SUBJECTS = {"train": (1, 2), "val": (11, 12), "test": (21, 22)}


def make_lcc_fixture(
    root: Path, *, flat_evaluation: bool = False, omit_split: str | None = None,
    empty_split: str | None = None, one_class_split: str | None = None,
    anonymous_evaluation: bool = False,
) -> None:
    """Mirror official split names and give every picture distinct image bytes."""
    for split, folder in OFFICIAL_SPLITS.items():
        if split == omit_split:
            continue
        directory = root / folder
        directory.mkdir(parents=True)
        for kind in ("real", "spoof"):
            label_directory = directory if split == "test" and flat_evaluation else directory / kind
            label_directory.mkdir(exist_ok=True)
            if split == empty_split or (split == one_class_split and kind == "spoof"):
                continue
            for subject in SUBJECTS[split]:
                for frame in range(2):
                    device = "webcam" if kind == "real" else "phone"
                    name = f"{device}_id{subject}_s0_{frame}.png"
                    if split == "test" and anonymous_evaluation:
                        name = f"{kind}_{(subject - SUBJECTS['test'][0]) * 2 + frame}.png"
                    elif split == "test" and flat_evaluation:
                        name = f"{kind}_{name}"
                    color = (subject * 7 + frame, 210 if kind == "real" else 35, subject + frame * 60)
                    Image.new("RGB", (40, 36), color).save(label_directory / name)


class LCCAdapterChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lcc_synthetic_")
        self.base = Path(self.temporary.name)
        self.root = self.base / "LCC_FASD"
        self.manifests = self.base / "manifests"
        make_lcc_fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_preserves_official_splits_canonical_labels_and_actual_subject_ids(self):
        summary = prepare_manifests(self.root, self.manifests)
        self.assertEqual(summary["dataset"], "LCC-FASD")
        self.assertEqual(summary["crop_mode"], "full")
        for folder in OFFICIAL_SPLITS.values():
            self.assertIn(folder, json.dumps(summary["official_split_mapping"]))
        subjects_by_split = []
        for split, folder in OFFICIAL_SPLITS.items():
            records = load_manifest(self.manifests / f"{split}.csv")
            self.assertEqual(len(records), 8)
            expected_subjects = {f"id{subject}" for subject in SUBJECTS[split]}
            self.assertEqual({record["subject_id"] for record in records}, expected_subjects)
            subjects_by_split.append(expected_subjects)
            self.assertEqual({record["label"] for record in records}, {0, 1})
            for record in records:
                path = Path(record["image_path"])
                self.assertIn(folder, path.parts)
                self.assertEqual(record["label"], int(path.parent.name == "real"))
                self.assertEqual(record["source_label"], 1 - record["label"])
                self.assertEqual(record["subject_id_source"], "filename")
            manifest = self.manifests / f"{split}.csv"
            self.assertEqual(summary["manifests"][split]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertFalse(subjects_by_split[0] & subjects_by_split[1])
        self.assertFalse(subjects_by_split[0] & subjects_by_split[2])
        self.assertFalse(subjects_by_split[1] & subjects_by_split[2])
        validation = validate_manifest_splits(
            self.manifests / "train.csv", self.manifests / "val.csv", self.manifests / "test.csv",
        )
        self.assertTrue(validation["disjoint_paths"])
        self.assertTrue(validation["disjoint_subjects"])

    def test_subject_parser_ignores_device_frame_and_leading_zeros(self):
        self.assertEqual(extract_subject_id("webcam_id0007_s0_0.png"), "id7")
        self.assertEqual(extract_subject_id("phone_id7_s0_99.png"), "id7")
        for filename in ("webcam_0007_s0_0.png", "real_0.png"):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                extract_subject_id(filename)

    def test_memory_cache_preserves_pixels_labels_and_avoids_reopening_sources(self):
        prepare_manifests(self.root, self.manifests)
        dataset = LivenessDataset(self.manifests / "train.csv", image_size=32, crop_mode="full")
        expected = [(image.clone(), label) for image, label in dataset]
        self.assertEqual(dataset.cache_in_memory(), len(dataset) * 3 * 32 * 32 * 4)
        Path(dataset.records[0]["image_path"]).unlink()
        for index, (expected_image, expected_label) in enumerate(expected):
            image, label = dataset[index]
            self.assertTrue(torch.equal(image, expected_image))
            self.assertEqual(label, expected_label)

    def test_case_insensitive_layout_and_id_tokens_still_validate_manifest_ids(self):
        alternate = self.base / "mixed_case"
        for split, folder in OFFICIAL_SPLITS.items():
            for kind in ("real", "spoof"):
                destination = alternate / folder.lower() / kind
                destination.mkdir(parents=True)
                subject = SUBJECTS[split][0]
                Image.new("RGB", (40, 36), (subject, 90 if kind == "real" else 20, 7)).save(
                    destination / f"webcam_ID00{subject}_s0_0.png"
                )
        prepare_manifests(alternate, self.manifests)
        manifest = self.manifests / "train.csv"
        records = load_manifest(manifest)
        self.assertEqual({row["subject_id"] for row in records}, {"id1"})
        with manifest.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["subject_id"] = "id999"
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "subject_id conflicts"):
            load_manifest(manifest)

    def test_split_reader_and_single_wrapper_root_are_supported(self):
        self.assertEqual(resolve_data_root(self.root), self.root.resolve())
        self.assertEqual(resolve_data_root(self.base), self.root.resolve())
        for split in OFFICIAL_SPLITS:
            with self.subTest(split=split):
                records = read_split_records(self.root, split=split)
                self.assertEqual({row["subject_id"] for row in records},
                                 {f"id{subject}" for subject in SUBJECTS[split]})

    def test_training_overlap_with_development_filters_all_matching_subject_records(self):
        conflict = self.root / OFFICIAL_SPLITS["val"] / "real" / "tablet_id0001_s0_97.png"
        Image.new("RGB", (40, 36), (13, 17, 19)).save(conflict)
        summary = prepare_manifests(self.root, self.manifests)
        train = load_manifest(self.manifests / "train.csv")
        val = load_manifest(self.manifests / "val.csv")
        self.assertEqual(len(train), 4)
        self.assertEqual({row["subject_id"] for row in train}, {"id2"})
        self.assertEqual({row["label"] for row in train}, {0, 1})
        self.assertEqual(len(val), 9)
        self.assertIn(str(conflict.resolve()), {row["image_path"] for row in val})
        self.assertEqual(summary["original_splits"]["train"]["samples"], 8)
        excluded = summary["excluded_train_overlap"]
        self.assertEqual(excluded["subject_ids"], ["id1"])
        self.assertEqual(excluded["samples"], 4)
        self.assertEqual({row["subject_id"] for row in excluded["records"]}, {"id1"})
        self.assertEqual({row["label"] for row in excluded["records"]}, {0, 1})
        self.assertTrue(summary["protocol_modified"])
        for description in ("official", "overlap", "removed"):
            self.assertIn(description, summary["split_strategy"])

    def test_identified_evaluation_overlap_is_still_rejected(self):
        conflict = self.root / OFFICIAL_SPLITS["test"] / "real" / "tablet_id0001_s0_97.png"
        Image.new("RGB", (40, 36), (13, 17, 19)).save(conflict)
        with self.assertRaisesRegex(ValueError, r"(?i)(subject|identity|overlap)"):
            prepare_manifests(self.root, self.manifests)

    def test_identical_cross_split_content_is_removed_without_test_filtering_training(self):
        train_duplicate = self.root / OFFICIAL_SPLITS["train"] / "real/webcam_id1_s0_0.png"
        retained_train = self.root / OFFICIAL_SPLITS["train"] / "real/webcam_id2_s0_0.png"
        val_duplicate = self.root / OFFICIAL_SPLITS["val"] / "real/webcam_id501_s0_99.png"
        val_duplicate.write_bytes(train_duplicate.read_bytes())
        copies = (
            ("real/real_990.png", train_duplicate),
            ("real/real_991.png", retained_train),
            ("spoof/spoof_992.png", self.root / OFFICIAL_SPLITS["val"] / "spoof/phone_id11_s0_0.png"),
        )
        for destination, source in copies:
            (self.root / OFFICIAL_SPLITS["test"] / destination).write_bytes(source.read_bytes())
        summary = prepare_manifests(self.root, self.manifests)
        self.assertEqual(summary["excluded_train_content_overlap"]["samples"], 1)
        self.assertEqual(summary["excluded_test_content_overlap"]["samples"], 3)
        self.assertEqual(summary["actual_splits"]["train"]["samples"], 7)
        self.assertEqual(summary["actual_splits"]["val"]["samples"], 9)
        self.assertEqual(summary["actual_splits"]["test"]["samples"], 8)
        self.assertFalse(summary["official_development_and_evaluation_preserved"])
        self.assertTrue(summary["official_development_preserved"])
        train_paths = {row["image_path"] for row in load_manifest(self.manifests / "train.csv")}
        self.assertNotIn(str(train_duplicate.resolve()), train_paths)
        self.assertIn(str(retained_train.resolve()), train_paths)

    def test_identical_bytes_with_conflicting_labels_are_rejected(self):
        source = self.root / OFFICIAL_SPLITS["train"] / "real/webcam_id1_s0_0.png"
        destination = self.root / OFFICIAL_SPLITS["val"] / "spoof/phone_id501_s0_99.png"
        destination.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting class labels"):
            prepare_manifests(self.root, self.manifests)

    def test_overlap_filter_cannot_produce_an_empty_training_split(self):
        for subject in SUBJECTS["train"]:
            conflict = self.root / OFFICIAL_SPLITS["val"] / "real" / f"tablet_id{subject}_s0_98.png"
            Image.new("RGB", (40, 36), (subject, 21, 23)).save(conflict)
        with self.assertRaises(ValueError):
            prepare_manifests(self.root, self.manifests)

    def test_unknown_subject_ids_in_training_and_development_are_rejected(self):
        for split in ("train", "val"):
            invalid = self.root / OFFICIAL_SPLITS[split] / "real" / "webcam_subject777_s0_0.png"
            Image.new("RGB", (40, 36), (14, 18, 20)).save(invalid)
            try:
                with self.subTest(split=split), self.assertRaises(ValueError):
                    read_split_records(self.root, split=split)
            finally:
                invalid.unlink()

    def test_conflicting_filename_prefix_and_nested_class_directories_are_rejected(self):
        for relative in ("real/spoof_webcam_id1_s0_91.png", "real/spoof/webcam_id1_s0_92.png"):
            invalid = self.root / OFFICIAL_SPLITS["train"] / relative
            invalid.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 36), (15, 19, 21)).save(invalid)
            try:
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    read_split_records(self.root, split="train")
            finally:
                invalid.unlink()

    def test_unlabelled_image_directories_are_not_silently_ignored(self):
        invalid = self.root / OFFICIAL_SPLITS["train"] / "unlabelled" / "webcam_id1_s0_96.png"
        invalid.parent.mkdir()
        Image.new("RGB", (40, 36), (19, 23, 29)).save(invalid)
        with self.assertRaises(ValueError):
            read_split_records(self.root, split="train")

    def test_missing_empty_and_one_class_splits_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            prepare_manifests(self.base / "absent", self.manifests)
        for name, options, error in (
            ("missing", {"omit_split": "test"}, FileNotFoundError),
            ("empty", {"empty_split": "val"}, ValueError),
            ("one_class", {"one_class_split": "val"}, ValueError),
        ):
            root = self.base / name / "LCC_FASD"
            make_lcc_fixture(root, **options)
            with self.subTest(case=name), self.assertRaises(error):
                prepare_manifests(root, self.base / f"{name}_manifests")

    def test_flat_evaluation_prefixes_preserve_real_spoof_labels_and_subjects(self):
        root = self.base / "flat" / "LCC_FASD"
        make_lcc_fixture(root, flat_evaluation=True)
        manifests = self.base / "flat_manifests"
        prepare_manifests(root, manifests)
        records = load_manifest(manifests / "test.csv")
        self.assertEqual(len(records), 8)
        self.assertEqual({record["subject_id"] for record in records}, {"id21", "id22"})
        for record in records:
            path = Path(record["image_path"])
            self.assertEqual(path.parent.name, OFFICIAL_SPLITS["test"])
            self.assertEqual(record["label"], int(path.name.startswith("real_")))

    def test_anonymous_official_evaluation_has_no_fabricated_subject_ids(self):
        root = self.base / "anonymous" / "LCC_FASD"
        make_lcc_fixture(root, anonymous_evaluation=True)
        manifests = self.base / "anonymous_manifests"
        prepare_manifests(root, manifests)
        records = load_manifest(manifests / "test.csv")
        self.assertEqual(len(records), 8)
        self.assertEqual({record["subject_id"] for record in records}, {""})
        self.assertEqual({record["subject_id_source"] for record in records}, {"unavailable"})
        self.assertEqual({record["label"] for record in records}, {0, 1})
        summary = validate_manifest_splits(manifests / "train.csv", manifests / "val.csv", manifests / "test.csv")
        self.assertTrue(summary["disjoint_paths"])
        self.assertIsNone(summary["disjoint_subjects"])
        self.assertEqual(summary["subject_identity_check"], "evaluation subject IDs unavailable")
        self.assertEqual(summary["splits"]["test"]["subjects"], 0)
        self.assertEqual(summary["splits"]["test"]["samples_without_subject_id"], 8)
        # Explicitly unavailable IDs are permitted for evaluation, never for learning or calibration.
        for train, val in (("test.csv", "val.csv"), ("train.csv", "test.csv")):
            with self.subTest(train=train, val=val), self.assertRaises(ValueError):
                validate_manifest_splits(manifests / train, manifests / val)

    def test_every_image_loads_in_explicit_full_mode_without_bbox_sidecars(self):
        prepare_manifests(self.root, self.manifests)
        hashes = set()
        for split in OFFICIAL_SPLITS:
            dataset = LivenessDataset(self.manifests / f"{split}.csv", image_size=32, crop_mode="full")
            for index, record in enumerate(dataset.records):
                path = Path(record["image_path"])
                self.assertFalse(path.with_name(path.stem + "_BB.txt").exists())
                tensor, label = dataset[index]
                self.assertEqual(tuple(tensor.shape), (3, 32, 32))
                self.assertEqual(tensor.dtype, torch.float32)
                self.assertEqual(label, record["label"])
                with Image.open(path) as image:
                    expected_pixel = np.asarray(image.convert("RGB"))[0, 0].astype(np.float32) / 127.5 - 1
                np.testing.assert_allclose(tensor[:, 0, 0].numpy(), expected_pixel, atol=1e-6)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertNotIn(digest, hashes, "Synthetic images must not trigger held-out content-overlap rejection.")
                hashes.add(digest)


class LCCEndToEndChecks(unittest.TestCase):
    def run_cli(self, *arguments: str, expected: int = 0):
        completed = subprocess.run([sys.executable, "-m", *arguments], cwd=PROJECT_ROOT,
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, expected, completed.stdout + "\n" + completed.stderr)
        return completed

    def test_full_mode_train_calibration_and_held_out_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="lcc_synthetic_e2e_") as temporary:
            base = Path(temporary)
            data, manifests = base / "LCC_FASD", base / "manifests"
            training, evaluation = base / "training", base / "evaluation"
            make_lcc_fixture(data, anonymous_evaluation=True)
            self.run_cli("anti_spoofing.prepare_lcc_fasd", "--data-root", str(data),
                         "--output-dir", str(manifests))
            self.run_cli("anti_spoofing.train", "--train-manifest", str(manifests / "train.csv"),
                         "--val-manifest", str(manifests / "val.csv"), "--output-dir", str(training),
                         "--crop-mode", "full", "--epochs", "1", "--batch-size", "4",
                         "--image-size", "32", "--feature-dim", "16", "--max-batches", "1",
                         "--seed", "7", "--device", "cpu", "--num-threads", "1")
            checkpoint = training / "best_model.pt"
            before_bytes = checkpoint.read_bytes()
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["config"]["crop_mode"], "full")
            self.assertTrue(saved["config"]["threshold_calibrated"])
            self.assertTrue(saved["training"]["smoke_test"])
            self.assertEqual(saved["training"]["calibration"]["validation_samples_used"], 8)
            self.assertNotIn("test", saved["training"]["provenance"])
            self.assertTrue(all("bbox_sha256" not in record
                                for split in saved["training"]["provenance"].values()
                                for record in split["samples"]))
            self.run_cli("anti_spoofing.evaluate", "--checkpoint", str(checkpoint),
                         "--manifest", str(manifests / "test.csv"), "--output-dir", str(evaluation),
                         "--device", "cpu", "--num-threads", "1", "--batch-size", "4")
            report = json.loads((evaluation / "metrics.json").read_text())
            self.assertEqual(report["crop_mode"], "full")
            self.assertEqual(report["leakage_check"]["status"],
                             "image-disjoint; evaluation subject IDs unavailable")
            self.assertFalse(report["leakage_check"]["subject_disjoint_verified"])
            for overlap in report["leakage_check"]["overlaps"].values():
                self.assertEqual(overlap["subject_ids"], [])
            self.assertIn("held-out", report["evaluation_scope"])
            self.assertIn("subject separation cannot be verified", report["evaluation_scope"])
            self.assertFalse(report["threshold_retuned_on_evaluation"])
            self.assertEqual(report["threshold"], saved["config"]["threshold"])
            self.assertTrue(report["training_smoke_test"])
            self.assertEqual(report["evaluation_manifest"]["sample_count"], 8)
            self.assertFalse(report["evaluation_manifest"]["subject_ids_complete"])
            self.assertEqual(report["evaluation_manifest"]["unknown_subject_samples"], 8)
            self.assertEqual(report["evaluation_manifest"]["subject_ids"], [])
            self.assertTrue((evaluation / "predictions.csv").is_file())
            self.assertTrue((evaluation / "roc_curve.svg").is_file())
            self.assertEqual(checkpoint.read_bytes(), before_bytes)
            # Unknown evaluation identities must not disable the independent content-overlap check.
            source = Path(load_manifest(manifests / "train.csv")[0]["image_path"])
            copied = Path(load_manifest(manifests / "test.csv")[0]["image_path"])
            copied.write_bytes(source.read_bytes())
            rejected = self.run_cli(
                "anti_spoofing.evaluate", "--checkpoint", str(checkpoint),
                "--manifest", str(manifests / "test.csv"), "--output-dir", str(base / "forbidden"),
                "--device", "cpu", "--num-threads", "1", expected=2,
            )
            self.assertIn("overlap", rejected.stderr.casefold())
            self.assertIn("hash", rejected.stderr.casefold())
            self.assertEqual(checkpoint.read_bytes(), before_bytes)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
