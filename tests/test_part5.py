"""Section 5 identity import/audit tests; all pictures are synthetic noise."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from face_verification.audit_data import audit_dataset, sha256_file
from face_verification.audit_archive import audit_archive
from face_verification.import_data import find_dataset_root, import_dataset


def fixture(root):
    index = 0
    for split, count in (("train_data", 3), ("val_data", 1), ("test_data", 1)):
        for identity in ("id_a", "id_b"):
            folder = root / "classification_data" / split / identity
            folder.mkdir(parents=True)
            for number in range(count):
                array = np.random.default_rng(index).integers(0, 256, (24, 24, 3), dtype=np.uint8)
                Image.fromarray(array).save(folder / f"{number}.png")
                index += 1
    for identity in ("unseen_a", "unseen_b"):
        folder = root / "verification_data" / identity
        folder.mkdir(parents=True)
        for number in range(2):
            array = np.random.default_rng(index).integers(0, 256, (24, 24, 3), dtype=np.uint8)
            Image.fromarray(array).save(folder / f"{number}.png")
            index += 1
    pairs = "verification_data/unseen_a/0.png verification_data/unseen_a/1.png 1\nverification_data/unseen_a/0.png verification_data/unseen_b/0.png 0\n"
    (root / "verification_pairs_val.txt").write_text(pairs, encoding="utf-8")


class IdentityInputChecks(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parents[1] / "tmp"
        self.test_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="identity_import_synthetic_", dir=self.test_root)
        self.base = Path(self.temporary.name)
        self.assertTrue(self.base.resolve().is_relative_to(self.test_root.resolve()))
        self.source = self.base / "original"
        fixture(self.source)
        self.output = self.base / "imported"

    def tearDown(self):
        self.assertTrue(self.base.resolve().is_relative_to(self.test_root.resolve()))
        self.temporary.cleanup()

    def test_complete_read_only_audit_preserves_pairs(self):
        pair = self.source / "verification_pairs_val.txt"
        before = pair.read_bytes()
        report = audit_dataset(self.source)
        self.assertEqual(report["audit_status"], "ready")
        self.assertTrue(report["all_referenced_images_decoded"])
        self.assertEqual(report["images_decoded"], 14)  # 10 classification + all 4 verification images
        self.assertEqual(report["verification_images_present"], 4)
        self.assertEqual(report["pair_integrity"]["sha256"], sha256_file(pair))
        self.assertEqual(pair.read_bytes(), before)

    def test_missing_inputs_do_not_create_fake_data(self):
        absent = self.base / "absent"
        report = audit_dataset(absent)
        self.assertEqual(report["audit_status"], "not_ready")
        self.assertFalse(absent.exists())

    def test_unreadable_image_and_pair_label_conflicts_are_reported(self):
        (self.source / "classification_data/train_data/id_a/0.png").write_bytes(b"broken")
        pair = self.source / "verification_pairs_val.txt"
        original = pair.read_text()
        pair.write_text(original + "verification_data/unseen_a/1.png verification_data/unseen_a/0.png 0\n")
        report = audit_dataset(self.source)
        self.assertEqual(report["audit_status"], "not_ready")
        self.assertEqual(len(report["unreadable_images"]), 1)
        self.assertEqual(len(report["pair_integrity"]["conflicting_pairs"]), 1)
        self.assertEqual(report["pairs_removed"], 0)

    def test_pixel_duplicates_are_flagged_even_with_different_file_encoding(self):
        original = self.source / "classification_data/train_data/id_a/0.png"
        duplicate = self.source / "classification_data/val_data/id_b/duplicate.bmp"
        with Image.open(original) as image:
            image.save(duplicate)
        report = audit_dataset(self.source)
        self.assertEqual(report["audit_status"], "ready_with_warnings")
        self.assertTrue(report["decoded_pixel_overlap"])
        self.assertTrue(report["identical_pixels_under_different_ids"])
        self.assertTrue(duplicate.exists())

    def test_directory_copy_keeps_root_prefix_and_exact_pair_bytes(self):
        pair = self.source / "verification_pairs_val.txt"
        pair.write_text(pair.read_text().replace("verification_data/", "original/verification_data/"))
        report = import_dataset(self.source, self.output)
        copied = Path(report["data_root"])
        self.assertEqual(copied, self.output / "original")
        self.assertEqual((copied / pair.name).read_bytes(), pair.read_bytes())
        self.assertEqual(audit_dataset(copied)["audit_status"], "ready")
        self.assertFalse(report["pairs_regenerated"])

    def test_wrapped_zip_import_is_byte_preserving_and_repeatable(self):
        archive = self.base / "official.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in self.source.rglob("*"):
                if path.is_file():
                    handle.write(path, "wrapper/" + path.relative_to(self.source).as_posix())
        report = import_dataset(archive, self.output)
        self.assertEqual(Path(report["data_root"]), self.output / "wrapper")
        self.assertEqual(audit_dataset(report["data_root"])["audit_status"], "ready")
        self.assertEqual(import_dataset(archive, self.output)["archive_sha256"], report["archive_sha256"])
        self.assertFalse(list(self.output.rglob("*.importing")))
        archive_audit = audit_archive(archive, self.base / "archive_audit")
        self.assertEqual(archive_audit["audit_status"], "ready")
        self.assertTrue(archive_audit["all_images_decoded"])
        self.assertEqual(archive_audit["images_decoded"], 14)
        self.assertEqual(archive_audit["verification"]["pairs"], 2)

    def test_zip_traversal_and_windows_path_aliases_are_rejected_before_writing(self):
        for index, names in enumerate((("../escape.txt",), ("C:/escape.txt",), ("CON.txt",), ("data/A.png", "data/a.png"))):
            archive = self.base / f"unsafe{index}.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for name in names:
                    handle.writestr(name, "bad")
            with self.subTest(names=names), self.assertRaises(ValueError):
                import_dataset(archive, self.base / f"out{index}")
        self.assertFalse((self.base / "escape.txt").exists())

    def test_existing_different_image_is_never_overwritten(self):
        import_dataset(self.source, self.output)
        destination = self.output / "original/classification_data/train_data/id_a/0.png"
        destination.write_bytes(b"user content")
        with self.assertRaises(FileExistsError):
            import_dataset(self.source, self.output)
        self.assertEqual(destination.read_bytes(), b"user content")

    def test_ambiguous_dataset_roots_are_rejected(self):
        second = self.source / "another"
        fixture(second)
        with self.assertRaises(ValueError):
            find_dataset_root(self.source)


if __name__ == "__main__":
    unittest.main()
