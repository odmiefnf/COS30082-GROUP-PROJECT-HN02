"""Submission archives contain only the selected hand-in artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.package_submission import build_package, PREFIX, REQUIRED


class SubmissionPackageChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="package_fixture_", dir=base)
        self.root = Path(self.temp.name)
        for name in REQUIRED:
            self.put(name, b"fixture content")
        for directory in ("anti_spoofing", "attendance_app", "emotion_recognition", "face_verification", "tools", "tests"):
            self.put(directory+"/__init__.py", b"")

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value=b"private fixture"):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def test_archive_is_self_contained_allowlist_with_verified_hashes_and_no_private_data(self):
        self.put("docs/usage.md", b"Run the local GUI")
        self.put("results/comparison/triplet/roc_curve.svg", b"<svg/>")
        self.put("results/readiness_20261108/model_startup.json", b"{}")
        excluded = (
            "data/attendance/profiles.sqlite3", "data/downloads/raw.zip", "data/faces/photo.jpg",
            "tmp/private.json", ".env", "kaggle.json", "credentials.json", "reports/archive/old.docx",
            "attendance_app/__pycache__/gui.py", "results/unrelated/private.json",
            "results/readiness_20261108/profiles.sqlite3", "results/classification/credentials.json",
            "results/classification/last_model.pt", "results/classification/photo.jpg",
            "results/classification/personal/config.json",
        )
        for name in excluded:
            self.put(name)
        output = self.root/"submission.zip"
        result = build_package(self.root, output)
        self.assertEqual(result["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertTrue(all(name.startswith(PREFIX+"/") for name in names))
            self.assertTrue(all(f"{PREFIX}/{name}" not in names for name in excluded))
            self.assertIn(f"{PREFIX}/attendance_app/gui.py", names)
            self.assertIn(f"{PREFIX}/results/comparison/triplet/roc_curve.svg", names)
            self.assertIn(f"{PREFIX}/docs/usage.md", names)
            manifest = json.loads(archive.read(f"{PREFIX}/SUBMISSION_MANIFEST.json"))
            self.assertEqual(len(names), len(manifest["files"])+1)
            for name, metadata in manifest["files"].items():
                content = archive.read(f"{PREFIX}/{name}")
                self.assertEqual(metadata, {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        self.assertEqual(list(self.root.glob("submission.zip.*.tmp")), [])

    def test_missing_trained_model_fails_without_creating_output(self):
        (self.root/"results/triplet/best_model.pt").unlink()
        output = self.root/"submission.zip"
        with self.assertRaisesRegex(FileNotFoundError, "triplet/best_model.pt"):
            build_package(self.root, output)
        self.assertFalse(output.exists())

    def test_existing_archive_is_never_overwritten(self):
        output = self.put("submission.zip", b"existing hand-in")
        with self.assertRaises(FileExistsError):
            build_package(self.root, output)
        self.assertEqual(output.read_bytes(), b"existing hand-in")

    def test_unfinished_candidate_is_excluded_and_completed_candidate_is_optional(self):
        directory = "results/liveness_regularized_20261108/"
        self.put(directory+"best_model.pt")
        self.put(directory+"experiment_status.json", b'{"status":"training"}')
        first = self.root/"first.zip"
        build_package(self.root, first)
        with zipfile.ZipFile(first) as archive:
            self.assertNotIn(PREFIX+"/"+directory+"best_model.pt", archive.namelist())
        self.put(directory+"experiment_status.json", b'{"status":"completed"}')
        second = self.root/"second.zip"
        build_package(self.root, second)
        with zipfile.ZipFile(second) as archive:
            self.assertIn(PREFIX+"/"+directory+"best_model.pt", archive.namelist())


if __name__ == "__main__":
    unittest.main()
