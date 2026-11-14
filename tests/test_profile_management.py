"""Profile changes and reference import use temporary fixtures, not real identities."""
from contextlib import closing
from pathlib import Path
import queue
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from attendance_app.recognition import FaceRecognizer
from attendance_app.gui import CameraWorker
from attendance_app.reference_import import ReferenceImporter
from attendance_app.store import ProfileStore


class ProfileManagementChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.store = ProfileStore(self.root/"profiles.sqlite3", "a"*64, 3)
        self.person = self.store.enroll("Fixture A", [[1., 0., 0.]]*3)
        self.store.record_attendance("Class", self.person, .99, .9)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def test_rename_preserves_identity_samples_attendance_and_makes_backup(self):
        backup = self.store.rename_profile(self.person, "Kiệt / S123")
        profile = self.store.profiles()[0]
        self.assertEqual(profile["id"], self.person)
        self.assertEqual(profile["name"], "Kiệt / S123")
        self.assertEqual(profile["sample_count"], 3)
        self.assertEqual(self.store.attendance("Class")[0]["name"], "Kiệt / S123")
        with closing(sqlite3.connect(backup)) as source:
            self.assertEqual(source.execute("SELECT name FROM profiles").fetchone()[0], "Fixture A")
            self.assertEqual(source.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 1)
        self.assertIsNone(self.store.rename_profile(self.person, "Kiệt / S123"))

    def test_duplicate_name_last_group_and_foreign_group_are_rejected(self):
        other = self.store.enroll("Other", [[0., 1., 0.]]*3)
        for name in ("OTHER", "", "x\ny"):
            with self.assertRaises(ValueError):
                self.store.rename_profile(self.person, name)
        with self.assertRaises(ValueError):
            self.store.remove_appearance(self.person, "original:"+self.person)
        with self.assertRaises(ValueError):
            self.store.remove_appearance(self.person, "original:"+other)
        self.assertEqual(len(self.store.attendance("Class")), 1)

    def test_removing_original_retains_other_group_and_attendance_with_backup(self):
        group = self.store.add_appearance(self.person, "Without glasses", [[0., 1., 0.]]*3)
        backup = self.store.remove_appearance(self.person, "original:"+self.person)
        current = self.store.profiles()[0]
        self.assertEqual(current["sample_count"], 3)
        self.assertEqual(current["appearances"][0]["id"], group)
        self.assertEqual(len(self.store.attendance("Class")), 1)
        with closing(sqlite3.connect(backup)) as source:
            self.assertEqual(source.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 6)

    def test_failed_backup_aborts_change(self):
        self.store.add_appearance(self.person, "Other view", [[0., 1., 0.]]*3)
        with patch.object(self.store, "_backup_before_change", side_effect=OSError("Disk full")):
            with self.assertRaises(OSError):
                self.store.rename_profile(self.person, "Renamed")
            with self.assertRaises(OSError):
                self.store.remove_appearance(self.person, "original:"+self.person)
        self.assertEqual(self.store.profiles()[0]["name"], "Fixture A")
        self.assertEqual(self.store.profiles()[0]["sample_count"], 6)

    def test_worker_catalog_contains_no_embeddings_and_camera_blocks_management(self):
        updates = queue.Queue()
        worker = CameraWorker(SimpleNamespace(detector="haar"), queue.Queue(), updates)
        worker.service = SimpleNamespace(store=self.store, session="Class", profiles=[], pending_name=None,
                                         pending_appearance=None, review_open=False, appearance_dialog_open=False)
        worker.require_profile_management()
        worker.publish_profiles()
        data = updates.get_nowait()
        self.assertEqual(data["event"], "profiles_data")
        self.assertEqual(data["profiles"][0]["sample_count"], 3)
        self.assertNotIn("embedding", repr(data["profiles"]))
        worker.camera = Mock()
        with self.assertRaises(ValueError):
            worker.require_profile_management()


class ReferenceImportChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.store = ProfileStore(self.root/"profiles.sqlite3", "a"*64, 3)
        self.person = self.store.enroll("Fixture A", [[1., 0., 0.]]*3)
        self.recognizer = FaceRecognizer.__new__(FaceRecognizer)
        self.recognizer.dimension, self.recognizer.threshold, self.recognizer.margin = 3, .8, .05
        self.recognizer.fingerprint = "a"*64
        self.recognizer.embed = Mock(return_value=np.array([0., 1., 0.], dtype=np.float32))
        self.live = Mock()
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .9}
        self.locator = Mock()
        self.locator.detect.return_value = [(20, 20, 120, 120)]
        self.importer = ReferenceImporter(self.store, self.recognizer, self.live, self.locator)
        self.files = []
        rng = np.random.default_rng(42)
        for index in range(3):
            path = self.root/f"photo{index}.png"
            Image.fromarray(rng.integers(0, 256, (180, 180, 3), dtype=np.uint8)).save(path)
            self.files.append(str(path))
        self.operation = "fixture-operation-123456"

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def prepare(self, **changes):
        fields = dict(operation_id=self.operation, files=self.files, person_id=self.person, label="Without glasses", now=10.)
        fields.update(changes)
        return self.importer.prepare(**fields)

    def commit(self, checked, **changes):
        fields = dict(operation_id=self.operation, token=checked["token"], confirmed=True, now=11.)
        fields.update(changes)
        return self.importer.commit(**fields)

    def test_check_then_commit_adds_one_group_without_attendance_or_training(self):
        checked = self.prepare()
        self.assertTrue(checked["valid"])
        self.assertEqual(self.store.profiles()[0]["sample_count"], 3)
        self.commit(checked)
        self.assertEqual(len(self.store.profiles()), 1)
        self.assertEqual(self.store.profiles()[0]["sample_count"], 6)
        self.assertFalse(self.store.attendance("Class"))
        with self.assertRaises(ValueError):
            self.commit(checked)

    def test_context_pad_import_preserves_tight_identity_crop(self):
        self.live.config = {"context_crop": True}
        self.importer = ReferenceImporter(self.store, self.recognizer, self.live, self.locator)
        self.assertTrue(self.prepare()["valid"])
        with Image.open(self.files[0]) as image:
            pixels = np.asarray(image)
        pad_crop = self.live.predict_crop.call_args_list[0].args[0]
        identity_crop = self.recognizer.embed.call_args_list[0].args[0]
        np.testing.assert_array_equal(np.asarray(pad_crop), pixels)
        np.testing.assert_array_equal(np.asarray(identity_crop), pixels[20:140, 20:140])

    def test_new_person_and_duplicate_existing_identity_handling(self):
        checked = self.prepare(person_id=None, name="New person")
        result = self.commit(checked)
        self.assertNotEqual(result["person_id"], self.person)
        self.assertEqual(len(self.store.profiles()), 2)
        with self.assertRaises(ValueError):
            self.prepare(person_id=None, name="NEW PERSON")
        self.assertFalse(self.prepare(person_id=None, name="Duplicate face")["valid"])

    def test_duplicate_pixels_or_any_failed_gate_prevent_partial_writes(self):
        duplicate = self.root/"copy.png"
        duplicate.write_bytes(Path(self.files[0]).read_bytes())
        self.assertFalse(self.prepare(files=[self.files[0], str(duplicate), self.files[2]])["valid"])
        for result in (False, None):
            self.live.predict_crop.return_value = {"is_live": result}
            self.assertFalse(self.prepare()["valid"])
        self.assertEqual(self.store.profiles()[0]["sample_count"], 3)

    def test_missing_multi_face_low_quality_and_roi_are_rejected(self):
        for boxes in ([], [(0, 0, 100, 100)]*2, [(0, 0, 20, 20)]):
            self.locator.detect.return_value = boxes
            self.assertFalse(self.prepare()["valid"])
        self.locator.detect.return_value = [(20, 20, 120, 120)]
        blank = self.root/"blank.png"
        Image.new("RGB", (180, 180)).save(blank)
        self.assertFalse(self.prepare(files=[str(blank), self.files[1], self.files[2]])["valid"])
        self.locator.mode = "roi"
        self.assertFalse(self.prepare()["valid"])

    def test_inconsistent_or_other_known_person_is_not_attached(self):
        self.recognizer.embed.side_effect = [np.array(v, dtype=np.float32) for v in ([0,1,0],[0,0,1],[0,1,0])]
        self.assertFalse(self.prepare()["valid"])
        self.recognizer.embed.side_effect = None
        self.store.enroll("Other person", [[0., 1., 0.]]*3)
        self.assertFalse(self.prepare()["valid"])

    def test_confirmation_token_expiry_and_cancellation_are_enforced(self):
        checked = self.prepare()
        for fields in ({"confirmed": False}, {"token": "wrong"}, {"operation_id": "wrong-operation"}):
            with self.assertRaises(ValueError):
                self.commit(checked, **fields)
        with self.assertRaises(ValueError):
            self.commit(checked, now=131.)
        checked = self.prepare()
        self.importer.cancel("different-operation")
        self.assertIsNotNone(self.importer.pending)
        self.importer.cancel(self.operation)
        with self.assertRaises(ValueError):
            self.commit(checked)

    def test_checked_snapshot_is_used_and_target_changes_require_recheck(self):
        checked = self.prepare()
        for path in self.files:
            Image.new("RGB", (180, 180)).save(path)
        self.commit(checked)
        self.assertEqual(self.store.profiles()[0]["sample_count"], 6)
        self.assertEqual(len(list(self.root.glob("*.png"))), 3)

    def test_target_rename_or_new_conflict_after_check_requires_recheck(self):
        checked = self.prepare()
        self.store.rename_profile(self.person, "Renamed")
        with self.assertRaises(ValueError):
            self.commit(checked)
        checked = self.prepare()
        self.store.enroll("Other person", [[0., 1., 0.]]*3)
        with self.assertRaises(ValueError):
            self.commit(checked)
        self.assertEqual(next(p for p in self.store.profiles() if p["id"]==self.person)["sample_count"], 3)

    def test_encoder_mismatch_is_blocked(self):
        self.recognizer.fingerprint = "b"*64
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.store.profiles()[0]["sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
