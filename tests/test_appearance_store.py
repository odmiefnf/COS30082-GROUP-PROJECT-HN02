"""Appearance persistence/migration checks using vector fixtures, not real faces."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from attendance_app.store import ProfileStore


class AppearanceStoreChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temporary.name)/"profiles.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def legacy(self):
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL,name_key TEXT NOT NULL UNIQUE,created_utc TEXT NOT NULL);
            CREATE TABLE samples(person_id TEXT NOT NULL REFERENCES profiles(id),embedding BLOB NOT NULL);
            CREATE TABLE attendance(session TEXT NOT NULL,person_id TEXT NOT NULL,recorded_utc TEXT NOT NULL,
                cosine REAL NOT NULL,live_score REAL NOT NULL,PRIMARY KEY(session,person_id));
        """)
        connection.executemany("INSERT INTO metadata VALUES (?,?)", [("schema", "1"), ("encoder_sha256", "a"*64), ("dimension", "3")])
        connection.execute("INSERT INTO profiles VALUES (?,?,?,?)", ("old-id", "Kiệt / S1", "kiệt / s1", "2026-10-01T00:00:00+00:00"))
        blob = np.asarray([1., 0., 0.], dtype="<f4").tobytes()
        connection.executemany("INSERT INTO samples VALUES (?,?)", [("old-id", blob)]*4)
        connection.execute("INSERT INTO attendance VALUES (?,?,?,?,?)", ("old-session", "old-id", "2026-10-01T00:00:00+00:00", .95, .9))
        connection.commit()
        connection.close()
        return blob

    def test_old_schema_migrates_additively_and_keeps_one_original_group(self):
        blob = self.legacy()
        store = ProfileStore(self.path, "a"*64, 3)
        try:
            profile = store.profiles()[0]
            self.assertEqual(profile["id"], "old-id")
            self.assertEqual(profile["sample_count"], 4)
            self.assertEqual(len(profile["appearances"]), 1)
            self.assertEqual(profile["appearances"][0]["label"], "Original")
            self.assertEqual(profile["appearances"][0]["sample_count"], 4)
            self.assertEqual(store.connection.execute("SELECT embedding,appearance_id FROM samples").fetchall(), [(blob, None)]*4)
            self.assertEqual(dict(store.connection.execute("SELECT key,value FROM metadata")),
                             {"schema": "1", "encoder_sha256": "a"*64, "dimension": "3"})
            self.assertEqual(store.attendance("old-session")[0]["source"], "legacy_unrecorded")
        finally:
            store.close()

    def test_two_variants_persist_without_changing_original_centroid_or_identity(self):
        store = ProfileStore(self.path, "a"*64, 3)
        person = store.enroll("Test / S1", [[1., 0., 0.]]*3)
        original = store.profiles()[0]["embedding"].copy()
        first = store.add_appearance(person, "With glasses", [[0., 1., 0.]]*3)
        second = store.add_appearance(person, "Without glasses", [[0., 0., 1.]]*3)
        store.close()
        reopened = ProfileStore(self.path, "a"*64, 3)
        try:
            profile = reopened.profiles()[0]
            self.assertEqual(profile["id"], person)
            self.assertEqual(profile["sample_count"], 9)
            self.assertEqual([value["id"] for value in profile["appearances"]], ["original:"+person, first, second])
            self.assertEqual([value["sample_count"] for value in profile["appearances"]], [3, 3, 3])
            np.testing.assert_array_equal(profile["embedding"], original)
            np.testing.assert_array_equal(profile["appearances"][1]["embedding"], [0., 1., 0.])
        finally:
            reopened.close()

    def test_invalid_addition_never_changes_rows(self):
        store = ProfileStore(self.path, "a"*64, 3)
        try:
            person = store.enroll("Test", [[1., 0., 0.]]*3)
            invalid = [([], "Extra"), ([[0., 1., 0.]]*2, "Extra"), ([[0., 0., 0.]]*3, "Extra"),
                       ([[float("nan"), 1., 0.]]*3, "Extra"), ([[1., 0.]]*3, "Extra"),
                       ([[0., 1., 0.]]*3, ""), ([[1., 0., 0.], [-1., 0., 0.]]*2, "Extra")]
            for vectors, label in invalid:
                with self.subTest(label=label), self.assertRaises(ValueError):
                    store.add_appearance(person, label, vectors)
            with self.assertRaises(ValueError):
                store.add_appearance("missing", "Extra", [[0., 1., 0.]]*3)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM appearances").fetchone()[0], 0)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 3)
        finally:
            store.close()

    def test_duplicate_labels_and_four_group_limit_are_enforced(self):
        store = ProfileStore(self.path, "a"*64, 3)
        try:
            person = store.enroll("Test", [[1., 0., 0.]]*3)
            with self.assertRaises(ValueError):
                store.add_appearance(person, "ORIGINAL", [[0., 1., 0.]]*3)
            store.add_appearance(person, "Glasses", [[0., 1., 0.]]*3)
            with self.assertRaises(ValueError):
                store.add_appearance(person, " glasses ", [[0., 1., 0.]]*3)
            store.add_appearance(person, "No glasses", [[0., 0., 1.]]*3)
            store.add_appearance(person, "Other lighting", [[1., 1., 0.]]*3)
            with self.assertRaises(ValueError):
                store.add_appearance(person, "Fifth", [[1., 0., 1.]]*3)
            self.assertEqual(len(store.profiles()[0]["appearances"]), 4)
            self.assertEqual(store.profiles()[0]["sample_count"], 12)
        finally:
            store.close()

    def test_sample_insert_failure_rolls_back_the_new_group(self):
        store = ProfileStore(self.path, "a"*64, 3)
        try:
            person = store.enroll("Test", [[1., 0., 0.]]*3)
            store.connection.execute("CREATE TRIGGER fail_appearance BEFORE INSERT ON samples WHEN NEW.appearance_id IS NOT NULL BEGIN SELECT RAISE(ABORT,'fixture insert failure'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                store.add_appearance(person, "Glasses", [[0., 1., 0.]]*3)
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM appearances").fetchone()[0], 0)
            self.assertEqual(store.profiles()[0]["sample_count"], 3)
        finally:
            store.close()

    def test_wrong_fingerprint_is_rejected_before_any_schema_migration(self):
        self.legacy()
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            ProfileStore(self.path, "b"*64, 3)
        self.assertEqual(self.path.read_bytes(), before)
        connection = sqlite3.connect(self.path)
        try:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='appearances'").fetchone())
            self.assertNotIn("appearance_id", [row[1] for row in connection.execute("PRAGMA table_info(samples)")])
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
