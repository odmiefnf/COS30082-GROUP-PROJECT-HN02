"""History/export correctness on temporary databases, never real attendance."""
from __future__ import annotations

import csv
from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from attendance_app.history import (CSV_COLUMNS, export_attendance_csv, filter_attendance, local_timestamp,
                                    read_attendance_snapshot, spreadsheet_text, summarize_attendance)
from attendance_app.store import ProfileStore


class HistoryChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temp.name)
        self.database = self.root/"attendance.sqlite3"
        with_store = ProfileStore(self.database, "a"*64, 3)
        self.a = with_store.enroll("Nguyễn Kiệt / S123", [[1., 0., 0.]]*3)
        self.b = with_store.enroll("=1+2", [[0., 1., 0.]]*3)
        with_store.record_attendance("Class A", self.a, .95, .91)
        with_store.record_manual_attendance("Class A", self.b, .5, .9, reviewer="+Operator",
            reason='Checked, with "quoted" note', review_id="fixture-review-token-12345",
            observed_utc="2026-10-30T01:00:00+00:00")
        with_store.record_attendance("Class B", self.a, .96, .9)
        with_store.close()

    def tearDown(self):
        self.temp.cleanup()

    def digest(self):
        return hashlib.sha256(self.database.read_bytes()).hexdigest()

    def test_snapshot_reads_all_sessions_and_provenance_without_changing_database(self):
        before = self.digest()
        rows = read_attendance_snapshot(self.database)
        self.assertEqual(len(rows), 3)
        self.assertEqual({r["session"] for r in rows}, {"Class A", "Class B"})
        manual = next(r for r in rows if r["source"] == "manual_review")
        self.assertEqual(manual["reviewer"], "+Operator")
        self.assertIn("quoted", manual["reason"])
        self.assertNotIn("embedding", manual)
        self.assertNotIn("live_score", manual)
        self.assertEqual(self.digest(), before)

    def test_filtering_and_totals_preserve_snapshot_and_exact_session(self):
        rows = read_attendance_snapshot(self.database)
        selected = filter_attendance(rows, session="Class A", source="manual_review")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["person_id"], self.b)
        self.assertEqual(len(filter_attendance(rows, search="NGUYỄN KIỆT")), 2)
        self.assertFalse(filter_attendance(rows, session="Class"))
        self.assertEqual(summarize_attendance(rows), {"sessions": 2, "records": 3,
                         "automatic": 2, "manual_review": 1, "legacy_unrecorded": 0})
        selected[0]["name"] = "Local copy"
        self.assertNotEqual(next(r for r in rows if r["person_id"] == self.b)["name"], "Local copy")

    def test_legacy_database_has_unknown_source_and_is_not_migrated(self):
        legacy = self.root/"legacy.sqlite3"
        with closing(sqlite3.connect(legacy)) as connection:
            connection.executescript("CREATE TABLE profiles(id TEXT PRIMARY KEY,name TEXT);"
                "CREATE TABLE attendance(session TEXT,person_id TEXT,recorded_utc TEXT);")
            connection.execute("INSERT INTO profiles VALUES (?,?)", ("legacy", "Older record"))
            connection.execute("INSERT INTO attendance VALUES (?,?,?)", ("Old class", "legacy", "2025-01-01T01:00:00+00:00"))
            connection.commit()
        before = hashlib.sha256(legacy.read_bytes()).hexdigest()
        row = read_attendance_snapshot(legacy)[0]
        self.assertEqual(row["source"], "legacy_unrecorded")
        self.assertIsNone(row["reviewer"])
        self.assertEqual(hashlib.sha256(legacy.read_bytes()).hexdigest(), before)
        with closing(sqlite3.connect(legacy)) as connection:
            self.assertIsNone(connection.execute("SELECT 1 FROM sqlite_master WHERE name='attendance_audit'").fetchone())

    def test_missing_database_is_not_created_and_empty_database_returns_no_rows(self):
        missing = self.root/"missing.sqlite3"
        with self.assertRaises(FileNotFoundError):
            read_attendance_snapshot(missing)
        self.assertFalse(missing.exists())
        empty = self.root/"empty.sqlite3"
        store = ProfileStore(empty, "a"*64, 3)
        store.close()
        self.assertEqual(read_attendance_snapshot(empty), [])

    def test_csv_exports_filtered_snapshot_with_unicode_bom_and_literal_text(self):
        rows = filter_attendance(read_attendance_snapshot(self.database), session="Class A")
        before = self.digest()
        destination = self.root/"class.csv"
        self.assertEqual(export_attendance_csv(destination, rows), 2)
        self.assertTrue(destination.read_bytes().startswith(b"\xef\xbb\xbf"))
        with destination.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            exported = list(reader)
            self.assertEqual(tuple(reader.fieldnames), CSV_COLUMNS)
        self.assertEqual({r["session"] for r in exported}, {"Class A"})
        self.assertTrue(any(r["name"] == "Nguyễn Kiệt / S123" for r in exported))
        manual = next(r for r in exported if r["source"] == "Manual")
        self.assertEqual(manual["name"], "'=1+2")
        self.assertEqual(manual["reviewer"], "'+Operator")
        self.assertEqual(manual["reason"], 'Checked, with "quoted" note')
        self.assertEqual(manual["source_code"], "manual_review")
        self.assertTrue(manual["recorded_local"].endswith("+07:00"))
        self.assertEqual(self.digest(), before)
        self.assertEqual(next(r for r in rows if r["source"] == "manual_review")["name"], "=1+2")

    def test_csv_does_not_silently_overwrite_and_rejects_database_extension(self):
        destination = self.root/"existing.csv"
        destination.write_text("Existing document", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            export_attendance_csv(destination, [])
        self.assertEqual(destination.read_text(), "Existing document")
        self.assertEqual(export_attendance_csv(destination, [], overwrite=True), 0)
        with self.assertRaises(ValueError):
            export_attendance_csv(self.database, [], overwrite=True)

    def test_snapshot_is_stable_until_explicit_refresh(self):
        snapshot = read_attendance_snapshot(self.database)
        store = ProfileStore(self.database, "a"*64, 3)
        store.record_attendance("Later class", self.a, .9, .9)
        store.close()
        self.assertEqual(len(snapshot), 3)
        self.assertEqual(len(read_attendance_snapshot(self.database)), 4)
        output = self.root/"snapshot.csv"
        export_attendance_csv(output, snapshot)
        with output.open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_spreadsheet_text_blocks_formulas_even_after_whitespace_or_bom(self):
        for value in ("=1+2", "+SUM(A1)", "-1+2", "@SUM(A1)", "  =1", "\tPlain", "\n=1", "\ufeff \u2003=1"):
            with self.subTest(value=value):
                self.assertEqual(spreadsheet_text(value), "'"+value)
        self.assertEqual(spreadsheet_text("Nguyễn Kiệt"), "Nguyễn Kiệt")
        self.assertEqual(spreadsheet_text(None), "")
        self.assertEqual(local_timestamp("2026-10-29T20:00:00+00:00"), "2026-10-30T03:00:00+07:00")


if __name__ == "__main__":
    unittest.main()
