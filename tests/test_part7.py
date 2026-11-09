"""Manual-review lifecycle, audit integrity and legacy-database compatibility."""
from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from attendance_app.camera import FrameProcessor
from attendance_app.service import AttendanceService
from attendance_app.store import ProfileStore


class ManualReviewChecks(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temporary.name)
        self.store = ProfileStore(self.root/"profiles.sqlite3", "a"*64, 3)
        self.person_a = self.store.enroll("Test A", [[1., 0., 0.]]*3)
        self.person_b = self.store.enroll("Test B", [[0., 0., 1.]]*3)
        self.identity = Mock(dimension=3, threshold=.8)
        self.identity.embed.return_value = np.array([.6, .8, 0.], dtype=np.float32)
        self.identity.match.return_value = {"status": "unknown", "person_id": None, "name": None, "cosine": .6}
        self.live = Mock()
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .93}
        self.emotion = Mock()
        self.emotion.predict_crop.return_value = {"label": "angry"}
        self.service = AttendanceService(self.store, self.identity, self.live, self.emotion, "session-test")
        self.crop = Image.fromarray(np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8))

    def tearDown(self):
        self.store.close()
        self.temporary.cleanup()

    def request(self):
        self.service.analyze(self.crop, now=10.)
        return self.service.request_manual_review(now=10.1)

    def confirm(self, request, **changes):
        fields = {"review_id": request["id"], "person_id": self.person_a, "reviewer": "Test operator",
                  "reason": "Direct visual check with known test roster", "confirmed": True, "now": 10.2}
        fields.update(changes)
        return self.service.confirm_manual_review(**fields)

    def test_low_confidence_human_decision_preserves_score_and_records_provenance(self):
        request = self.request()
        original = request["crop"].tobytes()
        self.service.analyze(self.crop, now=10.15)
        result = self.confirm(request)
        self.assertTrue(result["recorded"])
        row = self.store.attendance("session-test")[0]
        self.assertEqual(row["source"], "manual_review")
        self.assertEqual(row["reviewer"], "Test operator")
        self.assertIn("visual check", row["reason"])
        self.assertAlmostEqual(row["cosine"], .6, places=6)
        self.assertLess(row["cosine"], self.identity.threshold)
        self.assertNotIn("expression", row)
        self.assertEqual(request["crop"].tobytes(), original)
        self.assertEqual(self.identity.match.return_value["status"], "unknown")
        self.assertEqual([p.name for p in self.root.iterdir()], ["profiles.sqlite3"])

    def test_explicit_selection_reviewer_reason_and_confirmation_required(self):
        request = self.request()
        for changes in ({"confirmed": False}, {"reviewer": ""}, {"reason": " "}, {"person_id": "missing"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.confirm(request, **changes)
            self.assertFalse(self.store.attendance("session-test"))
        self.assertTrue(self.confirm(request)["recorded"])

    def test_no_evidence_or_certain_identity_cannot_request_review(self):
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.)
        self.identity.match.return_value = {"status": "known", "person_id": self.person_a, "cosine": .95}
        self.service.analyze(self.crop, now=10.)
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.1)

    def test_spoof_or_missing_live_threshold_invalidates_ticket_and_blocks_writes(self):
        for value in (False, None):
            with self.subTest(value=value):
                self.live.predict_crop.return_value = {"is_live": True, "live_score": .93}
                request = self.request()
                self.live.predict_crop.return_value = {"is_live": value}
                self.service.analyze(self.crop, now=10.15)
                with self.assertRaises(ValueError):
                    self.confirm(request)
                self.assertIsNone(self.service.review.latest)
                self.assertFalse(self.store.attendance("session-test"))
                self.service.cancel_manual_review(request["id"])

    def test_changed_person_invalidates_ticket_and_holds_automatic_recording(self):
        request = self.request()
        self.identity.embed.return_value = np.array([0., 0., 1.], dtype=np.float32)
        self.identity.match.return_value = {"status": "known", "person_id": self.person_b, "cosine": .99}
        for index in range(4):
            result = self.service.analyze(self.crop, now=10.15+index*.1)
            self.assertTrue(result["manual_review_pending"])
            self.assertFalse(result["manual_review_valid"])
        with self.assertRaises(ValueError):
            self.confirm(request, now=10.5)
        self.assertFalse(self.store.attendance("session-test"))

    def test_no_face_or_multiple_faces_invalidates_review(self):
        locator = Mock()
        processor = FrameProcessor(locator, self.service)
        for boxes in ([], [(0, 0, 100, 100)]*2):
            request = self.request()
            locator.detect.return_value = boxes
            processor.process(np.asarray(self.crop))
            with self.assertRaises(ValueError):
                self.confirm(request)
            self.assertFalse(self.store.attendance("session-test"))
            self.service.cancel_manual_review(request["id"])

    def test_stale_latest_frame_and_expired_ticket_cannot_confirm(self):
        for now in (12., 41.):
            request = self.request()
            with self.assertRaises(ValueError):
                self.confirm(request, now=now)
            self.assertFalse(self.store.attendance("session-test"))
            self.service.cancel_manual_review(request["id"])

    def test_session_change_wrong_token_and_replay_cannot_write(self):
        request = self.request()
        with self.assertRaises(ValueError):
            self.confirm(request, review_id="wrong-token-12345678")
        self.service.session = "other-session"
        with self.assertRaises(ValueError):
            self.confirm(request)
        self.assertFalse(self.store.attendance("session-test"))
        self.assertFalse(self.store.attendance("other-session"))
        self.service.cancel_manual_review(request["id"])
        self.service.session = "session-test"
        request = self.request()
        self.confirm(request)
        with self.assertRaises(ValueError):
            self.confirm(request)
        self.assertEqual(len(self.store.attendance("session-test")), 1)

    def test_pending_review_suppresses_auto_until_cancel_and_preserves_auto_source(self):
        request = self.request()
        self.identity.match.return_value = {"status": "known", "person_id": self.person_a, "cosine": .95}
        for index in range(4):
            self.service.analyze(self.crop, now=10.2+index*.1)
        self.assertFalse(self.store.attendance("session-test"))
        self.service.cancel_manual_review("stale-dialog")
        self.assertTrue(self.service.review_open)
        self.service.cancel_manual_review(request["id"])
        for index in range(3):
            self.service.analyze(self.crop, now=10.6+index*.1)
        self.assertEqual(self.store.attendance("session-test")[0]["source"], "automatic")

    def test_existing_automatic_record_is_not_relabelled_as_manual(self):
        self.store.record_attendance("session-test", self.person_a, .95, .93)
        request = self.request()
        self.assertFalse(self.confirm(request)["recorded"])
        row = self.store.attendance("session-test")[0]
        self.assertEqual(row["source"], "automatic")
        self.assertIsNone(row["reviewer"])
        self.assertAlmostEqual(row["cosine"], .95)

    def test_register_and_review_are_mutually_exclusive(self):
        request = self.request()
        with self.assertRaises(ValueError):
            self.service.begin_registration("New test person")
        self.service.cancel_manual_review(request["id"])
        self.service.begin_registration("New test person")
        self.service.analyze(self.crop, now=10.3)
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.4)

    def test_audit_failure_rolls_back_attendance_and_profile_change_is_not_allowed(self):
        request = self.request()
        new_id = self.store.enroll("New test profile", [[1., 0., 0.]]*3)
        with self.assertRaises(ValueError):
            self.confirm(request, person_id=new_id)
        self.store.connection.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON attendance_audit BEGIN SELECT RAISE(ABORT,'test-only failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.confirm(request)
        self.assertFalse(self.store.attendance("session-test"))

    def test_legacy_rows_survive_and_wrong_encoder_does_not_modify_schema(self):
        legacy = self.root/"legacy.sqlite3"
        backup = sqlite3.connect(legacy)
        try:
            self.store.connection.backup(backup)
        finally:
            backup.close()
        connection = sqlite3.connect(legacy)
        connection.execute("DROP TABLE attendance_audit")
        connection.execute("INSERT INTO attendance VALUES (?,?,?,?,?)",
                           ("old-session", self.person_a, "2026-10-01T00:00:00+00:00", .9, .95))
        connection.commit(); connection.close()
        with self.assertRaises(ValueError):
            ProfileStore(legacy, "b"*64, 3)
        probe = sqlite3.connect(legacy)
        try:
            self.assertIsNone(probe.execute("SELECT name FROM sqlite_master WHERE name='attendance_audit'").fetchone())
        finally:
            probe.close()
        reopened = ProfileStore(legacy, "a"*64, 3)
        try:
            row = reopened.attendance("old-session")[0]
            self.assertEqual(row["source"], "legacy_unrecorded")
            self.assertEqual(row["name"], "Test A")
            self.assertIsNone(row["reviewer"])
        finally:
            reopened.close()


if __name__ == "__main__":
    unittest.main()
