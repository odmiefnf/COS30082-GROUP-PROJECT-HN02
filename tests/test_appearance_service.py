"""Explicit additional appearance capture; generated fixtures are not face benchmarks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from attendance_app.recognition import FaceRecognizer
from attendance_app.service import AttendanceService
from attendance_app.store import ProfileStore
from attendance_app.review import ManualReview


class AppearanceServiceChecks(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]/"tmp"
        root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.store = ProfileStore(Path(self.temp.name)/"test.sqlite3", "a"*64, 3)
        self.person = self.store.enroll("Sample", [[1., 0., 0.]]*3)
        self.recognizer = FaceRecognizer.__new__(FaceRecognizer)
        self.recognizer.dimension, self.recognizer.threshold, self.recognizer.margin = 3, .8, .05
        self.recognizer.embed = Mock(return_value=np.array([0., 1., 0.], dtype=np.float32))
        self.live = Mock()
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .9}
        self.service = AttendanceService(self.store, self.recognizer, self.live, Mock(), "test")

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def crop(self, color):
        return Image.new("RGB", (100, 100), (color, 50, 80))

    def begin(self):
        self.service.analyze(self.crop(1), now=10.)
        request = self.service.request_appearance(now=10.05)
        self.service.begin_appearance(self.person, "Without glasses", True, request_id=request["id"], now=10.1)

    def test_three_distinct_live_views_add_to_same_person_and_match_both_modes(self):
        self.begin()
        for index in range(3):
            result = self.service.analyze(self.crop(20+index), now=10.2+index*.5)
            self.assertFalse(result["attendance_recorded"])
        self.assertTrue(result["appearance_added"])
        profiles = self.store.profiles()
        self.assertEqual(len(profiles), 1)
        self.assertEqual(len(profiles[0]["appearances"]), 2)
        self.assertEqual(profiles[0]["sample_count"], 6)
        for query in ([1., 0., 0.], [0., 1., 0.]):
            self.assertEqual(self.recognizer.match(query, profiles)["person_id"], self.person)
        self.assertFalse(self.store.attendance("test"))

    def test_spoof_reset_and_duplicate_pixels_cannot_complete_capture(self):
        self.begin()
        for now in (10.2, 10.8, 11.4):
            result = self.service.analyze(self.crop(20), now=now)
        self.assertEqual(result["registration_samples"], 1)
        self.live.predict_crop.return_value = {"is_live": False, "live_score": .01}
        self.service.analyze(self.crop(21), now=11.5)
        self.live.predict_crop.return_value = {"is_live": True, "live_score": .9}
        result = self.service.analyze(self.crop(22), now=11.6)
        self.assertEqual(result["registration_samples"], 1)
        self.service.cancel_registration()
        self.assertEqual(len(self.store.profiles()[0]["appearances"]), 1)

    def test_explicit_confirmation_fresh_live_evidence_and_nonconflicting_identity_required(self):
        for confirmed in (False, 1):
            with self.assertRaises(ValueError):
                self.service.begin_appearance(self.person, "Without glasses", confirmed, now=10.)
        with self.assertRaises(ValueError):
            self.service.begin_appearance(self.person, "Without glasses", True, now=10.)
        self.service.analyze(self.crop(1), now=10.)
        with self.assertRaises(ValueError):
            self.service.begin_appearance(self.person, "Without glasses", True, now=12.)
        self.store.enroll("Other", [[0., 1., 0.]]*3)
        self.service.analyze(self.crop(1), now=13.)
        with self.assertRaises(ValueError):
            self.service.begin_appearance(self.person, "Without glasses", True, now=13.1)

    def test_switch_person_expiry_and_session_change_do_not_add_samples(self):
        for mode in ("person", "expired", "session"):
            with self.subTest(mode=mode):
                self.recognizer.embed.return_value = np.array([0., 1., 0.], dtype=np.float32)
                self.begin()
                if mode == "person":
                    self.recognizer.embed.return_value = np.array([0., 0., 1.], dtype=np.float32)
                if mode == "session":
                    self.service.session = "other"
                result = self.service.analyze(self.crop(2), now=41. if mode == "expired" else 10.2)
                self.assertIn(result["status"], ("appearance_person_changed", "appearance_expired"))
                self.assertIsNone(self.service.pending_appearance)
                self.assertEqual(len(self.store.profiles()[0]["appearances"]), 1)
                self.service.session = "test"

    def test_capture_and_open_dialog_hold_attendance_and_block_other_enrollment(self):
        self.recognizer.embed.return_value = np.array([1., 0., 0.], dtype=np.float32)
        self.service.appearance_dialog_open = True
        for index in range(4):
            self.assertFalse(self.service.analyze(self.crop(index), now=10.+index*.1)["attendance_recorded"])
        with self.assertRaises(ValueError):
            self.service.begin_registration("New")
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.4)
        self.service.cancel_registration()
        self.begin()
        with self.assertRaises(ValueError):
            self.service.begin_registration("New")
        with self.assertRaises(ValueError):
            self.service.request_manual_review(now=10.2)

    def test_manual_review_audit_uses_best_template_of_the_selected_person(self):
        self.store.add_appearance(self.person, "Without glasses", [[0., 1., 0.]]*3)
        review = ManualReview(3, .9)
        review.observe(self.crop(1), [.6, .8, 0.], .9, "unknown", "test", now=10.)
        ticket = review.begin(self.store.profiles(), now=10.1)
        result = review.confirm(self.store, review_id=ticket["id"], person_id=self.person,
                                reviewer="Fixture operator", reason="Software fixture only", confirmed=True,
                                session="test", now=10.2)
        self.assertAlmostEqual(result["cosine"], .8, places=6)
        self.assertAlmostEqual(self.store.attendance("test")[0]["cosine"], .8, places=6)


class AppearanceMatchingChecks(unittest.TestCase):
    def recognizer(self):
        r = FaceRecognizer.__new__(FaceRecognizer)
        r.dimension, r.threshold, r.margin = 3, .8, .05
        return r

    def test_templates_of_same_person_do_not_create_ambiguity(self):
        person = {"id": "a", "name": "A", "embedding": [1, 0, 0], "appearances": [
            {"embedding": [1, 0, 0]}, {"embedding": [.99, .01, 0]}, {"embedding": [0, 1, 0]}]}
        self.assertEqual(self.recognizer().match([0, 1, 0], [person])["status"], "known")
        other = {"id": "b", "name": "B", "embedding": [0, .99, .01]}
        self.assertEqual(self.recognizer().match([0, 1, 0], [person, other])["status"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
