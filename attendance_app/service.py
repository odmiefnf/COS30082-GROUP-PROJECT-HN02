"""Live-only recognition, explicit registration and repeated-frame attendance."""
from __future__ import annotations

import hashlib
import secrets
import time

import numpy as np

from .store import clean_text
from .review import ManualReview


class AttendanceService:
    def __init__(self, store, recognizer, liveness, emotion, session, stable_frames=3):
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive.")
        self.store, self.recognizer = store, recognizer
        self.liveness, self.emotion = liveness, emotion
        self.session = clean_text(session, "Session")
        self.stable_frames = stable_frames
        self.pending_name = None
        self.pending_appearance = None
        self.appearance_dialog_open = False
        self.appearance_dialog_id = None
        self.appearance_request = None
        self.appearance_request_reason = ""
        self.capture_message = ""
        self.enrollment_samples = []
        self.enrollment_hashes = set()
        self.last_sample_time = float("-inf")
        self.last_person = None
        self.streak = 0
        self.last_frame_time = None
        self.profiles = store.profiles()
        self.review = ManualReview(recognizer.dimension, recognizer.threshold)
        self.review_open = False
        self.review_id = None

    def reset_tracking(self, reason="The face observation was interrupted. Cancel and try again."):
        self.review.invalidate()
        if self.appearance_dialog_open:
            self.appearance_request = None
            self.appearance_request_reason = reason
        self.last_person, self.streak, self.last_frame_time = None, 0, None
        self.enrollment_samples.clear()
        self.enrollment_hashes.clear()
        self.last_sample_time = float("-inf")

    def frame_state(self, now=None):
        """Lifecycle checks also run on frames rejected before CNN inference."""
        now = self.review.clock(now)
        if self.pending_appearance is not None and (
                now >= self.pending_appearance["expires"] or self.session != self.pending_appearance["session"]):
            self.pending_appearance = None
            self.enrollment_samples.clear()
            self.enrollment_hashes.clear()
            self.last_sample_time = float("-inf")
            self.last_person, self.streak = None, 0
            self.capture_message = "Appearance capture expired or the session changed. Start again."
        request = self.appearance_request
        if self.appearance_dialog_open and request is not None:
            latest = self.review.latest
            reason = ""
            if now >= request["expires"] or now < request["opened"]:
                reason = "This request has expired. Cancel and capture a new observation."
            elif self.session != request["session"]:
                reason = "The session changed. Cancel and start again."
            elif latest is None or not 0 <= now-latest.now <= 1.:
                reason = "The face observation is stale. Cancel and try again."
            if reason:
                self.appearance_request = None
                self.appearance_request_reason = reason
        return {
            "appearance_dialog_pending": self.appearance_dialog_open,
            "appearance_request_valid": self.appearance_dialog_open and self.appearance_request is not None,
            "appearance_request_reason": self.appearance_request_reason,
            "appearance_pending": self.pending_appearance is not None,
            "appearance_seconds_left": (max(0., self.pending_appearance["expires"]-now)
                                        if self.pending_appearance is not None else 0.),
            "registration_samples": len(self.enrollment_samples),
            "enrollment_pending": self.pending_name is not None or self.pending_appearance is not None,
            "capture_message": self.capture_message,
            "manual_review_pending": self.review_open,
            "manual_review_valid": self.review.active(now),
        }

    def request_appearance(self, now=None):
        now = self.review.clock(now)
        if self.pending_name is not None or self.pending_appearance is not None or self.review_open or self.appearance_dialog_open:
            raise ValueError("Finish or cancel the current capture or review first.")
        latest = self.review.latest
        if latest is None or not 0 <= now-latest.now <= 1. or latest.status not in ("unknown", "known"):
            raise ValueError("A fresh face that passes liveness is required.")
        self.profiles = self.store.profiles()
        if not self.profiles:
            raise ValueError("Register a profile before adding an appearance.")
        request_id = secrets.token_urlsafe(24)
        anchor = latest.embedding.copy()
        anchor.setflags(write=False)
        self.appearance_request = {"id": request_id, "anchor": anchor, "session": self.session,
                                   "opened": now, "expires": now+30., "allowed_ids": {p["id"] for p in self.profiles}}
        self.appearance_dialog_id = request_id
        self.appearance_dialog_open = True
        self.appearance_request_reason = ""
        self.capture_message = ""
        self.last_person, self.streak = None, 0
        return {"id": request_id, "expires_at": now+30.,
                "profiles": [{"id": p["id"], "name": p["name"]} for p in self.profiles]}

    def cancel_appearance_request(self, request_id):
        if self.appearance_dialog_id is not None and secrets.compare_digest(str(request_id), self.appearance_dialog_id):
            self.appearance_dialog_open = False
            self.appearance_dialog_id = self.appearance_request = None
            self.appearance_request_reason = ""
            self.last_person, self.streak = None, 0

    def begin_registration(self, name):
        if self.pending_appearance is not None or self.appearance_dialog_open:
            raise ValueError("Cancel appearance capture before registering a new person.")
        if self.review_open:
            raise ValueError("Close manual review before registering a new person.")
        name = clean_text(name)
        if any(p["name"].casefold() == name.casefold() for p in self.profiles):
            raise ValueError("This name is already registered. Add a student ID to distinguish people.")
        self.reset_tracking()
        self.capture_message = ""
        self.pending_name = name

    def begin_appearance(self, person_id, label, confirmed, request_id=None, now=None):
        """An operator explicitly binds a fresh live view to an existing profile.

        The operator's confirmation is enrollment intent, not automatic proof of
        identity. Unknown faces must never be attached without this action.
        """
        now = self.review.clock(now)
        if confirmed is not True:
            raise ValueError("Verify this is the person in the selected profile before adding an appearance.")
        if self.pending_name is not None or self.pending_appearance is not None or self.review_open:
            raise ValueError("Finish or cancel the current capture or manual review first.")
        self.frame_state(now)
        request = self.appearance_request
        if (not self.appearance_dialog_open or request is None
                or not secrets.compare_digest(str(request_id), request["id"])):
            raise ValueError(self.appearance_request_reason or "Open a fresh Add appearance request first.")
        label = clean_text(label, "Appearance label")
        self.profiles = self.store.profiles()
        selected = next((p for p in self.profiles if p["id"] == person_id), None)
        if selected is None or person_id not in request["allowed_ids"]:
            raise ValueError("Select an existing profile.")
        appearances = selected.get("appearances", [])
        if len(appearances) >= 4:
            raise ValueError("This profile already has four appearances, including the original.")
        if label.casefold() == "original" or any(a["label"].casefold() == label.casefold() for a in appearances):
            raise ValueError("This appearance label already exists. Use a different label.")
        latest = self.review.latest
        if latest is None or not 0 <= now-latest.now <= 1.:
            raise ValueError("A fresh face that passes liveness is required.")
        if float(np.dot(request["anchor"], latest.embedding)) < self.recognizer.threshold:
            self.appearance_request = None
            self.appearance_request_reason = "The face changed. Cancel and start again with the same person."
            raise ValueError(self.appearance_request_reason)
        identity = self.recognizer.match(latest.embedding, self.profiles)
        if (identity["status"] not in ("unknown", "known")
                or identity["status"] == "known" and identity["person_id"] != person_id):
            raise ValueError("This face matches another profile or is ambiguous. Check the selected person.")
        anchor = request["anchor"].copy()
        self.appearance_dialog_open = False
        self.appearance_dialog_id = self.appearance_request = None
        self.reset_tracking()
        self.pending_appearance = {"person_id": person_id, "label": label, "anchor": anchor,
                                   "expires": now+30., "session": self.session}
        self.appearance_dialog_open = False
        self.capture_message = ""

    def cancel_registration(self):
        self.pending_name = None
        self.pending_appearance = None
        self.appearance_dialog_open = False
        self.appearance_dialog_id = self.appearance_request = None
        self.appearance_request_reason = ""
        self.capture_message = ""
        self.review_open, self.review_id = False, None
        self.reset_tracking()

    def request_manual_review(self, now=None):
        if self.pending_name is not None or self.pending_appearance is not None or self.appearance_dialog_open:
            raise ValueError("Cancel registration before opening manual review.")
        if self.review_open:
            raise ValueError("Close the current manual review first.")
        ticket = self.review.begin(self.profiles, now)
        self.review_open, self.review_id = True, ticket["id"]
        self.last_person, self.streak = None, 0
        return ticket

    def cancel_manual_review(self, review_id):
        if review_id == self.review_id:
            self.review.cancel()
            self.review_open, self.review_id = False, None
            self.last_person, self.streak = None, 0

    def confirm_manual_review(self, **fields):
        result = self.review.confirm(self.store, session=self.session, **fields)
        self.review_open, self.review_id = False, None
        self.last_person, self.streak = None, 0
        return result

    def analyze(self, crop, now=None, max_observation_age=None):
        now = self.review.clock(now)
        was_capturing_appearance = self.pending_appearance is not None
        state = self.frame_state(now)
        appearance_expired = was_capturing_appearance and self.pending_appearance is None
        if self.last_frame_time is not None and now - self.last_frame_time > 1.5:
            self.reset_tracking()
        self.last_frame_time = now
        live = self.liveness.predict_crop(crop)
        result = {"status": "spoof" if live.get("is_live") is False else "liveness_unavailable",
                  "liveness": live, "identity": None, "expression": None,
                  "registered": False, "attendance_recorded": False, "registration_samples": 0,
                  "appearance_added": False, "appearance_pending": self.pending_appearance is not None,
                  "enrollment_pending": self.pending_name is not None or self.pending_appearance is not None,
                  "manual_review_pending": self.review_open, "manual_review_valid": False}
        try:
            result["expression"] = self.emotion.predict_crop(crop)
        except (OSError, ValueError, RuntimeError) as exc:
            result["expression_error"] = str(exc)
        if live.get("is_live") is True:
            embedding = self.recognizer.embed(crop)
            identity = self.recognizer.match(embedding, self.profiles)
        # Async camera input can age while a slow model runs. Recheck before
        # storing an observation, enrollment samples or attendance. Historical
        # offline/test calls retain their explicit deterministic clock.
        if max_observation_age is not None and not 0 <= time.monotonic()-now <= max_observation_age:
            self.reset_tracking("The camera observation is stale. Capture a fresh face.")
            result.update(status="observation_stale", liveness=None, identity=None, expression=None)
            result.update(self.frame_state())
            return result
        if live.get("is_live") is not True:
            self.reset_tracking("Liveness no longer passes. Cancel and capture a new observation.")
            result.update(self.frame_state(now))
            return result
        self.review.observe(crop, embedding, live["live_score"], identity["status"], self.session, now)
        if (self.appearance_dialog_open and self.appearance_request is not None
                and float(np.dot(self.appearance_request["anchor"], embedding)) < self.recognizer.threshold):
            self.appearance_request = None
            self.appearance_request_reason = "The face changed. Cancel and start again with the same person."
        result.update(status=identity["status"], identity=identity)
        result.update(self.frame_state(now))
        if appearance_expired:
            result["status"] = "appearance_expired"
            return result
        result["manual_review_pending"] = self.review_open
        result["manual_review_valid"] = self.review.active(now)
        if result["manual_review_pending"]:
            self.last_person, self.streak = None, 0
            return result
        if self.appearance_dialog_open:
            self.last_person, self.streak = None, 0
            return result
        if self.pending_appearance is not None:
            pending = self.pending_appearance
            if now >= pending["expires"] or self.session != pending["session"]:
                self.cancel_registration()
                return {**result, "status": "appearance_expired", "appearance_pending": False, "enrollment_pending": False}
            if (identity["status"] == "ambiguous"
                    or identity["status"] == "known" and identity["person_id"] != pending["person_id"]):
                self.cancel_registration()
                return {**result, "status": "appearance_conflict", "appearance_pending": False, "enrollment_pending": False}
            if float(np.dot(pending["anchor"], embedding)) < self.recognizer.threshold:
                self.cancel_registration()
                return {**result, "status": "appearance_person_changed", "appearance_pending": False, "enrollment_pending": False}
            digest = hashlib.sha256(np.asarray(crop).tobytes()).hexdigest()
            if digest not in self.enrollment_hashes and now-self.last_sample_time >= .45:
                self.enrollment_samples.append(embedding.copy())
                self.enrollment_hashes.add(digest)
                self.last_sample_time = now
            result.update(status="appearance_capturing", registration_samples=len(self.enrollment_samples))
            self.last_person, self.streak = None, 0
            if len(self.enrollment_samples) >= 3:
                appearance_id = self.store.add_appearance(pending["person_id"], pending["label"], self.enrollment_samples)
                self.profiles = self.store.profiles()
                self.cancel_registration()
                result.update(status="appearance_added", appearance_added=True, appearance_id=appearance_id,
                              person_id=pending["person_id"], appearance_pending=False, enrollment_pending=False)
            return result
        if self.pending_name is not None:
            if identity["status"] != "unknown":
                self.reset_tracking()
                result["status"] = "registration_existing_or_ambiguous"
                return result
            pixels = np.asarray(crop)
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            # Require a coherent sequence of live views of the same person.
            if self.enrollment_samples and float(np.dot(self.enrollment_samples[0], embedding)) < self.recognizer.threshold:
                self.reset_tracking()
                result["status"] = "registration_person_changed"
                return result
            if digest not in self.enrollment_hashes and now - self.last_sample_time >= .45:
                self.enrollment_samples.append(embedding.copy())
                self.enrollment_hashes.add(digest)
                self.last_sample_time = now
            result.update(status="registering", registration_samples=len(self.enrollment_samples))
            if len(self.enrollment_samples) >= 3:
                person_id = self.store.enroll(self.pending_name, self.enrollment_samples)
                result.update(status="registered", registered=True, registered_name=self.pending_name, person_id=person_id)
                self.profiles = self.store.profiles()
                self.cancel_registration()
            return result
        if identity["status"] == "known":
            self.streak = self.streak + 1 if identity["person_id"] == self.last_person else 1
            self.last_person = identity["person_id"]
            if self.streak >= self.stable_frames:
                result["attendance_recorded"] = self.store.record_attendance(
                    self.session, identity["person_id"], identity["cosine"], live["live_score"])
                result["status"] = "attendance_confirmed"
        else:
            self.last_person, self.streak = None, 0
        result["stable_frames"] = self.streak
        return result
