"""Short-lived human review of a live, uncertain face; never changes model predictions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import secrets
import time

import numpy as np
from PIL import Image

from .store import clean_text, unit_vector
from .templates import profile_similarity


@dataclass(frozen=True)
class Observation:
    embedding: np.ndarray
    crop: Image.Image
    live_score: float
    status: str
    session: str
    now: float
    utc: str


class ManualReview:
    """One ticket per service; fresh live evidence and continuity required at commit."""
    def __init__(self, dimension, binding_threshold, ticket_seconds=30., freshness_seconds=1.5):
        if (not math.isfinite(binding_threshold) or not -1 <= binding_threshold <= 1
                or not math.isfinite(ticket_seconds) or not math.isfinite(freshness_seconds)
                or ticket_seconds <= 0 or freshness_seconds <= 0):
            raise ValueError("Invalid review timing or continuity threshold.")
        self.dimension, self.binding_threshold = dimension, binding_threshold
        self.ticket_seconds, self.freshness_seconds = ticket_seconds, freshness_seconds
        self.latest = None
        self.ticket = None

    def invalidate(self):
        self.latest = self.ticket = None

    def cancel(self):
        self.ticket = None

    @staticmethod
    def clock(now):
        value = time.monotonic() if now is None else float(now)
        if not math.isfinite(value):
            raise ValueError("Review clock must be finite.")
        return value

    def observe(self, crop, embedding, live_score, status, session, now=None):
        now = self.clock(now)
        if self.latest is not None and not 0 <= now - self.latest.now <= self.freshness_seconds:
            self.cancel()
        embedding = unit_vector(embedding, self.dimension).copy()
        embedding.setflags(write=False)
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(crop)
        if not isinstance(crop, Image.Image) or not 0 <= live_score <= 1:
            raise ValueError("Invalid live review observation.")
        self.latest = Observation(embedding, crop.convert("RGB").copy(), float(live_score), status,
                                  session, now, datetime.now(timezone.utc).isoformat())
        if self.ticket is not None:
            original = self.ticket["observation"]
            if (now >= self.ticket["expires"] or session != original.session
                    or now < original.now
                    or float(np.dot(embedding, original.embedding)) < self.binding_threshold):
                self.cancel()

    def active(self, now=None):
        now = self.clock(now)
        if self.ticket is not None and now >= self.ticket["expires"]:
            self.cancel()
        return self.ticket is not None

    def _fresh(self, now):
        return self.latest is not None and 0 <= now - self.latest.now <= self.freshness_seconds

    def begin(self, profiles, now=None):
        now = self.clock(now)
        if self.active(now):
            raise ValueError("A review is already open. Complete or cancel it first.")
        if not self._fresh(now) or self.latest.status not in ("unknown", "ambiguous"):
            raise ValueError("A fresh face that passes liveness with an uncertain identity is required.")
        if not profiles:
            raise ValueError("No profiles are available. Register a person first.")
        options = [{"id": p["id"], "name": p["name"]} for p in profiles]
        self.ticket = {"id": secrets.token_urlsafe(24), "observation": self.latest,
                       "expires": now + self.ticket_seconds, "allowed_ids": {p["id"] for p in profiles}}
        return {"id": self.ticket["id"], "expires_at": self.ticket["expires"],
                "observed_utc": self.latest.utc, "profiles": options, "crop": self.latest.crop.copy()}

    def confirm(self, store, *, review_id, person_id, reviewer, reason, confirmed, session, now=None):
        now = self.clock(now)
        if not self.active(now) or not secrets.compare_digest(str(review_id), self.ticket["id"]):
            raise ValueError("This review is no longer valid. Capture a fresh face that passes liveness.")
        ticket = self.ticket
        original = ticket["observation"]
        if (session != original.session or not self._fresh(now)
                or self.latest.session != original.session
                or float(np.dot(self.latest.embedding, original.embedding)) < self.binding_threshold):
            self.invalidate()
            raise ValueError("The face or session changed, or the observation expired. Start a new review.")
        if confirmed is not True:
            raise ValueError("Confirm that you checked the person directly against the selected profile.")
        reviewer, reason = clean_text(reviewer, "Reviewer"), clean_text(reason, "Reason")
        selected = next((p for p in store.profiles() if p["id"] == person_id), None)
        if person_id not in ticket["allowed_ids"] or selected is None:
            raise ValueError("Select a valid profile from the list.")
        # The score remains a score, even when the human decision disagrees.
        cosine = profile_similarity(self.latest.embedding, selected, self.dimension)
        recorded = store.record_manual_attendance(session, person_id, cosine, self.latest.live_score,
            reviewer=reviewer, reason=reason, review_id=review_id, observed_utc=original.utc)
        self.cancel()
        return {"recorded": recorded, "person_id": person_id, "name": selected["name"],
                "source": "manual_review", "cosine": cosine, "reviewer": reviewer, "reason": reason}
