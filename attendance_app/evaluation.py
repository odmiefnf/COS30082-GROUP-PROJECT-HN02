"""Aggregate one operator-labelled webcam trial using a disposable gallery copy.

Correlated frames in one trial are not independent accuracy, FAR or FRR trials.
This module never saves camera pixels, crops or embeddings to its report.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import math
import numbers
from pathlib import Path
import sqlite3
import tempfile
import time

from .device_check import camera_index
from .store import ProfileStore, clean_text


CONDITIONS = ("glasses", "no-glasses", "unknown", "screen-replay", "print")
PRESENTATIONS = ("live-person", "phone-still-photo", "recorded-video", "printed-photo")
GENUINE = ("glasses", "no-glasses")
ATTACKS = ("screen-replay", "print")
COUNT_FIELDS = ("all_frames", "exactly_one_face", "quality_pass", "pad_scored", "pad_pass",
                "pad_not_passed", "pad_unavailable", "identity_scored", "identity_known",
                "identity_unknown", "identity_ambiguous", "identity_correct", "identity_wrong",
                "unknown_matched", "attack_target_matched", "attack_any_matched", "expression_scored")


def validate_ground_truth(condition, expected_profile, profiles):
    if condition not in CONDITIONS:
        raise ValueError("Unsupported webcam trial condition.")
    if not profiles:
        raise ValueError("The gallery must contain at least one enrolled profile for this evaluation.")
    if condition == "unknown":
        if expected_profile:
            raise ValueError("An unknown-person trial cannot name an enrolled expected profile.")
        return None
    if not expected_profile:
        raise ValueError("Confirm the exact enrolled profile name before running this trial.")
    name = clean_text(expected_profile)
    matches = [p for p in profiles if p["name"].casefold() == name.casefold()]
    if len(matches) != 1:
        raise ValueError("The expected profile name must match exactly one enrolled person.")
    return matches[0]["id"]


def ratio(numerator, denominator):
    return numerator/denominator if denominator else None


def validate_presentation(condition, presentation):
    if condition == "screen-replay":
        if presentation not in ("phone-still-photo", "recorded-video"):
            raise ValueError("Declare screen presentation as phone-still-photo or recorded-video.")
    elif condition == "print":
        if presentation != "printed-photo":
            raise ValueError("A print trial requires the declared presentation printed-photo.")
    elif presentation is not None and presentation != "live-person":
        raise ValueError("Genuine/unknown-person trials can only declare live-person presentation.")
    return presentation


class TrialAccumulator:
    def __init__(self, condition, expected_id):
        self.condition, self.expected_id = condition, expected_id
        self.counts = Counter({key: 0 for key in COUNT_FIELDS})
        self.statuses, self.quality_reasons = Counter(), Counter()
        self.frame_sizes = Counter()
        self.first_attendance_seconds = None

    def observe(self, result, elapsed, frame_size=None):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Measured frame time must be finite and nonnegative.")
        live = result.get("liveness")
        if live is not None:
            score, decision = live.get("live_score"), live.get("is_live")
            if score is not None and (isinstance(score, bool) or not isinstance(score, numbers.Real)
                                      or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError("PAD scored-crop denominator requires a finite score in [0,1].")
            if decision is not True and decision is not False and decision is not None:
                raise ValueError("PAD decision must be True, False or unavailable.")
            if decision is not None and score is None:
                raise ValueError("A PAD decision requires a finite scored-crop value.")
        c = self.counts
        c["all_frames"] += 1
        if frame_size is not None:
            self.frame_sizes[tuple(frame_size)] += 1
        self.statuses[result["status"]] += 1
        self.quality_reasons.update(result.get("quality_reasons", []))
        if len(result.get("boxes", [])) == 1:
            c["exactly_one_face"] += 1
            if (result.get("sharpness", -1) >= 20
                    and 20 <= result.get("brightness", -1) <= 235
                    and result["status"] != "face_too_small_or_outside"):
                c["quality_pass"] += 1
        if live is not None:
            if live.get("live_score") is not None:
                c["pad_scored"] += 1
            if live.get("is_live") is True:
                c["pad_pass"] += 1
            elif live.get("is_live") is False:
                c["pad_not_passed"] += 1
            else:
                c["pad_unavailable"] += 1
        identity = result.get("identity")
        if identity is not None:
            c["identity_scored"] += 1
            status = identity["status"]
            if status in ("known", "unknown", "ambiguous"):
                c["identity_"+status] += 1
            if status == "known":
                if self.condition in GENUINE:
                    c["identity_correct" if identity["person_id"] == self.expected_id else "identity_wrong"] += 1
                elif self.condition == "unknown":
                    c["unknown_matched"] += 1
                else:
                    c["attack_any_matched"] += 1
                    c["attack_target_matched"] += int(identity["person_id"] == self.expected_id)
        if result.get("expression") is not None:
            c["expression_scored"] += 1
        if result.get("attendance_recorded") and self.first_attendance_seconds is None:
            self.first_attendance_seconds = float(elapsed)

    def summary(self, elapsed, attendance_rows=()):
        c = self.counts
        target = sum(row["person_id"] == self.expected_id for row in attendance_rows) if self.expected_id else None
        counts = dict(c)
        if self.condition not in GENUINE:
            counts.update(identity_correct=None, identity_wrong=None)
        if self.condition != "unknown":
            counts["unknown_matched"] = None
        if self.condition not in ATTACKS:
            counts.update(attack_target_matched=None, attack_any_matched=None)
        return {
            "counts": counts, "frame_statuses": dict(self.statuses),
            "frame_sizes": [{"width": width, "height": height, "frames": count}
                            for (width, height), count in self.frame_sizes.items()],
            "quality_reasons": dict(self.quality_reasons), "duration_seconds": float(elapsed),
            "rates": {
                "one_face_per_all_frames": ratio(c["exactly_one_face"], c["all_frames"]),
                "quality_pass_per_all_frames": ratio(c["quality_pass"], c["all_frames"]),
                "pad_pass_per_scored_crop": ratio(c["pad_pass"], c["pad_scored"]),
                "pad_pass_per_all_frames": ratio(c["pad_pass"], c["all_frames"]),
                "correct_identity_per_scored_probe": ratio(c["identity_correct"], c["identity_scored"]) if self.condition in GENUINE else None,
                "correct_identity_per_all_frames": ratio(c["identity_correct"], c["all_frames"]) if self.condition in GENUINE else None,
                "unknown_match_per_scored_probe": ratio(c["unknown_matched"], c["identity_scored"]) if self.condition == "unknown" else None,
                "attack_match_per_scored_probe": ratio(c["attack_any_matched"], c["identity_scored"]) if self.condition in ATTACKS else None,
            },
            "simulated_attendance": {"scope": "Only in disposable gallery copy; not production attendance",
                "record_count": len(attendance_rows), "target_record_count": target,
                "other_record_count": len(attendance_rows)-target if target is not None else None,
                "correct_record_count": target if self.condition in GENUINE else None,
                "unknown_accept_record_count": len(attendance_rows) if self.condition == "unknown" else None,
                "attack_accept_record_count": len(attendance_rows) if self.condition in ATTACKS else None,
                "first_record_seconds": self.first_attendance_seconds},
        }


@contextmanager
def gallery_snapshot(source, encoder_sha256, dimension, temporary_root):
    """Read-only SQLite backup; all model-side writes stay in validated scratch."""
    source, temporary_root = Path(source).resolve(), Path(temporary_root).resolve()
    if not source.is_file():
        raise ValueError("The existing gallery database does not exist.")
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="webcam_trial_", dir=temporary_root)
    directory = Path(temporary.name).resolve()
    if not directory.is_relative_to(temporary_root):
        raise ValueError("Trial directory escaped the designated temporary root.")
    store = None
    try:
        destination = directory/"gallery.sqlite3"
        with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(destination)) as copied:
                original.backup(copied)
        store = ProfileStore(destination, encoder_sha256, dimension)
        yield store
    finally:
        if store is not None:
            store.close()
        # Checked absolute directory above; cleanup never targets the source.
        temporary.cleanup()


def collect_frames(camera_factory, processor, accumulator, *, camera=0, seconds=15., max_frames=300,
                   warmup_frames=8, clock=time.monotonic):
    """Capture bounded frames using the application processor; always release.

    OpenCV driver open/read calls can block independently of this duration.
    Warm-up frames are excluded from denominators and inference.
    """
    import cv2
    index = camera_index(camera)
    if not math.isfinite(seconds) or not 0 < seconds <= 120 or not 1 <= max_frames <= 5000:
        raise ValueError("Use a duration in (0,120] seconds and 1–5000 measured frames.")
    device = None
    try:
        device = camera_factory(index)
        device.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        device.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not device.isOpened():
            raise RuntimeError("The selected webcam could not be opened.")
        for _ in range(warmup_frames):
            ok, _ = device.read()
            if not ok:
                raise RuntimeError("The camera failed during warm-up.")
        started = clock()
        print("CAPTURE_STARTED: keep the declared person/condition in view.", flush=True)
        while accumulator.counts["all_frames"] < max_frames and clock()-started < seconds:
            ok, bgr = device.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            result = processor.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            accumulator.observe(result, clock()-started, (bgr.shape[1], bgr.shape[0]))
        if not accumulator.counts["all_frames"]:
            raise RuntimeError("No measured frames were processed; this trial is not completed.")
        return clock()-started
    finally:
        if device is not None:
            device.release()


def report_header(condition, expected_profile):
    return {"schema_version": 1, "status": "not_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One operator-labelled convenience trial; correlated frames are not independent accuracy/FAR/FRR trials",
        "operator_ground_truth": {"condition": condition, "expected_profile": expected_profile},
        "trial_count": 1, "production_database_read_only": True,
        "images_or_video_saved": False, "camera_embeddings_in_report": False,
        "profiles_or_attendance_written_to_production": False,
        "thresholds_retuned": False}
