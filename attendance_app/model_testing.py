"""Paired PAD reports and a cancellable runner; never access the attendance store."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from uuid import uuid4

from .device_check import camera_index
from .evaluation import ATTACKS, GENUINE, validate_presentation
from .history import spreadsheet_text


SCENARIOS = {
    "Live person · with glasses": ("glasses", "live-person"),
    "Live person · without glasses": ("no-glasses", "live-person"),
    "Phone still photo": ("screen-replay", "phone-still-photo"),
    "Recorded video": ("screen-replay", "recorded-video"),
    "Printed photo": ("print", "printed-photo"),
}
SCOPE = "Operator-labelled, correlated crops from one trial; not population accuracy or APCER/BPCER."


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Report counts must be nonnegative integers.")
    return value


def report_rows(report):
    """Validate the existing CLI report format before deriving display/export counts."""
    if not isinstance(report, dict):
        raise ValueError("Choose a paired PAD comparison JSON report.")
    condition = report.get("condition_supplied_by_operator")
    if condition not in (*GENUINE, *ATTACKS, "unspecified"):
        raise ValueError("Unsupported comparison condition.")
    validate_presentation(condition, report.get("presentation_type_supplied_by_operator"))
    same_crop = report.get("same_original_rgb_crop_for_both_models")
    if type(same_crop) is not bool or (not same_crop and report.get("same_original_rgb_frame_for_both_models") is not True):
        raise ValueError("The comparison requires paired crops from the same frame.")
    for field, expected in (("images_or_video_saved", False),
                            ("profiles_or_attendance_written", False),
                            ("thresholds_retuned_on_camera", False)):
        if report.get(field) is not expected:
            raise ValueError(f"Unsupported comparison report: {field}.")
    if "operator_cancelled" in report and type(report["operator_cancelled"]) is not bool:
        raise ValueError("A recorded cancellation flag must be boolean.")
    frames = _count(report.get("frame_count"))
    statuses = report.get("frame_statuses")
    if not isinstance(statuses, dict) or set(statuses) - {
        "scored", "no_face", "multiple_faces", "face_too_small_or_outside", "image_quality"
    } or sum(_count(value) for value in statuses.values()) != frames:
        raise ValueError("Frame-status counts do not match the captured-frame total.")
    started = report.get("started_utc")
    if not isinstance(started, str) or datetime.fromisoformat(started).tzinfo is None:
        raise ValueError("The report requires a timestamp with a timezone.")
    models = report.get("models")
    if not isinstance(models, dict) or set(models) != {"baseline", "candidate"}:
        raise ValueError("The report must contain baseline and candidate models.")
    rows = []
    for name in ("baseline", "candidate"):
        model = models[name]
        if not isinstance(model, dict):
            raise ValueError("Invalid model report.")
        threshold = model.get("threshold")
        if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                or not 0 <= threshold <= 1 or model.get("threshold_source") != "validation"):
            raise ValueError("Each model needs a finite validation-calibrated threshold.")
        if not isinstance(model.get("checkpoint"), str) or not model["checkpoint"]:
            raise ValueError("Each model requires a checkpoint path.")
        if not isinstance(model.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", model["sha256"]):
            raise ValueError("Each model requires a checkpoint SHA256.")
        scored, passed, rejected = (_count(model.get(key)) for key in ("scored_crops", "passed", "not_passed"))
        if passed + rejected != scored or scored != statuses.get("scored", 0):
            raise ValueError("Both models must score the same frames with consistent decision totals.")
        crop_policy = model.get("crop_policy", "tight")
        if crop_policy not in ("tight", "context-2x"):
            raise ValueError("Unsupported model crop policy.")
        correct = passed if condition in GENUINE else rejected if condition in ATTACKS else None
        rows.append({"model": name, "checkpoint": model["checkpoint"], "sha256": model["sha256"],
                     "threshold": threshold, "crop_policy": crop_policy, "scored_crops": scored, "passed": passed,
                     "not_passed": rejected, "correct": correct,
                     "incorrect": scored-correct if correct is not None else None,
                     "acceptance_rate": passed/scored if scored else None})
    if same_crop != (rows[0]["crop_policy"] == rows[1]["crop_policy"]):
        raise ValueError("Crop policies disagree with the reported pairing.")
    return rows


def load_report(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        content = source.read(2_000_001)
    if len(content) > 2_000_000:
        raise ValueError("The comparison report is too large.")
    report = json.loads(content)
    report_rows(report)
    json.dumps(report, allow_nan=False)
    return report


def export_report(destination, report):
    """Export one trial, protecting existing evidence and spreadsheet text."""
    rows = report_rows(report)
    destination = Path(destination)
    if destination.suffix.lower() == ".json":
        text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    elif destination.suffix.lower() == ".csv":
        buffer = io.StringIO(newline="")
        fields = ("started_utc", "condition", "presentation_type", "operator_cancelled", "frame_count",
                  *rows[0], "scope")
        writer = csv.DictWriter(buffer, fieldnames=fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            record = {"started_utc": report["started_utc"],
                      "condition": report["condition_supplied_by_operator"],
                      "presentation_type": report.get("presentation_type_supplied_by_operator"),
                      "operator_cancelled": report.get("operator_cancelled", "not-recorded"),
                      "frame_count": report["frame_count"], **row, "scope": SCOPE}
            writer.writerow({key: spreadsheet_text(value) for key, value in record.items()})
        text = buffer.getvalue()
    else:
        raise ValueError("Choose a new .json or .csv file.")
    with destination.open("x", encoding="utf-8-sig" if destination.suffix.lower() == ".csv" else "utf-8",
                          newline="") as target:
        target.write(text)


class ComparisonRun:
    """Run the existing camera tool outside Tk; cancellation remains cooperative."""
    def __init__(self, project, baseline, candidate, scenario, camera, seconds):
        self.project = Path(project).resolve()
        self.baseline, self.candidate = Path(baseline).resolve(), Path(candidate).resolve()
        if not self.baseline.is_file() or not self.candidate.is_file():
            raise ValueError("Choose existing baseline and candidate checkpoint files.")
        if scenario not in SCENARIOS:
            raise ValueError("Choose a supported presentation scenario.")
        if not math.isfinite(seconds) or not 0 < seconds <= 120:
            raise ValueError("Use a measured duration greater than 0 and at most 120 seconds.")
        self.scenario, self.camera, self.seconds = scenario, camera_index(camera), seconds
        self.process, self.log = None, None
        self.cancel_requested = False
        directory = self.project/"results/model_tests"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid4().hex[:8]
        self.output = directory/f"{run_id}.json"
        self.cancel_file = directory/f"{run_id}.cancel"

    def start(self):
        if self.process is not None:
            raise RuntimeError("A comparison is already running.")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        condition, presentation = SCENARIOS[self.scenario]
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        command = [str(executable), "-m", "tools.check_liveness_camera",
                   "--baseline", str(self.baseline), "--candidate", str(self.candidate),
                   "--condition", condition, "--presentation-type", presentation,
                   "--camera", str(self.camera), "--seconds", str(self.seconds), "--frames", "5000",
                   "--preview", "--output", str(self.output), "--cancel-file", str(self.cancel_file)]
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        try:
            self.process = subprocess.Popen(command, cwd=self.project, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError:
            self.log.close()
            self.log = None
            raise

    def cancel(self):
        if self.process is not None and self.process.poll() is None:
            self.cancel_file.touch(exist_ok=True)
            self.cancel_requested = True

    def poll(self):
        """Return (report or None, message) on completion, or None while running."""
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None:
            return None
        self.process = None
        try:
            self.log.seek(0)
            detail = self.log.read()[-4000:]
        finally:
            self.log.close()
            self.log = None
            self.cancel_file.unlink(missing_ok=True)
        if code:
            raise RuntimeError(detail.strip().splitlines()[-1][-500:] if detail.strip()
                               else f"Comparison stopped with exit code {code}.")
        if self.output.is_file():
            report = load_report(self.output)
            state = "Cancelled · Partial trial" if report["operator_cancelled"] else "Completed"
            return report, f"{state} · Saved {self.output}"
        if self.cancel_requested:
            return None, "Cancelled before opening the camera; no trial was recorded."
        raise RuntimeError("The comparison exited without a report. "+detail.strip())
