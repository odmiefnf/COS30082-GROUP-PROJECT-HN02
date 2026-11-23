"""Fast local HTTP server for the Face Attendance app (http://127.0.0.1:8000).

Starts the localhost UI immediately while loading the three local CNN models in
the background, reusing CameraWorker, ProfileStore, AttendanceService and
history helpers without any extra third-party web framework dependencies.
"""
from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import queue
import shutil
import socket
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

from PIL import Image, ImageDraw, ImageFont

from .device_check import camera_index, probe_camera
from .diagnostics import COMPONENTS, frame_details, frame_guidance
from .gui import (BOX_MAX_AGE, CameraWorker, DEFAULT_LIVENESS_CHECKPOINT, FACE_PENDING, FACE_RECOGNIZED,
                  MESSAGES, PROJECT, force_exit, format_expression, message_tone, stat_tone)
from .history import (CSV_COLUMNS, SOURCE_LABELS, filter_attendance, local_timestamp,
                      read_attendance_snapshot, read_profile_names, spreadsheet_text, summarize_attendance)
from .history_dialog import preview_history_rows
from .web_ui import HTML_PAGE


def _scan_progress(text, tone):
    lowered = text.casefold()
    if "3/3" in text or text in (MESSAGES["attendance_confirmed"], MESSAGES["registered"], MESSAGES["appearance_added"]):
        return 1.0, "#10b981"
    if text == MESSAGES["known"]:
        return 0.72, "#12b896"
    if "2/3" in text:
        return 0.67, "#12b896"
    if "1/3" in text:
        return 0.34, "#12b896"
    if tone == "danger":
        return 1.0, "#ef4444"
    if tone == "warning":
        return 0.50, "#f59e0b"
    if lowered.startswith(("analyzing", "face the camera", "testing")):
        return 0.38, "#12b896"
    if lowered.startswith(("models ready", "opened")):
        return 0.16, "#12b896"
    return 0.0, "#12b896"


def _quality_map(result=None):
    if result is None:
        return {"face": "neutral", "lighting": "neutral", "focus": "neutral", "liveness": "neutral"}
    status = result.get("status", "")
    boxes = result.get("boxes", [])
    reasons = set(result.get("quality_reasons") or [])
    face = ("danger" if status == "multiple_faces"
            else "warning" if status in ("no_face", "face_too_small_or_outside") or len(boxes) != 1
            else "success")
    lighting = ("neutral" if "brightness" not in result
                else "warning" if reasons & {"dark", "bright"} else "success")
    focus = ("neutral" if "sharpness" not in result
             else "warning" if "blur" in reasons else "success")
    live = result.get("liveness")
    liveness = ("neutral" if live is None
                else "success" if live.get("is_live") is True
                else "danger" if live.get("is_live") is False else "warning")
    return {"face": face, "lighting": lighting, "focus": focus, "liveness": liveness}


def _measured_fps(samples, now):
    samples.append(now)
    while len(samples) > 2 and samples[0] < now - 2:
        samples.pop(0)
    return (len(samples) - 1) / (samples[-1] - samples[0]) if len(samples) > 1 and samples[-1] > samples[0] else 0.


def _format_session_rows(rows):
    formatted = []
    for row in rows:
        local = datetime.fromisoformat(row["recorded_utc"]).astimezone().strftime("%H:%M")
        source = row.get("source")
        label = {"automatic": "Automatic", "manual_review": "Manual"}.get(source, "Unknown")
        formatted.append({
            "name": row["name"],
            "time_local": local,
            "source": source or "legacy_unrecorded",
            "source_label": label,
            "reviewer": row.get("reviewer") or "",
            "reason": row.get("reason") or "",
        })
    return formatted


def _draw_corners(draw, box, color, width=4):
    """Mark a face with four corner brackets instead of a full rectangle."""
    x, y, w, h = box
    arm = max(8, round(min(w, h) * .22))
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        draw.line([(cx + dx * arm, cy), (cx, cy), (cx, cy + dy * arm)], fill=color, width=width, joint="curve")


_LABEL_FONT = []


def _label_font():
    if not _LABEL_FONT:
        try:
            _LABEL_FONT.append(ImageFont.truetype("arial.ttf", 20))  # covers Vietnamese names on Windows
        except OSError:
            _LABEL_FONT.append(ImageFont.load_default())
    return _LABEL_FONT[0]


def _draw_label(draw, box, text, color):
    """Name tag above the face box, or just inside its top edge when there is no room above."""
    x, y, w, h = box
    font = _label_font()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    ty = y - (bottom - top) - 12 if y - (bottom - top) - 12 >= 0 else y + 4
    draw.rectangle((x, ty, x + right - left + 12, ty + bottom - top + 8), fill=color)
    draw.text((x + 6 - left, ty + 4 - top), text, fill="#0b1220", font=font)


def _face_marks(result):
    """(box, colour, name) per detected face: green when live and recognized, yellow otherwise."""
    if result.get("faces"):
        marks = [(f["box"], FACE_RECOGNIZED if f["recognized"] else FACE_PENDING, f["name"]) for f in result["faces"]]
        scored = [list(f["box"]) for f in result["faces"]]
        return marks + [(box, FACE_PENDING, None) for box in result["boxes"] if list(box) not in scored]
    identity = result.get("identity") or {}
    recognized = (len(result["boxes"]) == 1 and (result.get("liveness") or {}).get("is_live") is True
                  and identity.get("status") == "known" and identity.get("person_id") is not None)
    return [(box, FACE_RECOGNIZED if recognized else FACE_PENDING, identity.get("name") if recognized else None)
            for box in result["boxes"]]


def _crop_to_data_url(crop, size=(220, 170), mirror=False):
    picture = crop.copy()
    if mirror:
        picture = picture.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    picture.thumbnail(size, Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    picture.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_csv_bytes(rows):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in rows:
        source = row.get("source", "legacy_unrecorded")
        record = {key: row.get(key) for key in CSV_COLUMNS}
        record.update(recorded_local=local_timestamp(row.get("recorded_utc")),
                      source=SOURCE_LABELS.get(source, "Unknown"),
                      source_code=source if source in SOURCE_LABELS else "legacy_unrecorded")
        writer.writerow({key: spreadsheet_text(record[key]) for key in CSV_COLUMNS})
    return buf.getvalue().encode("utf-8-sig")


class WebController:
    """Owns the background CameraWorker and exposes thread-safe web state & frames."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.closing = False
        self.commands = queue.Queue()
        self.updates = queue.Queue()
        self.previews = queue.Queue(maxsize=1)
        self.analyses = queue.Queue(maxsize=1)
        self.worker = None
        self.camera_probe = None

        self.stream_id = None
        self.last_analysis = None
        self.preview_times = []
        self.analysis_times = []
        self.preview_fps = 0.0
        self.analysis_fps = 0.0
        self.latest_jpeg = None
        self.latest_plain_jpeg = None  # same frame without face boxes, for the Register page
        self.plain_viewers = 0
        self.jpeg_seq = 0

        self.models_ready = False
        self.setup_loading = not args.preview
        self.running = False
        self.camera_starting = False
        self.probe_busy = False
        self.session = args.session
        self.camera = args.camera
        self.camera_status = "Not tested"
        self.camera_summary = f"Camera {args.camera} · Not tested"
        self.state_text = "Preview · Camera off" if args.preview else "Loading three models…"
        self.component_status = {}
        self.last_frame_details = ""

        self.feedback = "Click Start camera when you are ready"
        self.guidance = "Keep one face centered in good lighting. Identity, liveness and expression update live."
        self.last_status = ""
        self.identity_text = "—"
        self.liveness_text = "—"
        self.expression_text = "—"
        self.quality = _quality_map(None)
        self.pad_score = None
        self.pad_raw_score = None  # single-frame P(live) before smoothing
        self.pad_threshold = None
        self.id_cosine = None
        self.id_threshold = None
        self.top_matches = []
        self.stable_frames = 0
        self.stable_frames_target = 3
        # Average PAD over 5 frames by default (the sample app uses 8); 5 is this UI's maximum.
        self.liveness_smooth_window = 5
        self.registration_samples = 0
        self.processing_ms = None
        self.mirror_preview = True
        self.show_hud = True
        self.emotion_scores = {}
        self.emotion_top = None
        self.profile_count = 0
        self.attendance = []
        self.spoof_events = []
        self.activity_log = []
        self._last_spoof_log_at = float("-inf")
        self._seen_attendance_keys = set()
        self.model_info = {
            "identity_threshold": None, "identity_margin": None, "identity_dim": 128,
            "identity_checkpoint": "triplet/best_model.pt",
            "liveness_threshold": None, "liveness_checkpoint": Path(args.liveness_checkpoint).name if hasattr(args, "liveness_checkpoint") else "best_model.pt",
            "emotion_checkpoint": Path(args.emotion_checkpoint).name if hasattr(args, "emotion_checkpoint") else "best_model.pt",
            "stable_frames": 3, "device": getattr(args, "device", "cpu"),
            "num_threads": getattr(args, "num_threads", 4),
        }
        self.can_register = False
        self.can_capture_shot = False
        self.can_appearance = False
        self.can_review = False
        self.capture_pending = False
        self.guided_shots = []
        self.photo_import_available = args.detector != "roi"

        self.notice_id = 0
        self.notice = None
        self.review_req = None
        self.appearance_req = None

        # Synchronous waiter slots for REST profile operations
        self._rpc_lock = threading.Lock()
        self._pending_rpc = None
        self._import_dirs = {}

        if args.preview:
            self.component_status = {
                "Runtime": {"status": "Ready", "detail": "PREVIEW · No models or hardware loaded"},
                "Identity": {"status": "Ready", "detail": "PREVIEW · Triplet CNN (disabled in preview)"},
                "Liveness": {"status": "Ready", "detail": "PREVIEW · PAD CNN (disabled in preview)"},
                "Expression": {"status": "Ready", "detail": "PREVIEW · FER2013 CNN (disabled in preview)"},
                "Detector": {"status": "Ready", "detail": "PREVIEW · Haar cascade"},
                "Profiles": {"status": "Ready", "detail": "PREVIEW · Fictional gallery only"},
            }
            self.model_info.update(identity_threshold=0.86954, identity_margin=0.05, liveness_threshold=0.52)
            self.id_threshold = 0.86954
            self.profile_count = 2
            self.spoof_events = [{
                "time_local": "09:04:12", "session": args.session,
                "pad_score": 0.14, "threshold": 0.52, "note": "Preview example · Spoof blocked (memory only)",
            }]
            self.activity_log = [
                {"time": "09:05:00", "kind": "manual", "text": "Demo student B checked in (Manual review)"},
                {"time": "09:04:12", "kind": "spoof", "text": "Blocked spoof attempt (PAD 0.14 < 0.52)"},
                {"time": "09:01:00", "kind": "checkin", "text": "Demo student A checked in (Automatic)"},
            ]
        else:
            self._start_worker()
            self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
            self._pump_thread.start()

    def _start_worker(self):
        self.commands = queue.Queue()
        self.worker = CameraWorker(self.args, self.commands, self.updates, self.previews, self.analyses)
        self.worker.start()

    def _send(self, command, value=None):
        if self.worker is not None and self.worker.is_alive():
            self.commands.put((command, value))

    def _set_notice(self, message):
        self.notice_id += 1
        self.notice = {"id": self.notice_id, "message": message}

    def _log_activity(self, kind, text):
        stamp = datetime.now().astimezone().strftime("%H:%M:%S")
        self.activity_log.insert(0, {"time": stamp, "kind": kind, "text": text})
        del self.activity_log[30:]

    def _resolve_rpc(self, kind, payload):
        waiter = self._pending_rpc
        if waiter is not None and kind in waiter["expect"]:
            waiter["result"] = (kind, payload)
            waiter["event"].set()
            return True
        return False

    def _call_profile_rpc(self, command, value, expect, timeout=30.0):
        if self.args.preview:
            raise ValueError("Preview mode: no database or photo changes are performed.")
        if not self.models_ready:
            raise ValueError("Models are still loading. Wait a moment and try again.")
        if self.running or self.camera_starting:
            raise ValueError("Stop the camera before managing profiles.")
        with self._rpc_lock:
            event = threading.Event()
            waiter = {"expect": set(expect), "event": event, "result": None}
            self._pending_rpc = waiter
            try:
                self._send(command, value)
                if not event.wait(timeout):
                    raise RuntimeError("Timed out waiting for profile operation.")
                return waiter["result"]
            finally:
                self._pending_rpc = None

    def _pump_loop(self):
        while not self.closing:
            worked = False
            while True:
                try:
                    update = self.updates.get_nowait()
                except queue.Empty:
                    break
                worked = True
                self._handle_event(update)
            try:
                analysis = self.analyses.get_nowait()
                worked = True
                self._handle_frame(analysis)
            except queue.Empty:
                pass
            try:
                preview = self.previews.get_nowait()
                worked = True
                self._handle_preview(preview)
            except queue.Empty:
                pass
            self._expire_stale()
            if not worked:
                time.sleep(0.015)

    def _handle_event(self, update):
        event = update["event"]
        with self.lock:
            if event == "component":
                self.component_status[update["component"]] = {
                    "status": update["status"], "detail": update["detail"]
                }
            elif event == "camera_probe":
                self.probe_busy = False
                report = update["report"]
                self.camera_status = report["message"]
                summary = f"Ready · {report['width']}×{report['height']}" if report["ok"] else "Test failed · See Diagnostics"
                self.camera_summary = f"Camera {report['index']} · {summary}"
                self.feedback = report["message"]
                self._set_notice(report["message"])
            elif event == "ready":
                self.setup_loading = False
                self.models_ready = True
                self.state_text = "Ready · Camera off"
                self.profile_count = update["profiles"]
                self.attendance = _format_session_rows(update["attendance"])
                self._seen_attendance_keys = {(self.session, r["name"]) for r in self.attendance}
                if update.get("model_info"):
                    self.model_info.update(update["model_info"])
                    self.id_threshold = self.model_info.get("identity_threshold")
                    self.stable_frames_target = int(self.model_info.get("stable_frames", 3))
                self._send("set_liveness_smooth", self.liveness_smooth_window)
                self.feedback = "Models ready · Click Start camera to begin"
                self.guidance = "Keep one face centered in good lighting, or open the Register tab to enroll someone new."
            elif event == "config_updated":
                if "stable_frames" in update:
                    self.stable_frames_target = int(update["stable_frames"])
                    self.model_info["stable_frames"] = self.stable_frames_target
                if "liveness_smooth_window" in update:
                    self.liveness_smooth_window = int(update["liveness_smooth_window"])
            elif event == "started":
                self.camera_starting = False
                self.running = True
                self.stream_id = update["stream_id"]
                self.last_analysis = None
                self.preview_times.clear()
                self.analysis_times.clear()
                self.preview_fps = self.analysis_fps = 0.0
                self.camera = update.get("camera", self.camera)
                self.camera_status = "Opened successfully"
                self.camera_summary = f"Camera {self.camera} · Running"
                self.state_text = "Camera running"
                self.attendance = _format_session_rows(update["attendance"])
                self._seen_attendance_keys = {(self.session, r["name"]) for r in self.attendance}
            elif event in ("stopped", "camera_error"):
                self.camera_starting = False
                self.running = False
                self.stream_id = None
                self.last_analysis = None
                self.review_req = None
                self.appearance_req = None
                self.state_text = "Camera off"
                self.feedback = update.get("message", "Camera stopped")
                self.camera_status = update.get("message", "Stopped; device released")
                self.camera_summary = f"Camera {self.camera} · {'Error · See Diagnostics' if event == 'camera_error' else 'Off'}"
                self.last_frame_details = ""
                self.last_status = ""
                self.guidance = "Click Start camera to resume live verification, or open Register / Profiles."
                self.identity_text = self.liveness_text = self.expression_text = "—"
                self.quality = _quality_map(None)
                self.pad_score = self.pad_raw_score = self.pad_threshold = self.id_cosine = None
                self.top_matches = []
                self.stable_frames = 0
                self.registration_samples = 0
                self.processing_ms = None
                self.emotion_scores = {}
                self.emotion_top = None
                self.can_register = self.can_capture_shot = self.can_appearance = self.can_review = False
                self.guided_shots = []
                self.preview_fps = self.analysis_fps = 0.0
                if event == "camera_error":
                    self._set_notice(self.feedback)
                with self.frame_cond:
                    self.frame_cond.notify_all()
            elif event == "fatal":
                self.setup_loading = False
                self.models_ready = False
                self.state_text = "Not ready"
                comp = update.get("component", "Runtime")
                self.feedback = f"{comp} setup failed · Open Diagnostics for details"
                self.guidance = COMPONENTS.get(comp, ("", "Check the project environment."))[1]
            elif event == "notice":
                self._set_notice(update["message"])
            elif event == "profiles_data":
                self.profile_count = len(update["profiles"])
                self.attendance = _format_session_rows(update["attendance"])
                self._resolve_rpc("profiles_data", update)
            elif event == "profiles_snapshot":
                self._resolve_rpc("profiles_snapshot", update)
            elif event == "profiles_error":
                if not self._resolve_rpc("profiles_error", update):
                    self._set_notice(update["message"])
            elif event == "profile_import_checked":
                self._resolve_rpc("profile_import_checked", update)
            elif event == "profile_import_done":
                self._cleanup_import_dir(update.get("operation_id"))
            elif event == "appearance_open":
                req = update["request"]
                self.appearance_req = {
                    "id": req["id"], "expires_at": req["expires_at"],
                    "profiles": req["profiles"], "valid": True, "reason": ""
                }
            elif event == "appearance_started":
                self.appearance_req = None
                self.feedback = "Capturing an additional appearance · Keep the same person in view"
            elif event == "review_open":
                req = update["request"]
                observed_local = datetime.fromisoformat(req["observed_utc"]).astimezone().strftime("%H:%M:%S")
                self.review_req = {
                    "id": req["id"], "expires_at": req["expires_at"],
                    "observed_local": observed_local,
                    "crop_data_url": _crop_to_data_url(req["crop"]),
                    "profiles": req["profiles"], "valid": True, "reason": ""
                }
            elif event == "review_done":
                self.review_req = None
                self.attendance = _format_session_rows(update["attendance"])
                if update["result"]["recorded"]:
                    for row in self.attendance:
                        key = (self.session, row["name"])
                        if key not in self._seen_attendance_keys:
                            self._seen_attendance_keys.add(key)
                            self._log_activity("manual", f"{row['name']} checked in (Manual review)")
                msg = "Manual attendance recorded" if update["result"]["recorded"] else "Attendance already recorded for this session"
                self.state_text = msg
                self._set_notice(msg)
                self._resolve_rpc("review_done", update)
            elif event == "review_error":
                if not self._resolve_rpc("review_error", update):
                    self._set_notice(update["message"])
            elif event == "guided_state":
                mirror = self.mirror_preview
                self.guided_shots = [{
                    "id": s["id"],
                    "slot": s["slot"],
                    "live_score": round(float(s["live_score"]), 4),
                    "is_live": bool(s.get("is_live", True)),
                    "thumb_url": _crop_to_data_url(s["crop"], size=(148, 148), mirror=mirror),
                } for s in update["shots"]]
                self._resolve_rpc("guided_state", {"shots": list(self.guided_shots)})
            elif event == "guided_committed":
                res = update["result"]
                if "duplicate" not in res:
                    self.guided_shots = []
                    self.profile_count = update["profiles"]
                    self.attendance = _format_session_rows(update["attendance"])
                    if res.get("mode") == "appearance":
                        msg = f"Added appearance '{res['label']}' to {res['name']} ({res['used_shots']} live shots)"
                    else:
                        msg = f"Registered {res['name']} ({res['used_shots']} live shots)"
                    self.feedback = msg
                    self._set_notice(msg)
                self._resolve_rpc("guided_committed", update)
            elif event == "guided_error":
                if not self._resolve_rpc("guided_error", update):
                    self._set_notice(update["message"])

    def _handle_frame(self, update):
        with self.lock:
            if not self.running or update["stream_id"] != self.stream_id:
                return
            if time.monotonic() - update["captured_at"] > 1:
                return
            self.last_analysis = update
            self.analysis_fps = _measured_fps(self.analysis_times, time.monotonic())
            result = update["result"]
            status = result["status"]
            self.last_status = status
            message = MESSAGES.get(status, status)
            if status == "registering":
                message = f"Sample {result['registration_samples']}/3 · Turn your face slightly"
            if result.get("appearance_pending"):
                seconds_left = max(0., result.get("appearance_seconds_left", 0.))
                progress = f"Appearance sample {result['registration_samples']}/3 · {seconds_left:.0f}s left"
                message = f"{progress} · Turn your face slightly" if status == "appearance_capturing" else f"{progress} · {message}"
            if self.appearance_req is not None and result.get("appearance_dialog_pending") and not result.get("appearance_request_valid"):
                self.appearance_req["valid"] = False
                self.appearance_req["reason"] = result.get("appearance_request_reason") or "The observation is no longer valid."
            if result.get("manual_review_pending"):
                message = "Manual review in progress · Keep the same person in view"
                if self.review_req is not None and not result.get("manual_review_valid"):
                    self.review_req["valid"] = False
                    self.review_req["reason"] = "The face or liveness is no longer valid. Cancel and try again."
            if result.get("faces"):
                names = [f["name"] for f in result["faces"] if f["recognized"]]
                message = (f"{result['face_count']} faces · "
                           + (f"Recognized: {', '.join(names)}" if names else "No one recognized yet"))
            self.feedback = message
            self.guidance = frame_guidance(result)
            self.last_frame_details = (
                frame_details(result)
                + f"\nAnalysis: {update['processing_seconds']*1000:.1f} ms"
                + f"\nObservation age: {(time.monotonic()-update['captured_at'])*1000:.0f} ms"
                + f"\nPreview: {self.preview_fps:.1f} FPS; analysis: {self.analysis_fps:.1f} FPS"
            )
            ident = result.get("identity")
            self.identity_text = (ident or {}).get("name") or ("Not recognized" if ident is not None else "Not checked")
            live = result.get("liveness")
            self.liveness_text = ("Not checked" if live is None else "Passed" if live.get("is_live") is True
                                  else "Unavailable" if live.get("is_live") is None else "Not passed")
            expr = result.get("expression")
            self.expression_text = format_expression(expr, result.get("expression_error"))
            self.quality = _quality_map(result)
            self.pad_score = (live or {}).get("live_score")
            self.pad_raw_score = (live or {}).get("raw_live_score", self.pad_score)
            self.pad_threshold = (live or {}).get("threshold")
            self.id_cosine = (ident or {}).get("cosine")
            self.id_threshold = (ident or {}).get("threshold", self.id_threshold)
            self.top_matches = (ident or {}).get("top_matches") or []
            self.stable_frames = result.get("stable_frames", 0)
            self.registration_samples = result.get("registration_samples", 0)
            self.processing_ms = round(update["processing_seconds"] * 1000.0, 1)
            self.emotion_scores = (expr or {}).get("scores") or {}
            self.emotion_top = (expr or {}).get("label")
            self.profile_count = update["profiles"]
            self.attendance = _format_session_rows(update["attendance"])
            # Log new automatic check-ins and debounced blocked spoof attempts (memory only, no images).
            if result.get("attendance_recorded") and (ident or {}).get("name"):
                key = (self.session, ident["name"])
                if key not in self._seen_attendance_keys:
                    self._seen_attendance_keys.add(key)
                    self._log_activity("checkin", f"{ident['name']} checked in (Automatic)")
                    self._set_notice(f"Checked in: {ident['name']}")
            if status == "spoof":
                now_mono = time.monotonic()
                if now_mono - self._last_spoof_log_at >= 3.0:
                    self._last_spoof_log_at = now_mono
                    stamp = datetime.now().astimezone().strftime("%H:%M:%S")
                    score_txt = f"{self.pad_score:.2f}" if self.pad_score is not None else "—"
                    thr_txt = f"{self.pad_threshold:.2f}" if self.pad_threshold is not None else "—"
                    self.spoof_events.insert(0, {
                        "time_local": stamp, "session": self.session,
                        "pad_score": self.pad_score, "threshold": self.pad_threshold,
                        "note": f"PAD {score_txt} < {thr_txt} · Blocked (no image saved)",
                    })
                    del self.spoof_events[50:]
                    self._log_activity("spoof", f"Blocked spoof attempt (PAD {score_txt} < {thr_txt})")
            capture_pending = update.get("capture_pending", False)
            dialog_open = self.appearance_req is not None or self.review_req is not None
            self.capture_pending = capture_pending
            # Registration, appearance and review stay single-person even in multi-face mode.
            single = len(result.get("boxes") or []) == 1
            self.can_register = single and status == "unknown" and not capture_pending and not dialog_open
            # One visible face is enough to take a shot; PAD is checked per shot and filtered on save.
            self.can_capture_shot = (len(result.get("boxes") or []) == 1
                                     and not capture_pending and not dialog_open
                                     and not result.get("manual_review_pending"))
            self.can_appearance = (single and status in ("unknown", "known", "attendance_confirmed")
                                   and (live or {}).get("is_live") is True and update["profiles"] > 0
                                   and not capture_pending and not dialog_open
                                   and not result.get("manual_review_pending"))
            self.can_review = (single and status in ("unknown", "ambiguous") and update["profiles"] > 0
                               and not result.get("manual_review_pending") and not capture_pending
                               and not dialog_open)

    def _handle_preview(self, update):
        with self.lock:
            if not self.running or update["stream_id"] != self.stream_id:
                return
            sample = update["sample"]
            now = time.monotonic()
            if now - sample.captured_at > 1:
                return
            analysis = self.last_analysis
            mirror = self.mirror_preview
            self.preview_fps = _measured_fps(self.preview_times, now)
        image = Image.fromarray(sample.rgb)
        if mirror:  # mirror before drawing so name labels read normally
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        plain = image.copy() if self.plain_viewers else None
        if (analysis is not None and analysis["stream_id"] == update["stream_id"]
                and analysis["rgb"].shape == sample.rgb.shape
                and sample.sequence >= analysis["sequence"]
                and 0 <= sample.captured_at - analysis["captured_at"] <= BOX_MAX_AGE
                and 0 <= now - analysis["captured_at"] <= BOX_MAX_AGE):
            draw = ImageDraw.Draw(image)
            for box, color, name in _face_marks(analysis["result"]):
                x, y, w, h = box
                box = (image.width - x - w, y, w, h) if mirror else box
                _draw_corners(draw, box, color)
                if name:
                    _draw_label(draw, box, name, color)
        def encode(picture):
            buf = io.BytesIO()
            picture.save(buf, format="JPEG", quality=82)
            return buf.getvalue()
        jpeg = encode(image)
        plain_jpeg = encode(plain) if plain is not None else None
        with self.frame_cond:
            self.latest_jpeg = jpeg
            self.latest_plain_jpeg = plain_jpeg
            self.jpeg_seq += 1
            self.frame_cond.notify_all()

    def _expire_stale(self):
        with self.lock:
            now = time.monotonic()
            if self.review_req is not None and now >= self.review_req["expires_at"]:
                self.review_req["valid"] = False
                self.review_req["reason"] = "This request has expired. Cancel and capture a new observation."
            if self.appearance_req is not None and now >= self.appearance_req["expires_at"]:
                self.appearance_req["valid"] = False
                self.appearance_req["reason"] = "This request has expired. Cancel and capture a new observation."
            if not self.running or (self.last_analysis is not None and now - self.last_analysis["captured_at"] <= 1):
                return
            self.identity_text = "Not checked"
            self.liveness_text = "Not checked"
            self.expression_text = "—"
            self.quality = _quality_map(None)
            self.pad_score = self.pad_raw_score = self.pad_threshold = self.id_cosine = None
            self.stable_frames = 0
            self.emotion_scores = {}
            self.emotion_top = None
            self.feedback = "Analyzing the current camera view…"
            self.guidance = "Keep one face in view. Identity, liveness and expression results appear below the camera."
            self.can_register = self.can_capture_shot = self.can_appearance = self.can_review = False
            self.last_status = ""

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            tone = message_tone(self.feedback)
            ratio, color = _scan_progress(self.feedback, tone)
            mode = "PREVIEW ONLY · No hardware or models loaded" if self.args.preview else f"Inference device: {self.args.device}"
            runtime = (f"{mode} · Selected camera: {self.camera} · "
                       f"{'Running' if self.running else 'Opening' if self.camera_starting else 'Off'} · "
                       f"Check: {self.camera_status}")
            components = []
            for key, (title, hint) in COMPONENTS.items():
                item = self.component_status.get(key, {})
                st = item.get("status", "Not checked")
                detail = item.get("detail", "")
                if st == "Error":
                    detail = f"{detail} — Next step: {hint}".strip()
                components.append({"key": key, "title": title, "status": st, "detail": detail})
            review_out = None
            if self.review_req is not None:
                rem = max(0.0, self.review_req["expires_at"] - now)
                review_out = {**self.review_req, "seconds_left": rem, "valid": self.review_req["valid"] and rem > 0}
            app_out = None
            if self.appearance_req is not None:
                rem = max(0.0, self.appearance_req["expires_at"] - now)
                app_out = {**self.appearance_req, "seconds_left": rem, "valid": self.appearance_req["valid"] and rem > 0}
            can_retry = (not self.args.preview and not self.setup_loading and not self.models_ready
                         and not self.running and not self.camera_starting and not self.probe_busy)
            return {
                "preview": self.args.preview,
                "models_ready": self.models_ready,
                "setup_loading": self.setup_loading,
                "running": self.running,
                "camera_starting": self.camera_starting,
                "probe_busy": self.probe_busy,
                "session": self.session,
                "camera": self.camera,
                "camera_summary": self.camera_summary,
                "state_text": self.state_text,
                "speed": (f"Preview {self.preview_fps:.0f} FPS · Analysis {self.analysis_fps:.0f} FPS"
                          if self.running else "FPS not measured"),
                "feedback": self.feedback,
                "feedback_tone": tone,
                "status": self.last_status,
                "scan_ratio": ratio,
                "scan_color": color,
                "guidance": self.guidance,
                "identity_text": self.identity_text,
                "identity_tone": stat_tone("identity", self.identity_text),
                "liveness_text": self.liveness_text,
                "liveness_tone": stat_tone("liveness", self.liveness_text),
                "expression_text": self.expression_text,
                "expression_tone": stat_tone("expression", self.expression_text),
                "quality": self.quality,
                "pad_score": self.pad_score,
                "pad_raw_score": self.pad_raw_score,
                "pad_threshold": self.pad_threshold,
                "id_cosine": self.id_cosine,
                "id_threshold": self.id_threshold,
                "top_matches": self.top_matches,
                "stable_frames": self.stable_frames,
                "stable_frames_target": self.stable_frames_target,
                "liveness_smooth_window": self.liveness_smooth_window,
                "registration_samples": self.registration_samples,
                "capture_pending": self.capture_pending,
                "processing_ms": self.processing_ms,
                "mirror_preview": self.mirror_preview,
                "show_hud": self.show_hud,
                "model_info": self.model_info,
                "spoof_count": len(self.spoof_events),
                "activity": self.activity_log[:12],
                "emotion_scores": self.emotion_scores,
                "emotion_top": self.emotion_top,
                "profile_count": self.profile_count,
                "attendance": self.attendance,
                "can_register": self.can_register,
                "can_capture_shot": self.can_capture_shot,
                "guided_shots_count": len(self.guided_shots),
                "can_appearance": self.can_appearance,
                "can_review": self.can_review,
                "photo_import_available": self.photo_import_available,
                "runtime_summary": runtime,
                "components": components,
                "last_frame_details": self.last_frame_details,
                "can_retry_setup": can_retry,
                "review_request": review_out,
                "appearance_request": app_out,
                "notice": self.notice,
            }

    # Actions invoked by HTTP routes
    def start_camera(self, session, cam_idx):
        if self.args.preview:
            raise ValueError("Preview mode never opens the camera.")
        idx = camera_index(cam_idx)
        with self.lock:
            if not self.models_ready:
                raise ValueError("Models are still loading.")
            if self.running or self.camera_starting or self.probe_busy:
                raise ValueError("Camera is busy.")
            self.session = session
            self.camera = idx
            self.camera_starting = True
            self.last_frame_details = ""
            self.state_text = "Opening camera…"
        self._send("profiles_close")
        self._send("start", {"session": session, "camera": idx})

    def stop_camera(self):
        self._send("stop")

    def test_camera(self, cam_idx):
        if self.args.preview:
            raise ValueError("Preview mode never opens the camera.")
        idx = camera_index(cam_idx)
        with self.lock:
            if self.running or self.camera_starting or self.probe_busy:
                raise ValueError("Stop the camera before testing.")
            self.camera = idx
            self.probe_busy = True
            self.camera_status = "Testing selected camera…"
            self.camera_summary = f"Camera {idx} · Testing…"
            self.feedback = "Testing one camera frame · No attendance is recorded"

        def check():
            report = probe_camera(idx, backend=getattr(self.args, "camera_backend", None))
            self.updates.put_nowait({"event": "camera_probe", "report": report})

        self.camera_probe = threading.Thread(target=check, daemon=True)
        self.camera_probe.start()

    def retry_setup(self):
        with self.lock:
            if self.args.preview or self.setup_loading or self.models_ready or self.running:
                raise ValueError("Setup retry is not needed right now.")
            self.setup_loading = True
            self.component_status = {}
            self.state_text = "Loading models…"
            self.feedback = "Retrying model setup · Camera stays off"
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=3.0)
        self._start_worker()

    def register(self, name):
        self._send("register", name)

    def _call_guided_rpc(self, command, value, expect, timeout=10.0):
        if self.args.preview:
            raise ValueError("Preview mode: guided capture is disabled.")
        if not self.running:
            raise ValueError("Start the camera before capturing registration shots.")
        with self._rpc_lock:
            event = threading.Event()
            waiter = {"expect": set(expect), "event": event, "result": None}
            self._pending_rpc = waiter
            try:
                self._send(command, value)
                if not event.wait(timeout):
                    raise RuntimeError("Timed out waiting for registration action.")
                kind, payload = waiter["result"]
                if kind == "guided_error":
                    raise ValueError(payload["message"])
                return payload
            finally:
                self._pending_rpc = None

    def guided_shots_snapshot(self):
        with self.lock:
            return {"shots": list(self.guided_shots)}

    def guided_capture(self, slot=None):
        return self._call_guided_rpc("guided_capture", {"slot": slot}, ("guided_state", "guided_error"))

    def guided_remove(self, shot_id=None, slot=None):
        return self._call_guided_rpc("guided_remove", {"shot_id": shot_id, "slot": slot}, ("guided_state", "guided_error"))

    def guided_clear(self):
        if not self.running:
            with self.lock:
                self.guided_shots = []
            return {"shots": []}
        return self._call_guided_rpc("guided_clear", None, ("guided_state", "guided_error"))

    def guided_commit(self, name="", force=False, appearance_person_id=None, appearance_label=""):
        payload = self._call_guided_rpc(
            "guided_commit",
            {
                "name": name,
                "force": bool(force),
                "appearance_person_id": appearance_person_id,
                "appearance_label": appearance_label,
            },
            ("guided_committed", "guided_error"),
        )
        return payload["result"]

    def cancel_capture(self):
        self._send("cancel")

    def request_review(self):
        self._send("review")

    def cancel_review(self, review_id):
        with self.lock:
            self.review_req = None
        self._send("review_cancel", review_id)

    def confirm_review(self, fields):
        if self.args.preview:
            raise ValueError("Preview mode: no data is recorded.")
        with self._rpc_lock:
            event = threading.Event()
            waiter = {"expect": {"review_done", "review_error"}, "event": event, "result": None}
            self._pending_rpc = waiter
            try:
                self._send("review_confirm", fields)
                if not event.wait(10.0):
                    raise RuntimeError("Timed out confirming manual review.")
                kind, payload = waiter["result"]
                if kind == "review_error":
                    raise ValueError(payload["message"])
                return payload["result"]
            finally:
                self._pending_rpc = None

    def request_appearance(self):
        self._send("appearance_request")

    def cancel_appearance(self, request_id):
        with self.lock:
            self.appearance_req = None
        self._send("appearance_cancel", request_id)

    def begin_appearance(self, fields):
        self._send("appearance_begin", fields)

    def list_profiles(self):
        if self.args.preview:
            return {
                "editable": False, "photo_import_available": True,
                "profiles": [{
                    "id": "demo-person-a", "name": "Demo person A",
                    "created_local": "2026-11-20T09:00:00+07:00", "sample_count": 6,
                    "appearances": [
                        {"id": "orig", "label": "Original", "sample_count": 3},
                        {"id": "noglass", "label": "Without glasses", "sample_count": 3},
                    ],
                }],
            }
        if self.running or self.camera_starting:
            profiles = []
            if self.running:
                profiles = self._call_guided_rpc("profiles_snapshot", None, ("profiles_snapshot",))["profiles"]
            return {"editable": False, "photo_import_available": self.photo_import_available,
                    "profiles": [{**p, "created_local": local_timestamp(p["created_utc"])} for p in profiles]}
        kind, payload = self._call_profile_rpc("profiles_list", None, ("profiles_data", "profiles_error"))
        if kind == "profiles_error":
            raise ValueError(payload["message"])
        profiles = [{**p, "created_local": local_timestamp(p["created_utc"])} for p in payload["profiles"]]
        return {"editable": True, "photo_import_available": payload["photo_import_available"], "profiles": profiles}

    def rename_profile(self, person_id, name):
        self._call_profile_rpc("profiles_list", None, ("profiles_data", "profiles_error"))
        kind, payload = self._call_profile_rpc("profile_rename", {"person_id": person_id, "name": name},
                                               ("profiles_data", "profiles_error"))
        if kind == "profiles_error":
            raise ValueError(payload["message"])
        return {"message": payload.get("message") or "Name updated."}

    def remove_appearance(self, person_id, appearance_id):
        self._call_profile_rpc("profiles_list", None, ("profiles_data", "profiles_error"))
        kind, payload = self._call_profile_rpc("profile_remove",
                                               {"person_id": person_id, "appearance_id": appearance_id},
                                               ("profiles_data", "profiles_error"))
        if kind == "profiles_error":
            raise ValueError(payload["message"])
        return {"message": payload.get("message") or "Appearance removed."}

    def delete_profile(self, person_id):
        self._call_profile_rpc("profiles_list", None, ("profiles_data", "profiles_error"))
        kind, payload = self._call_profile_rpc("profile_delete", {"person_id": person_id},
                                               ("profiles_data", "profiles_error"))
        if kind == "profiles_error":
            raise ValueError(payload["message"])
        return {"message": payload.get("message") or "Profile deleted."}

    def _cleanup_import_dir(self, operation_id):
        path = self._import_dirs.pop(operation_id, None)
        if path is not None:
            shutil.rmtree(path, ignore_errors=True)

    def import_check(self, person_id, name, label, files):
        from .reference_import import MAX_PHOTOS, MIN_PHOTOS
        if not MIN_PHOTOS <= len(files) <= MAX_PHOTOS:
            raise ValueError(f"Choose {MIN_PHOTOS} to {MAX_PHOTOS} reference photos.")
        op_id = uuid.uuid4().hex
        temp_dir = Path(tempfile.mkdtemp(prefix="ref_import_"))
        self._import_dirs[op_id] = temp_dir
        paths = []
        try:
            for idx, item in enumerate(files):
                safe = Path(str(item.get("name", f"photo_{idx}.jpg"))).name
                dest = temp_dir / f"{idx}_{safe}"
                dest.write_bytes(base64.b64decode(item["data"]))
                paths.append(str(dest))
            self._call_profile_rpc("profiles_list", None, ("profiles_data", "profiles_error"))
            kind, payload = self._call_profile_rpc(
                "profile_import_check",
                {"operation_id": op_id, "files": paths, "person_id": person_id, "name": name, "label": label},
                ("profile_import_checked", "profiles_error"),
            )
            if kind == "profiles_error":
                self._cleanup_import_dir(op_id)
                raise ValueError(payload["message"])
            report = payload["report"]
            if not report.get("token"):
                self._cleanup_import_dir(op_id)
            return report
        except Exception:
            self._cleanup_import_dir(op_id)
            raise

    def import_commit(self, operation_id, token, confirmed):
        try:
            kind, payload = self._call_profile_rpc(
                "profile_import_commit",
                {"operation_id": operation_id, "token": token, "confirmed": bool(confirmed)},
                ("profiles_data", "profiles_error"),
            )
            if kind == "profiles_error":
                raise ValueError(payload["message"])
            return {"message": payload.get("message") or "Reference samples imported."}
        finally:
            self._cleanup_import_dir(operation_id)

    def update_settings(self, payload):
        with self.lock:
            if "mirror_preview" in payload:
                self.mirror_preview = bool(payload["mirror_preview"])
            if "show_hud" in payload:
                self.show_hud = bool(payload["show_hud"])
            if "stable_frames" in payload:
                frames = int(payload["stable_frames"])
                if not 1 <= frames <= 5:
                    raise ValueError("Confirmation frames must be between 1 and 5.")
                self.stable_frames_target = frames
                self.model_info["stable_frames"] = frames
                if not self.args.preview:
                    self._send("set_stable_frames", frames)
            if "liveness_smooth_window" in payload:
                window = int(payload["liveness_smooth_window"])
                if not 1 <= window <= 5:
                    raise ValueError("Liveness smoothing window must be between 1 and 5.")
                self.liveness_smooth_window = window
                if not self.args.preview:
                    self._send("set_liveness_smooth", window)
        return {
            "mirror_preview": self.mirror_preview,
            "show_hud": self.show_hud,
            "stable_frames": self.stable_frames_target,
            "liveness_smooth_window": self.liveness_smooth_window,
        }

    def history(self, session="", source="", query=""):
        if self.args.preview:
            all_rows = preview_history_rows()
            all_profiles = [
                {"id": "fictional-a", "name": "Demo student A", "created_utc": "2026-10-30T02:00:00+00:00"},
                {"id": "fictional-b", "name": "Demo student B", "created_utc": "2026-10-30T02:00:00+00:00"},
                {"id": "fictional-c", "name": "Demo student C", "created_utc": "2026-10-30T02:00:00+00:00"},
            ]
        else:
            try:
                all_rows = read_attendance_snapshot(self.args.database)
            except FileNotFoundError:
                all_rows = []
            all_profiles = read_profile_names(self.args.database)
        sessions = list(dict.fromkeys(r["session"] for r in all_rows))
        filtered = filter_attendance(all_rows, session or None, source or None, query or "")
        rows = [{
            **r,
            "recorded_local": local_timestamp(r["recorded_utc"]),
            "source_label": SOURCE_LABELS.get(r["source"], "Unknown"),
        } for r in filtered]
        # Compute Absent list for the chosen session (or active session if none chosen)
        target_session = session or (sessions[0] if sessions else self.session)
        present_ids = {r["person_id"] for r in all_rows if r["session"] == target_session}
        absent = [{"id": p["id"], "name": p["name"], "created_local": local_timestamp(p["created_utc"])}
                  for p in all_profiles if p["id"] not in present_ids]
        with self.lock:
            spoofs = [dict(e) for e in self.spoof_events if not session or e["session"] == session]
        summary = summarize_attendance(filtered)
        summary["absent"] = len(absent)
        summary["spoofs"] = len(spoofs)
        summary["registered"] = len(all_profiles)
        summary["target_session"] = target_session
        return {"sessions": sessions, "summary": summary, "rows": rows, "absent": absent,
                "spoof_events": spoofs, "raw": filtered}

    def close(self):
        self.closing = True
        self._send("quit")
        for op_id in list(self._import_dirs):
            self._cleanup_import_dir(op_id)
        with self.frame_cond:
            self.frame_cond.notify_all()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=1.2)  # then main() force-exits rather than hang on a camera driver


def make_handler(controller: WebController):
    page_bytes = HTML_PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass  # keep terminal output clean

        def _send_bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK, extra_headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for k, v in extra_headers.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj, status=HTTPStatus.OK):
            self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                             "application/json; charset=utf-8", status=status)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            if length > 160 * 1024 * 1024:  # up to 10 phone photos, base64-encoded
                raise ValueError("Request body is too large.")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)
            try:
                if path in ("/", "/index.html"):
                    self._send_bytes(page_bytes, "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._send_json(controller.snapshot())
                elif path == "/api/register/shots":
                    self._send_json(controller.guided_shots_snapshot())
                elif path == "/api/profiles":
                    self._send_json(controller.list_profiles())
                elif path == "/api/history":
                    data = controller.history(
                        qs.get("session", [""])[0], qs.get("source", [""])[0], qs.get("q", [""])[0]
                    )
                    self._send_json({k: v for k, v in data.items() if k != "raw"})
                elif path == "/api/history/export.csv":
                    data = controller.history(
                        qs.get("session", [""])[0], qs.get("source", [""])[0], qs.get("q", [""])[0]
                    )
                    csv_bytes = render_csv_bytes(data["raw"])
                    filename = f"attendance_{datetime.now():%Y%m%d_%H%M%S}.csv"
                    self._send_bytes(csv_bytes, "text/csv; charset=utf-8",
                                     extra_headers={"Content-Disposition": f'attachment; filename="{filename}"'})
                elif path == "/api/video":
                    self._stream_video(parse_qs(urlparse(self.path).query).get("boxes") == ["0"])
                else:
                    self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError, OSError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                if path == "/api/camera/start":
                    controller.start_camera(body.get("session", controller.session), body.get("camera", 0))
                    self._send_json({"ok": True})
                elif path == "/api/camera/stop":
                    controller.stop_camera()
                    self._send_json({"ok": True})
                elif path == "/api/camera/test":
                    controller.test_camera(body.get("camera", 0))
                    self._send_json({"ok": True})
                elif path == "/api/setup/retry":
                    controller.retry_setup()
                    self._send_json({"ok": True})
                elif path == "/api/settings":
                    res = controller.update_settings(body)
                    self._send_json({"ok": True, **res})
                elif path == "/api/register":
                    controller.register(body.get("name", ""))
                    self._send_json({"ok": True})
                elif path == "/api/register/capture":
                    res = controller.guided_capture(body.get("slot"))
                    self._send_json({"ok": True, **res})
                elif path == "/api/register/remove":
                    res = controller.guided_remove(body.get("shot_id"), body.get("slot"))
                    self._send_json({"ok": True, **res})
                elif path == "/api/register/clear":
                    res = controller.guided_clear()
                    self._send_json({"ok": True, **res})
                elif path == "/api/register/commit":
                    res = controller.guided_commit(
                        name=body.get("name", ""),
                        force=bool(body.get("force", False)),
                        appearance_person_id=body.get("appearance_person_id"),
                        appearance_label=body.get("appearance_label", ""),
                    )
                    if "duplicate" in res:
                        self._send_json(res, HTTPStatus.CONFLICT)
                    else:
                        self._send_json({"ok": True, "result": res})
                elif path == "/api/cancel":
                    controller.cancel_capture()
                    self._send_json({"ok": True})
                elif path == "/api/review/request":
                    controller.request_review()
                    self._send_json({"ok": True})
                elif path == "/api/review/cancel":
                    controller.cancel_review(body.get("review_id"))
                    self._send_json({"ok": True})
                elif path == "/api/review/confirm":
                    res = controller.confirm_review(body)
                    self._send_json({"ok": True, "result": res})
                elif path == "/api/appearance/request":
                    controller.request_appearance()
                    self._send_json({"ok": True})
                elif path == "/api/appearance/cancel":
                    controller.cancel_appearance(body.get("request_id"))
                    self._send_json({"ok": True})
                elif path == "/api/appearance/begin":
                    controller.begin_appearance(body)
                    self._send_json({"ok": True})
                elif path == "/api/profiles/rename":
                    res = controller.rename_profile(body.get("person_id"), body.get("name", ""))
                    self._send_json(res)
                elif path == "/api/profiles/remove-appearance":
                    res = controller.remove_appearance(body.get("person_id"), body.get("appearance_id"))
                    self._send_json(res)
                elif path == "/api/profiles/delete":
                    res = controller.delete_profile(body.get("person_id"))
                    self._send_json(res)
                elif path == "/api/profiles/import-check":
                    res = controller.import_check(body.get("person_id"), body.get("name", ""),
                                                  body.get("label", ""), body.get("files", []))
                    self._send_json(res)
                elif path == "/api/profiles/import-commit":
                    res = controller.import_commit(body.get("operation_id"), body.get("token"),
                                                   body.get("confirmed", False))
                    self._send_json(res)
                else:
                    self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def _stream_video(self, plain=False):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = 0
            with controller.frame_cond:
                controller.plain_viewers += int(plain)
            try:
                while not controller.closing and controller.running:
                    with controller.frame_cond:
                        if controller.jpeg_seq == seen:
                            controller.frame_cond.wait(timeout=0.5)
                        if not controller.running or controller.closing:
                            break
                        frame = controller.latest_plain_jpeg if plain else controller.latest_jpeg
                        if controller.jpeg_seq == seen or frame is None:
                            continue
                        seen = controller.jpeg_seq
                    chunk = (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                             + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                with controller.frame_cond:
                    controller.plain_viewers -= int(plain)

    return Handler


def _pick_port(host: str, preferred: int) -> int:
    for port in (preferred, *range(preferred + 1, preferred + 15)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    return preferred


def main(argv=None):
    parser = argparse.ArgumentParser(description="Localhost web app for COS30082 Face Attendance.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="Open the localhost URL in the default browser.")
    parser.add_argument("--identity-config", type=Path, default=PROJECT / "results/identity_app_config.json")
    parser.add_argument("--liveness-checkpoint", type=Path, default=DEFAULT_LIVENESS_CHECKPOINT)
    parser.add_argument("--emotion-checkpoint", type=Path, default=PROJECT / "results/emotion_fer_mnv3/best_model.pt")
    parser.add_argument("--database", type=Path, default=PROJECT / "data/attendance/profiles.sqlite3")
    parser.add_argument("--session", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto",
                        help="OpenCV video backend preference (auto uses fast MSMF on Windows).")
    parser.add_argument("--detector", choices=("yunet", "haar", "roi"), default="yunet")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--preview", action="store_true", help="Preview mode; never loads models, database or camera.")
    args = parser.parse_args(argv)
    if args.num_threads < 1 or not 0 <= args.camera <= 99:
        parser.error("Positive threads and a camera index from 0 to 99 required.")

    port = _pick_port(args.host, args.port)
    controller = WebController(args)
    server = ThreadingHTTPServer((args.host, port), make_handler(controller))
    url = f"http://{args.host}:{port}"
    print(f"Face Attendance localhost server running at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if args.open:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()
    return 0


if __name__ == "__main__":
    force_exit(main() or 0)
