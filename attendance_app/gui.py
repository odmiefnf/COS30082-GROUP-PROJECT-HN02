"""English Tkinter webcam app. Camera starts only after the Start button."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import queue
import sqlite3
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .device_check import CAMERA_HELP, camera_index, probe_camera
from .diagnostics import COMPONENTS, frame_details, frame_guidance
from .theme import (BG, BORDER, CANVAS, CANVAS_LIVE, CARD, FONT, HEADER_BG, INK, MINT, MUTED,
                    PILL_BG, PRIMARY, PRIMARY_INK, PRIMARY_SOFT, SUBTLE, SURFACE, TONES,
                    FlatButton, apply_style, card, entry)
from .video_stream import LatestCamera, put_latest

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LIVENESS_CHECKPOINT = PROJECT/"results/liveness_mobilenetv3_20261120/best_model.pt"
BLUE = PRIMARY
FACE_PENDING, FACE_RECOGNIZED = "#fbbf24", "#4ade80"
EXPRESSION_ICONS = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😮",
    "neutral": "😐",
}


def format_expression(expression, error=None):
    """Format an expression dictionary with its icon, label and confidence."""
    if expression is None:
        return "Unavailable" if error else "—"
    label = str(expression.get("label", "")).strip()
    icon = EXPRESSION_ICONS.get(label.lower(), "🙂")
    return f"{icon} {label.title()} · {expression['confidence']:.0%}"

MESSAGES = {
    "no_face": "Face the camera and keep your face in view",
    "multiple_faces": "Only one person should be in front of the camera",
    "face_too_small_or_outside": "Move closer and keep your whole face in the frame",
    "image_quality": "Hold still and adjust the lighting for a clearer face",
    "spoof": "Liveness not passed · Attendance not recorded",
    "liveness_unavailable": "A calibrated liveness threshold is unavailable",
    "observation_stale": "Analysis took too long · Waiting for a fresh camera frame",
    "unknown": "Not recognized · New person: Register · Already registered: Add appearance",
    "ambiguous": "Match is uncertain · Try another viewing angle",
    "known": "Recognized · Hold still to confirm",
    "attendance_confirmed": "Attendance confirmed for this session",
    "registering": "Capturing 3 samples · Turn slightly and keep the lighting steady",
    "registered": "Registration complete · Keep looking at the camera to confirm attendance",
    "registration_existing_or_ambiguous": "Cannot register: too similar to an existing profile",
    "registration_person_changed": "Samples are inconsistent · Keep one person in view",
    "appearance_capturing": "Capturing an additional appearance · Keep the same person in view",
    "appearance_added": "Appearance added · Keep looking at the camera to test recognition",
    "appearance_conflict": "Cannot add this appearance: it matches another profile",
    "appearance_person_changed": "Samples are inconsistent · Keep the same person in view",
    "appearance_expired": "Appearance capture expired · Start again",
}

# Presentation-only colour coding of the feedback banner and result cards.
_SUCCESS = {MESSAGES[key] for key in ("known", "attendance_confirmed", "registered", "appearance_added")}
_DANGER = {MESSAGES[key] for key in ("spoof", "liveness_unavailable", "registration_existing_or_ambiguous",
                                     "registration_person_changed", "appearance_conflict",
                                     "appearance_person_changed", "appearance_expired")}
_WARNING = {MESSAGES[key] for key in ("unknown", "ambiguous", "registering", "appearance_capturing",
                                      "observation_stale", "multiple_faces", "face_too_small_or_outside",
                                      "image_quality")}


def message_tone(text):
    """Map a feedback message to a banner tone; never used for any decision."""
    lowered = text.casefold()
    if text in _SUCCESS:
        return "success"
    if text in _DANGER or "failed" in lowered or "error" in lowered:
        return "danger"
    if text in _WARNING or lowered.startswith(("sample ", "appearance sample", "manual review in progress",
                                               "capturing")):
        return "warning"
    if lowered.startswith(("models ready", "testing", "analyzing", "click start", "retrying", "face the camera",
                           "loading", "opened")):
        return "info"
    return "neutral"


def stat_tone(key, text):
    if text in ("", "—", "Not checked"):
        return "neutral"
    if key == "liveness":
        return {"Passed": "success", "Not passed": "danger", "Unavailable": "warning"}.get(text, "neutral")
    if key == "identity":
        return "warning" if text == "Not recognized" else "success"
    return "neutral" if text == "Unavailable" else "info"


class CameraWorker(threading.Thread):
    """Serialize models/SQLite; camera capture and Tk rendering run independently."""
    def __init__(self, args, commands, updates, previews=None, analyses=None):
        super().__init__(daemon=True)
        self.args, self.commands, self.updates = args, commands, updates
        self.previews = previews if previews is not None else queue.Queue(maxsize=1)
        self.analyses = analyses if analyses is not None else queue.Queue(maxsize=1)
        self.stream_id = 0
        self.last_sequence = 0
        self.camera = None
        self.service = None
        self.latest = self.latest_rgb = None
        self.latest_time = float("-inf")
        self.profiles_open = False
        self.reference_importer = None
        self.startup_stage = "Runtime"

    def load_component(self, key, detail, load):
        self.startup_stage = key
        self.publish(event="component", component=key, status="Loading", detail=detail)
        value = load()
        self.publish(event="component", component=key, status="Ready", detail=detail)
        return value

    def require_profile_management(self):
        if self.camera is not None or self.service is None:
            raise ValueError("Stop the camera before managing profiles.")
        if (self.service.pending_name is not None or self.service.pending_appearance is not None
                or self.service.review_open or self.service.appearance_dialog_open):
            raise ValueError("Finish or cancel the current capture or review first.")

    def profile_rows(self):
        """Names and sample counts only; embeddings never leave the worker."""
        return [{"id": p["id"], "name": p["name"], "created_utc": p["created_utc"],
                 "sample_count": p["sample_count"], "appearances": [
                     {"id": a["id"], "label": a["label"], "sample_count": a["sample_count"]}
                     for a in p["appearances"]]} for p in self.service.profiles]

    def publish_profiles(self, message=None, selected_id=None):
        self.service.profiles = self.service.store.profiles()
        profiles = self.profile_rows()
        self.publish(event="profiles_data", profiles=profiles, message=message, selected_id=selected_id,
                     photo_import_available=self.args.detector != "roi",
                     attendance=self.service.store.attendance(self.service.session))

    def publish(self, **update):
        # Video/analysis retain the newest observation. Ordered control events
        # have their own queue and cannot be displaced by video traffic.
        if update.get("event") == "frame":
            put_latest(self.analyses, update)
            return
        self.updates.put_nowait(update)

    def preview(self, sample, stream_id):
        put_latest(self.previews, {"sample": sample, "stream_id": stream_id})

    def stop_camera(self):
        self.stream_id += 1
        if self.camera is not None:
            camera, self.camera = self.camera, None
            try:
                camera.release()
            except Exception as exc:
                self.publish(event="camera_error", message=f"Camera release failed: {exc}. Close the app before retrying.")
        self.latest = None
        self.latest_time = float("-inf")
        self.last_sequence = 0
        for channel in (self.previews, self.analyses):
            try:
                channel.get_nowait()
            except queue.Empty:
                pass
        if self.service is not None:
            self.service.cancel_registration()
            if hasattr(self.service, "clear_guided_shots"):
                self.service.clear_guided_shots()

    def require_appearance_face(self):
        """Use only a fresh live observation, with no capture or review in progress."""
        if (self.latest is None or time.monotonic() - self.latest_time > 1
                or self.latest.get("status") not in ("unknown", "known", "attendance_confirmed")
                or (self.latest.get("liveness") or {}).get("is_live") is not True):
            raise ValueError("Adding an appearance requires a current face that has passed liveness.")
        if (self.service.pending_name is not None
                or getattr(self.service, "pending_appearance", None) is not None
                or self.service.review_open):
            raise ValueError("Finish or cancel the current capture or manual review first.")
        if not self.service.profiles:
            raise ValueError("Register a profile before adding an appearance.")

    def run(self):
        store = None
        try:
            self.publish(event="component", component="Runtime", status="Loading", detail="Loading local dependencies")
            import cv2
            import torch
            from anti_spoofing.infer import LivenessDetector
            from emotion_recognition.infer import EmotionRecognizer
            from .recognition import FaceRecognizer
            from .store import ProfileStore
            from .service import AttendanceService, pad_crop_scale, uses_pad_context
            from .camera import FaceLocator, FrameProcessor
            from .reference_import import ReferenceImporter
            torch.set_num_threads(self.args.num_threads)
            self.publish(event="component", component="Runtime", status="Ready", detail=f"Device: {self.args.device}; threads: {self.args.num_threads}")
            recognizer = self.load_component("Identity", str(self.args.identity_config),
                lambda: FaceRecognizer(self.args.identity_config, self.args.device))

            def validated_model(name, load):
                model = load()
                metadata = model.config if name == "Liveness" else model.payload
                if name == "Liveness" and model.threshold is None:
                    raise ValueError("Liveness checkpoint requires a validation-calibrated threshold.")
                training = metadata.get("training", {})
                allowed = ("random_from_scratch", "partial_fine_tuning_imagenet")
                if training.get("smoke_test") is not False or training.get("initialization") not in allowed:
                    raise ValueError(f"{name} requires a completed trained checkpoint.")
                return model

            liveness = self.load_component("Liveness", str(self.args.liveness_checkpoint),
                lambda: validated_model("Liveness", lambda: LivenessDetector(self.args.liveness_checkpoint, device=self.args.device)))
            emotion = self.load_component("Expression", str(self.args.emotion_checkpoint),
                lambda: validated_model("Expression", lambda: EmotionRecognizer(self.args.emotion_checkpoint, self.args.device)))
            locator = self.load_component("Detector", self.args.detector,
                lambda: FaceLocator(self.args.detector, self.args.roi))
            self.startup_stage = "Profiles"
            self.publish(event="component", component="Profiles", status="Loading", detail=str(self.args.database))
            store = ProfileStore(self.args.database, recognizer.fingerprint, recognizer.dimension)
            # 3-frame PAD smoothing by default; the web app sends its own window setting on start.
            self.service = AttendanceService(store, recognizer, liveness, emotion, self.args.session,
                                             liveness_smooth_window=3)
            self.reference_importer = ReferenceImporter(store, recognizer, liveness, locator)
            scale = pad_crop_scale(liveness)
            crop_options = ({"pad_scale": scale} if scale is not None
                            else {"pad_context": True} if uses_pad_context(liveness) else {})
            processor = FrameProcessor(locator, self.service, **crop_options)
            self.publish(event="component", component="Profiles", status="Ready", detail=str(self.args.database))
            def _num(val):
                return float(val) if isinstance(val, (int, float)) else None
            ckpt = getattr(recognizer, "checkpoint", None)
            stable = getattr(self.service, "stable_frames", 3)
            model_info = {
                "identity_threshold": _num(getattr(recognizer, "threshold", None)),
                "identity_margin": _num(getattr(recognizer, "margin", None)),
                "identity_dim": int(recognizer.dimension) if isinstance(getattr(recognizer, "dimension", None), int) else 128,
                "identity_checkpoint": ckpt.name if isinstance(ckpt, Path) else str(ckpt or ""),
                "liveness_threshold": _num(getattr(liveness, "threshold", None)),
                "liveness_arch": str((getattr(liveness, "config", None) or {}).get("arch", "liveness_cnn")),
                "liveness_crop_scale": (getattr(liveness, "config", None) or {}).get("crop_scale"),
                "liveness_checkpoint": Path(str(getattr(self.args, "liveness_checkpoint", ""))).name,
                "emotion_checkpoint": Path(str(getattr(self.args, "emotion_checkpoint", ""))).name,
                "stable_frames": int(stable) if isinstance(stable, int) else 3,
                "device": str(getattr(self.args, "device", "cpu")),
                "num_threads": int(self.args.num_threads) if isinstance(getattr(self.args, "num_threads", None), int) else 4,
            }
            self.publish(event="ready", profiles=len(self.service.profiles),
                         attendance=store.attendance(self.service.session), model_info=model_info)
            while True:
                try:
                    command, value = self.commands.get(timeout=.08 if self.camera is None else .001)
                except queue.Empty:
                    command, value = None, None
                if command == "quit":
                    break
                if command == "set_stable_frames":
                    frames = max(1, min(5, int(value)))
                    self.service.stable_frames = frames
                    self.publish(event="config_updated", stable_frames=frames)
                if command == "set_liveness_smooth":
                    window = max(1, min(5, int(value)))
                    self.service.set_liveness_smooth_window(window)
                    self.publish(event="config_updated", liveness_smooth_window=window)
                if command == "stop":
                    self.stop_camera()
                    self.publish(event="stopped")
                if command == "start":
                    try:
                        if self.profiles_open:
                            raise ValueError("Close the Profiles window before starting the camera.")
                        from .store import clean_text
                        if isinstance(value, dict):
                            session = clean_text(value["session"], "Session")
                            selected_camera = camera_index(value["camera"])
                        else:
                            session = clean_text(value, "Session")
                            selected_camera = camera_index(self.args.camera)
                        self.stop_camera()
                        self.service.session = session
                        stream_id = self.stream_id
                        self.camera = LatestCamera(selected_camera,
                            on_preview=lambda sample, token=stream_id: self.preview(sample, token),
                            backend=getattr(self.args, "camera_backend", None))
                        self.camera.open()
                        self.publish(event="started", camera=selected_camera, stream_id=stream_id,
                                     attendance=store.attendance(session))
                    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                        self.stop_camera()
                        self.publish(event="camera_error", message=str(exc))
                if command == "register":
                    try:
                        if self.latest is None or self.latest["status"] != "unknown" or time.monotonic() - self.latest_time > 1:
                            raise ValueError("Registration requires a current, unrecognized face that has passed liveness.")
                        self.service.begin_registration(value)
                        self.publish(event="registering")
                    except ValueError as exc:
                        self.publish(event="notice", message=str(exc))
                if command == "cancel":
                    self.service.cancel_registration()
                if command == "profiles_snapshot":
                    # Read-only view while the camera runs; editing still needs the camera stopped.
                    self.publish(event="profiles_snapshot", profiles=self.profile_rows())
                if command == "guided_capture":
                    try:
                        slot = value.get("slot") if isinstance(value, dict) else None
                        latest, rgb = self.latest, self.latest_rgb
                        if latest is None or rgb is None or time.monotonic() - self.latest_time > 1.5:
                            raise ValueError("Start the camera and face it before capturing.")
                        if len(latest.get("boxes") or []) != 1:
                            raise ValueError("Exactly one face must be in view to capture a shot.")
                        shots = self.service.capture_guided_shot(rgb, tuple(latest["boxes"][0]), slot=slot)
                        self.publish(event="guided_state", shots=[
                            {"id": s["id"], "slot": s["slot"], "live_score": s["live_score"],
                             "is_live": s["is_live"], "crop": s["crop"].copy()}
                            for s in shots
                        ])
                    except ValueError as exc:
                        self.publish(event="guided_error", message=str(exc))
                if command == "guided_remove":
                    val = value if isinstance(value, dict) else {}
                    shots = self.service.remove_guided_shot(shot_id=val.get("shot_id"), slot=val.get("slot"))
                    self.publish(event="guided_state", shots=[
                        {"id": s["id"], "slot": s["slot"], "live_score": s["live_score"], "is_live": s["is_live"], "crop": s["crop"].copy()}
                        for s in shots
                    ])
                if command == "guided_clear":
                    self.service.clear_guided_shots()
                    self.publish(event="guided_state", shots=[])
                if command == "guided_commit":
                    try:
                        val = value if isinstance(value, dict) else {}
                        result = self.service.commit_guided_registration(
                            val.get("name", ""),
                            force=bool(val.get("force", False)),
                            appearance_person_id=val.get("appearance_person_id"),
                            appearance_label=val.get("appearance_label", ""),
                        )
                        self.publish(event="guided_committed", result=result,
                                     profiles=len(self.service.profiles),
                                     attendance=store.attendance(self.service.session))
                    except (ValueError, sqlite3.Error) as exc:
                        self.publish(event="guided_error", message=str(exc))
                if command in ("profiles_list", "profile_rename", "profile_remove", "profile_delete",
                               "profile_import_check", "profile_import_commit"):
                    try:
                        self.require_profile_management()
                        if command != "profiles_list" and not self.profiles_open:
                            raise ValueError("Open Profiles before making changes.")
                        if command == "profiles_list":
                            self.profiles_open = True
                            self.publish_profiles()
                        elif command == "profile_rename":
                            backup = store.rename_profile(**value)
                            self.reference_importer.cancel()
                            self.publish_profiles("Name updated. Backup: "+backup if backup else "Name unchanged.", value["person_id"])
                        elif command == "profile_remove":
                            backup = store.remove_appearance(**value)
                            self.reference_importer.cancel()
                            self.publish_profiles("Appearance removed. Backup: "+backup, value["person_id"])
                        elif command == "profile_delete":
                            backup = store.delete_profile(**value)
                            self.reference_importer.cancel()
                            self.service.cancel_registration()
                            self.publish_profiles("Profile and its attendance deleted. Backup: "+backup)
                        elif command == "profile_import_check":
                            report = self.reference_importer.prepare(**value)
                            self.publish(event="profile_import_checked", report=report)
                        else:
                            result = self.reference_importer.commit(**value)
                            self.service.cancel_registration()
                            self.publish(event="profile_import_done", operation_id=value["operation_id"])
                            self.publish_profiles("Three reference samples imported. No attendance recorded.", result["person_id"])
                    except (OSError, ValueError, RuntimeError, sqlite3.Error, cv2.error) as exc:
                        operation_id = value.get("operation_id") if isinstance(value, dict) else None
                        if operation_id:
                            self.reference_importer.cancel(operation_id)
                        self.publish(event="profiles_error", message=str(exc), operation_id=operation_id)
                if command == "profile_import_cancel":
                    self.reference_importer.cancel(value)
                if command == "profiles_close":
                    self.reference_importer.cancel()
                    self.profiles_open = False
                if command == "appearance_request":
                    try:
                        self.require_appearance_face()
                        self.publish(event="appearance_open", request=self.service.request_appearance())
                    except ValueError as exc:
                        self.publish(event="notice", message=str(exc))
                if command == "appearance_begin":
                    try:
                        self.require_appearance_face()
                        self.service.begin_appearance(**value)
                        self.publish(event="appearance_started")
                    except (ValueError, sqlite3.Error) as exc:
                        self.service.cancel_appearance_request(value["request_id"])
                        self.publish(event="notice", message=str(exc))
                if command == "appearance_cancel":
                    self.service.cancel_appearance_request(value)
                if command == "review":
                    try:
                        self.publish(event="review_open", request=self.service.request_manual_review())
                    except ValueError as exc:
                        self.publish(event="notice", message=str(exc))
                if command == "review_cancel":
                    self.service.cancel_manual_review(value)
                if command == "review_confirm":
                    try:
                        reviewed = self.service.confirm_manual_review(**value)
                        self.publish(event="review_done", result=reviewed, attendance=store.attendance(self.service.session))
                    except (ValueError, sqlite3.Error) as exc:
                        self.publish(event="review_error", message=str(exc))
                if self.camera is None:
                    continue
                try:
                    started = time.perf_counter()
                    frame_stage = "Camera read"
                    sample = self.camera.read_latest(self.last_sequence, timeout=.03)
                    if sample is None:
                        continue
                    self.last_sequence = sample.sequence
                    if time.monotonic() - sample.captured_at > 1:
                        self.service.reset_tracking("The camera observation is stale. Try again.")
                        self.latest = None
                        continue
                    rgb = sample.rgb
                    frame_stage = "Frame processing"
                    processing_started = time.perf_counter()
                    result = processor.process(rgb, observed_at=sample.captured_at)
                    self.latest, self.latest_time, self.latest_rgb = result, sample.captured_at, rgb
                    elapsed = time.perf_counter() - started
                    self.publish(event="frame", rgb=rgb, result=result, elapsed=elapsed,
                                 processing_seconds=time.perf_counter()-processing_started,
                                 captured_at=sample.captured_at, sequence=sample.sequence, stream_id=self.stream_id,
                                 profiles=len(self.service.profiles), attendance=store.attendance(self.service.session),
                                 capture_pending=self.service.pending_name is not None
                                 or getattr(self.service, "pending_appearance", None) is not None)
                except (OSError, ValueError, RuntimeError, cv2.error, sqlite3.Error) as exc:
                    self.stop_camera()
                    self.publish(event="camera_error", message=f"{frame_stage}: {exc}. "
                                 + (CAMERA_HELP if frame_stage == "Camera read" else "Open Diagnostics; try again after checking the models and detector."))
        except Exception as exc:
            self.publish(event="component", component=self.startup_stage, status="Error", detail=str(exc))
            self.publish(event="fatal", component=self.startup_stage, message=str(exc))
        finally:
            self.stop_camera()
            if store is not None:
                store.close()


class AttendanceWindow:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root, self.args = tk, ttk, root, args
        self.commands, self.updates = queue.Queue(), queue.Queue()
        self.previews, self.analyses = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self.stream_id = None
        self.last_analysis = None
        self.preview_times = []
        self.analysis_times = []
        self.preview_fps = self.analysis_fps = 0.
        self.canvas_image = None
        self.worker = None
        self.running, self.closing = False, False
        self.review_dialog = None
        self.appearance_dialog = None
        self.history_dialog = None
        self.model_test_dialog = None
        self.profiles_dialog = None
        self.profiles_opening = False
        self.models_ready = False
        self.setup_loading = not args.preview
        self.camera_starting = False
        self.probe_busy = False
        self.camera_probe = None
        self.camera_status = "Not tested"
        self.component_status = {}
        self.last_frame_details = ""
        self.diagnostics_dialog = None
        root.title("Layout preview · COS30082" if args.preview else "Face Attendance · COS30082")
        width = min(1440, max(800, root.winfo_screenwidth()-60))
        height = min(960, max(600, root.winfo_screenheight()-85))
        root.geometry(f"{width}x{height}")
        root.minsize(min(1100, width), min(700, height))
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        apply_style(root)
        self.state_text = tk.StringVar(value="Preview · Camera off" if args.preview else "Loading three models…")
        self.build_app_bar(root)
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=18, pady=(14, 8))

        # Toolbar: session + camera controls on the left, secondary tools on the right.
        control_card = card(outer, padx=14, pady=10)
        control_card.pack(fill="x", pady=(0, 12))
        bar = tk.Frame(control_card, bg=CARD)
        bar.pack(fill="x")
        self.caption(bar, "SESSION").pack(side="left", padx=(0, 10))
        self.session = tk.StringVar(value=args.session)
        self.session_entry = entry(bar, self.session, 11, width=22)
        self.session_entry.pack(side="left", ipady=6, ipadx=6)
        self.start_button = self.button(bar, "▶  Start camera", self.start, "primary")
        self.start_button.pack(side="left", padx=(14, 6))
        self.stop_button = self.button(bar, "■  Stop", lambda: self.send("stop"), "danger")
        self.stop_button.pack(side="left")
        self.history_button = self.button(bar, "History / CSV", self.open_history, "secondary")
        self.history_button.pack(side="right")
        tk.Frame(control_card, bg=BORDER, height=1).pack(fill="x", pady=9)
        device_bar = tk.Frame(control_card, bg=CARD)
        device_bar.pack(fill="x")
        self.caption(device_bar, "CAMERA").pack(side="left", padx=(0, 10))
        self.camera_choice = tk.StringVar(value=str(args.camera))
        indices = sorted(set(range(10)) | {args.camera})
        self.camera_selector = ttk.Combobox(device_bar, textvariable=self.camera_choice,
            values=[str(index) for index in indices], width=4, state="readonly", font=(FONT, 10))
        self.camera_selector.pack(side="left")
        self.camera_selector.bind("<<ComboboxSelected>>", self.camera_changed)
        self.test_camera_button = self.button(device_bar, "Test camera", self.test_camera, "secondary", 9)
        self.test_camera_button.pack(side="left", padx=(8, 0))
        self.camera_summary = tk.StringVar(value=f"Camera {args.camera} · Not tested")
        self.label(device_bar, variable=self.camera_summary, size=9, fg=MUTED).pack(side="left", padx=12)
        self.model_test_button = self.button(device_bar, "Model tests…", self.open_model_tests, "ghost", 9)
        self.model_test_button.pack(side="right")
        self.button(device_bar, "Diagnostics…", self.open_diagnostics, "ghost", 9).pack(side="right", padx=(0, 4))

        self.label(outer, "Stored locally  •  Camera images are not saved  •  Expressions are display only  •  Educational prototype",
                   8, bg=BG, fg=SUBTLE).pack(side="bottom", anchor="w", pady=(8, 0))
        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3, minsize=320)
        content.rowconfigure(0, weight=1)
        # A preferred column size is independent of live text. Grid still
        # allocates the actual size when the user resizes the window.
        left = tk.Frame(content, bg=BG, width=720)
        left.pack_propagate(False)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        # Live camera card with a colour-coded feedback banner of fixed height.
        camera_card = card(left, padx=14, pady=12)
        camera_card.columnconfigure(0, weight=1)
        camera_card.rowconfigure(1, weight=1)
        camera_top = tk.Frame(camera_card, bg=CARD)
        camera_top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.label(camera_top, "Live camera", 11, True).pack(side="left")
        self.speed = tk.StringVar(value="FPS not measured")
        speed_label = self.label(camera_top, variable=self.speed, size=9, fg=MUTED, width=34)
        speed_label.configure(anchor="e")
        speed_label.pack(side="right")
        self.canvas = tk.Canvas(camera_card, bg=CANVAS, highlightthickness=1, highlightbackground=BORDER, height=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self.placeholder() if self.canvas_image is None else None)
        checks_row = tk.Frame(camera_card, bg=CARD)
        checks_row.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.quality_chips = {}
        for idx, (key, title) in enumerate((("face", "Single face"), ("lighting", "Good lighting"),
                                            ("focus", "Sharp focus"), ("liveness", "Liveness"))):
            checks_row.columnconfigure(idx, weight=1, uniform="checks")
            chip = tk.Frame(checks_row, bg=SURFACE, padx=8, pady=4, highlightthickness=1, highlightbackground=BORDER)
            chip.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 6, 0))
            dot = tk.Label(chip, text="●", bg=SURFACE, fg=SUBTLE, font=(FONT, 8))
            dot.pack(side="left", padx=(0, 5))
            txt = tk.Label(chip, text=title, bg=SURFACE, fg=MUTED, font=(FONT, 8, "bold"), anchor="w")
            txt.pack(side="left", fill="x", expand=True)
            self.quality_chips[key] = (chip, dot, txt)
        self.scan_bar = tk.Canvas(camera_card, bg="#e5eaf0", highlightthickness=0, height=5)
        self.scan_bar.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.scan_bar.bind("<Configure>", lambda event: self.paint_scan_bar())
        self.scan_ratio, self.scan_color = 0.0, MINT
        soft, accent, ink = TONES["neutral"]
        self.feedback_box = tk.Frame(camera_card, bg=soft)
        self.feedback_box.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        self.feedback_box.columnconfigure(1, weight=1)
        self.feedback_accent = tk.Frame(self.feedback_box, bg=accent, width=4)
        self.feedback_accent.grid(row=0, column=0, rowspan=2, sticky="ns")
        self.feedback = tk.StringVar(value="Click Start camera when you are ready")
        self.feedback_label = self.label(self.feedback_box, variable=self.feedback, size=11, bold=True,
                                         bg=soft, fg=ink, width=1, height=2, wraplength=680)
        self.feedback_label.configure(anchor="nw")
        self.feedback_label.grid(row=0, column=1, sticky="ew", padx=(12, 10), pady=(7, 0))
        self.guidance = tk.StringVar(value="Choose a camera index while stopped. Test camera reads one frame without recording attendance.")
        self.guidance_label = self.label(self.feedback_box, variable=self.guidance, size=9, bg=soft, fg=MUTED,
                                         width=1, height=2, wraplength=650)
        self.guidance_label.configure(anchor="nw")
        self.guidance_label.grid(row=1, column=1, sticky="ew", padx=(12, 10), pady=(0, 7))
        self.feedback_label.bind("<Configure>", lambda event: self.feedback_label.configure(wraplength=max(100, event.width-8)))
        self.guidance_label.bind("<Configure>", lambda event: self.guidance_label.configure(wraplength=max(100, event.width-8)))
        self.feedback.trace_add("write", lambda *_: self.paint_feedback())

        # Result cards: a coloured top strip and value text show the state at a glance.
        stats = tk.Frame(left, bg=BG)
        stats.pack(side="bottom", fill="x", pady=(12, 0))
        self.identity_text, self.live_text, self.expression_text = (tk.StringVar(value="—") for _ in range(3))
        self.stat_vars = {"identity": self.identity_text, "liveness": self.live_text, "expression": self.expression_text}
        self.stat_cards = {}
        for index, (key, title) in enumerate((("identity", "IDENTITY  ·  click for details"),
                                              ("liveness", "LIVENESS"), ("expression", "EXPRESSION"))):
            stats.columnconfigure(index, weight=1, uniform="stats")
            frame = card(stats, padx=0, pady=0)
            frame.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 10, 0))
            strip = tk.Frame(frame, bg=TONES["neutral"][1], height=4)
            strip.pack(fill="x")
            inner = tk.Frame(frame, bg=CARD, padx=14, pady=10)
            inner.pack(fill="both", expand=True)
            self.caption(inner, title).pack(anchor="w")
            value_label = self.label(inner, variable=self.stat_vars[key], size=13, bold=True,
                                     fg=SUBTLE, width=1, height=2, wraplength=200)
            value_label.configure(anchor="nw")
            value_label.pack(fill="x", pady=(4, 0))
            value_label.bind("<Configure>", lambda event: event.widget.configure(
                wraplength=max(100, event.width-8)))
            if index == 0:
                # The fixed slot must not grow for an 80-character profile.
                # Keep the complete value available by mouse and keyboard.
                value_label.configure(cursor="hand2", takefocus=True,
                                      highlightthickness=1, highlightbackground=CARD, highlightcolor=BLUE)
                value_label.bind("<Button-1>", self.show_identity_details)
                value_label.bind("<Return>", self.show_identity_details)
            self.stat_cards[key] = (strip, value_label)
            self.stat_vars[key].trace_add("write", lambda *_, name=key: self.paint_stat(name))
        camera_card.pack(fill="both", expand=True)

        # Right column: tabbed panel (Registration | Model details) + Present this session.
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1, minsize=150)
        panel_card = card(right, padx=14, pady=10)
        panel_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        panel_card.columnconfigure(0, weight=1)
        tab_bar = tk.Frame(panel_card, bg=SURFACE, padx=3, pady=3, highlightthickness=1, highlightbackground=BORDER)
        tab_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tab_bar.columnconfigure((0, 1), weight=1, uniform="tabs")
        self.tab_buttons = {}
        self.tab_frames = {}
        for idx, (key, title) in enumerate((("register", "People & registration"), ("models", "Live model scores"))):
            btn = tk.Button(tab_bar, text=title, command=lambda k=key: self.select_tab(k),
                            relief="flat", bd=0, highlightthickness=0, pady=5, cursor="hand2",
                            font=(FONT, 9, "bold"))
            btn.grid(row=0, column=idx, sticky="ew", padx=(0 if idx == 0 else 2, 0))
            self.tab_buttons[key] = btn
        tab_stack = tk.Frame(panel_card, bg=CARD)
        tab_stack.grid(row=1, column=0, sticky="nsew")
        tab_stack.columnconfigure(0, weight=1)
        tab_stack.rowconfigure(0, weight=1)

        register = tk.Frame(tab_stack, bg=CARD)
        register.grid(row=0, column=0, sticky="nsew")
        self.tab_frames["register"] = register
        hint = self.label(register, "New person: register with name / student ID. "
                          "Changed look: add an appearance to an existing profile.", 9, fg=MUTED, wraplength=280)
        hint.pack(fill="x", anchor="w", pady=(0, 8))
        hint.bind("<Configure>", lambda event: event.widget.configure(wraplength=max(120, event.width-4)))
        self.caption(register, "NAME / STUDENT ID").pack(anchor="w", pady=(0, 3))
        self.name = tk.StringVar()
        entry(register, self.name, 11).pack(fill="x", ipady=5, ipadx=6)
        register_actions = tk.Frame(register, bg=CARD)
        register_actions.pack(fill="x", pady=(8, 0))
        self.register_button = self.button(register_actions, "Register from camera",
                                           lambda: self.send("register", self.name.get()), "primary")
        self.register_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        cancel_button = self.button(register_actions, "Cancel capture", lambda: self.send("cancel"), "secondary")
        cancel_button.pack(side="right")
        existing_actions = tk.Frame(register, bg=CARD)
        existing_actions.pack(fill="x", pady=(6, 0))
        existing_actions.columnconfigure((0, 1), weight=1, uniform="actions")
        self.appearance_button = self.button(existing_actions, "Add appearance", lambda: self.send("appearance_request"))
        self.appearance_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.review_button = self.button(existing_actions, "Manual review", lambda: self.send("review"))
        self.review_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        tk.Frame(register, bg=BORDER, height=1).pack(fill="x", pady=(9, 6))
        self.profile_count = tk.StringVar(value="0 registered profiles")
        profile_bar = tk.Frame(register, bg=CARD)
        profile_bar.pack(fill="x")
        self.label(profile_bar, variable=self.profile_count, size=9, fg=MUTED).pack(side="left")
        self.profiles_button = self.button(profile_bar, "Profiles…", self.open_profiles, "ghost", 9)
        self.profiles_button.pack(side="right")

        models_tab = tk.Frame(tab_stack, bg=CARD)
        models_tab.grid(row=0, column=0, sticky="nsew")
        self.tab_frames["models"] = models_tab
        self.pad_detail = tk.StringVar(value="Waiting for live frame")
        self.id_detail = tk.StringVar(value="Waiting for live frame")
        self.build_telemetry_tab(models_tab)
        self.select_tab("register")

        attendance = card(right, padx=14, pady=10)
        attendance.grid(row=1, column=0, sticky="nsew")
        attendance_head = tk.Frame(attendance, bg=CARD)
        attendance_head.pack(fill="x", pady=(0, 6))
        self.label(attendance_head, "Present this session", 12, True).pack(side="left")
        self.present_count = tk.StringVar(value="0 present")
        tk.Label(attendance_head, textvariable=self.present_count, bg=PRIMARY_SOFT, fg=PRIMARY_INK,
                 font=(FONT, 9, "bold"), padx=10, pady=2).pack(side="right")
        tree_wrap = tk.Frame(attendance, bg=CARD)
        self.tree = ttk.Treeview(tree_wrap, columns=("name", "time", "source"), show="headings", height=3)
        self.tree.heading("name", text="NAME / STUDENT ID", anchor="w")
        self.tree.heading("time", text="TIME")
        self.tree.heading("source", text="SOURCE")
        self.tree.column("name", width=165, stretch=True)
        self.tree.column("time", width=56, stretch=False, anchor="center")
        self.tree.column("source", width=84, stretch=False, anchor="center")
        self.tree.tag_configure("odd", background=SURFACE)
        self.tree.tag_configure("manual", foreground="#b45309")
        self.tree.bind("<Double-1>", self.show_attendance_detail)
        tree_scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.label(attendance, "One record per session · Double-click for details", 8,
                   fg=SUBTLE).pack(side="bottom", anchor="w", pady=(7, 0))
        tree_wrap.pack(fill="both", expand=True)
        self.photo = None
        self.attendance_rows = None
        self.controls(False, False)
        if args.preview:
            self.profiles_button.configure(state="normal")
        root.after(150, self.placeholder)
        if getattr(args, "preview_review", False):
            root.after(300, self.preview_review)
        elif getattr(args, "preview_appearance", False):
            root.after(300, self.preview_appearance)
        elif getattr(args, "preview_history", False):
            root.after(300, self.open_history)
        elif getattr(args, "preview_profiles", False):
            root.after(300, self.open_profiles)
        elif getattr(args, "preview_diagnostics", False):
            self.component_status = {key: {"status": "Not checked", "detail": "Preview data only"} for key in COMPONENTS}
            self.component_status["Identity"] = {"status": "Ready", "detail": "PREVIEW · Example configuration loaded"}
            self.component_status["Liveness"] = {"status": "Error", "detail": "PREVIEW · Example missing checkpoint; no files were read"}
            root.after(300, self.open_diagnostics)
        elif getattr(args, "preview_model_tests", False):
            root.after(300, self.open_model_tests)
        if not args.preview:
            self.worker = CameraWorker(args, self.commands, self.updates, self.previews, self.analyses)
            self.worker.start()
        root.after(60, self.poll)

    def label(self, parent, text=None, size=11, bold=False, bg=CARD, fg=INK, variable=None, **kwargs):
        return self.tk.Label(parent, text=text, textvariable=variable, bg=bg, fg=fg, anchor="w", justify="left",
                             font=(FONT, size, "bold" if bold else "normal"), **kwargs)

    def caption(self, parent, text, bg=CARD):
        return self.tk.Label(parent, text=text, bg=bg, fg=MUTED, anchor="w", font=(FONT, 8, "bold"))

    def build_app_bar(self, root):
        tk = self.tk
        bar = tk.Frame(root, bg=HEADER_BG, padx=20, pady=10)
        bar.pack(fill="x")
        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")
        logo = tk.Canvas(bar, width=40, height=40, bg=HEADER_BG, highlightthickness=0)
        logo.create_polygon(20, 3, 35, 11.5, 35, 28.5, 20, 37, 5, 28.5, 5, 11.5,
                            fill=PRIMARY_SOFT, outline=MINT, width=1.8)
        logo.create_oval(15, 11, 25, 22, outline=PRIMARY_INK, width=1.6)
        logo.create_arc(11, 22, 29, 36, start=15, extent=150, style="arc", outline=PRIMARY_INK, width=1.6)
        logo.create_arc(7, 17, 33, 27, start=20, extent=140, style="arc", outline=MINT, width=1.6)
        logo.pack(side="left", padx=(0, 12))
        titles = tk.Frame(bar, bg=HEADER_BG)
        titles.pack(side="left")
        brand_row = tk.Frame(titles, bg=HEADER_BG)
        brand_row.pack(anchor="w")
        tk.Label(brand_row, text="Face Attendance", bg=HEADER_BG, fg=INK, font=(FONT, 15, "bold")).pack(side="left")
        tk.Label(brand_row, text="  ·  Live Verification", bg=HEADER_BG, fg=PRIMARY_INK,
                 font=(FONT, 10, "bold")).pack(side="left", pady=(3, 0))
        tk.Label(titles, text="Identity verification  ·  Anti-spoofing liveness  ·  Facial expression",
                 bg=HEADER_BG, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        tk.Label(bar, text="COS30082 · LOCAL DEMO", bg=HEADER_BG, fg=SUBTLE, font=(FONT, 8, "bold")).pack(side="right", padx=(16, 0))
        pill = tk.Frame(bar, bg=PILL_BG, padx=12, pady=5, highlightthickness=1, highlightbackground=BORDER)
        pill.pack(side="right")
        self.state_dot = tk.Canvas(pill, width=10, height=10, bg=PILL_BG, highlightthickness=0)
        self.state_dot_item = self.state_dot.create_oval(1, 1, 9, 9, fill=SUBTLE, outline="")
        self.state_dot.pack(side="left", padx=(0, 8))
        tk.Label(pill, textvariable=self.state_text, bg=PILL_BG, fg=INK, font=(FONT, 9, "bold")).pack(side="left")
        self.state_text.trace_add("write", lambda *_: self.paint_state())
        self.paint_state()

    def select_tab(self, active):
        self.active_tab = active
        for key, button in self.tab_buttons.items():
            if key == active:
                button.configure(bg=CARD, fg=PRIMARY_INK)
            else:
                button.configure(bg=SURFACE, fg=MUTED)
        self.tab_frames[active].tkraise()

    def build_telemetry_tab(self, parent):
        tk = self.tk
        self.caption(parent, "LIVENESS (PAD) & IDENTITY").pack(anchor="w")
        self.telemetry_bars = {}
        for key, title, var in (("pad", "Liveness score", self.pad_detail),
                                ("identity", "Identity match", self.id_detail)):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=(3, 0))
            tk.Label(row, text=title, bg=CARD, fg=INK, font=(FONT, 8, "bold")).pack(side="left")
            tk.Label(row, textvariable=var, bg=CARD, fg=MUTED, font=(FONT, 8)).pack(side="right")
            bar = tk.Canvas(parent, bg="#e5eaf0", highlightthickness=0, height=5)
            bar.pack(fill="x", pady=(2, 2))
            bar.bind("<Configure>", lambda event, k=key: self.paint_mini_bar(k))
            self.telemetry_bars[key] = [bar, 0.0, MINT]
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=(4, 4))
        self.caption(parent, "EXPRESSIONS (FER2013 · DISPLAY ONLY)").pack(anchor="w", pady=(0, 2))
        self.emotion_bars = {}
        for name in ("happy", "neutral", "surprise", "sad", "angry", "fear", "disgust"):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=0)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text=name.title(), bg=CARD, fg=MUTED, font=(FONT, 8), width=8, anchor="w").grid(row=0, column=0, sticky="w")
            bar = tk.Canvas(row, bg="#edf1f5", highlightthickness=0, height=6)
            bar.grid(row=0, column=1, sticky="ew", padx=(4, 6))
            pct = tk.Label(row, text="—", bg=CARD, fg=SUBTLE, font=(FONT, 8), width=4, anchor="e")
            pct.grid(row=0, column=2, sticky="e")
            bar.bind("<Configure>", lambda event, k=name: self.paint_emotion_bar(k))
            self.emotion_bars[name] = [bar, pct, 0.0, False]

    def paint_mini_bar(self, key):
        bar, ratio, color = self.telemetry_bars[key]
        bar.delete("all")
        w, h = max(1, bar.winfo_width()), max(1, bar.winfo_height())
        bar.create_rectangle(0, 0, w, h, fill="#e5eaf0", outline="")
        if ratio > 0:
            bar.create_rectangle(0, 0, max(4, int(w * min(1.0, ratio))), h, fill=color, outline="")

    def paint_emotion_bar(self, key):
        bar, _, ratio, top = self.emotion_bars[key]
        bar.delete("all")
        w, h = max(1, bar.winfo_width()), max(1, bar.winfo_height())
        bar.create_rectangle(0, 0, w, h, fill="#edf1f5", outline="")
        if ratio > 0:
            bar.create_rectangle(0, 0, max(3, int(w * min(1.0, ratio))), h,
                                 fill=MINT if top else "#93c5b8", outline="")

    def set_chip(self, key, tone):
        chip, dot, txt = self.quality_chips[key]
        if tone == "neutral":
            bg, border, dot_c, fg = SURFACE, BORDER, SUBTLE, MUTED
        else:
            bg, dot_c, fg = TONES[tone]
            border = dot_c if tone == "success" else BORDER
        chip.configure(bg=bg, highlightbackground=border)
        dot.configure(bg=bg, fg=dot_c)
        txt.configure(bg=bg, fg=fg)

    def update_quality_chips(self, result=None):
        if result is None:
            for key in self.quality_chips:
                self.set_chip(key, "neutral")
            return
        status = result.get("status", "")
        boxes = result.get("boxes", [])
        reasons = set(result.get("quality_reasons") or [])
        if status == "multiple_faces":
            self.set_chip("face", "danger")
        elif status in ("no_face", "face_too_small_or_outside") or len(boxes) != 1:
            self.set_chip("face", "warning")
        else:
            self.set_chip("face", "success")
        if "brightness" in result:
            self.set_chip("lighting", "warning" if reasons & {"dark", "bright"} else "success")
        else:
            self.set_chip("lighting", "neutral")
        if "sharpness" in result:
            self.set_chip("focus", "warning" if "blur" in reasons else "success")
        else:
            self.set_chip("focus", "neutral")
        live = result.get("liveness")
        if live is None:
            self.set_chip("liveness", "neutral")
        elif live.get("is_live") is True:
            self.set_chip("liveness", "success")
        elif live.get("is_live") is False:
            self.set_chip("liveness", "danger")
        else:
            self.set_chip("liveness", "warning")

    def update_telemetry(self, result=None):
        if result is None:
            self.pad_detail.set("Waiting for live frame")
            self.id_detail.set("Waiting for live frame")
            for key in self.telemetry_bars:
                self.telemetry_bars[key][1], self.telemetry_bars[key][2] = 0.0, MINT
                self.paint_mini_bar(key)
            for name, item in self.emotion_bars.items():
                item[2], item[3] = 0.0, False
                item[1].configure(text="—", fg=SUBTLE)
                self.paint_emotion_bar(name)
            return
        live = result.get("liveness") or {}
        score, threshold = live.get("live_score"), live.get("threshold")
        if score is not None and threshold is not None:
            self.pad_detail.set(f"{score:.2f}  (thr {threshold:.2f})")
            self.telemetry_bars["pad"][1] = max(0.0, min(1.0, float(score)))
            self.telemetry_bars["pad"][2] = MINT if live.get("is_live") is True else "#ef4444"
        else:
            self.pad_detail.set("Not checked")
            self.telemetry_bars["pad"][1] = 0.0
        self.paint_mini_bar("pad")
        ident = result.get("identity") or {}
        cos = ident.get("cosine")
        streak = result.get("stable_frames", 0)
        if cos is not None:
            suffix = f" · {streak}/3" if ident.get("status") == "known" else ""
            self.id_detail.set(f"cos {cos:.2f}{suffix}")
            self.telemetry_bars["identity"][1] = max(0.0, min(1.0, (float(cos) + 1.0) / 2.0))
            self.telemetry_bars["identity"][2] = MINT if ident.get("status") == "known" else "#f59e0b"
        else:
            self.id_detail.set("Not checked")
            self.telemetry_bars["identity"][1] = 0.0
        self.paint_mini_bar("identity")
        expr = result.get("expression") or {}
        scores = expr.get("scores") or {}
        top_label = expr.get("label")
        for name, item in self.emotion_bars.items():
            val = float(scores.get(name, 0.0)) if scores else 0.0
            is_top = bool(scores) and name == top_label
            item[2], item[3] = val, is_top
            item[1].configure(text=f"{val:.0%}" if scores else "—",
                              fg=PRIMARY_INK if is_top else MUTED if scores else SUBTLE)
            self.paint_emotion_bar(name)

    def paint_state(self):
        text = self.state_text.get().casefold()
        if "not ready" in text or "error" in text:
            color = "#ef4444"
        elif "running" in text or "recorded" in text or text.startswith("ready"):
            color = MINT
        elif any(word in text for word in ("loading", "opening", "closing", "retry")):
            color = "#f59e0b"
        else:
            color = SUBTLE
        self.state_dot.itemconfigure(self.state_dot_item, fill=color)

    def paint_scan_bar(self):
        bar = self.scan_bar
        bar.delete("all")
        width = max(1, bar.winfo_width())
        height = max(1, bar.winfo_height())
        bar.create_rectangle(0, 0, width, height, fill="#e5eaf0", outline="")
        if self.scan_ratio > 0:
            fill_width = max(8, int(width * min(1.0, self.scan_ratio)))
            bar.create_rectangle(0, 0, fill_width, height, fill=self.scan_color, outline="")

    def paint_feedback(self):
        text = self.feedback.get()
        tone = message_tone(text)
        soft, accent, ink = TONES[tone]
        self.feedback_box.configure(bg=soft)
        self.feedback_accent.configure(bg=accent)
        self.feedback_label.configure(bg=soft, fg=ink)
        self.guidance_label.configure(bg=soft)
        lowered = text.casefold()
        if "3/3" in text or text in (MESSAGES["attendance_confirmed"], MESSAGES["registered"], MESSAGES["appearance_added"]):
            self.scan_ratio, self.scan_color = 1.0, "#10b981"
        elif text == MESSAGES["known"]:
            self.scan_ratio, self.scan_color = 0.72, MINT
        elif "2/3" in text:
            self.scan_ratio, self.scan_color = 0.67, MINT
        elif "1/3" in text:
            self.scan_ratio, self.scan_color = 0.34, MINT
        elif tone == "danger":
            self.scan_ratio, self.scan_color = 1.0, "#ef4444"
        elif tone == "warning":
            self.scan_ratio, self.scan_color = 0.50, "#f59e0b"
        elif lowered.startswith(("analyzing", "face the camera", "testing")):
            self.scan_ratio, self.scan_color = 0.38, MINT
        elif lowered.startswith(("models ready", "opened")):
            self.scan_ratio, self.scan_color = 0.16, MINT
        else:
            self.scan_ratio, self.scan_color = 0.0, MINT
        self.paint_scan_bar()

    def paint_stat(self, key):
        strip, value_label = self.stat_cards[key]
        tone = stat_tone(key, self.stat_vars[key].get())
        strip.configure(bg=TONES[tone][1])
        value_label.configure(fg=SUBTLE if tone == "neutral" else TONES[tone][2])

    def camera_changed(self, event=None):
        self.camera_status = "Not tested"
        self.camera_summary.set(f"Camera {self.camera_choice.get()} · Not tested")
        self.last_frame_details = ""
        self.refresh_diagnostics()

    def device_locked(self):
        return (self.running or self.camera_starting or self.probe_busy or self.closing
                or self.model_test_dialog is not None)

    def refresh_device_controls(self):
        locked = self.device_locked()
        profiles_busy = self.profiles_dialog is not None or self.profiles_opening
        self.camera_selector.configure(state="disabled" if locked or profiles_busy else "readonly")
        self.test_camera_button.configure(state="normal" if not locked and not profiles_busy and not self.args.preview else "disabled")
        self.start_button.configure(state="normal" if self.models_ready and not locked and not profiles_busy else "disabled")
        self.profiles_button.configure(state="normal" if (self.models_ready or self.args.preview) and not locked else "disabled")
        self.model_test_button.configure(state="normal" if not locked and not profiles_busy else "disabled")
        self.session_entry.configure(state="disabled" if locked else "normal")

    def test_camera(self):
        if self.device_locked() or self.profiles_dialog is not None or self.profiles_opening or self.args.preview:
            return
        index = camera_index(self.camera_choice.get())
        self.probe_busy = True
        self.camera_status = "Testing selected camera…"
        self.camera_summary.set(f"Camera {index} · Testing…")
        self.feedback.set("Testing one camera frame · No attendance is recorded")
        self.refresh_device_controls()
        self.refresh_diagnostics()

        def check():
            report = probe_camera(index, backend=getattr(self.args, "camera_backend", None))
            self.updates.put_nowait({"event": "camera_probe", "report": report})

        self.camera_probe = threading.Thread(target=check, daemon=True)
        self.camera_probe.start()

    def can_retry_setup(self):
        return (not self.args.preview and not self.setup_loading and not self.models_ready
                and not self.device_locked() and self.profiles_dialog is None and not self.profiles_opening)

    def retry_setup(self):
        if not self.can_retry_setup():
            return
        self.setup_loading = True
        self.component_status = {}
        self.state_text.set("Loading models…")
        self.feedback.set("Retrying model setup · Camera stays off")
        self.refresh_diagnostics()
        self.root.after(50, self.start_setup_worker)

    def start_setup_worker(self):
        if self.closing:
            return
        if self.worker is not None and self.worker.is_alive():
            self.root.after(80, self.start_setup_worker)
            return
        # A failed setup has exited; never run two SQLite/model workers together.
        self.commands = queue.Queue()
        self.worker = CameraWorker(self.args, self.commands, self.updates, self.previews, self.analyses)
        self.worker.start()

    def open_diagnostics(self):
        if self.diagnostics_dialog is not None:
            self.diagnostics_dialog.window.lift()
        else:
            from .diagnostics_dialog import DiagnosticsDialog
            self.diagnostics_dialog = DiagnosticsDialog(self.root, self.retry_setup,
                lambda: setattr(self, "diagnostics_dialog", None), preview=self.args.preview)
        self.refresh_diagnostics()

    def refresh_diagnostics(self):
        if self.diagnostics_dialog is not None:
            mode = "PREVIEW ONLY · No hardware or models loaded" if self.args.preview else f"Inference device: {self.args.device}"
            runtime = (f"{mode}\nSelected camera: {self.camera_choice.get()} · "
                       f"{'Running' if self.running else 'Opening' if self.camera_starting else 'Off'}\n"
                       f"Camera check: {self.camera_status}\n"
                       "Test camera opens only the selected device for one frame. No images or attendance are saved.")
            self.diagnostics_dialog.update(self.component_status, runtime, self.last_frame_details, self.can_retry_setup())

    def preview_review(self):
        from .review_dialog import ReviewDialog
        from datetime import timezone
        crop = Image.new("RGB", (180, 130), "#eef2f6")
        draw = ImageDraw.Draw(crop)
        draw.text((42, 58), "PREVIEW ONLY", fill="#6b7280")
        request = {"id": "preview-only-never-saved", "expires_at": time.monotonic()+30.,
                   "observed_utc": datetime.now(timezone.utc).isoformat(), "crop": crop,
                   "profiles": [{"id": "demo-a", "name": "Demo profile A"}, {"id": "demo-b", "name": "Demo profile B"}]}
        self.review_dialog = ReviewDialog(self.root, request,
            lambda fields: self.review_dialog.show_error("Preview only: no data is recorded."),
            lambda token: self.cancel_review_dialog(token))
        self.review_dialog.window.title("Manual review · PREVIEW, NO DATA SAVED")

    def preview_appearance(self):
        from .appearance_dialog import AppearanceDialog
        request = {"id": "preview-only-never-saved", "expires_at": time.monotonic()+30.,
                   "profiles": [{"id": "demo-a", "name": "Demo profile A"},
                                {"id": "demo-b", "name": "Demo profile B"}]}
        self.appearance_dialog = AppearanceDialog(self.root, request,
            self.preview_appearance_submitted, self.cancel_appearance_dialog)
        self.appearance_dialog.window.title("Add appearance · PREVIEW, NO DATA SAVED")

    def preview_appearance_submitted(self, fields):
        self.appearance_dialog = None
        self.feedback.set("Preview only · No appearance samples or data are captured or saved")

    def open_history(self):
        if self.history_dialog is not None:
            self.history_dialog.window.lift()
            return
        from .history_dialog import HistoryDialog, preview_history_rows
        self.history_dialog = HistoryDialog(self.root, self.args.database, self.history_closed,
            preview_rows=preview_history_rows() if self.args.preview else None)

    def history_closed(self):
        self.history_dialog = None

    def open_model_tests(self):
        if self.model_test_dialog is not None:
            self.model_test_dialog.window.lift()
            return
        if (self.device_locked() or self.profiles_dialog is not None or self.profiles_opening
                or self.appearance_dialog is not None or self.review_dialog is not None):
            self.feedback.set("Stop the camera and finish any open capture or profile operation before model tests.")
            return
        from .model_test_dialog import ModelTestDialog
        self.model_test_dialog = ModelTestDialog(self.root, PROJECT,
            getattr(self.args, "liveness_checkpoint", DEFAULT_LIVENESS_CHECKPOINT),
            camera_index(self.camera_choice.get()), self.model_tests_closed, preview=self.args.preview)
        self.refresh_device_controls()

    def model_tests_closed(self):
        self.model_test_dialog = None
        if not self.closing:
            self.refresh_device_controls()

    def open_profiles(self):
        if self.profiles_dialog is not None:
            self.profiles_dialog.window.lift()
            return
        if self.device_locked():
            return
        if self.args.preview:
            from .profiles_dialog import ProfilesDialog
            profiles = [{"id": "demo-person-a", "name": "Demo person A", "created_utc": "2026-10-30T02:00:00+00:00",
                "sample_count": 6, "appearances": [{"id": "original:demo-person-a", "label": "Original", "sample_count": 3},
                                                  {"id": "demo-look", "label": "Without glasses", "sample_count": 3}]}]
            self.profiles_dialog = ProfilesDialog(self.root, profiles, self.send, self.profiles_closed, preview=True)
            return
        if not self.models_ready or self.device_locked() or self.profiles_opening:
            return
        self.profiles_opening = True
        self.refresh_device_controls()
        self.profiles_button.configure(state="disabled")
        self.send("profiles_list")

    def profiles_closed(self):
        self.send("profiles_close")
        self.profiles_dialog = None
        self.profiles_opening = False
        if not self.closing:
            self.refresh_device_controls()

    def button(self, parent, text, action, variant="secondary", size=10):
        return FlatButton(parent, text, action, variant, size)

    def controls(self, ready, running):
        self.running = running
        self.start_button.configure(state="normal" if ready and not running and self.profiles_dialog is None and not self.profiles_opening else "disabled")
        self.stop_button.configure(state="normal" if running else "disabled")
        self.register_button.configure(state="disabled")
        self.appearance_button.configure(state="disabled")
        self.review_button.configure(state="disabled")
        self.profiles_button.configure(state="normal" if ready and not running else "disabled")
        self.session_entry.configure(state="disabled" if running else "normal")
        self.refresh_device_controls()

    def send(self, command, value=None):
        if self.worker is not None and self.worker.is_alive():
            self.commands.put((command, value))

    def start(self):
        if not self.models_ready or self.device_locked() or self.profiles_dialog is not None or self.profiles_opening:
            return
        self.camera_starting = True
        self.last_frame_details = ""
        self.refresh_device_controls()
        self.state_text.set("Opening camera…")
        self.send("start", {"session": self.session.get(), "camera": camera_index(self.camera_choice.get())})

    def placeholder(self):
        if not self.running:
            canvas = self.canvas
            canvas.configure(bg=CANVAS)
            canvas.delete("all")
            width, height = canvas.winfo_width(), canvas.winfo_height()
            cx = width / 2
            cy = max(24, (height - 58) / 2)
            scale = max(.26, min(1.0, (height - 78) / 195))
            # Soft framing corner brackets around the center scan zone
            bw, bh, arm = 96*scale, 84*scale, 18*scale
            for sx in (-1, 1):
                for sy in (-1, 1):
                    x, y = cx + sx*bw, cy + sy*bh
                    canvas.create_line(x, y, x - sx*arm, y, fill="#cbd5e1", width=2, capstyle="round")
                    canvas.create_line(x, y, x, y - sy*arm, fill="#cbd5e1", width=2, capstyle="round")
            # Geometric hexagon + minimalist face outline + mint scan arc
            rx, ry = 58*scale, 66*scale
            hex_pts = [cx, cy-ry, cx+rx, cy-ry*0.5, cx+rx, cy+ry*0.5,
                       cx, cy+ry, cx-rx, cy+ry*0.5, cx-rx, cy-ry*0.5]
            canvas.create_polygon(*hex_pts, fill="#ffffff", outline="#94a3b8", width=2)
            canvas.create_oval(cx-24*scale, cy-36*scale, cx+24*scale, cy+20*scale,
                               outline="#475569", width=2)
            canvas.create_arc(cx-40*scale, cy+16*scale, cx+40*scale, cy+84*scale,
                              start=20, extent=140, style="arc", outline="#475569", width=2)
            canvas.create_arc(cx-78*scale, cy-16*scale, cx+78*scale, cy+34*scale,
                              start=24, extent=132, style="arc", outline=MINT, width=2.4)
            canvas.create_text(cx, height-48, text="Live face detection", fill=INK, font=(FONT, 14, "bold"))
            canvas.create_text(cx, height-24, text="Click “Start camera” to scan your face and verify attendance",
                               fill=MUTED, font=(FONT, 10))

    def render_attendance(self, rows):
        if rows == self.attendance_rows:
            return
        self.attendance_rows = rows
        self.present_count.set(f"{len(rows)} present")
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(rows):
            local = datetime.fromisoformat(row["recorded_utc"]).astimezone().strftime("%H:%M")
            source = {"automatic": "Automatic", "manual_review": "Manual"}.get(row.get("source"), "Unknown")
            tags = (("odd",) if index % 2 else ()) + (("manual",) if row.get("source") == "manual_review" else ())
            self.tree.insert("", "end", iid=str(index), values=(row["name"], local, source), tags=tags)

    def show_attendance_detail(self, event=None):
        selection = self.tree.selection()
        if selection:
            from tkinter import messagebox
            row = self.attendance_rows[int(selection[0])]
            source = {"automatic": "Automatic", "manual_review": "Manual review"}.get(row.get("source"), "Source not recorded")
            text = f"Name: {row['name']}\nSource: {source}\nReviewer: {row.get('reviewer') or '—'}\nReason: {row.get('reason') or '—'}"
            messagebox.showinfo("Attendance details", text, parent=self.root)

    def show_identity_details(self, event=None):
        from tkinter import messagebox
        messagebox.showinfo("Identity details", self.identity_text.get(), parent=self.root)
        return "break"

    def render_frame(self, update):
        if not self.running or update["stream_id"] != self.stream_id:
            return
        if time.monotonic() - update["captured_at"] > 1:
            return
        self.last_analysis = update
        self.analysis_fps = self.measured_fps(self.analysis_times, time.monotonic())
        result = update["result"]
        status = result["status"]
        message = MESSAGES.get(status, status)
        if status == "registering":
            message = f"Sample {result['registration_samples']}/3 · Turn your face slightly"
        if result.get("appearance_pending"):
            seconds_left = max(0., result.get("appearance_seconds_left", 0.))
            progress = f"Appearance sample {result['registration_samples']}/3 · {seconds_left:.0f}s left"
            message = f"{progress} · Turn your face slightly" if status == "appearance_capturing" else f"{progress} · {message}"
        if (self.appearance_dialog is not None and result.get("appearance_dialog_pending")
                and not result.get("appearance_request_valid")):
            self.appearance_dialog.invalidate(result.get("appearance_request_reason")
                                             or "The observation is no longer valid. Cancel and try again.")
        if result.get("manual_review_pending"):
            message = "Manual review in progress · Keep the same person in view"
            if self.review_dialog is not None and not result.get("manual_review_valid"):
                self.review_dialog.invalidate("The face or liveness is no longer valid. Cancel and capture a new observation.")
        self.feedback.set(message)
        self.guidance.set(frame_guidance(result))
        self.last_frame_details = frame_details(result)
        self.last_frame_details += (f"\nAnalysis: {update['processing_seconds']*1000:.1f} ms"
                                    f"\nObservation age: {(time.monotonic()-update['captured_at'])*1000:.0f} ms"
                                    f"\nPreview: {self.preview_fps:.1f} FPS; analysis: {self.analysis_fps:.1f} FPS")
        self.identity_text.set((result.get("identity") or {}).get("name") or
                               ("Not recognized" if result.get("identity") is not None else "Not checked"))
        live = result.get("liveness")
        self.live_text.set("Not checked" if live is None else "Passed" if live.get("is_live") is True
                           else "Unavailable" if live.get("is_live") is None else "Not passed")
        expression = result.get("expression")
        self.expression_text.set(format_expression(expression, result.get("expression_error")))
        self.update_quality_chips(result)
        self.update_telemetry(result)
        self.profile_count.set(f"{update['profiles']} registered profiles")
        self.render_attendance(update["attendance"])
        capture_pending = update.get("capture_pending", False)
        dialog_open = self.appearance_dialog is not None or self.review_dialog is not None
        self.register_button.configure(state="normal" if status == "unknown" and not capture_pending and not dialog_open else "disabled")
        self.appearance_button.configure(state="normal" if status in ("unknown", "known", "attendance_confirmed")
                                         and (live or {}).get("is_live") is True and update["profiles"] > 0
                                         and not capture_pending and not dialog_open
                                         and not result.get("manual_review_pending") else "disabled")
        self.review_button.configure(state="normal" if status in ("unknown", "ambiguous") and update["profiles"] > 0
                                     and not result.get("manual_review_pending") and not capture_pending
                                     and not dialog_open else "disabled")
        if result.get("registered"):
            self.name.set("")

    @staticmethod
    def measured_fps(samples, now):
        samples.append(now)
        while len(samples) > 2 and samples[0] < now-2:
            samples.pop(0)
        return (len(samples)-1)/(samples[-1]-samples[0]) if len(samples) > 1 and samples[-1] > samples[0] else 0.

    def render_preview(self, update):
        if not self.running or update["stream_id"] != self.stream_id:
            return
        sample = update["sample"]
        now = time.monotonic()
        if now - sample.captured_at > 1:
            return
        image = Image.fromarray(sample.rgb)
        analysis = self.last_analysis
        # The outline reflects only the latest fresh analysis. Recognition/PAD
        # decisions still come exclusively from the analyzed original crop.
        # Never put a previous run, resolution or future frame's box here.
        if (analysis is not None and analysis["stream_id"] == update["stream_id"]
                and analysis["rgb"].shape == sample.rgb.shape
                and sample.sequence >= analysis["sequence"]
                and 0 <= sample.captured_at-analysis["captured_at"] <= .2
                and 0 <= now-analysis["captured_at"] <= .2):
            draw = ImageDraw.Draw(image)
            result = analysis["result"]
            identity = result.get("identity") or {}
            recognized = (len(result["boxes"]) == 1
                          and (result.get("liveness") or {}).get("is_live") is True
                          and identity.get("status") == "known"
                          and identity.get("person_id") is not None)
            color = FACE_RECOGNIZED if recognized else FACE_PENDING
            for x, y, width, height in result["boxes"]:
                draw.rectangle((x, y, x+width, y+height), outline=color, width=3)
        image.thumbnail((max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(image)
        if self.canvas_image is None:
            self.canvas.delete("all")
            self.canvas_image = self.canvas.create_image(0, 0, image=self.photo)
        else:
            self.canvas.itemconfigure(self.canvas_image, image=self.photo)
        self.canvas.coords(self.canvas_image, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2)
        self.preview_fps = self.measured_fps(self.preview_times, now)
        self.speed.set(f"Preview {self.preview_fps:.0f} FPS · Analysis {self.analysis_fps:.0f} FPS")

    def expire_analysis(self):
        if not self.running or (self.last_analysis is not None
                and time.monotonic()-self.last_analysis["captured_at"] <= 1):
            return
        self.identity_text.set("Not checked")
        self.live_text.set("Not checked")
        self.expression_text.set("—")
        self.update_quality_chips(None)
        self.update_telemetry(None)
        self.feedback.set("Analyzing the current camera view…")
        self.guidance.set("Keep one face in view. Identity, liveness and expression results appear below the camera.")
        for button in (self.register_button, self.appearance_button, self.review_button):
            button.configure(state="disabled")

    def poll(self):
        if self.closing:
            return
        try:
            while True:
                update = self.updates.get_nowait()
                event = update["event"]
                if event == "component":
                    self.component_status[update["component"]] = {"status": update["status"], "detail": update["detail"]}
                elif event == "camera_probe":
                    self.probe_busy = False
                    report = update["report"]
                    self.camera_status = report["message"]
                    summary = f"Ready · {report['width']}×{report['height']}" if report["ok"] else "Test failed · See Diagnostics"
                    self.camera_summary.set(f"Camera {report['index']} · {summary}")
                    self.feedback.set(report["message"])
                    self.refresh_device_controls()
                elif event == "ready":
                    self.setup_loading = False
                    self.models_ready = True
                    self.controls(True, False)
                    self.state_text.set("Ready · Camera off")
                    self.profile_count.set(f"{update['profiles']} registered profiles")
                    self.render_attendance(update["attendance"])
                    self.feedback.set("Models ready · Choose a camera, then click Start camera")
                    self.guidance.set("Choose a camera index while stopped. Test camera reads one frame without recording attendance.")
                elif event == "started":
                    self.camera_starting = False
                    self.stream_id = update["stream_id"]
                    self.last_analysis = None
                    self.preview_times.clear()
                    self.analysis_times.clear()
                    self.preview_fps = self.analysis_fps = 0.
                    self.canvas_image = None
                    self.canvas.configure(bg=CANVAS_LIVE)
                    self.controls(True, True)
                    self.camera_status = "Opened successfully"
                    self.camera_summary.set(f"Camera {update.get('camera', self.camera_choice.get())} · Running")
                    self.state_text.set("Camera running")
                    self.render_attendance(update["attendance"])
                elif event in ("stopped", "camera_error"):
                    self.camera_starting = False
                    self.stream_id = None
                    self.last_analysis = None
                    self.canvas_image = None
                    if self.review_dialog is not None:
                        self.review_dialog.finish()
                        self.review_dialog = None
                    if self.appearance_dialog is not None:
                        self.appearance_dialog.finish()
                        self.appearance_dialog = None
                    self.controls(True, False)
                    self.state_text.set("Camera off")
                    self.feedback.set(update.get("message", "Camera stopped"))
                    self.camera_status = update.get("message", "Stopped; device released")
                    self.camera_summary.set(f"Camera {self.camera_choice.get()} · {'Error · See Diagnostics' if event == 'camera_error' else 'Off'}")
                    self.last_frame_details = ""
                    self.guidance.set("Choose another camera or use Test camera. Open Diagnostics for setup and device status.")
                    for value in (self.identity_text, self.live_text, self.expression_text):
                        value.set("—")
                    self.update_quality_chips(None)
                    self.update_telemetry(None)
                    self.speed.set("FPS not measured")
                    self.placeholder()
                elif event == "fatal":
                    self.setup_loading = False
                    self.models_ready = False
                    self.controls(False, False)
                    self.state_text.set("Not ready")
                    component = update.get("component", "Runtime")
                    self.feedback.set(f"{component} setup failed · Open Diagnostics for details")
                    self.guidance.set(COMPONENTS.get(component, ("", "Check the project environment."))[1])
                    self.feedback_label.configure(wraplength=680)
                elif event == "notice":
                    from tkinter import messagebox
                    messagebox.showinfo("Capture", update["message"], parent=self.root)
                elif event == "profiles_data":
                    if self.profiles_opening and self.profiles_dialog is None:
                        from .profiles_dialog import ProfilesDialog
                        self.profiles_dialog = ProfilesDialog(self.root, update["profiles"], self.send, self.profiles_closed,
                                                              photo_import_available=update["photo_import_available"])
                    elif self.profiles_dialog is not None:
                        self.profiles_dialog.update_profiles(update["profiles"], update.get("selected_id"), update.get("message"))
                    self.profiles_opening = False
                    self.profile_count.set(f"{len(update['profiles'])} registered profiles")
                    self.render_attendance(update["attendance"])
                    self.profiles_button.configure(state="normal" if not self.running else "disabled")
                elif event == "profiles_error":
                    self.profiles_opening = False
                    if self.profiles_dialog is not None:
                        self.profiles_dialog.error(update["message"], update.get("operation_id"))
                    else:
                        self.feedback.set(update["message"])
                        self.profiles_closed()
                elif event == "profile_import_checked":
                    if self.profiles_dialog is not None and self.profiles_dialog.import_dialog is not None:
                        self.profiles_dialog.import_dialog.checked(update["report"])
                elif event == "profile_import_done":
                    if self.profiles_dialog is not None and self.profiles_dialog.import_dialog is not None:
                        dialog = self.profiles_dialog.import_dialog
                        if dialog.operation_id == update["operation_id"]:
                            dialog.close()
                elif event == "appearance_open":
                    if self.appearance_dialog is None and self.review_dialog is None:
                        from .appearance_dialog import AppearanceDialog
                        self.appearance_dialog = AppearanceDialog(self.root, update["request"],
                            self.begin_appearance_capture, self.cancel_appearance_dialog)
                    else:
                        self.send("appearance_cancel", update["request"]["id"])
                elif event == "appearance_started":
                    self.feedback.set("Capturing an additional appearance · Keep the same person in view")
                elif event == "review_open":
                    from .review_dialog import ReviewDialog
                    self.review_dialog = ReviewDialog(self.root, update["request"],
                        lambda fields: self.send("review_confirm", fields),
                        lambda token: self.cancel_review_dialog(token))
                elif event == "review_done":
                    if self.review_dialog is not None:
                        self.review_dialog.finish()
                        self.review_dialog = None
                    self.render_attendance(update["attendance"])
                    self.state_text.set("Manual attendance recorded" if update["result"]["recorded"] else "Attendance already recorded for this session")
                elif event == "review_error":
                    if self.review_dialog is not None:
                        self.review_dialog.show_error(update["message"])
        except queue.Empty:
            pass
        try:
            self.render_frame(self.analyses.get_nowait())
        except queue.Empty:
            pass
        try:
            self.render_preview(self.previews.get_nowait())
        except queue.Empty:
            pass
        self.expire_analysis()
        self.refresh_diagnostics()
        self.root.after(15, self.poll)

    def close(self):
        self.closing = True
        if self.model_test_dialog is not None:
            self.model_test_dialog.close()
        if self.diagnostics_dialog is not None:
            self.diagnostics_dialog.close()
        if self.profiles_dialog is not None:
            self.profiles_dialog.close()
        if self.history_dialog is not None:
            self.history_dialog.close()
        if self.review_dialog is not None:
            self.review_dialog.finish()
            self.review_dialog = None
        if self.appearance_dialog is not None:
            self.appearance_dialog.finish()
            self.appearance_dialog = None
        self.send("quit")
        self.state_text.set("Closing camera…")
        self._close_deadline = time.monotonic() + 1.2  # a stuck camera driver must not hang the window
        self.await_close()

    def cancel_review_dialog(self, token):
        self.send("review_cancel", token)
        self.review_dialog = None

    def begin_appearance_capture(self, fields):
        self.send("appearance_begin", fields)
        self.appearance_dialog = None

    def cancel_appearance_dialog(self, request_id):
        self.send("appearance_cancel", request_id)
        self.appearance_dialog = None

    def await_close(self):
        timed_out = hasattr(self, "_close_deadline") and time.monotonic() > self._close_deadline
        if timed_out or ((self.worker is None or not self.worker.is_alive()) and (self.camera_probe is None or not self.camera_probe.is_alive()) and self.model_test_dialog is None):
            for finish in (self.root.quit, self.root.destroy):
                try:
                    finish()
                except Exception:
                    pass
        else:
            self.root.after(80, self.await_close)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-config", type=Path, default=PROJECT/"results/identity_app_config.json")
    parser.add_argument("--liveness-checkpoint", type=Path, default=DEFAULT_LIVENESS_CHECKPOINT)
    parser.add_argument("--emotion-checkpoint", type=Path, default=PROJECT/"results/emotion_fer/best_model.pt")
    parser.add_argument("--database", type=Path, default=PROJECT/"data/attendance/profiles.sqlite3")
    parser.add_argument("--session", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto",
                        help="OpenCV video backend preference (auto uses fast MSMF on Windows).")
    parser.add_argument("--detector", choices=("yunet", "haar", "roi"), default="yunet")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--preview", action="store_true", help="Layout preview; never loads models, database or camera.")
    parser.add_argument("--preview-review", action="store_true", help="Preview the manual-review dialog with fictional data; never writes records.")
    parser.add_argument("--preview-appearance", action="store_true", help="Preview the appearance dialog with fictional data; never loads models, opens a camera or writes records.")
    parser.add_argument("--preview-history", action="store_true", help="Preview history with fictional rows; never loads models, opens a camera or reads the database.")
    parser.add_argument("--preview-profiles", action="store_true", help="Preview profile management with fictional data; no models, database or camera.")
    parser.add_argument("--preview-diagnostics", action="store_true", help="Preview an example setup error; no models, database or camera.")
    parser.add_argument("--preview-model-tests", action="store_true", help="Preview model tests; no files, models, database or camera.")
    args = parser.parse_args(argv)
    if sum((args.preview_review, args.preview_appearance, args.preview_history, args.preview_profiles, args.preview_diagnostics, args.preview_model_tests)) > 1:
        parser.error("Choose one dialog preview at a time.")
    if args.preview_review or args.preview_appearance or args.preview_history or args.preview_profiles or args.preview_diagnostics or args.preview_model_tests:
        args.preview = True
    if args.num_threads < 1 or not 0 <= args.camera <= 99:
        parser.error("Positive threads and a camera index from 0 to 99 required.")
    if args.roi and args.detector != "roi":
        parser.error("--roi requires --detector roi.")
    if args.roi and (min(args.roi[:2]) < 0 or min(args.roi[2:]) < 1):
        parser.error("ROI needs nonnegative x/y and positive width/height.")
    import tkinter as tk
    root = tk.Tk()
    AttendanceWindow(root, args)
    root.mainloop()
    return 0


def force_exit(code=0):
    """End the process even if a camera driver thread is blocked; SQLite writes are already committed."""
    try:
        import sys
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), int(code or 0))
    except Exception:
        pass
    import os
    os._exit(int(code or 0))


if __name__ == "__main__":
    force_exit(main() or 0)
