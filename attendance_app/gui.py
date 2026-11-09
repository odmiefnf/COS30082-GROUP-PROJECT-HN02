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
from .video_stream import LatestCamera, put_latest

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_LIVENESS_CHECKPOINT = PROJECT/"results/liveness_camera_matched/best_model.pt"
BG, CARD, INK, MUTED, BLUE = "#eef3f8", "#ffffff", "#172b45", "#61748b", "#1765d1"
FACE_PENDING, FACE_RECOGNIZED = "#fbbf24", "#4ade80"

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
        self.latest = None
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

    def publish_profiles(self, message=None, selected_id=None):
        self.service.profiles = self.service.store.profiles()
        profiles = [{"id": p["id"], "name": p["name"], "created_utc": p["created_utc"],
                     "sample_count": p["sample_count"], "appearances": [
                         {"id": a["id"], "label": a["label"], "sample_count": a["sample_count"]}
                         for a in p["appearances"]]} for p in self.service.profiles]
        self.publish(event="profiles_data", profiles=profiles, message=message, selected_id=selected_id,
                     photo_import_available=self.args.detector == "haar",
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
            from .service import AttendanceService
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
                if training.get("smoke_test") is not False or training.get("initialization") != "random_from_scratch":
                    raise ValueError(f"{name} requires a completed from-scratch training checkpoint.")
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
            self.service = AttendanceService(store, recognizer, liveness, emotion, self.args.session)
            self.reference_importer = ReferenceImporter(store, recognizer, liveness, locator)
            processor = FrameProcessor(locator, self.service)
            self.publish(event="component", component="Profiles", status="Ready", detail=str(self.args.database))
            self.publish(event="ready", profiles=len(self.service.profiles), attendance=store.attendance(self.service.session))
            while True:
                try:
                    command, value = self.commands.get(timeout=.08 if self.camera is None else .001)
                except queue.Empty:
                    command, value = None, None
                if command == "quit":
                    break
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
                            on_preview=lambda sample, token=stream_id: self.preview(sample, token))
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
                if command in ("profiles_list", "profile_rename", "profile_remove", "profile_import_check", "profile_import_commit"):
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
                    self.latest, self.latest_time = result, sample.captured_at
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
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("Treeview", background=CARD, foreground=INK, fieldbackground=CARD, rowheight=34,
                        borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#edf3fb", foreground=MUTED)
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=18, pady=12)
        header = tk.Frame(outer, bg=BG)
        header.pack(fill="x")
        self.label(header, "FACE ATTENDANCE", 19, True, BG).pack(side="left")
        self.label(header, "COS30082  /  LOCAL DEMO", 10, True, BG, MUTED).pack(side="right", pady=4)
        self.label(outer, "Identity recognition · Liveness check · Facial expression display", 11, bg=BG, fg=MUTED).pack(anchor="w", pady=(2, 8))
        bar = tk.Frame(outer, bg=CARD, padx=12, pady=8)
        bar.pack(fill="x", pady=(0, 10))
        control_card = bar
        bar = tk.Frame(control_card, bg=CARD)
        bar.pack(fill="x")
        self.label(bar, "Session", 10, True).pack(side="left", padx=(0, 12))
        self.session = tk.StringVar(value=args.session)
        self.session_entry = tk.Entry(bar, textvariable=self.session, font=("Segoe UI", 11), width=24,
                                      relief="solid", bd=1, bg="#f8fafc", fg=INK)
        self.session_entry.pack(side="left", ipady=6)
        self.start_button = self.button(bar, "Start camera", self.start, BLUE)
        self.start_button.pack(side="left", padx=(16, 8))
        self.stop_button = self.button(bar, "Stop", lambda: self.send("stop"), "#e7edf5", INK)
        self.stop_button.pack(side="left")
        self.history_button = self.button(bar, "History / CSV", self.open_history, "#e7edf5", INK)
        self.history_button.pack(side="left", padx=(8, 0))
        for button in (self.start_button, self.stop_button, self.history_button):
            button.configure(pady=6)
        self.state_text = tk.StringVar(value="Preview · Camera off" if args.preview else "Loading three models…")
        self.label(bar, variable=self.state_text, size=10, fg=MUTED).pack(side="right", padx=5)
        device_bar = tk.Frame(control_card, bg=CARD)
        device_bar.pack(fill="x", pady=(6, 0))
        self.label(device_bar, "Camera index", 10, True).pack(side="left", padx=(0, 10))
        self.camera_choice = tk.StringVar(value=str(args.camera))
        indices = sorted(set(range(10)) | {args.camera})
        self.camera_selector = ttk.Combobox(device_bar, textvariable=self.camera_choice,
            values=[str(index) for index in indices], width=5, state="readonly", font=("Segoe UI", 11))
        self.camera_selector.pack(side="left")
        self.camera_selector.bind("<<ComboboxSelected>>", self.camera_changed)
        self.test_camera_button = ttk.Button(device_bar, text="Test camera", command=self.test_camera)
        self.test_camera_button.pack(side="left", padx=10)
        ttk.Button(device_bar, text="Diagnostics…", command=self.open_diagnostics).pack(side="left")
        self.camera_summary = tk.StringVar(value=f"Camera {args.camera} · Not tested")
        self.label(device_bar, variable=self.camera_summary, size=10, fg=MUTED).pack(side="left", padx=14)
        self.label(outer, "Stored locally · Camera images are not saved · Expressions are display only · Educational prototype", 9, bg=BG, fg=MUTED).pack(side="bottom", anchor="w", pady=(8, 0))
        content = tk.Frame(outer, bg=BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=7)
        content.columnconfigure(1, weight=3, minsize=310)
        content.rowconfigure(0, weight=1)
        # A preferred column size is independent of live text. Grid still
        # allocates the actual size when the user resizes the window.
        left = tk.Frame(content, bg=BG, width=720)
        left.pack_propagate(False)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        camera_card = tk.Frame(left, bg=CARD, padx=12, pady=12)
        camera_card.columnconfigure(0, weight=1)
        camera_card.rowconfigure(1, weight=1)
        camera_top = tk.Frame(camera_card, bg=CARD)
        camera_top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.label(camera_top, "LIVE CAMERA", 10, True).pack(side="left")
        self.speed = tk.StringVar(value="FPS not measured")
        speed_label = self.label(camera_top, variable=self.speed, size=9, fg=MUTED, width=42)
        speed_label.configure(anchor="e")
        speed_label.pack(side="right")
        self.canvas = tk.Canvas(camera_card, bg="#102039", highlightthickness=0, height=1)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.feedback = tk.StringVar(value="Click Start camera when you are ready")
        self.feedback_label = self.label(camera_card, variable=self.feedback, size=11, bold=True,
                                         width=1, height=2, wraplength=680)
        self.feedback_label.configure(anchor="nw")
        self.feedback_label.grid(row=2, column=0, sticky="ew", pady=(12, 3))
        self.guidance = tk.StringVar(value="Choose a camera index while stopped. Test camera reads one frame without recording attendance.")
        self.guidance_label = self.label(camera_card, variable=self.guidance, size=10, fg=MUTED,
                                         width=1, height=2, wraplength=650)
        self.guidance_label.configure(anchor="nw")
        self.guidance_label.grid(row=3, column=0, sticky="ew", pady=(2, 0))
        self.feedback_label.bind("<Configure>", lambda event: self.feedback_label.configure(wraplength=max(100, event.width-8)))
        self.guidance_label.bind("<Configure>", lambda event: self.guidance_label.configure(wraplength=max(100, event.width-8)))
        stats = tk.Frame(left, bg=BG)
        stats.pack(side="bottom", fill="x", pady=(10, 0))
        self.identity_text, self.live_text, self.expression_text = (tk.StringVar(value="—") for _ in range(3))
        for index, (title, value) in enumerate((("IDENTITY", self.identity_text), ("LIVENESS", self.live_text), ("EXPRESSION", self.expression_text))):
            stats.columnconfigure(index, weight=1, uniform="stats")
            card = tk.Frame(stats, bg=CARD, padx=13, pady=12)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 9, 0))
            self.label(card, title + (" · details" if index == 0 else ""), 9, True, fg=MUTED).pack(anchor="w")
            value_label = self.label(card, variable=value, size=12, bold=True,
                                     width=1, height=2, wraplength=200)
            value_label.configure(anchor="nw")
            value_label.pack(fill="x", pady=(5, 0))
            value_label.bind("<Configure>", lambda event: event.widget.configure(
                wraplength=max(100, event.width-8)))
            if index == 0:
                # The fixed slot must not grow for an 80-character profile.
                # Keep the complete value available by mouse and keyboard.
                value_label.configure(cursor="hand2", takefocus=True,
                                      highlightthickness=1, highlightbackground=CARD, highlightcolor=BLUE)
                value_label.bind("<Button-1>", self.show_identity_details)
                value_label.bind("<Return>", self.show_identity_details)
        camera_card.pack(fill="both", expand=True)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1, minsize=140)
        register = tk.Frame(right, bg=CARD, padx=18, pady=12)
        register.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.label(register, "People & registration", 15, True).pack(anchor="w")
        self.label(register, "New person: register with name / student ID. Changed look: Add appearance to the existing profile.", 10, fg=MUTED, wraplength=300).pack(anchor="w", pady=(5, 8))
        self.name = tk.StringVar()
        tk.Entry(register, textvariable=self.name, font=("Segoe UI", 12), relief="solid", bd=1,
                 bg="#f8fafc", fg=INK).pack(fill="x", ipady=4)
        register_actions = tk.Frame(register, bg=CARD)
        register_actions.pack(fill="x", pady=(8, 0))
        self.register_button = self.button(register_actions, "Register from camera", lambda: self.send("register", self.name.get()), BLUE)
        self.register_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        cancel_button = self.button(register_actions, "Cancel capture", lambda: self.send("cancel"), "#e7edf5", INK)
        cancel_button.configure(pady=6, padx=10)
        cancel_button.pack(side="right")
        existing_actions = tk.Frame(register, bg=CARD)
        existing_actions.pack(fill="x", pady=(6, 0))
        existing_actions.columnconfigure((0, 1), weight=1)
        self.appearance_button = self.button(existing_actions, "Add appearance", lambda: self.send("appearance_request"), "#e7edf5", INK)
        self.appearance_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.review_button = self.button(existing_actions, "Manual review", lambda: self.send("review"), "#e7edf5", INK)
        self.review_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.profile_count = tk.StringVar(value="0 registered profiles")
        profile_bar = tk.Frame(register, bg=CARD)
        profile_bar.pack(fill="x", pady=(8, 0))
        self.label(profile_bar, variable=self.profile_count, size=10, fg=MUTED).pack(side="left")
        self.profiles_button = self.button(profile_bar, "Profiles…", self.open_profiles, "#e7edf5", INK)
        self.profiles_button.pack(side="right")
        for button in (self.register_button, self.appearance_button, self.review_button, self.profiles_button):
            button.configure(pady=6, padx=10)
        attendance = tk.Frame(right, bg=CARD, padx=16, pady=16)
        attendance.grid(row=1, column=0, sticky="nsew")
        self.label(attendance, "Present this session", 15, True).pack(anchor="w", pady=(0, 10))
        self.tree = ttk.Treeview(attendance, columns=("name", "time", "source"), show="headings", height=3)
        self.tree.heading("name", text="Name / student ID")
        self.tree.heading("time", text="Time")
        self.tree.heading("source", text="Source")
        self.tree.column("name", width=185, stretch=True)
        self.tree.column("time", width=62, stretch=False, anchor="center")
        self.tree.column("source", width=90, stretch=False, anchor="center")
        self.tree.bind("<Double-1>", self.show_attendance_detail)
        self.label(attendance, "One record per person per session.", 9, fg=MUTED).pack(side="bottom", anchor="w", pady=(9, 0))
        self.tree.pack(fill="both", expand=True)
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
        if not args.preview:
            self.worker = CameraWorker(args, self.commands, self.updates, self.previews, self.analyses)
            self.worker.start()
        root.after(60, self.poll)

    def label(self, parent, text=None, size=11, bold=False, bg=CARD, fg=INK, variable=None, **kwargs):
        return self.tk.Label(parent, text=text, textvariable=variable, bg=bg, fg=fg, anchor="w", justify="left",
                             font=("Segoe UI", size, "bold" if bold else "normal"), **kwargs)

    def camera_changed(self, event=None):
        self.camera_status = "Not tested"
        self.camera_summary.set(f"Camera {self.camera_choice.get()} · Not tested")
        self.last_frame_details = ""
        self.refresh_diagnostics()

    def device_locked(self):
        return self.running or self.camera_starting or self.probe_busy or self.closing

    def refresh_device_controls(self):
        locked = self.device_locked()
        profiles_busy = self.profiles_dialog is not None or self.profiles_opening
        self.camera_selector.configure(state="disabled" if locked or profiles_busy else "readonly")
        self.test_camera_button.configure(state="normal" if not locked and not profiles_busy and not self.args.preview else "disabled")
        self.start_button.configure(state="normal" if self.models_ready and not locked and not profiles_busy else "disabled")
        self.profiles_button.configure(state="normal" if (self.models_ready or self.args.preview) and not locked else "disabled")
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
            report = probe_camera(index)
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
        crop = Image.new("RGB", (180, 130), "#e7edf5")
        draw = ImageDraw.Draw(crop)
        draw.text((38, 55), "PREVIEW ONLY", fill="#61748b")
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

    def open_profiles(self):
        if self.profiles_dialog is not None:
            self.profiles_dialog.window.lift()
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

    def button(self, parent, text, action, background, foreground="white"):
        return self.tk.Button(parent, text=text, command=action, bg=background, fg=foreground,
                              activebackground=background, activeforeground=foreground, bd=0,
                              font=("Segoe UI", 11, "bold"), padx=17, pady=9, cursor="hand2")

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
            self.canvas.delete("all")
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            cx, cy = width / 2, height / 2 - 40
            scale = max(.25, min(1., (height-90)/200))
            self.canvas.create_rectangle(cx-78*scale, cy-95*scale, cx+78*scale, cy+80*scale, outline="#41658a", width=2)
            self.canvas.create_oval(cx-30*scale, cy-60*scale, cx+30*scale, cy, outline="#88a7c8", width=3)
            self.canvas.create_arc(cx-55*scale, cy, cx+55*scale, cy+90*scale, start=0, extent=180, style="arc", outline="#88a7c8", width=3)
            self.canvas.create_text(cx, height-58, text="CAMERA OFF", fill="#d7e6f7", font=("Segoe UI", 15, "bold"))
            self.canvas.create_text(cx, height-28, text="The camera preview will appear here", fill="#88a7c8", font=("Segoe UI", 11))

    def render_attendance(self, rows):
        if rows == self.attendance_rows:
            return
        self.attendance_rows = rows
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(rows):
            local = datetime.fromisoformat(row["recorded_utc"]).astimezone().strftime("%H:%M")
            source = {"automatic": "Automatic", "manual_review": "Manual"}.get(row.get("source"), "Unknown")
            self.tree.insert("", "end", iid=str(index), values=(row["name"], local, source))

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
        self.expression_text.set(("Unavailable" if result.get("expression_error") else "—") if expression is None
                                 else f"{expression['label'].title()} · {expression['confidence']:.0%}")
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
        if (self.worker is None or not self.worker.is_alive()) and (self.camera_probe is None or not self.camera_probe.is_alive()):
            self.root.destroy()
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
    parser.add_argument("--detector", choices=("haar", "roi"), default="haar")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--preview", action="store_true", help="Layout preview; never loads models, database or camera.")
    parser.add_argument("--preview-review", action="store_true", help="Preview the manual-review dialog with fictional data; never writes records.")
    parser.add_argument("--preview-appearance", action="store_true", help="Preview the appearance dialog with fictional data; never loads models, opens a camera or writes records.")
    parser.add_argument("--preview-history", action="store_true", help="Preview history with fictional rows; never loads models, opens a camera or reads the database.")
    parser.add_argument("--preview-profiles", action="store_true", help="Preview profile management with fictional data; no models, database or camera.")
    parser.add_argument("--preview-diagnostics", action="store_true", help="Preview an example setup error; no models, database or camera.")
    args = parser.parse_args(argv)
    if sum((args.preview_review, args.preview_appearance, args.preview_history, args.preview_profiles, args.preview_diagnostics)) > 1:
        parser.error("Choose one dialog preview at a time.")
    if args.preview_review or args.preview_appearance or args.preview_history or args.preview_profiles or args.preview_diagnostics:
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


if __name__ == "__main__":
    raise SystemExit(main())
