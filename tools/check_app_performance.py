"""Measure the actual visible Tk app against a disposable gallery copy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import uuid

from attendance_app.evaluation import gallery_snapshot
from tools.profile_camera_performance import timing_summary


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=12.)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--database", type=Path, default=ROOT/"data/attendance/profiles.sqlite3")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.seconds) or not 0 < args.seconds <= 60
            or not 0 <= args.camera <= 99 or not 1 <= args.num_threads <= 64):
        parser.error("Use duration (0,60], camera 0–99 and 1–64 CPU threads.")
    output, source = args.output.resolve(), args.database.resolve()
    if output.suffix.lower() != ".json" or output.exists():
        parser.error("Choose a new JSON output file; existing evidence is not overwritten.")
    report = {"schema_version": 1, "status": "not_completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual visible AttendanceWindow, CameraWorker, LatestCamera and Tk rendering; performance only",
        "device_requested": args.device, "camera_requested": args.camera,
        "duration_requested_seconds": args.seconds, "production_database_read_only": True,
        "images_or_video_saved": False, "camera_embeddings_in_report": False,
        "profiles_or_attendance_written_to_production": False, "thresholds_retuned": False,
        "automatic_actions": ["Start camera after model setup", "Close after requested measurement duration"],
        "registration_or_profile_actions_invoked": False}
    timings = defaultdict(list)
    previews, captures, processed, displayed = [], [], [], []
    statuses, control_events, capture_sizes = Counter(), Counter(), Counter()
    errors = []
    root, app = None, None
    first_preview, measurement_end = None, None
    auto_closed, start_requested = False, False
    session = "gui-performance-"+uuid.uuid4().hex
    stage = "Read-only gallery metadata"

    def stop_owned_workers():
        if app is not None and app.worker is not None and app.worker.is_alive():
            app.commands.put(("quit", None))
            app.worker.join()
        if app is not None and app.camera_probe is not None and app.camera_probe.is_alive():
            app.camera_probe.join()

    try:
        if not source.is_file():
            raise ValueError("The existing gallery database was not found.")
        report["production_database_sha256_before"] = file_sha256(source)
        with closing(sqlite3.connect(source.as_uri()+"?mode=ro", uri=True)) as connection:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        with ExitStack() as lifecycle:
            snapshot = lifecycle.enter_context(gallery_snapshot(source, metadata["encoder_sha256"], int(metadata["dimension"]), ROOT/"tmp"))
            # ExitStack invokes this before closing/deleting the earlier-entered
            # snapshot, including exceptions that happen before mainloop starts.
            lifecycle.callback(stop_owned_workers)
            import tkinter as tk
            from attendance_app.gui import AttendanceWindow, DEFAULT_LIVENESS_CHECKPOINT
            app_args = SimpleNamespace(identity_config=ROOT/"results/identity_app_config.json",
                liveness_checkpoint=DEFAULT_LIVENESS_CHECKPOINT,
                emotion_checkpoint=ROOT/"results/emotion_fer/best_model.pt", database=snapshot.path,
                session=session, camera=args.camera, detector="haar", roi=None,
                device=args.device, num_threads=args.num_threads, preview=False,
                preview_review=False, preview_appearance=False, preview_history=False,
                preview_profiles=False, preview_diagnostics=False)
            stage = "Visible app / model setup"
            root = tk.Tk()
            app = AttendanceWindow(root, app_args)
            root.title("Performance check · Face Attendance")
            created = time.monotonic()
            original_render_preview, original_render_frame = app.render_preview, app.render_frame
            original_publish, original_preview = app.worker.publish, app.worker.preview

            def render_preview(update):
                nonlocal first_preview
                started = time.monotonic()
                previous_photo = app.photo
                original_render_preview(update)
                finished = time.monotonic()
                if app.photo is not previous_photo:
                    sample = update["sample"]
                    previews.append((sample.sequence, finished))
                    timings["tk_preview_render"].append(finished-started)
                    timings["capture_to_preview_completion"].append(finished-sample.captured_at)
                    if first_preview is None:
                        first_preview = finished
                        print("GUI_CAPTURE_STARTED: actual visible app; scratch gallery only.", flush=True)

            def render_frame(update):
                started = time.monotonic()
                original_render_frame(update)
                finished = time.monotonic()
                if app.last_analysis is update:
                    displayed.append((update["sequence"], finished))
                    timings["tk_analysis_render"].append(finished-started)
                    timings["capture_to_analysis_display"].append(finished-update["captured_at"])

            def publish(**update):
                if update.get("event") == "frame":
                    now = time.monotonic()
                    processed.append((update["sequence"], now))
                    timings["worker_processing"].append(update["processing_seconds"])
                    timings["capture_to_analysis_publication"].append(now-update["captured_at"])
                    statuses[update["result"]["status"]] += 1
                else:
                    event = update.get("event", "unknown")
                    control_events[event] += 1
                    if event in ("camera_error", "fatal"):
                        errors.append({"event": event, "message": update.get("message", "")})
                return original_publish(**update)

            def preview(sample, stream_id):
                captures.append((sample.sequence, sample.captured_at))
                capture_sizes[(sample.rgb.shape[1], sample.rgb.shape[0])] += 1
                return original_preview(sample, stream_id)

            app.render_preview, app.render_frame = render_preview, render_frame
            app.worker.publish, app.worker.preview = publish, preview

            def callback_exception(exception_type, value, traceback):
                errors.append({"event": "tk_callback_error", "message": f"{exception_type.__name__}: {value}"})
                if not app.closing:
                    app.close()

            root.report_callback_exception = callback_exception

            def tick():
                nonlocal auto_closed, measurement_end, start_requested, stage
                if app.closing:
                    return
                now = time.monotonic()
                if errors:
                    measurement_end = now
                    app.close()
                    return
                if app.models_ready and not start_requested:
                    start_requested = True
                    stage = "Camera and actual Tk rendering"
                    app.start()
                if first_preview is not None and now-first_preview >= args.seconds:
                    auto_closed, measurement_end = True, now
                    app.close()
                    return
                if first_preview is None and now-created > 120:
                    errors.append({"event": "setup_timeout", "message": "No preview arrived within 120 seconds."})
                    measurement_end = now
                    app.close()
                    return
                root.after(30, tick)

            root.after(30, tick)
            try:
                root.mainloop()
            finally:
                # Native close may end the mainloop early; finish camera/SQLite
                # ownership before gallery_snapshot is allowed to delete scratch.
                stop_owned_workers()
                measurement_end = measurement_end or time.monotonic()
            report["worker_stopped_before_scratch_cleanup"] = app.worker is None or not app.worker.is_alive()
            report["simulated_attendance_records"] = len(snapshot.attendance(session))
            report["control_events"] = dict(control_events)
            report["frame_statuses"] = dict(statuses)
            report["frame_sizes"] = [{"width": w, "height": h, "frames": count}
                                     for (w, h), count in capture_sizes.items()]
            report["ended_automatically_after_duration"] = auto_closed
            report["model_components"] = app.component_status
            if first_preview is not None:
                duration = measurement_end-first_preview
                in_window = lambda values: [row for row in values if first_preview <= row[1] <= measurement_end]
                measured_previews = in_window(previews)
                measured_captures = in_window(captures)
                measured_processed = in_window(processed)
                measured_displayed = in_window(displayed)
                for before, after in zip(measured_previews, measured_previews[1:]):
                    timings["actual_preview_interval"].append(after[1]-before[1])
                report.update(duration_seconds=duration, startup_to_first_preview_seconds=first_preview-created,
                    preview_render_count=len(measured_previews), preview_fps=len(measured_previews)/duration,
                    captured_frames=len(measured_captures), capture_fps=len(measured_captures)/duration,
                    worker_analysis_count=len(measured_processed), worker_analysis_fps=len(measured_processed)/duration,
                    displayed_analysis_count=len(measured_displayed), displayed_analysis_fps=len(measured_displayed)/duration,
                    analysis_duplicate_sequences=len(measured_processed)-len({row[0] for row in measured_processed}),
                    preview_duplicate_sequences=len(measured_previews)-len({row[0] for row in measured_previews}))
                if auto_closed and measured_previews and measured_processed and not errors:
                    report["status"] = "completed"
            report["timings"] = {name: timing_summary(values) for name, values in timings.items()}
            report["timing_note"] = "Actual Tk render calls are measured. Nested timings overlap; capture/read/GPU setup and first inference may be cold. No identity/PAD accuracy claim."
    except Exception as exc:
        errors.append({"event": "harness_error", "stage": stage, "message": str(exc)})
    finally:
        stop_owned_workers()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        if source.is_file():
            report["production_database_sha256_after"] = file_sha256(source)
            report["production_database_file_unchanged"] = (report.get("production_database_sha256_before")
                                                          == report["production_database_sha256_after"])
    if errors:
        report["status"], report["errors"] = "not_completed", errors
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "preview_fps": report.get("preview_fps"),
                      "worker_analysis_fps": report.get("worker_analysis_fps"), "errors": errors}, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
