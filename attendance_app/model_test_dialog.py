"""Model comparison UI; the attendance camera stays stopped while this is open."""
from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .history import local_timestamp
from .model_testing import ComparisonRun, SCENARIOS, export_report, load_report, report_rows


class ModelTestDialog:
    def __init__(self, parent, project, baseline, camera, on_close, *, preview=False):
        self.project, self.baseline, self.camera = Path(project), baseline, camera
        self.on_close, self.preview = on_close, preview
        self.job, self.report, self.report_path = None, None, None
        self.closed, self.closing = False, False
        self.window = tk.Toplevel(parent)
        self.window.title("Model tests · PREVIEW ONLY" if preview else "Model tests · COS30082")
        width, height = min(1100, parent.winfo_screenwidth()-80), min(740, parent.winfo_screenheight()-90)
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(850, width), min(590, height))
        self.window.configure(bg="#eef3f8")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", lambda event: self.close())
        body = tk.Frame(self.window, bg="#eef3f8", padx=20, pady=16)
        body.pack(fill="both", expand=True)
        self.label(body, "Model tests", 20, True).pack(anchor="w")
        self.label(body, "Compare two liveness models on the same camera crops. No attendance is recorded.").pack(anchor="w", pady=(3, 12))
        form = tk.Frame(body, bg="white", padx=12, pady=10)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        self.label(form, "App baseline", bold=True, bg="white").grid(row=0, column=0, sticky="w", padx=(0, 12))
        self.baseline_text = tk.StringVar(value=str(baseline))
        ttk.Entry(form, textvariable=self.baseline_text, state="readonly").grid(row=0, column=1, columnspan=2, sticky="ew", pady=4)
        self.label(form, "Candidate", bold=True, bg="white").grid(row=1, column=0, sticky="w")
        self.candidate = tk.StringVar()
        self.candidate_entry = ttk.Entry(form, textvariable=self.candidate)
        self.candidate_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self.browse_button = ttk.Button(form, text="Browse…", command=self.browse)
        self.browse_button.grid(row=1, column=2, padx=(8, 0))
        self.label(form, "Scenario", bold=True, bg="white").grid(row=2, column=0, sticky="w")
        options = tk.Frame(form, bg="white")
        options.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        options.columnconfigure(0, weight=1)
        self.scenario = tk.StringVar(value=next(iter(SCENARIOS)))
        self.scenario_box = ttk.Combobox(options, textvariable=self.scenario, values=list(SCENARIOS), state="readonly", width=32)
        self.scenario_box.grid(row=0, column=0, sticky="ew")
        self.label(options, "Seconds", bg="white").grid(row=0, column=1, padx=(15, 6))
        self.seconds = tk.StringVar(value="15")
        self.duration_box = ttk.Spinbox(options, from_=1, to=120, textvariable=self.seconds, width=6)
        self.duration_box.grid(row=0, column=2)
        self.label(options, f"Camera {camera}", bg="white").grid(row=0, column=3, padx=(14, 0))
        self.ready = tk.BooleanVar(value=False)
        self.ready_box = ttk.Checkbutton(form, text="Participant agrees, is ready, and the selected scenario matches this presentation.",
                                         variable=self.ready)
        self.ready_box.grid(row=3, column=0, columnspan=3, sticky="w", pady=(9, 2))
        actions = tk.Frame(body, bg="#eef3f8")
        actions.pack(fill="x", pady=(12, 8))
        self.start_button = ttk.Button(actions, text="Start comparison", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Cancel test", command=self.cancel)
        self.cancel_button.pack(side="left", padx=8)
        self.load_button = ttk.Button(actions, text="Open report…", command=self.open_report)
        self.load_button.pack(side="left")
        ttk.Button(actions, text="Close", command=self.close).pack(side="right")
        self.progress = ttk.Progressbar(body, mode="indeterminate")
        self.progress.pack(fill="x")
        self.status = tk.StringVar(value="PREVIEW ONLY · No files, models or camera are accessed." if preview
                                 else "Select a candidate and confirm readiness. Framing lasts 10 seconds before measurement.")
        self.status_label = self.label(body, variable=self.status)
        self.status_label.pack(fill="x", pady=(7, 10))
        self.summary = tk.StringVar(value="No trial loaded")
        self.label(body, variable=self.summary, bold=True).pack(fill="x", pady=(0, 7))
        table = tk.Frame(body)
        columns = ("model", "threshold", "scored", "accepted", "rejected", "correct", "wrong", "rate")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=3)
        for key, text in zip(columns, ("Model", "Threshold", "Scored crops", "Accepted", "Rejected", "Correct*", "Wrong*", "Accepted %")):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=105, minwidth=85, anchor="center", stretch=True)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=horizontal.set)
        self.tree.pack(fill="both", expand=True)
        horizontal.pack(fill="x")
        table.pack(fill="both", expand=True)
        self.details = tk.StringVar(value="Both models keep their validation-calibrated thresholds.")
        self.label(body, variable=self.details).pack(fill="x", pady=(8, 3))
        self.label(body, "* Correct/Wrong use the operator's scenario. No-face and low-quality frames are excluded, not PAD rejections. "
                        "Frames are correlated; this is not population accuracy or a dataset ACER/AUC benchmark.").pack(fill="x", pady=(2, 9))
        exports = tk.Frame(body, bg="#eef3f8")
        exports.pack(fill="x")
        self.json_button = ttk.Button(exports, text="Export JSON…", command=lambda: self.export(".json"))
        self.json_button.pack(side="left")
        self.csv_button = ttk.Button(exports, text="Export CSV…", command=lambda: self.export(".csv"))
        self.csv_button.pack(side="left", padx=8)
        self.candidate.trace_add("write", lambda *_: self.controls())
        self.ready.trace_add("write", lambda *_: self.controls())
        self.controls()

    def label(self, parent, text=None, size=10, bold=False, bg="#eef3f8", variable=None):
        label = tk.Label(parent, text=text, textvariable=variable, bg=bg, fg="#172b45", anchor="w",
                         justify="left", wraplength=920, font=("Segoe UI", size, "bold" if bold else "normal"))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width-4)))
        return label

    def controls(self):
        busy = self.job is not None
        editable = not busy and not self.preview and not self.closing
        self.start_button.configure(state="normal" if editable and self.ready.get() and self.candidate.get().strip() else "disabled")
        self.cancel_button.configure(state="normal" if busy and not self.job.cancel_requested else "disabled")
        for button in (self.browse_button, self.load_button):
            button.configure(state="normal" if editable else "disabled")
        for button in (self.json_button, self.csv_button):
            button.configure(state="normal" if editable and self.report is not None else "disabled")
        self.candidate_entry.configure(state="normal" if editable else "disabled")
        self.scenario_box.configure(state="readonly" if editable else "disabled")
        self.duration_box.configure(state="normal" if editable else "disabled")
        self.ready_box.configure(state="normal" if editable else "disabled")

    def browse(self):
        path = filedialog.askopenfilename(parent=self.window, title="Select a trained liveness checkpoint",
            initialdir=str(self.project/"results"), filetypes=[("PyTorch checkpoints", "*.pt *.pth")])
        if path:
            self.candidate.set(path)

    def start(self):
        if self.preview or self.job is not None or not self.ready.get():
            return
        try:
            job = ComparisonRun(self.project, self.baseline, self.candidate.get().strip(),
                                self.scenario.get(), self.camera, float(self.seconds.get()))
            job.start()
        except (OSError, ValueError, RuntimeError) as exc:
            self.status.set(f"Could not start: {exc}")
            return
        self.job, self.report, self.report_path = job, None, None
        self.tree.delete(*self.tree.get_children())
        self.summary.set("Comparison running")
        self.details.set("Thresholds stay fixed. No images, profiles or attendance are saved by the test.")
        self.status.set("Loading models, then 10 seconds of framing and the measured trial. The preview window also accepts Q / Esc to cancel.")
        self.progress.start(12)
        self.controls()
        self.window.after(150, self.poll)

    def cancel(self):
        if self.job is not None:
            try:
                self.job.cancel()
            except OSError as exc:
                self.status.set(f"Could not request cancellation: {exc}. Use Q / Esc in the camera preview.")
            else:
                self.status.set("Cancelling · Waiting for the camera tool to release the device…")
            self.controls()

    def poll(self):
        try:
            result = self.job.poll()
        except (OSError, ValueError, RuntimeError) as exc:
            result = (None, f"Comparison failed: {exc}")
        if result is None:
            self.window.after(150, self.poll)
            return
        report, message = result
        self.report_path = self.job.output
        self.job = None
        self.progress.stop()
        self.ready.set(False)
        if report is not None:
            self.show_report(report)
        else:
            self.summary.set("No completed trial report")
        self.status.set(message)
        self.controls()
        if self.closing:
            self.close()

    def show_report(self, report):
        rows = report_rows(report)
        self.report = report
        self.tree.delete(*self.tree.get_children())
        for row in rows:
            self.tree.insert("", "end", values=(row["model"].title(), f"{row['threshold']:.6f}", row["scored_crops"],
                row["passed"], row["not_passed"], row["correct"] if row["correct"] is not None else "—",
                row["incorrect"] if row["incorrect"] is not None else "—",
                f"{row['acceptance_rate']:.1%}" if row["acceptance_rate"] is not None else "—"))
        statuses = report["frame_statuses"]
        one_face = sum(statuses.get(key, 0) for key in ("scored", "image_quality", "face_too_small_or_outside"))
        declared = (report["condition_supplied_by_operator"], report.get("presentation_type_supplied_by_operator"))
        scenario = next((label for label, pair in SCENARIOS.items() if pair == declared), declared[0])
        cancellation = report.get("operator_cancelled")
        prefix = "CANCELLED · Partial trial · " if cancellation is True else "Cancellation not recorded · " if cancellation is None else ""
        self.summary.set(f"Report · {prefix}{scenario} · "
                         f"Captured {report['frame_count']} · One face {one_face} · Scored {rows[0]['scored_crops']}\n"
                         f"Started {local_timestamp(report['started_utc'])}")
        self.details.set("\n".join(f"{row['model'].title()}: {Path(row['checkpoint']).name} · {row['crop_policy']} · SHA256 {row['sha256']}" for row in rows))
        self.controls()

    def open_report(self):
        path = filedialog.askopenfilename(parent=self.window, title="Open a paired PAD comparison report",
            initialdir=str(self.project/"results"), filetypes=[("JSON reports", "*.json")])
        if path:
            try:
                report = load_report(path)
                self.show_report(report)
            except (OSError, ValueError) as exc:
                self.status.set(f"Could not open report: {exc}")
            else:
                self.report_path = Path(path)
                self.status.set(f"Saved report loaded: {path}")

    def export(self, suffix):
        if self.preview or self.report is None:
            return
        path = filedialog.asksaveasfilename(parent=self.window, title="Export this trial to a new file",
            initialfile="model_comparison"+suffix, defaultextension=suffix,
            filetypes=[(suffix[1:].upper()+" files", "*"+suffix)], confirmoverwrite=True)
        if path:
            try:
                export_report(path, self.report)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Export failed", str(exc)+"\nChoose a new filename; existing evidence is protected.", parent=self.window)
            else:
                self.status.set(f"Exported this trial to {path}")

    def close(self):
        if self.closed:
            return
        if self.job is not None:
            self.closing = True
            self.cancel()
            return
        self.closed = True
        self.window.destroy()
        self.on_close()
