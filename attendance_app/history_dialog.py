"""Attendance history viewer; browsing never changes the active camera/session."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .history import (SOURCE_LABELS, export_attendance_csv, filter_attendance, local_timestamp,
                      read_attendance_snapshot, summarize_attendance)


class HistoryDialog:
    def __init__(self, parent, database, on_close, preview_rows=None):
        self.database, self.on_close, self.preview_rows = database, on_close, preview_rows
        self.all_rows, self.visible_rows, self.sessions = [], [], []
        self.closed = False
        self.window = tk.Toplevel(parent)
        self.window.title("Attendance history · PREVIEW DATA" if preview_rows is not None else "Attendance history · COS30082")
        width, height = min(1100, parent.winfo_screenwidth()-80), min(650, parent.winfo_screenheight()-100)
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(850, width), min(480, height))
        self.window.configure(bg="#eef3f8")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg="#eef3f8", padx=20, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Attendance history", font=("Segoe UI", 20, "bold"),
                 bg="#eef3f8", fg="#172b45").pack(anchor="w")
        tk.Label(body, text="Filter saved records and export the displayed snapshot. Times use UTC+07:00.",
                 bg="#eef3f8", fg="#61748b", font=("Segoe UI", 10), wraplength=950, justify="left").pack(anchor="w", pady=(2, 12))
        filters = tk.Frame(body, bg="#eef3f8")
        filters.pack(fill="x")
        for index, text in enumerate(("Session", "Source", "Search name or session")):
            tk.Label(filters, text=text, bg="#eef3f8", fg="#172b45", font=("Segoe UI", 10, "bold")).grid(row=0, column=index, sticky="w")
        filters.columnconfigure(0, weight=2)
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(2, weight=2)
        self.session_box = ttk.Combobox(filters, state="readonly", font=("Segoe UI", 10))
        self.session_box.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(4, 0))
        self.source_box = ttk.Combobox(filters, state="readonly", values=["All sources", "Automatic", "Manual", "Unknown (legacy)"],
                                       font=("Segoe UI", 10), width=18)
        self.source_box.current(0)
        self.source_box.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(4, 0))
        self.search = tk.StringVar()
        tk.Entry(filters, textvariable=self.search, font=("Segoe UI", 10), relief="solid", bd=1).grid(row=1, column=2, sticky="ew", pady=(4, 0))
        self.session_box.bind("<<ComboboxSelected>>", lambda event: self.apply_filters())
        self.source_box.bind("<<ComboboxSelected>>", lambda event: self.apply_filters())
        self.search.trace_add("write", lambda *_: self.apply_filters())
        self.summary = tk.StringVar()
        tk.Label(body, textvariable=self.summary, bg="#eef3f8", fg="#172b45", font=("Segoe UI", 10, "bold"),
                 wraplength=1000, justify="left").pack(anchor="w", pady=(14, 8))
        table = tk.Frame(body, bg="white")
        columns = ("session", "name", "time", "source", "reviewer")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=6)
        for key, heading, size in (("session", "Session", 170), ("name", "Name / student ID", 220),
                                   ("time", "Recorded (UTC+07:00)", 215), ("source", "Source", 100),
                                   ("reviewer", "Reviewer", 155)):
            self.tree.heading(key, text=heading)
            self.tree.column(key, width=size, minwidth=80, stretch=True)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)
        self.tree.bind("<Double-1>", self.show_detail)
        self.status = tk.StringVar()
        status_label = tk.Label(body, textvariable=self.status, bg="#eef3f8", fg="#61748b", font=("Segoe UI", 9),
                                wraplength=1000, justify="left", anchor="w")
        actions = tk.Frame(body, bg="#eef3f8")
        actions.pack(side="bottom", fill="x")
        ttk.Button(actions, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(actions, text="View details", command=self.show_detail).pack(side="left", padx=8)
        self.export_button = ttk.Button(actions, text="Export displayed rows…", command=self.export)
        self.export_button.pack(side="left")
        ttk.Button(actions, text="Close", command=self.close).pack(side="right")
        status_label.pack(side="bottom", fill="x", pady=(10, 8))
        table.pack(fill="both", expand=True)
        self.refresh()

    def refresh(self):
        current = self.session_box.current()
        selected = self.sessions[current-1] if 0 < current <= len(self.sessions) else None
        try:
            rows = ([dict(row) for row in self.preview_rows] if self.preview_rows is not None
                    else read_attendance_snapshot(self.database))
        except FileNotFoundError as exc:
            rows = []
            self.status.set(str(exc))
        except (OSError, ValueError, sqlite3.Error) as exc:
            self.status.set(f"Could not refresh history: {exc}. The previous snapshot is still displayed.")
            return
        else:
            prefix = "Fictional preview data" if self.preview_rows is not None else "Snapshot loaded"
            self.status.set(f"{prefix} · {datetime.now().astimezone().isoformat(timespec='seconds')}")
        self.all_rows = rows
        self.sessions = list(dict.fromkeys(row["session"] for row in rows))
        self.session_box.configure(values=["All sessions"]+[f"Session: {name}" for name in self.sessions])
        self.session_box.current(self.sessions.index(selected)+1 if selected in self.sessions else 0)
        self.apply_filters()

    def apply_filters(self):
        index = self.session_box.current()
        session = self.sessions[index-1] if 0 < index <= len(self.sessions) else None
        source = [None, "automatic", "manual_review", "legacy_unrecorded"][max(0, self.source_box.current())]
        self.visible_rows = filter_attendance(self.all_rows, session, source, self.search.get())
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(self.visible_rows):
            self.tree.insert("", "end", iid=str(index), values=(row["session"], row["name"],
                local_timestamp(row["recorded_utc"]), SOURCE_LABELS[row["source"]], row.get("reviewer") or "—"))
        totals = summarize_attendance(self.visible_rows)
        self.summary.set(f"{totals['sessions']} sessions · {totals['records']} records · Automatic {totals['automatic']} · "
                         f"Manual {totals['manual_review']} · Unknown {totals['legacy_unrecorded']}")
        self.export_button.configure(state="normal" if self.visible_rows else "disabled")

    def show_detail(self, event=None):
        selected = self.tree.selection()
        if not selected:
            return
        row = self.visible_rows[int(selected[0])]
        text = (f"Session: {row['session']}\nName: {row['name']}\nProfile ID: {row['person_id']}\n"
                f"Recorded (UTC+07:00): {local_timestamp(row['recorded_utc'])}\nRecorded (UTC): {row['recorded_utc']}\n"
                f"Source: {SOURCE_LABELS[row['source']]}\nReviewer: {row.get('reviewer') or '—'}\nReason: {row.get('reason') or '—'}")
        messagebox.showinfo("Attendance details", text, parent=self.window)

    def export(self):
        snapshot = [dict(row) for row in self.visible_rows]
        if not snapshot:
            return
        destination = filedialog.asksaveasfilename(parent=self.window, title=f"Export {len(snapshot)} displayed records",
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{datetime.now():%Y%m%d_%H%M%S}.csv", confirmoverwrite=True)
        if not destination:
            return
        try:
            if Path(destination).resolve() == Path(self.database).resolve():
                raise ValueError("The attendance database cannot be used as an export destination.")
            count = export_attendance_csv(destination, snapshot, overwrite=Path(destination).exists())
        except (OSError, ValueError, csv.Error) as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.window)
        else:
            self.status.set(f"Exported {count} displayed records to {destination}")

    def close(self):
        if not self.closed:
            self.closed = True
            self.window.destroy()
            self.on_close()


def preview_history_rows():
    """Clearly fictional rows for layout QA, not model or attendance results."""
    return [
        {"session": "Demo session A", "person_id": "fictional-a", "name": "Demo student A",
         "recorded_utc": "2026-10-30T02:00:00+00:00", "source": "automatic", "reviewer": None, "reason": None},
        {"session": "Demo session A", "person_id": "fictional-b", "name": "Demo student B",
         "recorded_utc": "2026-10-30T02:05:00+00:00", "source": "manual_review", "reviewer": "Demo operator",
         "reason": "Fictional layout example"},
        {"session": "Demo session B", "person_id": "fictional-a", "name": "Demo student A",
         "recorded_utc": "2026-10-29T02:00:00+00:00", "source": "legacy_unrecorded", "reviewer": None, "reason": None},
    ]
