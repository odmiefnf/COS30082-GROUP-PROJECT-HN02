"""Attendance history viewer; browsing never changes the active camera/session."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .history import (SOURCE_LABELS, export_attendance_csv, filter_attendance, local_timestamp,
                      preview_history_rows, read_attendance_snapshot, summarize_attendance)
from .theme import BG, CARD, FONT, INK, MUTED, SURFACE, FlatButton, card, entry


class HistoryDialog:
    def __init__(self, parent, database, on_close, preview_rows=None):
        self.database, self.on_close, self.preview_rows = database, on_close, preview_rows
        self.all_rows, self.visible_rows, self.sessions = [], [], []
        self.closed = False
        self.window = tk.Toplevel(parent)
        self.window.title("Attendance history · PREVIEW DATA" if preview_rows is not None else "Attendance history · COS30082")
        width, height = min(1100, parent.winfo_screenwidth()-80), min(660, parent.winfo_screenheight()-100)
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(850, width), min(480, height))
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Attendance history", font=(FONT, 17, "bold"),
                 bg=BG, fg=INK).pack(anchor="w")
        tk.Label(body, text="Filter saved records and export the displayed snapshot. Times use UTC+07:00.",
                 bg=BG, fg=MUTED, font=(FONT, 10), wraplength=950, justify="left").pack(anchor="w", pady=(2, 12))
        filters = card(body, padx=14, pady=12)
        filters.pack(fill="x")
        for index, text in enumerate(("SESSION", "SOURCE", "SEARCH NAME OR SESSION")):
            tk.Label(filters, text=text, bg=CARD, fg=MUTED, font=(FONT, 8, "bold")).grid(row=0, column=index, sticky="w")
        filters.columnconfigure(0, weight=2)
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(2, weight=2)
        self.session_box = ttk.Combobox(filters, state="readonly", font=(FONT, 10))
        self.session_box.grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(4, 0))
        self.source_box = ttk.Combobox(filters, state="readonly", values=["All sources", "Automatic", "Manual", "Unknown (legacy)"],
                                       font=(FONT, 10), width=18)
        self.source_box.current(0)
        self.source_box.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(4, 0))
        self.search = tk.StringVar()
        entry(filters, self.search, 10).grid(row=1, column=2, sticky="ew", ipady=4, ipadx=6, pady=(4, 0))
        self.session_box.bind("<<ComboboxSelected>>", lambda event: self.apply_filters())
        self.source_box.bind("<<ComboboxSelected>>", lambda event: self.apply_filters())
        self.search.trace_add("write", lambda *_: self.apply_filters())
        self.summary = tk.StringVar()
        tk.Label(body, textvariable=self.summary, bg=BG, fg=INK, font=(FONT, 10, "bold"),
                 wraplength=1000, justify="left").pack(anchor="w", pady=(12, 8))
        table = card(body, padx=2, pady=2)
        columns = ("session", "name", "time", "source", "reviewer")
        self.tree = ttk.Treeview(table, columns=columns, show="headings", selectmode="browse", height=6)
        for key, heading, size in (("session", "SESSION", 170), ("name", "NAME / STUDENT ID", 220),
                                   ("time", "RECORDED (UTC+07:00)", 215), ("source", "SOURCE", 100),
                                   ("reviewer", "REVIEWER", 155)):
            self.tree.heading(key, text=heading, anchor="w" if key in ("session", "name", "reviewer") else "center")
            self.tree.column(key, width=size, minwidth=80, stretch=True,
                             anchor="w" if key in ("session", "name", "reviewer") else "center")
        self.tree.tag_configure("odd", background=SURFACE)
        self.tree.tag_configure("manual", foreground="#b45309")
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
        status_label = tk.Label(body, textvariable=self.status, bg=BG, fg=MUTED, font=(FONT, 9),
                                wraplength=1000, justify="left", anchor="w")
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x")
        FlatButton(actions, "Refresh", self.refresh, "secondary", 10).pack(side="left")
        FlatButton(actions, "View details", self.show_detail, "secondary", 10).pack(side="left", padx=8)
        self.export_button = FlatButton(actions, "Export displayed rows…", self.export, "primary", 10)
        self.export_button.pack(side="left")
        FlatButton(actions, "Close", self.close, "secondary", 10).pack(side="right")
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
            tags = (("odd",) if index % 2 else ()) + (("manual",) if row["source"] == "manual_review" else ())
            self.tree.insert("", "end", iid=str(index), values=(row["session"], row["name"],
                local_timestamp(row["recorded_utc"]), SOURCE_LABELS[row["source"]], row.get("reviewer") or "—"), tags=tags)
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
