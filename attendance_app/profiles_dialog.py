"""Profile browser and an explicit check-then-import reference photo workflow."""
from __future__ import annotations

from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from .history import local_timestamp
from .store import clean_text


BG, INK = "#eef3f8", "#172b45"


def label(parent, text=None, variable=None, bold=False):
    widget = tk.Label(parent, text=text, textvariable=variable, bg=BG, fg=INK, anchor="w",
                      justify="left", wraplength=950, font=("Segoe UI", 10, "bold" if bold else "normal"))
    widget.pack(fill="x", pady=(0, 7))
    widget.bind("<Configure>", lambda event: widget.configure(wraplength=max(100, event.width)))
    return widget


class ProfilesDialog:
    def __init__(self, parent, profiles, send, on_close, photo_import_available=True, preview=False):
        self.send, self.on_close, self.preview = send, on_close, preview
        self.photo_import_available = photo_import_available
        self.closed, self.busy = False, False
        self.profiles, self.selected_id, self.import_dialog = [], None, None
        self.window = tk.Toplevel(parent)
        self.window.title("Profiles · PREVIEW DATA" if preview else "Profiles · COS30082")
        self.window.geometry(f"{min(1080,parent.winfo_screenwidth()-80)}x{min(650,parent.winfo_screenheight()-100)}")
        self.window.minsize(800, 480)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)
        label(body, "Registered people and appearances", bold=True)
        label(body, "One profile per person. Add separate looks, such as with and without glasses. Stop the camera to manage profiles.")
        search_row = tk.Frame(body, bg=BG)
        search_row.pack(fill="x", pady=(0, 10))
        self.search = tk.StringVar()
        tk.Label(search_row, text="Search name / student ID", bg=BG, fg=INK).pack(side="left", padx=(0, 10))
        tk.Entry(search_row, textvariable=self.search, font=("Segoe UI", 11)).pack(side="left", fill="x", expand=True)
        self.search.trace_add("write", lambda *_: self.render_people())
        self.status = tk.StringVar(value="Select a person to view their reference groups.")
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x")
        self.refresh_button = ttk.Button(actions, text="Refresh", command=lambda: self.request("profiles_list"))
        self.refresh_button.pack(side="left")
        self.new_button = ttk.Button(actions, text="New person from photos…", command=lambda: self.open_import(None))
        self.new_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Close", command=self.close).pack(side="right")
        tk.Label(body, textvariable=self.status, bg=BG, fg="#61748b", wraplength=1000,
                 justify="left", anchor="w").pack(side="bottom", fill="x", pady=10)
        content = tk.Frame(body, bg=BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        self.people = ttk.Treeview(content, columns=("name", "groups", "samples"), show="headings", selectmode="browse", height=5)
        for key, title, width in (("name", "Name / student ID", 250), ("groups", "Looks", 60), ("samples", "Samples", 75)):
            self.people.heading(key, text=title)
            self.people.column(key, width=width, minwidth=45, stretch=key=="name")
        self.people.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.people.bind("<<TreeviewSelect>>", self.choose_person)
        right = tk.Frame(content, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self.details = tk.StringVar(value="No profile selected")
        label(right, variable=self.details, bold=True)
        self.rename = tk.StringVar()
        self.rename_entry = tk.Entry(right, textvariable=self.rename, font=("Segoe UI", 11))
        self.rename_entry.pack(fill="x", pady=(0, 6))
        self.rename_button = ttk.Button(right, text="Rename selected person", command=self.rename_person)
        self.rename_button.pack(anchor="w", pady=(0, 12))
        self.groups = ttk.Treeview(right, columns=("label", "samples"), show="headings", selectmode="browse", height=4)
        self.groups.heading("label", text="Appearance")
        self.groups.heading("samples", text="Samples")
        self.groups.column("label", width=230)
        self.groups.column("samples", width=70, stretch=False)
        self.groups.bind("<<TreeviewSelect>>", lambda event: self.controls())
        group_actions = tk.Frame(right, bg=BG)
        group_actions.pack(side="bottom", fill="x", pady=(10, 0))
        self.import_button = ttk.Button(group_actions, text="Import another appearance…", command=self.import_existing)
        self.import_button.pack(side="left")
        self.remove_button = ttk.Button(group_actions, text="Remove selected look…", command=self.remove_group)
        self.remove_button.pack(side="left", padx=8)
        self.groups.pack(fill="both", expand=True)
        self.update_profiles(profiles)

    def selected_person(self):
        return next((p for p in self.profiles if p["id"] == self.selected_id), None)

    def controls(self):
        person = self.selected_person()
        idle = not self.busy and self.import_dialog is None
        self.rename_button.configure(state="normal" if idle and person else "disabled")
        self.rename_entry.configure(state="normal" if idle and person else "disabled")
        self.refresh_button.configure(state="normal" if idle else "disabled")
        self.new_button.configure(state="normal" if idle and self.photo_import_available else "disabled")
        self.import_button.configure(state="normal" if idle and person and len(person["appearances"]) < 4 and self.photo_import_available else "disabled")
        self.remove_button.configure(state="normal" if idle and person and len(person["appearances"]) > 1 and self.groups.selection() else "disabled")

    def update_profiles(self, profiles, selected_id=None, message=None):
        self.profiles, self.busy = profiles, False
        if selected_id is not None:
            self.selected_id = selected_id
        if self.selected_person() is None:
            self.selected_id = None
        self.render_people()
        self.render_groups()
        if message:
            self.status.set(message)
        elif not self.photo_import_available:
            self.status.set("Photo import requires automatic face checks. ROI mode supports camera enrollment.")
        self.controls()

    def render_people(self):
        needle = self.search.get().strip().casefold()
        self.people.delete(*self.people.get_children())
        for p in self.profiles:
            if not needle or needle in p["name"].casefold():
                self.people.insert("", "end", iid=p["id"], values=(p["name"], len(p["appearances"]), p["sample_count"]))
        if self.selected_id and self.people.exists(self.selected_id):
            self.people.selection_set(self.selected_id)
        elif self.selected_id is not None:
            self.selected_id = None
            self.render_groups()

    def choose_person(self, event=None):
        selected = self.people.selection()
        self.selected_id = selected[0] if selected else None
        self.render_groups()

    def render_groups(self):
        self.groups.delete(*self.groups.get_children())
        p = self.selected_person()
        self.details.set(f"{p['name']} · Created {local_timestamp(p['created_utc'])}" if p else "No profile selected")
        self.rename.set(p["name"] if p else "")
        if p:
            for group in p["appearances"]:
                self.groups.insert("", "end", iid=group["id"], values=(group["label"], group["sample_count"]))
        self.controls()

    def request(self, command, value=None):
        if self.preview:
            self.status.set("Preview only · No database changes or photo checks are performed.")
            return
        self.busy = True
        self.controls()
        self.send(command, value)

    def rename_person(self):
        p = self.selected_person()
        if p:
            self.request("profile_rename", {"person_id": p["id"], "name": self.rename.get()})

    def remove_group(self):
        p, selected = self.selected_person(), self.groups.selection()
        if not p or not selected:
            return
        group = next(a for a in p["appearances"] if a["id"] == selected[0])
        if self.preview:
            self.status.set("Preview only · No appearance is removed.")
            return
        if messagebox.askyesno("Remove appearance", f"Remove '{group['label']}' from {p['name']}?\n"
                "A database backup will be saved first. The person's attendance history will be kept.", parent=self.window):
            self.request("profile_remove", {"person_id": p["id"], "appearance_id": group["id"]})

    def import_existing(self):
        if self.selected_person():
            self.open_import(self.selected_person())

    def open_import(self, person):
        if self.import_dialog is None:
            self.import_dialog = PhotoImportDialog(self.window, person, self.send, self.import_closed, self.preview)
            self.controls()

    def import_closed(self):
        self.import_dialog = None
        self.controls()

    def error(self, message, operation_id=None):
        self.busy = False
        self.status.set(message)
        if self.import_dialog is not None and operation_id == self.import_dialog.operation_id:
            self.import_dialog.error(message)
        self.controls()

    def close(self):
        if not self.closed:
            self.closed = True
            if self.import_dialog is not None:
                self.import_dialog.close()
            self.window.destroy()
            self.on_close()


class PhotoImportDialog:
    def __init__(self, parent, person, send, on_close, preview=False):
        self.person, self.send, self.on_close, self.preview = person, send, on_close, preview
        self.files, self.token, self.expires, self.timer = [], None, 0., None
        self.operation_id = uuid.uuid4().hex
        self.closed, self.busy = False, False
        self.window = tk.Toplevel(parent)
        self.window.title("Import reference photos · PREVIEW" if preview else "Import reference photos · COS30082")
        self.window.geometry(f"{min(760,parent.winfo_screenwidth()-100)}x{min(650,parent.winfo_screenheight()-100)}")
        self.window.minsize(600, 500)
        self.window.configure(bg=BG)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)
        label(body, f"Add reference photos: {person['name']}" if person else "New person from reference photos", bold=True)
        label(body, "Choose exactly 3 clear photos of the same person and look. Checks do not prove live presence or record attendance.")
        label(body, "Appearance label (for example, Without glasses)" if person else "Name / student ID (initial appearance: Original)", bold=True)
        self.target_text = tk.StringVar()
        self.target_entry = tk.Entry(body, textvariable=self.target_text, font=("Segoe UI", 11))
        self.target_entry.pack(fill="x", pady=(0, 10))
        self.choose_button = ttk.Button(body, text="Choose 3 photos…", command=self.choose_files)
        self.choose_button.pack(anchor="w", pady=(0, 10))
        self.status = tk.StringVar(value="No photos selected. Nothing is saved until you confirm the checked photos.")
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x")
        self.check_button = ttk.Button(actions, text="Check photos", command=self.check)
        self.check_button.pack(side="left")
        self.import_button = ttk.Button(actions, text="Import checked photos", command=self.commit)
        self.import_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Cancel", command=self.close).pack(side="right")
        tk.Label(body, textvariable=self.status, bg=BG, fg="#61748b", wraplength=680, justify="left",
                 anchor="w").pack(side="bottom", fill="x", pady=10)
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="I verified that all 3 photos show this person and may be used as reference images.",
                       variable=self.confirmed, command=self.controls, bg=BG, fg=INK, wraplength=680,
                       justify="left", anchor="w").pack(side="bottom", fill="x", pady=(10, 0))
        self.photos = ttk.Treeview(body, columns=("file", "result"), show="headings", height=3)
        self.photos.heading("file", text="Photo")
        self.photos.heading("result", text="Check result")
        self.photos.column("file", width=190)
        self.photos.column("result", width=460)
        self.photos.pack(fill="both", expand=True)
        self.photos.bind("<Double-1>", self.photo_detail)
        self.window.grab_set()
        self.controls()
        self.tick()

    def controls(self):
        self.choose_button.configure(state="disabled" if self.busy else "normal")
        self.target_entry.configure(state="disabled" if self.busy or self.token else "normal")
        self.check_button.configure(state="normal" if not self.busy and len(self.files)==3 else "disabled")
        self.import_button.configure(state="normal" if not self.busy and self.token and self.confirmed.get() else "disabled")

    def photo_detail(self, event=None):
        selected = self.photos.selection()
        if selected:
            values = self.photos.item(selected[0], "values")
            messagebox.showinfo("Photo check", f"{values[0]}\n\n{values[1]}", parent=self.window)

    def choose_files(self):
        files = filedialog.askopenfilenames(parent=self.window, title="Choose exactly 3 reference photos",
            filetypes=[("Face photos", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not files:
            return
        if len(files) != 3:
            self.status.set("Choose exactly 3 photos for one appearance.")
            return
        self.send("profile_import_cancel", self.operation_id)
        self.operation_id, self.files, self.token = uuid.uuid4().hex, list(files), None
        self.confirmed.set(False)
        self.photos.delete(*self.photos.get_children())
        for index, path in enumerate(files):
            self.photos.insert("", "end", iid=str(index), values=(Path(path).name, "Not checked"))
        self.status.set("Photos selected. Enter the name/label and click Check photos.")
        self.controls()

    def check(self):
        try:
            text = clean_text(self.target_text.get(), "Appearance label" if self.person else "Name")
        except ValueError as exc:
            self.error(str(exc))
            return
        if self.preview:
            self.status.set("Preview only · No photos are read, checked or imported.")
            return
        self.busy, self.token = True, None
        self.confirmed.set(False)
        self.status.set("Checking three photos…")
        self.controls()
        self.send("profile_import_check", {"operation_id": self.operation_id, "files": list(self.files),
            "person_id": self.person["id"] if self.person else None,
            "name": text if not self.person else "", "label": text if self.person else ""})

    def checked(self, report):
        if self.closed or report["operation_id"] != self.operation_id:
            return
        self.busy, self.token = False, report["token"]
        self.expires = report["expires_at"] or 0.
        self.photos.delete(*self.photos.get_children())
        for index, item in enumerate(report["items"]):
            self.photos.insert("", "end", iid=str(index), values=(item["file"], item["message"]))
        self.status.set(report["message"])
        self.controls()

    def error(self, message):
        self.busy, self.token = False, None
        self.status.set(message)
        self.controls()

    def commit(self):
        if self.busy or not self.token or not self.confirmed.get():
            return
        self.busy = True
        self.status.set("Importing the checked reference templates…")
        self.controls()
        self.send("profile_import_commit", {"operation_id": self.operation_id, "token": self.token, "confirmed": True})

    def tick(self):
        if self.closed:
            return
        if self.token and time.monotonic() >= self.expires:
            self.token = None
            self.status.set("The checked photos expired. Click Check photos again.")
            self.controls()
        self.timer = self.window.after(250, self.tick)

    def close(self):
        if not self.closed:
            self.closed = True
            self.send("profile_import_cancel", self.operation_id)
            if self.timer is not None:
                self.window.after_cancel(self.timer)
            self.window.grab_release()
            self.window.destroy()
            self.on_close()
