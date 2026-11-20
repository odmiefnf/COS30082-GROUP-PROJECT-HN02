"""Profile browser and an explicit check-then-import reference photo workflow."""
from __future__ import annotations

from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from .history import local_timestamp
from .store import clean_text
from .theme import BG, BORDER, CARD, FONT, INK, MUTED, SURFACE, FlatButton, card, entry


def label(parent, text=None, variable=None, bold=False, bg=BG, fg=INK, size=10):
    widget = tk.Label(parent, text=text, textvariable=variable, bg=bg, fg=fg, anchor="w",
                      justify="left", wraplength=950, font=(FONT, size, "bold" if bold else "normal"))
    widget.pack(fill="x", pady=(0, 6))
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
        self.window.geometry(f"{min(1080,parent.winfo_screenwidth()-80)}x{min(660,parent.winfo_screenheight()-100)}")
        self.window.minsize(800, 480)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)
        label(body, "Registered people and appearances", bold=True, size=16)
        label(body, "One profile per person. Add separate looks, such as with and without glasses. Stop the camera to manage profiles.",
              fg=MUTED)
        search_row = card(body, padx=14, pady=10)
        search_row.pack(fill="x", pady=(4, 12))
        self.search = tk.StringVar()
        tk.Label(search_row, text="SEARCH NAME / STUDENT ID", bg=CARD, fg=MUTED,
                 font=(FONT, 8, "bold")).pack(side="left", padx=(0, 12))
        entry(search_row, self.search, 11).pack(side="left", fill="x", expand=True, ipady=4, ipadx=6)
        self.search.trace_add("write", lambda *_: self.render_people())
        self.status = tk.StringVar(value="Select a person to view their reference groups.")
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x")
        self.refresh_button = FlatButton(actions, "Refresh", lambda: self.request("profiles_list"), "secondary", 10)
        self.refresh_button.pack(side="left")
        self.new_button = FlatButton(actions, "New person from photos…", lambda: self.open_import(None), "primary", 10)
        self.new_button.pack(side="left", padx=8)
        FlatButton(actions, "Close", self.close, "secondary", 10).pack(side="right")
        tk.Label(body, textvariable=self.status, bg=BG, fg=MUTED, wraplength=1000,
                 font=(FONT, 9), justify="left", anchor="w").pack(side="bottom", fill="x", pady=10)
        content = tk.Frame(body, bg=BG)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)
        people_card = card(content, padx=2, pady=2)
        people_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.people = ttk.Treeview(people_card, columns=("name", "groups", "samples"), show="headings", selectmode="browse", height=5)
        for key, title, width in (("name", "NAME / STUDENT ID", 250), ("groups", "LOOKS", 64), ("samples", "SAMPLES", 78)):
            self.people.heading(key, text=title, anchor="w" if key == "name" else "center")
            self.people.column(key, width=width, minwidth=45, stretch=key=="name",
                               anchor="w" if key == "name" else "center")
        self.people.tag_configure("odd", background=SURFACE)
        self.people.pack(fill="both", expand=True)
        self.people.bind("<<TreeviewSelect>>", self.choose_person)
        right = card(content, padx=16, pady=14)
        right.grid(row=0, column=1, sticky="nsew")
        self.details = tk.StringVar(value="No profile selected")
        label(right, variable=self.details, bold=True, bg=CARD, size=11)
        rename_row = tk.Frame(right, bg=CARD)
        rename_row.pack(fill="x", pady=(0, 12))
        self.rename = tk.StringVar()
        self.rename_entry = entry(rename_row, self.rename, 10)
        self.rename_entry.pack(side="left", fill="x", expand=True, ipady=5, ipadx=6, padx=(0, 8))
        self.rename_button = FlatButton(rename_row, "Rename", self.rename_person, "secondary", 9)
        self.rename_button.pack(side="right")
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        self.groups = ttk.Treeview(right, columns=("label", "samples"), show="headings", selectmode="browse", height=4)
        self.groups.heading("label", text="APPEARANCE", anchor="w")
        self.groups.heading("samples", text="SAMPLES")
        self.groups.column("label", width=230)
        self.groups.column("samples", width=75, stretch=False, anchor="center")
        self.groups.bind("<<TreeviewSelect>>", lambda event: self.controls())
        group_actions = tk.Frame(right, bg=CARD)
        group_actions.pack(side="bottom", fill="x", pady=(10, 0))
        self.import_button = FlatButton(group_actions, "Import another appearance…", self.import_existing, "secondary", 9)
        self.import_button.pack(side="left")
        self.remove_button = FlatButton(group_actions, "Remove selected look…", self.remove_group, "danger", 9)
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
        label(body, f"Add reference photos: {person['name']}" if person else "New person from reference photos", bold=True, size=15)
        label(body, "Choose exactly 3 clear photos of the same person and look. Checks do not prove live presence or record attendance.",
              fg=MUTED)
        form = card(body, padx=16, pady=14)
        form.pack(fill="x", pady=(4, 12))
        tk.Label(form, text="APPEARANCE LABEL (E.G. WITHOUT GLASSES)" if person else "NAME / STUDENT ID (INITIAL APPEARANCE: ORIGINAL)",
                 bg=CARD, fg=MUTED, anchor="w", font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.target_text = tk.StringVar()
        self.target_entry = entry(form, self.target_text, 11)
        self.target_entry.pack(fill="x", ipady=5, ipadx=6, pady=(0, 10))
        self.choose_button = FlatButton(form, "Choose 3 photos…", self.choose_files, "secondary", 10)
        self.choose_button.pack(anchor="w")
        self.status = tk.StringVar(value="No photos selected. Nothing is saved until you confirm the checked photos.")
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x")
        self.check_button = FlatButton(actions, "Check photos", self.check, "secondary", 10)
        self.check_button.pack(side="left")
        self.import_button = FlatButton(actions, "Import checked photos", self.commit, "primary", 10)
        self.import_button.pack(side="left", padx=8)
        FlatButton(actions, "Cancel", self.close, "secondary", 10).pack(side="right")
        tk.Label(body, textvariable=self.status, bg=BG, fg=MUTED, font=(FONT, 9), wraplength=680, justify="left",
                 anchor="w").pack(side="bottom", fill="x", pady=10)
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="I verified that all 3 photos show this person and may be used as reference images.",
                       variable=self.confirmed, command=self.controls, bg=BG, activebackground=BG, fg=INK,
                       activeforeground=INK, selectcolor=SURFACE,
                       font=(FONT, 10), wraplength=680, justify="left", anchor="w").pack(side="bottom", fill="x", pady=(10, 0))
        photos_card = card(body, padx=2, pady=2)
        photos_card.pack(fill="both", expand=True)
        self.photos = ttk.Treeview(photos_card, columns=("file", "result"), show="headings", height=3)
        self.photos.heading("file", text="PHOTO", anchor="w")
        self.photos.heading("result", text="CHECK RESULT", anchor="w")
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
