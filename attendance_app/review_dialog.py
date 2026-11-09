"""Explicit operator decision over a temporary face crop, with no default identity."""
from __future__ import annotations

from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .store import clean_text


class ReviewDialog:
    def __init__(self, parent, request, submit, cancel):
        self.request, self.submit, self.cancel = request, submit, cancel
        self.invalid, self.busy, self.closed = False, False, False
        self.timer = None
        self.window = tk.Toplevel(parent)
        self.window.title("Manual review · COS30082")
        self.window.geometry("620x680")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.configure(bg="#ffffff")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg="#ffffff", padx=24, pady=18)
        body.pack(fill="both", expand=True)

        def label(text, size=10, bold=False, color="#172b45"):
            widget = tk.Label(body, text=text, bg="white", fg=color, anchor="w", justify="left",
                              wraplength=565, font=("Segoe UI", size, "bold" if bold else "normal"))
            widget.pack(fill="x", pady=(0, 7))
            return widget

        label("Review the person in view", 17, True)
        label("Keep one person in front of the camera. Select a profile only after checking it in person; this is an operator decision.")
        picture = request["crop"].copy()
        picture.thumbnail((180, 140), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(picture)
        tk.Label(body, image=self.photo, bg="#eef3f8").pack(pady=(0, 5))
        captured = datetime.fromisoformat(request["observed_utc"]).astimezone().strftime("%H:%M:%S")
        label(f"Face observed at {captured} · Image held temporarily in memory", 9, color="#61748b")
        self.countdown = label("", 10, True, "#1765d1")
        label("Profile you checked", 10, True)
        self.person = ttk.Combobox(body, values=[p["name"] for p in request["profiles"]], state="readonly",
                                   font=("Segoe UI", 11))
        self.person.pack(fill="x", ipady=4, pady=(0, 10))
        self.person.set("")
        label("Reviewer name", 10, True)
        self.reviewer = tk.StringVar()
        tk.Entry(body, textvariable=self.reviewer, font=("Segoe UI", 11), relief="solid", bd=1).pack(fill="x", ipady=5, pady=(0, 10))
        label("Reason for confirmation", 10, True)
        self.reason = tk.StringVar()
        tk.Entry(body, textvariable=self.reason, font=("Segoe UI", 11), relief="solid", bd=1).pack(fill="x", ipady=5, pady=(0, 10))
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="I have checked the person against the selected profile in person.", variable=self.confirmed,
                       bg="white", activebackground="white", fg="#172b45", font=("Segoe UI", 10),
                       anchor="w", wraplength=550, justify="left").pack(fill="x", pady=(0, 5))
        self.message = label("", 10, color="#a32b2b")
        actions = tk.Frame(body, bg="white")
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        self.confirm_button = tk.Button(actions, text="Record manual attendance", command=self.confirm, bg="#1765d1", fg="white",
            bd=0, padx=16, pady=10, font=("Segoe UI", 11, "bold"))
        self.confirm_button.pack(side="left", fill="x", expand=True, padx=(0, 10))
        tk.Button(actions, text="Cancel", command=self.close, bg="#e7edf5", fg="#172b45", bd=0,
                  padx=22, pady=10, font=("Segoe UI", 11)).pack(side="right")
        self.window.grab_set()
        self.tick()

    def tick(self):
        if self.closed:
            return
        remaining = max(0, self.request["expires_at"] - time.monotonic())
        self.countdown.configure(text=f"Request valid for {remaining:.0f} seconds")
        if remaining == 0:
            self.invalidate("This request has expired. Cancel and capture a new observation.")
        self.timer = self.window.after(200, self.tick)

    def invalidate(self, message):
        self.invalid = True
        self.message.configure(text=message)
        self.confirm_button.configure(state="disabled")

    def show_error(self, message):
        if self.closed:
            return
        self.busy = False
        self.message.configure(text=message)
        if not self.invalid:
            self.confirm_button.configure(state="normal")

    def confirm(self):
        if self.invalid or self.busy:
            return
        try:
            selected = self.person.current()
            if selected < 0:
                raise ValueError("Select the profile you checked; no profile is selected by default.")
            if self.confirmed.get() is not True:
                raise ValueError("Confirm that you have checked the person in person.")
            reviewer = clean_text(self.reviewer.get(), "Reviewer")
            reason = clean_text(self.reason.get(), "Reason")
        except ValueError as exc:
            self.show_error(str(exc))
            return
        self.busy = True
        self.confirm_button.configure(state="disabled")
        self.submit({"review_id": self.request["id"], "person_id": self.request["profiles"][selected]["id"],
                     "reviewer": reviewer, "reason": reason, "confirmed": True})

    def finish(self):
        if not self.closed:
            self.closed = True
            if self.timer is not None:
                self.window.after_cancel(self.timer)
            self.window.grab_release()
            self.window.destroy()
            self.photo = None

    def close(self):
        self.cancel(self.request["id"])
        self.finish()
