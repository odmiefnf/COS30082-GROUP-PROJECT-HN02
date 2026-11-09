"""Explicit selection of an existing profile before capturing another appearance."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import time

from .store import clean_text


class AppearanceDialog:
    def __init__(self, parent, request, submit, cancel):
        self.request, self.profiles = request, request["profiles"]
        self.submit, self.cancel = submit, cancel
        self.closed, self.invalid = False, False
        self.timer = None
        self.window = tk.Toplevel(parent)
        self.window.title("Add appearance · COS30082")
        self.window.geometry("620x480")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.configure(bg="white")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg="white", padx=24, pady=20)
        body.pack(fill="both", expand=True)

        def label(text, size=10, bold=False, color="#172b45"):
            widget = tk.Label(body, text=text, bg="white", fg=color, anchor="w", justify="left",
                              wraplength=565, font=("Segoe UI", size, "bold" if bold else "normal"))
            widget.pack(fill="x", pady=(0, 9))
            return widget

        label("Add an appearance to an existing profile", 16, True)
        label("Use this for a verified person with a different appearance, such as without glasses. "
              "Select their existing profile. The camera will capture 3 new live samples.")
        self.countdown = label("", bold=True, color="#1765d1")
        label("Existing profile", bold=True)
        self.person = ttk.Combobox(body, values=[profile["name"] for profile in self.profiles], state="readonly",
                                   font=("Segoe UI", 11))
        self.person.pack(fill="x", ipady=4, pady=(0, 14))
        self.person.set("")
        label("Appearance label (for example, Without glasses)", bold=True)
        self.appearance_label = tk.StringVar()
        tk.Entry(body, textvariable=self.appearance_label, font=("Segoe UI", 11),
                 relief="solid", bd=1).pack(fill="x", ipady=6, pady=(0, 12))
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(body, text="I have verified this is the person in the selected profile.",
                       variable=self.confirmed, bg="white", activebackground="white", fg="#172b45",
                       font=("Segoe UI", 10), anchor="w", wraplength=550, justify="left").pack(fill="x", pady=(0, 8))
        self.message = label("", color="#a32b2b")
        actions = tk.Frame(body, bg="white")
        actions.pack(side="bottom", fill="x", pady=(6, 0))
        self.confirm_button = tk.Button(actions, text="Begin capture", command=self.confirm, bg="#1765d1", fg="white",
                  bd=0, padx=18, pady=10, font=("Segoe UI", 11, "bold"))
        self.confirm_button.pack(side="left", fill="x", expand=True, padx=(0, 10))
        tk.Button(actions, text="Cancel", command=self.close, bg="#e7edf5", fg="#172b45",
                  bd=0, padx=22, pady=10, font=("Segoe UI", 11)).pack(side="right")
        self.window.grab_set()
        self.tick()

    def tick(self):
        if self.closed:
            return
        remaining = max(0., self.request["expires_at"] - time.monotonic())
        self.countdown.configure(text=f"Request valid for {remaining:.0f} seconds · Keep the same person in view")
        if remaining == 0:
            self.invalidate("This request has expired. Cancel and capture a new observation.")
        self.timer = self.window.after(200, self.tick)

    def invalidate(self, message):
        if not self.closed:
            self.invalid = True
            self.message.configure(text=message)
            self.confirm_button.configure(state="disabled")

    def confirm(self):
        if self.closed or self.invalid:
            return
        if time.monotonic() >= self.request["expires_at"]:
            self.invalidate("This request has expired. Cancel and capture a new observation.")
            return
        try:
            selected = self.person.current()
            if selected < 0:
                raise ValueError("Select the existing profile you have verified.")
            if self.confirmed.get() is not True:
                raise ValueError("Confirm that this is the person in the selected profile.")
            label = clean_text(self.appearance_label.get(), "Appearance label")
        except ValueError as exc:
            self.message.configure(text=str(exc))
            return
        fields = {"person_id": self.profiles[selected]["id"], "label": label, "confirmed": True,
                  "request_id": self.request["id"]}
        self.finish()
        self.submit(fields)

    def finish(self):
        if not self.closed:
            self.closed = True
            if self.timer is not None:
                self.window.after_cancel(self.timer)
            self.window.grab_release()
            self.window.destroy()

    def close(self):
        self.finish()
        self.cancel(self.request["id"])
