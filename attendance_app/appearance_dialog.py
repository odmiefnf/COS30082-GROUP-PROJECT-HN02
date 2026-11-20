"""Explicit selection of an existing profile before capturing another appearance."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import time

from .store import clean_text
from .theme import BG, BORDER, CARD, FONT, INK, MUTED, PRIMARY, SURFACE, TONES, FlatButton, card, entry


class AppearanceDialog:
    def __init__(self, parent, request, submit, cancel):
        self.request, self.profiles = request, request["profiles"]
        self.submit, self.cancel = submit, cancel
        self.closed, self.invalid = False, False
        self.timer = None
        self.window = tk.Toplevel(parent)
        self.window.title("Add appearance · COS30082")
        self.window.geometry("620x500")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=22, pady=18)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Add an appearance to an existing profile", bg=BG, fg=INK, anchor="w",
                 font=(FONT, 15, "bold")).pack(fill="x")
        tk.Label(body, text="Use this for a verified person with a different appearance, such as without glasses. "
                 "Select their existing profile. The camera will capture 3 new live samples.",
                 bg=BG, fg=MUTED, anchor="w", justify="left", wraplength=565,
                 font=(FONT, 10)).pack(fill="x", pady=(4, 12))

        form = card(body, padx=18, pady=16)
        form.pack(fill="both", expand=True)

        soft, accent, info_ink = TONES["info"]
        banner = tk.Frame(form, bg=soft)
        banner.pack(fill="x", pady=(0, 12))
        tk.Frame(banner, bg=accent, width=4).pack(side="left", fill="y")
        self.countdown = tk.Label(banner, text="", bg=soft, fg=info_ink, anchor="w", justify="left",
                                  wraplength=510, padx=10, pady=7, font=(FONT, 10, "bold"))
        self.countdown.pack(fill="x")

        tk.Label(form, text="EXISTING PROFILE", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.person = ttk.Combobox(form, values=[profile["name"] for profile in self.profiles], state="readonly",
                                   font=(FONT, 11))
        self.person.pack(fill="x", ipady=4, pady=(0, 14))
        self.person.set("")
        tk.Label(form, text="APPEARANCE LABEL (E.G. WITHOUT GLASSES)", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.appearance_label = tk.StringVar()
        entry(form, self.appearance_label, 11).pack(fill="x", ipady=6, ipadx=6, pady=(0, 12))
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(form, text="I have verified this is the person in the selected profile.",
                       variable=self.confirmed, bg=CARD, activebackground=CARD, fg=INK, activeforeground=INK,
                       selectcolor=SURFACE, font=(FONT, 10), anchor="w", wraplength=520, justify="left").pack(fill="x", pady=(0, 8))
        self.message = tk.Label(form, text="", bg=CARD, fg="#dc2626", anchor="w", justify="left",
                                wraplength=520, font=(FONT, 10))
        self.message.pack(fill="x", pady=(2, 0))

        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x", pady=(14, 0))
        self.confirm_button = FlatButton(actions, "Begin capture", self.confirm, "primary", 10, pady=9)
        self.confirm_button.pack(side="left", fill="x", expand=True, padx=(0, 10))
        FlatButton(actions, "Cancel", self.close, "secondary", 10, padx=22, pady=9).pack(side="right")
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
