"""Explicit operator decision over a temporary face crop, with no default identity."""
from __future__ import annotations

from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .store import clean_text
from .theme import BG, BORDER, CARD, FONT, INK, MUTED, SURFACE, TONES, FlatButton, card, entry


class ReviewDialog:
    def __init__(self, parent, request, submit, cancel):
        self.request, self.submit, self.cancel = request, submit, cancel
        self.invalid, self.busy, self.closed = False, False, False
        self.timer = None
        self.window = tk.Toplevel(parent)
        self.window.title("Manual review · COS30082")
        self.window.geometry("620x710")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=22, pady=18)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Review the person in view", bg=BG, fg=INK, anchor="w",
                 font=(FONT, 16, "bold")).pack(fill="x")
        tk.Label(body, text="Keep one person in front of the camera. Select a profile only after checking it in person; "
                 "this is an operator decision.", bg=BG, fg=MUTED, anchor="w", justify="left",
                 wraplength=565, font=(FONT, 10)).pack(fill="x", pady=(4, 12))

        form = card(body, padx=18, pady=16)
        form.pack(fill="both", expand=True)

        top_row = tk.Frame(form, bg=CARD)
        top_row.pack(fill="x", pady=(0, 12))
        picture = request["crop"].copy()
        picture.thumbnail((160, 125), Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(picture)
        tk.Label(top_row, image=self.photo, bg=SURFACE, highlightthickness=1,
                 highlightbackground=BORDER).pack(side="left", padx=(0, 14))
        meta = tk.Frame(top_row, bg=CARD)
        meta.pack(side="left", fill="both", expand=True)
        captured = datetime.fromisoformat(request["observed_utc"]).astimezone().strftime("%H:%M:%S")
        tk.Label(meta, text=f"Observed at {captured}", bg=CARD, fg=INK, anchor="w",
                 font=(FONT, 11, "bold")).pack(fill="x", pady=(4, 2))
        tk.Label(meta, text="Temporary face crop held in memory only.", bg=CARD, fg=MUTED, anchor="w",
                 justify="left", wraplength=340, font=(FONT, 9)).pack(fill="x", pady=(0, 10))
        soft, accent, info_ink = TONES["info"]
        banner = tk.Frame(meta, bg=soft)
        banner.pack(fill="x")
        tk.Frame(banner, bg=accent, width=4).pack(side="left", fill="y")
        self.countdown = tk.Label(banner, text="", bg=soft, fg=info_ink, anchor="w",
                                  padx=10, pady=6, font=(FONT, 10, "bold"))
        self.countdown.pack(fill="x")

        tk.Label(form, text="PROFILE YOU CHECKED", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.person = ttk.Combobox(form, values=[p["name"] for p in request["profiles"]], state="readonly",
                                   font=(FONT, 11))
        self.person.pack(fill="x", ipady=4, pady=(0, 12))
        self.person.set("")
        tk.Label(form, text="REVIEWER NAME", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.reviewer = tk.StringVar()
        entry(form, self.reviewer, 11).pack(fill="x", ipady=5, ipadx=6, pady=(0, 12))
        tk.Label(form, text="REASON FOR CONFIRMATION", bg=CARD, fg=MUTED, anchor="w",
                 font=(FONT, 8, "bold")).pack(fill="x", pady=(0, 4))
        self.reason = tk.StringVar()
        entry(form, self.reason, 11).pack(fill="x", ipady=5, ipadx=6, pady=(0, 12))
        self.confirmed = tk.BooleanVar(value=False)
        tk.Checkbutton(form, text="I have checked the person against the selected profile in person.", variable=self.confirmed,
                       bg=CARD, activebackground=CARD, fg=INK, activeforeground=INK, selectcolor=SURFACE, font=(FONT, 10),
                       anchor="w", wraplength=520, justify="left").pack(fill="x", pady=(0, 6))
        self.message = tk.Label(form, text="", bg=CARD, fg="#dc2626", anchor="w", justify="left",
                                wraplength=520, font=(FONT, 10))
        self.message.pack(fill="x", pady=(2, 0))

        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x", pady=(14, 0))
        self.confirm_button = FlatButton(actions, "Record manual attendance", self.confirm, "primary", 10, pady=9)
        self.confirm_button.pack(side="left", fill="x", expand=True, padx=(0, 10))
        FlatButton(actions, "Cancel", self.close, "secondary", 10, padx=22, pady=9).pack(side="right")
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
