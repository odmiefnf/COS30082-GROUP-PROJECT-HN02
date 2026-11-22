"""Read-only setup and camera status, with an explicit model setup retry."""
import tkinter as tk
from tkinter import ttk

from .diagnostics import COMPONENTS
from .theme import BG, CARD, FONT, INK, MUTED, FlatButton, card


class DiagnosticsDialog:
    def __init__(self, parent, retry, on_close, preview=False):
        self.on_close = on_close
        self.window = tk.Toplevel(parent)
        self.window.title("Diagnostics · PREVIEW DATA" if preview else "Diagnostics · COS30082")
        self.window.geometry("880x650")
        self.window.minsize(650, 480)
        self.window.configure(bg=BG)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg=BG, padx=20, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Camera and model diagnostics", bg=BG, fg=INK,
                 font=(FONT, 16, "bold"), anchor="w").pack(fill="x")
        tk.Label(body, text="Read-only component status, runtime details, and the latest frame summary.",
                 bg=BG, fg=MUTED, font=(FONT, 10), anchor="w").pack(fill="x", pady=(2, 12))
        actions = tk.Frame(body, bg=BG)
        actions.pack(side="bottom", fill="x", pady=(12, 0))
        self.retry = FlatButton(actions, "Retry model setup", retry, "primary", 10)
        self.retry.pack(side="left")
        FlatButton(actions, "Close", self.close, "secondary", 10).pack(side="right")
        panel = card(body, padx=2, pady=2)
        panel.pack(fill="both", expand=True)
        self.text = tk.Text(panel, font=("Consolas", 10), wrap="word", bg=CARD, fg=INK,
                            padx=14, pady=12, relief="flat", bd=0, state="disabled")
        scroll = ttk.Scrollbar(panel, orient="vertical", command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(fill="both", expand=True)
        self.last = None

    def update(self, components, runtime, frame, can_retry):
        self.retry.configure(state="normal" if can_retry else "disabled")
        sections = [runtime]
        for key, (title, hint) in COMPONENTS.items():
            item = components.get(key, {})
            status = item.get("status", "Not checked")
            detail = item.get("detail", "")
            sections.append(f"{title}: {status}\n{detail}" + (f"\nNext step: {hint}" if status == "Error" else ""))
        sections.append("Latest frame (memory only)\n"+(frame or "No frame processed in this camera run."))
        value = "\n\n".join(sections)
        if value != self.last:
            position = self.text.yview()[0]
            self.text.configure(state="normal")
            self.text.delete("1.0", "end")
            self.text.insert("1.0", value)
            self.text.configure(state="disabled")
            self.text.yview_moveto(position)
            self.last = value

    def close(self):
        self.window.destroy()
        self.on_close()
