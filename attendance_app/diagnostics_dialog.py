"""Read-only setup and camera status, with an explicit model setup retry."""
import tkinter as tk
from tkinter import ttk

from .diagnostics import COMPONENTS


class DiagnosticsDialog:
    def __init__(self, parent, retry, on_close, preview=False):
        self.on_close = on_close
        self.window = tk.Toplevel(parent)
        self.window.title("Diagnostics · PREVIEW DATA" if preview else "Diagnostics · COS30082")
        self.window.geometry("880x650")
        self.window.minsize(650, 480)
        self.window.configure(bg="#eef3f8")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        body = tk.Frame(self.window, bg="#eef3f8", padx=20, pady=18)
        body.pack(fill="both", expand=True)
        tk.Label(body, text="Camera and model diagnostics", bg="#eef3f8", fg="#172b45",
                 font=("Segoe UI", 15, "bold"), anchor="w").pack(fill="x", pady=(0, 10))
        actions = tk.Frame(body, bg="#eef3f8")
        actions.pack(side="bottom", fill="x", pady=(12, 0))
        self.retry = ttk.Button(actions, text="Retry model setup", command=retry)
        self.retry.pack(side="left")
        ttk.Button(actions, text="Close", command=self.close).pack(side="right")
        self.text = tk.Text(body, font=("Segoe UI", 11), wrap="word", bg="white", fg="#172b45",
                            padx=12, pady=12, relief="flat", state="disabled")
        scroll = ttk.Scrollbar(body, orient="vertical", command=self.text.yview)
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
