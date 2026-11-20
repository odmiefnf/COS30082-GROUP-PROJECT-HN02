"""Shared visual theme for the Tk attendance app: clean light minimalist + mint emerald.

Presentation only. Nothing here changes recognition, liveness, expression or
attendance decisions.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

FONT = "Segoe UI"

# Clean light minimalist + mint-emerald palette inspired by modern Face Verification UI.
BG = "#f1f3f7"            # soft cool light-grey window background
CARD = "#ffffff"          # pure crisp white card surface
SURFACE = "#f7f9fb"       # whisper-grey inset surface (inputs, alternating rows)
BORDER = "#e2e8f0"        # clean hairline border
BORDER_STRONG = "#cbd5e1" # input border
INK = "#111827"           # crisp near-black primary text
MUTED = "#6b7280"         # calm secondary text
SUBTLE = "#9ca3af"        # tertiary captions / hints
HEADER_BG = "#ffffff"     # clean white app bar
PILL_BG = "#f3f5f9"       # status pill background
MINT = "#12b896"          # signature mint-emerald accent
PRIMARY = "#12b896"       # primary button fill
PRIMARY_HOVER = "#0e9f81" # primary button hover
PRIMARY_SOFT = "#e6f9f4"  # soft mint badge / highlight fill
PRIMARY_INK = "#0b7a64"   # accessible emerald text on light surfaces
CANVAS = "#f7f9fb"        # clean light placeholder viewport
CANVAS_LIVE = "#111827"   # dark letterbox while live video streams
DISABLED_BG = "#edf1f5"
DISABLED_FG = "#a1acb9"

# tone -> (soft background, accent, text)
TONES = {
    "neutral": (SURFACE, "#cbd5e1", INK),
    "info": ("#eefbf7", MINT, "#0b6e5a"),
    "success": ("#ecfdf5", "#10b981", "#065f46"),
    "warning": ("#fffbeb", "#f59e0b", "#92400e"),
    "danger": ("#fef2f2", "#ef4444", "#991b1b"),
}

# variant -> (background, foreground, hover background, border)
BUTTON_VARIANTS = {
    "primary": (PRIMARY, "#ffffff", PRIMARY_HOVER, PRIMARY),
    "secondary": ("#f1f5f9", INK, "#e5eaf0", "#dbe2ea"),
    "danger": ("#fef2f2", "#b91c1c", "#fee2e2", "#fecaca"),
    "ghost": (CARD, PRIMARY_INK, PRIMARY_SOFT, CARD),
}


class FlatButton(tk.Button):
    """A clean flat button whose colours follow its state (normal/disabled) and hover."""

    def __init__(self, parent, text, command, variant="secondary", size=10, **kwargs):
        self._colors = BUTTON_VARIANTS[variant]
        background, foreground, hover, border = self._colors
        options = dict(text=text, command=command, bg=background, fg=foreground,
                       activebackground=hover, activeforeground=foreground,
                       disabledforeground=DISABLED_FG, relief="flat", bd=0,
                       highlightthickness=1, highlightbackground=border, highlightcolor=MINT,
                       font=(FONT, size, "bold"), padx=14, pady=7, cursor="hand2")
        options.update(kwargs)
        super().__init__(parent, **options)
        self.bind("<Enter>", lambda event: self._paint(hover=True), add="+")
        self.bind("<Leave>", lambda event: self._paint(hover=False), add="+")

    def configure(self, cnf=None, **kwargs):
        result = super().configure(cnf, **kwargs)
        if "state" in kwargs or (isinstance(cnf, dict) and "state" in cnf):
            self._paint()
        return result

    config = configure

    def _paint(self, hover=False):
        background, foreground, hover_background, border = self._colors
        if str(self.cget("state")) == "disabled":
            super().configure(bg=DISABLED_BG, highlightbackground=BORDER, cursor="arrow")
        else:
            super().configure(bg=hover_background if hover else background, fg=foreground,
                              highlightbackground=hover_background if hover and border != CARD else border,
                              cursor="hand2")


def card(parent, padx=16, pady=14, bg=CARD):
    """Pure white surface with a delicate hairline border."""
    return tk.Frame(parent, bg=bg, padx=padx, pady=pady, highlightthickness=1,
                    highlightbackground=BORDER, highlightcolor=BORDER)


def entry(parent, variable, size=11, width=None, bg=CARD):
    widget = tk.Entry(parent, textvariable=variable, font=(FONT, size), relief="flat", bd=0,
                      bg=bg, fg=INK, insertbackground=MINT, selectbackground=PRIMARY_SOFT,
                      selectforeground=INK, highlightthickness=1,
                      highlightbackground=BORDER_STRONG, highlightcolor=MINT,
                      disabledbackground=SURFACE, disabledforeground=SUBTLE)
    if width is not None:
        widget.configure(width=width)
    return widget


def apply_style(root):
    """Configure clean light mint-emerald ttk widgets once per Tk interpreter."""
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=(FONT, 10), background=BG, foreground=INK)
    style.configure("TButton", background="#f1f5f9", foreground=INK, bordercolor="#dbe2ea",
                    lightcolor="#f1f5f9", darkcolor="#f1f5f9", focuscolor="#f1f5f9",
                    relief="flat", padding=(12, 6), font=(FONT, 10, "bold"))
    style.map("TButton",
              background=[("disabled", DISABLED_BG), ("pressed", "#e2e8f0"), ("active", "#e5eaf0")],
              lightcolor=[("disabled", DISABLED_BG), ("active", "#e5eaf0")],
              darkcolor=[("disabled", DISABLED_BG), ("active", "#e5eaf0")],
              foreground=[("disabled", DISABLED_FG)],
              bordercolor=[("focus", MINT)])
    style.configure("Accent.TButton", background=PRIMARY, foreground="#ffffff", lightcolor=PRIMARY,
                    darkcolor=PRIMARY, bordercolor=PRIMARY, focuscolor=PRIMARY)
    style.map("Accent.TButton",
              background=[("disabled", DISABLED_BG), ("pressed", PRIMARY_HOVER), ("active", PRIMARY_HOVER)],
              lightcolor=[("disabled", DISABLED_BG), ("active", PRIMARY_HOVER)],
              darkcolor=[("disabled", DISABLED_BG), ("active", PRIMARY_HOVER)],
              foreground=[("disabled", DISABLED_FG)])
    style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=INK,
                    bordercolor=BORDER_STRONG, lightcolor=CARD, darkcolor=CARD, arrowcolor=MUTED,
                    selectbackground=CARD, selectforeground=INK, padding=(6, 4))
    style.map("TCombobox",
              fieldbackground=[("disabled", SURFACE), ("readonly", CARD)],
              foreground=[("disabled", SUBTLE)],
              bordercolor=[("focus", MINT)],
              arrowcolor=[("disabled", SUBTLE)])
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))
    root.option_add("*TCombobox*Listbox.background", CARD)
    root.option_add("*TCombobox*Listbox.foreground", INK)
    root.option_add("*TCombobox*Listbox.selectBackground", PRIMARY_SOFT)
    root.option_add("*TCombobox*Listbox.selectForeground", PRIMARY_INK)
    style.configure("Treeview", background=CARD, foreground=INK, fieldbackground=CARD, rowheight=32,
                    borderwidth=0, relief="flat", font=(FONT, 10), bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.map("Treeview", background=[("selected", PRIMARY_SOFT)], foreground=[("selected", INK)])
    style.configure("Treeview.Heading", font=(FONT, 9, "bold"), background=SURFACE, foreground=MUTED,
                    relief="flat", bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE, padding=(8, 6))
    style.map("Treeview.Heading", background=[("active", "#eef2f6")])
    style.configure("TScrollbar", background="#cbd5e1", troughcolor=SURFACE, bordercolor=SURFACE,
                    lightcolor="#cbd5e1", darkcolor="#cbd5e1", arrowcolor=MUTED, gripcount=0)
    style.map("TScrollbar",
              background=[("disabled", "#e5eaf0"), ("pressed", MINT), ("active", "#94a3b8")],
              lightcolor=[("disabled", "#e5eaf0"), ("active", "#94a3b8")],
              darkcolor=[("disabled", "#e5eaf0"), ("active", "#94a3b8")],
              arrowcolor=[("disabled", "#cbd5e1"), ("active", INK)])
    style.configure("Vertical.TScrollbar", background="#cbd5e1", troughcolor=SURFACE, bordercolor=SURFACE,
                    lightcolor="#cbd5e1", darkcolor="#cbd5e1", arrowcolor=MUTED, gripcount=0)
    style.configure("Horizontal.TScrollbar", background="#cbd5e1", troughcolor=SURFACE, bordercolor=SURFACE,
                    lightcolor="#cbd5e1", darkcolor="#cbd5e1", arrowcolor=MUTED, gripcount=0)
    style.configure("Horizontal.TProgressbar", background=MINT, troughcolor="#e5eaf0", bordercolor=BORDER,
                    lightcolor=MINT, darkcolor=MINT, thickness=8)
    style.configure("TEntry", fieldbackground=CARD, foreground=INK, bordercolor=BORDER_STRONG,
                    lightcolor=CARD, darkcolor=CARD, insertcolor=MINT, padding=(6, 4))
    style.map("TEntry", fieldbackground=[("readonly", SURFACE), ("disabled", SURFACE)],
              foreground=[("readonly", MUTED), ("disabled", SUBTLE)], bordercolor=[("focus", MINT)])
    style.configure("TSpinbox", fieldbackground=CARD, background=CARD, foreground=INK,
                    bordercolor=BORDER_STRONG, lightcolor=CARD, darkcolor=CARD, arrowcolor=MUTED, padding=(6, 4))
    style.configure("TCheckbutton", background=BG, foreground=INK)
    return style
