"""Real Tk layout regression checks; no camera, models or profile database."""
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
import unittest

from attendance_app.diagnostics import frame_guidance
from attendance_app.gui import AttendanceWindow, MESSAGES


class PreviewLayoutChecks(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.app = AttendanceWindow(self.root, SimpleNamespace(
            preview=True, camera=0, session="layout-regression", device="cpu",
            database=Path("unused-layout-database.sqlite3")))

    def tearDown(self):
        # The preview never creates a worker, opens a camera or reads a gallery.
        self.assertIsNone(self.app.worker)
        self.root.destroy()

    def settle(self):
        # Configure events update text wrapping. Repeated idle settlement
        # observes the resulting geometry, not just the first text request.
        for _ in range(4):
            self.root.update()
            self.root.update_idletasks()

    def canvas_geometry(self):
        canvas = self.app.canvas
        return (canvas.winfo_x(), canvas.winfo_y(), canvas.winfo_width(), canvas.winfo_height())

    def cycle_statuses(self, geometry):
        self.root.geometry(geometry)
        self.settle()
        expected = None
        # Alternate high-frequency outcomes first, then every other supported
        # message. A long valid profile name exercises the former stat-card
        # growth as recognition changes from known to unknown.
        statuses = ["known", "unknown", "no_face", "spoof", "known", "multiple_faces"] + list(MESSAGES)
        for index, status in enumerate(statuses):
            with self.subTest(window=geometry, status=status, index=index):
                self.app.feedback.set(MESSAGES[status])
                self.app.guidance.set(frame_guidance({"status": status}))
                self.app.identity_text.set(("Very long enrolled student name " * 3)[:80]
                    if status in ("known", "attendance_confirmed") else
                    "Not recognized" if status == "unknown" else "Not checked")
                self.app.live_text.set("Passed" if status in ("known", "unknown") else
                    "Not passed" if status == "spoof" else "Not checked")
                self.app.expression_text.set("Surprised · 100%" if status in ("known", "unknown") else "—")
                self.app.speed.set("Preview 30 FPS · Analysis 20 FPS" if index % 2 else "FPS not measured")
                self.settle()
                measured = self.canvas_geometry()
                self.assertGreater(measured[2], 100)
                self.assertGreater(measured[3], 100)
                if expected is None:
                    expected = measured
                self.assertEqual(measured, expected, "A result-text change resized or moved the camera viewport")
        return expected

    def test_camera_viewport_is_stable_while_status_text_changes(self):
        self.cycle_statuses("1200x790")
        self.cycle_statuses("1100x700")

    def test_real_window_resize_changes_viewport_but_new_statuses_do_not(self):
        initial = self.cycle_statuses("1100x700")
        expanded = self.cycle_statuses("1400x900")
        self.assertGreater(expanded[2], initial[2])
        self.assertGreater(expanded[3], initial[3])
        returned = self.cycle_statuses("1100x700")
        self.assertEqual(returned, initial)


if __name__ == "__main__":
    unittest.main()
