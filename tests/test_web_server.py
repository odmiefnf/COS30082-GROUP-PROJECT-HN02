"""Localhost HTTP web controller and route checks (no sockets, camera, models or DB)."""
import io
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from attendance_app.service import AttendanceService
from attendance_app.store import ProfileStore
from attendance_app.web import WebController, make_handler, render_csv_bytes


class _FakeSocket:
    def __init__(self, raw_request: bytes):
        self._in = io.BytesIO(raw_request)
        self._out = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self._in if "r" in mode else self._out

    def sendall(self, data: bytes):
        self._out.write(data)

    def close(self):
        pass


def dispatch(handler_cls, method: str, path: str, body: bytes = b""):
    headers = f"Host: 127.0.0.1\r\nContent-Length: {len(body)}\r\n"
    if body:
        headers += "Content-Type: application/json\r\n"
    raw = f"{method} {path} HTTP/1.1\r\n{headers}\r\n".encode("utf-8") + body
    sock = _FakeSocket(raw)
    handler_cls(sock, ("127.0.0.1", 0), None)
    head, _, payload = sock._out.getvalue().partition(b"\r\n\r\n")
    status = int(head.splitlines()[0].split()[1])
    return status, head.decode("latin-1"), payload


class LocalhostWebServerChecks(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(
            preview=True, camera=0, session="web-preview-session", device="cpu",
            detector="haar", database=Path("unused-web-preview.sqlite3"),
        )
        self.controller = WebController(args)
        self.handler = make_handler(self.controller)

    def tearDown(self):
        self.controller.close()
        self.assertIsNone(self.controller.worker)

    def test_root_html_and_preview_endpoints(self):
        status, _, body = dispatch(self.handler, "GET", "/")
        self.assertEqual(status, 200)
        html = body.decode("utf-8")
        self.assertIn("Face Attendance", html)
        self.assertIn("Live face detection", html)
        self.assertIn('id="page-register"', html)
        self.assertIn('data-page="register"', html)
        self.assertIn('id="registerForm"', html)
        self.assertIn('id="btnTakeShot"', html)
        self.assertIn('id="btnAutoCapture"', html)
        self.assertIn('id="shotsGrid"', html)
        self.assertIn('id="regDupBox"', html)
        for removed in ("cameraSelect", "btnTestCam", "cameraSummary"):
            self.assertNotIn(removed, html)
        html_ids = set(re.findall(r'id="([^"]+)"', html))
        script = html.split("<script>", 1)[1]
        for ref in re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', script):
            self.assertIn(ref, html_ids, f"JS references missing DOM id: {ref}")

        status, _, body = dispatch(self.handler, "GET", "/api/state")
        self.assertEqual(status, 200)
        state = json.loads(body.decode("utf-8"))
        self.assertTrue(state["preview"])
        self.assertFalse(state["running"])
        self.assertEqual(state["session"], "web-preview-session")
        self.assertEqual(len(state["components"]), 6)

        status, _, body = dispatch(self.handler, "GET", "/api/profiles")
        self.assertEqual(status, 200)
        profiles = json.loads(body.decode("utf-8"))
        self.assertFalse(profiles["editable"])
        self.assertEqual(len(profiles["profiles"]), 1)

        status, _, body = dispatch(self.handler, "GET", "/api/history")
        self.assertEqual(status, 200)
        hist = json.loads(body.decode("utf-8"))
        self.assertEqual(hist["summary"]["records"], 3)
        self.assertEqual(hist["summary"]["absent"], 1)
        self.assertEqual([p["name"] for p in hist["absent"]], ["Demo student C"])
        self.assertEqual(len(hist["spoof_events"]), 1)

        status, _, body = dispatch(self.handler, "POST", "/api/settings",
                                   b'{"mirror_preview":false,"show_hud":false,"stable_frames":4}')
        self.assertEqual(status, 200)
        updated = json.loads(body.decode("utf-8"))
        self.assertFalse(updated["mirror_preview"])
        self.assertFalse(updated["show_hud"])
        self.assertEqual(updated["stable_frames"], 4)

        status, _, _ = dispatch(self.handler, "POST", "/api/settings", b'{"stable_frames":99}')
        self.assertEqual(status, 400)

        status, _, body = dispatch(self.handler, "GET", "/api/history/export.csv")
        self.assertEqual(status, 200)
        csv_text = body.decode("utf-8-sig")
        self.assertIn("Demo student A", csv_text)
        self.assertIn("Manual", csv_text)

    def test_preview_refuses_camera_start_and_unknown_routes_return_404(self):
        status, _, body = dispatch(self.handler, "POST", "/api/camera/start",
                                   b'{"session":"s","camera":0}')
        self.assertEqual(status, 400)
        self.assertIn("Preview mode", json.loads(body.decode("utf-8"))["error"])

        status, _, _ = dispatch(self.handler, "GET", "/api/does-not-exist")
        self.assertEqual(status, 404)

        status, _, body = dispatch(self.handler, "GET", "/api/register/shots")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body.decode("utf-8"))["shots"], [])

        status, _, body = dispatch(self.handler, "POST", "/api/register/capture", b"{}")
        self.assertEqual(status, 400)
        self.assertIn("Preview mode", json.loads(body.decode("utf-8"))["error"])

    def test_profiles_list_is_read_only_but_not_empty_while_camera_runs(self):
        controller = self.controller
        controller.args.preview, controller.running = False, True
        sent = []
        row = {"id": "p1", "name": "Kiet", "created_utc": "2026-11-20T08:57:30+00:00", "sample_count": 6,
               "appearances": [{"id": "original:p1", "label": "Original", "sample_count": 6}]}

        def worker_reply(command, value=None):
            sent.append(command)
            controller._handle_event({"event": "profiles_snapshot", "profiles": [row]})

        controller._send = worker_reply
        status, _, body = dispatch(self.handler, "GET", "/api/profiles")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(sent, ["profiles_snapshot"])  # never the editing command while running
        self.assertFalse(data["editable"])
        self.assertEqual([p["name"] for p in data["profiles"]], ["Kiet"])
        self.assertIn("created_local", data["profiles"][0])
        controller.running = False

    def test_face_box_survives_slower_analysis_but_not_an_outdated_one(self):
        import time
        from attendance_app.video_stream import CameraSample
        controller = self.controller
        controller.running, controller.stream_id, controller.mirror_preview = True, 1, False
        rgb = np.zeros((120, 160, 3), dtype=np.uint8)
        now = time.monotonic()

        def drawn(analysis_age):
            controller.last_analysis = {"stream_id": 1, "sequence": 1, "rgb": rgb, "captured_at": now - analysis_age,
                                        "result": {"boxes": [(40, 20, 60, 60)], "liveness": None, "identity": None}}
            controller._handle_preview({"stream_id": 1, "sample": CameraSample(2, now, rgb)})
            frame = np.asarray(Image.open(io.BytesIO(controller.latest_jpeg)))
            return bool((frame[20:30, 40:50] > 60).any())  # top-left corner bracket

        self.assertTrue(drawn(0.3))   # ~100 ms analysis plus frame wait used to exceed the old 0.2 s limit
        self.assertFalse(drawn(0.6))  # an outdated result must not mark new pixels
        controller.running = False

    def test_each_face_gets_its_own_colour_and_recognized_faces_a_name_tag(self):
        import time
        from attendance_app.video_stream import CameraSample
        controller = self.controller
        controller.running, controller.stream_id, controller.mirror_preview = True, 1, False
        rgb = np.zeros((240, 400, 3), dtype=np.uint8)
        now = time.monotonic()
        faces = [{"box": [40, 80, 100, 100], "recognized": True, "name": "Kiet"},
                 {"box": [250, 80, 100, 100], "recognized": False, "name": None}]
        controller.last_analysis = {"stream_id": 1, "sequence": 1, "rgb": rgb, "captured_at": now,
                                    "result": {"boxes": [f["box"] for f in faces], "faces": faces}}
        controller._handle_preview({"stream_id": 1, "sample": CameraSample(2, now, rgb)})
        frame = np.asarray(Image.open(io.BytesIO(controller.latest_jpeg))).astype(int)
        green, yellow = frame[80:84, 40:60].mean(axis=(0, 1)), frame[80:84, 250:270].mean(axis=(0, 1))
        self.assertGreater(green[1], green[0] + 60)    # recognized: green corner
        self.assertGreater(yellow[0], yellow[2] + 100)  # not recognized: yellow corner
        self.assertGreater(frame[50:75, 45:90].mean(), 60)    # name tag above the recognized face
        self.assertLess(frame[50:75, 255:300].mean(), 20)     # no tag for the unknown face
        controller.running = False

    def test_csv_export_neutralizes_spreadsheet_formulas(self):
        csv_text = render_csv_bytes([{
            "session": "=SUM(1,2)", "person_id": "p1", "name": "@evil",
            "recorded_utc": "2026-11-20T02:00:00+00:00", "source": "automatic",
            "reviewer": None, "reason": None,
        }]).decode("utf-8-sig")
        self.assertIn("\"'=SUM(1,2)\"", csv_text)
        self.assertIn("\"'@evil\"", csv_text)


class _StubRecognizer:
    dimension = 4
    threshold = 0.85

    def __init__(self):
        self.vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def embed(self, crop):
        return self.vec / np.linalg.norm(self.vec)

    def match(self, embedding, profiles):
        best = None
        for p in profiles:
            score = max(float(np.dot(embedding, a["embedding"])) for a in p["appearances"])
            if best is None or score > best[0]:
                best = (score, p)
        if best is not None and best[0] >= self.threshold:
            return {"status": "known", "person_id": best[1]["id"], "name": best[1]["name"], "cosine": best[0]}
        return {"status": "unknown", "person_id": None, "name": None, "cosine": best[0] if best else 0.0}


class _StubLiveness:
    config = {}
    live = True

    def predict_crop(self, crop):
        score = .95 if self.live else .02
        return {"is_live": self.live, "live_score": score, "spoof_score": 1 - score, "threshold": 0.5,
                "status": "live" if self.live else "spoof"}


class _StubEmotion:
    def predict_crop(self, crop):
        return {"label": "neutral", "confidence": 0.9, "scores": {"neutral": 0.9}}


class GuidedEnrollmentChecks(unittest.TestCase):
    def test_guided_multi_angle_capture_retake_duplicate_and_appearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(Path(tmp) / "profiles.sqlite3", "a" * 64, 4)
            try:
                rec = _StubRecognizer()
                pad = _StubLiveness()
                svc = AttendanceService(store, rec, pad, _StubEmotion(), "2026-11-20")

                def shoot(seed, vec, live=True, slot=None):
                    # Capture straight from a frame and its one face box; no prior live observation.
                    rec.vec, pad.live = np.array(vec, dtype=np.float32), live
                    frame = np.full((64, 64, 3), seed, dtype=np.uint8)
                    return svc.capture_guided_shot(frame, (16, 16, 32, 32), slot=slot)

                # Capture needs no live observation; each shot records its own PAD result
                shots = shoot(10, [1.0, 0.0, 0.0, 0.0])
                self.assertEqual([(s["slot"], s["is_live"]) for s in shots], [(0, True)])

                # Reject exact same frame
                with self.assertRaises(ValueError):
                    shoot(10, [1.0, 0.0, 0.0, 0.0])

                # Reject different person stepping into frame
                with self.assertRaises(ValueError):
                    shoot(20, [0.0, 1.0, 0.0, 0.0])

                # A spoof is captured but marked; Person A fills the rest of the 10 poses
                shots = shoot(30, [0.98, 0.0, 0.0, 0.0], live=False)
                self.assertFalse(shots[1]["is_live"])
                for idx in range(2, 10):
                    shots = shoot(30 + idx, [0.98, 0.03 * (idx - 5), 0.0, 0.0])
                self.assertEqual([s["slot"] for s in shots], list(range(10)))

                # Reject 11th shot when full
                with self.assertRaises(ValueError):
                    shoot(99, [0.99, 0.01, 0.0, 0.0])

                # Remove slot 1 and retake -> fills slot 1 back
                svc.remove_guided_shot(slot=1)
                self.assertEqual(len(svc.guided_shots), 9)
                shots = shoot(50, [0.97, 0.2, 0.0, 0.0])
                self.assertEqual([s["slot"] for s in shots], list(range(10)))

                # Commit new profile (10 live shots)
                saved = svc.commit_guided_registration("Nguyen Van A · 104123456")
                self.assertEqual(saved["mode"], "new")
                self.assertEqual(saved["used_shots"], 10)
                self.assertEqual(len(svc.guided_shots), 0)
                self.assertEqual(len(store.profiles()), 1)

                # Fewer than three live shots cannot be saved, and spoof shots are never enrolled
                shoot(60, [1.0, 0.02, 0.0, 0.0])
                shoot(65, [0.99, 0.05, 0.0, 0.0], live=False)
                shoot(70, [0.99, 0.08, 0.0, 0.0])
                with self.assertRaises(ValueError):
                    svc.commit_guided_registration("Another Name")
                self.assertEqual(len(svc.guided_shots), 3)

                # A third live shot of the same person triggers duplicate detection without losing shots
                shoot(80, [0.99, -0.08, 0.0, 0.0])
                dup = svc.commit_guided_registration("Another Name")
                self.assertIn("duplicate", dup)
                self.assertEqual(dup["duplicate"]["name"], "Nguyen Van A · 104123456")
                self.assertEqual(len(svc.guided_shots), 4)

                # Save the live staged shots as a new appearance on the matched profile
                app = svc.commit_guided_registration(
                    appearance_person_id=dup["duplicate"]["person_id"],
                    appearance_label="Without glasses",
                )
                self.assertEqual(app["mode"], "appearance")
                self.assertEqual(app["label"], "Without glasses")
                self.assertEqual(app["used_shots"], 3)
                self.assertEqual(len(svc.guided_shots), 0)
                self.assertEqual(len(store.profiles()[0]["appearances"]), 2)

                # No cross-person check: the operator-confirmed profile gets the look even if
                # the shots score closer to someone else (the model often does for a new look)
                other = store.enroll("Tran Thi B", [[0.0, 0.0, 1.0, 0.0]] * 3)
                for seed, tilt in ((110, .01), (120, .03), (130, .05)):
                    shoot(seed, [1.0, tilt, 0.0, 0.0])
                app = svc.commit_guided_registration(appearance_person_id=other, appearance_label="With mask")
                self.assertEqual(app["name"], "Tran Thi B")
                self.assertEqual(len(next(p for p in store.profiles() if p["id"] == other)["appearances"]), 2)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
