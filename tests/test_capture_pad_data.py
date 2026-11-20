"""Webcam PAD data collection: session-disjoint splits and crop saving, no camera."""
from contextlib import ExitStack, redirect_stdout
import csv
import io
from itertools import count
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from tools.capture_pad_data import COLUMNS, SaveGate, main, split_sessions


def rows(session, label, count=3):
    return [{"image_path": f"{session}/{i}.png", "label": label, "subject_id": session,
             "source_label": str(1 - int(label)), "subject_id_source": "provided"} for i in range(count)]


class CapturePadDataChecks(unittest.TestCase):
    def capture_with_key(self, root, key, setup_seconds, context=False, extra=()):
        checkerboard = (50 + 150 * (np.indices((160, 160)).sum(axis=0) % 2)).astype(np.uint8)
        frame = np.repeat(checkerboard[:, :, None], 3, axis=2)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        locator = Mock()
        locator.detect.return_value = [(30, 30, 100, 100)]
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(patch("cv2.VideoCapture", return_value=camera))
            stack.enter_context(patch("cv2.waitKey", return_value=key))
            stack.enter_context(patch("tools.capture_pad_data.time.monotonic", side_effect=count()))
            stack.enter_context(patch("attendance_app.camera.FaceLocator", return_value=locator))
            for method in ("namedWindow", "resizeWindow", "imshow"):
                stack.enter_context(patch(f"cv2.{method}"))
            destroy_windows = stack.enter_context(patch("cv2.destroyAllWindows"))
            args = ["--root", str(root), "capture", "--presentation", "live-person",
                    "--setup-seconds", str(setup_seconds), "--every", "1"]
            main(args + (["--context-crop"] if context else []) + list(extra))
        camera.release.assert_called_once()
        destroy_windows.assert_called_once()
        return output.getvalue(), locator

    def test_setup_cancel_saves_nothing_and_leaves_existing_manifest_untouched(self):
        for key, existing_manifest in ((ord("q"), False), (27, True)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = root / "manifest.csv"
                contents = b"previous,manifest,row\n"
                if existing_manifest:
                    manifest.write_bytes(contents)
                output, locator = self.capture_with_key(root, key, setup_seconds=5)
                self.assertIn("Capture cancelled during setup.", output)
                locator.detect.assert_not_called()
                self.assertEqual(list(root.iterdir()), [manifest] if existing_manifest else [])
                if existing_manifest:
                    self.assertEqual(manifest.read_bytes(), contents)

    def test_capture_stop_keeps_samples_and_manifest_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, _ = self.capture_with_key(root, ord("q"), setup_seconds=0)
            self.assertIn('"saved_crops": 1', output)
            with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                captured = list(csv.DictReader(handle))
            self.assertEqual(len(captured), 1)
            image_path = Path(captured[0]["image_path"])
            self.assertTrue(image_path.is_file())
            self.assertTrue((image_path.parent / "session.json").is_file())

    def test_context_capture_saves_wider_image_and_labels_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.capture_with_key(root, ord("q"), setup_seconds=0, context=True)
            with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                image_path = Path(next(csv.DictReader(handle))["image_path"])
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (160, 160))
            metadata = json.loads((image_path.parent / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["crop_policy"], "context-2x")

    def test_context_and_legacy_splits_keep_each_geometry_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, base = root / "manifest.csv", root / "base.csv"
            webcam = []
            for policy in ("tight", "context-2x"):
                for presentation, label in (("live-person", "1"), ("phone-still-photo", "0")):
                    for date in ("20261113", "20261114"):
                        session = f"cam_{presentation}_{date}_{100000 if policy == 'tight' else 110000}"
                        row = rows(session, label, count=1)[0]
                        directory = root / session
                        directory.mkdir()
                        if policy == "context-2x" or date == "20261113":
                            metadata = {"crop_policy": policy} if policy == "context-2x" else {}
                            (directory / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
                        image_path = directory / "crop.png"
                        row["image_path"] = str(image_path.relative_to(root) if label == "0" else image_path)
                        webcam.append(row)
            for path, entries in ((manifest, webcam), (base, rows("base_fixture", "1", count=1))):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(entries)
            original = manifest.read_bytes()
            with redirect_stdout(io.StringIO()):
                main(["--root", str(root), "split", "--base-train", str(base)])
                main(["--root", str(root), "split", "--context-crop", "--base-train", str(root / "missing.csv")])
            for directory, time_suffix, includes_base in ((root / "splits", "100000", True),
                    (root / "splits/context-2x", "110000", False)):
                for name, date in (("train.csv", "20261113"), ("val.csv", "20261114")):
                    with (directory / name).open(encoding="utf-8", newline="") as handle:
                        split_rows = list(csv.DictReader(handle))
                    actual = {row["subject_id"] for row in split_rows}
                    expected = {f"cam_{presentation}_{date}_{time_suffix}"
                                for presentation in ("live-person", "phone-still-photo")}
                    if includes_base and name == "train.csv":
                        expected.add("base_fixture")
                    self.assertEqual(actual, expected)
                    self.assertTrue(all(Path(row["image_path"]).is_absolute() for row in split_rows
                                        if row["subject_id"] != "base_fixture"))
            self.assertEqual(manifest.read_bytes(), original)

    def test_scaled_capture_saves_the_app_pad_crop_and_labels_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.capture_with_key(root, ord("q"), setup_seconds=0, extra=["--crop-scale", "1.4"])
            with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
                image_path = Path(next(csv.DictReader(handle))["image_path"])
            with Image.open(image_path) as image:
                self.assertEqual(image.size, (140, 140))  # 1.4 x the 100 px face box
            metadata = json.loads((image_path.parent / "session.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["crop_policy"], "scale-1.4x")

    def test_scaled_split_holds_out_test_and_val_sessions_and_adds_lcc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            webcam = []
            for presentation, label in (("live-person", "1"), ("phone-still-photo", "0")):
                for hour in ("10", "11", "12", "13"):
                    session = f"cam_{presentation}_20261121_{hour}0000"
                    directory = root / session
                    directory.mkdir()
                    (directory / "session.json").write_text(json.dumps({"crop_policy": "scale-1.4x"}), encoding="utf-8")
                    row = rows(session, label, count=1)[0]
                    row["image_path"] = str(directory / "crop.png")
                    webcam.append(row)
            excluded = root / "cam_phone-still-photo_20261121_140000"  # newest, but reviewer-excluded
            excluded.mkdir()
            (excluded / "session.json").write_text(json.dumps({"crop_policy": "scale-1.4x", "excluded": "real face"}),
                                                  encoding="utf-8")
            webcam += [{**rows(excluded.name, "0", count=1)[0], "image_path": str(excluded / "crop.png")}]
            legacy = root / "cam_live-person_20261113_090000"  # tight session from an older run
            legacy.mkdir()
            webcam += [{**rows(legacy.name, "1", count=1)[0], "image_path": str(legacy / "crop.png")}]
            for path, entries in ((root / "manifest.csv", webcam), (root / "base.csv", rows("lcc_train", "1", 1)),
                                  (root / "base_val.csv", rows("lcc_val", "0", 1))):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
                    writer.writeheader()
                    writer.writerows(entries)
            with redirect_stdout(io.StringIO()):
                main(["--root", str(root), "split", "--crop-scale", "1.4", "--base-train", str(root / "base.csv"),
                      "--base-val", str(root / "base_val.csv")])

            def sessions(name):
                with (root / "splits/scale-1.4x" / name).open(encoding="utf-8", newline="") as handle:
                    return {row["subject_id"] for row in csv.DictReader(handle)}
            both = ("live-person", "phone-still-photo")
            self.assertEqual(sessions("test.csv"), {f"cam_{p}_20261121_130000" for p in both})
            self.assertEqual(sessions("webcam_val.csv"), {f"cam_{p}_20261121_120000" for p in both})
            self.assertEqual(sessions("val.csv"), sessions("webcam_val.csv") | {"lcc_val"})
            self.assertEqual(sessions("train.csv"),
                             {f"cam_{p}_20261121_{h}0000" for p in both for h in ("10", "11")} | {"lcc_train"})

    def test_latest_session_per_presentation_is_held_out(self):
        data = (rows("cam_live-person_20261113_100000", "1") + rows("cam_live-person_20261113_110000", "1")
                + rows("cam_phone-still-photo_20261113_100500", "0")
                + rows("cam_phone-still-photo_20261113_120000", "0")
                + rows("cam_printed-photo_20261113_130000", "0"))
        train, val = split_sessions(data)
        self.assertEqual({r["subject_id"] for r in val},
                         {"cam_live-person_20261113_110000", "cam_phone-still-photo_20261113_120000"})
        self.assertFalse({r["subject_id"] for r in train} & {r["subject_id"] for r in val})
        self.assertIn("cam_printed-photo_20261113_130000", {r["subject_id"] for r in train})
        self.assertEqual(len(train) + len(val), len(data))

    def test_validation_without_both_classes_is_refused(self):
        data = rows("cam_live-person_20261113_100000", "1") + rows("cam_live-person_20261113_110000", "1") \
            + rows("cam_phone-still-photo_20261113_100500", "0")
        with self.assertRaises(ValueError):
            split_sessions(data)

    def test_gate_keeps_every_nth_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = SaveGate(Path(directory), every=2)
            for _ in range(5):
                gate.analyze(Image.new("RGB", (8, 8)))
            self.assertEqual(gate.accepted, 5)
            self.assertEqual(len(gate.saved), 3)
            self.assertTrue(all(path.is_file() for path in gate.saved))


if __name__ == "__main__":
    unittest.main()
