"""Paired PAD trial contracts; all camera input is synthetic and mocked."""
from contextlib import redirect_stderr, redirect_stdout
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tools.check_liveness_camera import CompareGate, main
from attendance_app.camera import FrameProcessor


def trained_model():
    return Mock(threshold=.5, threshold_source="validation",
                config={"training": {"smoke_test": False, "initialization": "random_from_scratch"}},
                predict_crop=Mock(return_value={"live_score": .9, "is_live": True}))


class CameraComparisonChecks(unittest.TestCase):
    def test_real_frame_processor_rejection_paths_do_not_score_or_crash(self):
        models = {"baseline": trained_model(), "candidate": trained_model()}
        gate = CompareGate(models)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        for boxes, expected in (([], "no_face"), ([(0, 0, 90, 90)]*2, "multiple_faces"),
                                ([(0, 0, 50, 50)], "face_too_small_or_outside"),
                                ([(0, 0, 100, 100)], "image_quality")):
            with self.subTest(status=expected):
                locator = SimpleNamespace(detect=lambda rgb: boxes)
                self.assertEqual(FrameProcessor(locator, gate).process(frame)["status"], expected)
        for model in models.values():
            model.predict_crop.assert_not_called()
        self.assertEqual(gate.scores, {"baseline": [], "candidate": []})

    def setUp(self):
        root = Path(__file__).resolve().parents[1]/"tmp"
        root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, output, condition="glasses"):
        return ["--baseline", "baseline.pt", "--candidate", "candidate.pt",
                "--condition", condition, "--output", str(output)]

    def test_invalid_request_and_existing_evidence_are_rejected_before_models_or_camera(self):
        existing = self.root/"existing.json"
        existing.write_text("historical evidence", encoding="utf-8")
        requests = [self.arguments(existing), self.arguments(self.root/"wrong.txt"),
                    self.arguments(self.root/"missing-presentation.json", "screen-replay"),
                    self.arguments(self.root/"wrong-presentation.json", "print")
                    + ["--presentation-type", "phone-still-photo"]]
        for extra in (["--seconds", "nan"], ["--seconds", "inf"], ["--seconds", "0"],
                      ["--seconds", "121"], ["--frames", "0"], ["--frames", "5001"],
                      ["--camera", "-1"], ["--camera", "100"]):
            requests.append(self.arguments(self.root/"new.json")+extra)
        with patch("tools.check_liveness_camera.LivenessDetector") as load, \
             patch("tools.check_liveness_camera.cv2.VideoCapture") as camera, redirect_stderr(io.StringIO()):
            for arguments in requests:
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    main(arguments)
        load.assert_not_called()
        camera.assert_not_called()
        self.assertEqual(existing.read_text(encoding="utf-8"), "historical evidence")

    def test_uncalibrated_or_incomplete_models_never_open_camera(self):
        for field, value in (("threshold", None), ("threshold_source", "explicit_override"),
                             ("smoke_test", True), ("initialization", "pretrained")):
            bad = trained_model()
            if field in ("threshold", "threshold_source"):
                setattr(bad, field, value)
            else:
                bad.config["training"][field] = value
            with self.subTest(field=field), \
                 patch("tools.check_liveness_camera.LivenessDetector", side_effect=[trained_model(), bad]), \
                 patch("tools.check_liveness_camera.cv2.VideoCapture") as camera, self.assertRaises(ValueError):
                main(self.arguments(self.root/"new.json"))
            camera.assert_not_called()

    def test_same_crop_reaches_both_models_without_modification(self):
        models = {"baseline": trained_model(), "candidate": trained_model()}
        crop = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
        before = crop.copy()
        gate = CompareGate(models)
        self.assertEqual(gate.analyze(crop), {"status": "scored"})
        for model in models.values():
            self.assertIs(model.predict_crop.call_args.args[0], crop)
        np.testing.assert_array_equal(crop, before)
        self.assertEqual(gate.passed, {"baseline": 1, "candidate": 1})

    def test_context_checkpoint_scores_wider_pixels_from_the_same_frame(self):
        models = {"baseline": trained_model(), "candidate": trained_model()}
        models["candidate"].config["context_crop"] = True
        gate = CompareGate(models)
        rng = np.random.default_rng(42)
        frame = rng.integers(0, 256, (240, 240, 3), dtype=np.uint8)
        locator = SimpleNamespace(detect=lambda rgb: [(60, 60, 80, 80)])
        result = FrameProcessor(locator, gate, pad_context=True).process(frame)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["boxes"], [[60, 60, 80, 80]])
        tight = models["baseline"].predict_crop.call_args.args[0]
        wide = models["candidate"].predict_crop.call_args.args[0]
        np.testing.assert_array_equal(np.asarray(tight), frame[60:140, 60:140])
        np.testing.assert_array_equal(np.asarray(wide), frame[20:180, 20:180])

    def test_declared_screen_presentation_is_saved_and_camera_released(self):
        for presentation in ("phone-still-photo", "recorded-video"):
            camera = Mock()
            camera.isOpened.return_value = True
            camera.read.return_value = (True, np.zeros((80, 100, 3), dtype=np.uint8))
            output = self.root/f"{presentation}.json"
            with self.subTest(presentation=presentation), \
                 patch("tools.check_liveness_camera.LivenessDetector", side_effect=[trained_model(), trained_model()]), \
                 patch("tools.check_liveness_camera.FaceLocator"), \
                 patch("tools.check_liveness_camera.FrameProcessor", side_effect=lambda locator, gate: SimpleNamespace(process=gate.analyze)), \
                 patch("tools.check_liveness_camera.cv2.VideoCapture", return_value=camera), \
                 patch("tools.check_liveness_camera.sha256", return_value="a"*64), redirect_stdout(io.StringIO()):
                main(self.arguments(output, "screen-replay")
                     + ["--presentation-type", presentation, "--frames", "1"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["presentation_type_supplied_by_operator"], presentation)
            self.assertTrue(report["started_utc"].endswith("+00:00"))
            self.assertTrue(report["finished_utc"].endswith("+00:00"))
            self.assertEqual(report["models"]["candidate"]["threshold_source"], "validation")
            self.assertEqual(report["models"]["candidate"]["scored_crops"], 1)
            self.assertFalse(report["images_or_video_saved"])
            camera.release.assert_called_once()

    def test_visible_setup_is_unscored_and_drawing_never_changes_camera_pixels(self):
        for cancellation in (None, "setup", "measurement"):
            models = [trained_model(), trained_model()]
            source = np.zeros((160, 240, 3), dtype=np.uint8)
            original = source.copy()
            camera = Mock()
            camera.isOpened.return_value = True
            camera.read.return_value = (True, source)
            clock = itertools.count()
            output = self.root/f"preview-{cancellation}.json"

            def processor(locator, gate):
                return SimpleNamespace(process=lambda rgb: {**gate.analyze(rgb), "boxes": [(20, 20, 90, 90)]})

            def key(delay):
                return ord("q") if cancellation == "setup" else (
                    27 if cancellation == "measurement" and models[0].predict_crop.called else -1)

            with self.subTest(cancellation=cancellation), \
                 patch("tools.check_liveness_camera.LivenessDetector", side_effect=models), \
                 patch("tools.check_liveness_camera.FaceLocator"), \
                 patch("tools.check_liveness_camera.FrameProcessor", side_effect=processor), \
                 patch("tools.check_liveness_camera.cv2.VideoCapture", return_value=camera) as capture, \
                 patch("tools.check_liveness_camera.cv2.namedWindow") as create, \
                 patch("tools.check_liveness_camera.cv2.resizeWindow") as resize, \
                 patch("tools.check_liveness_camera.cv2.imshow") as show, \
                 patch("tools.check_liveness_camera.cv2.waitKey", side_effect=key), \
                 patch("tools.check_liveness_camera.cv2.destroyWindow") as destroy, \
                 patch("tools.check_liveness_camera.time.monotonic", side_effect=lambda: next(clock)), \
                 patch("tools.check_liveness_camera.sha256", return_value="a"*64), redirect_stdout(io.StringIO()):
                main(self.arguments(output)+["--preview", "--frames", "1", "--seconds", "15"])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["preview_enabled"])
            self.assertEqual(report["setup_seconds"], 10.)
            self.assertFalse(report["setup_frames_scored"])
            self.assertEqual(report["cancelled_during"], cancellation)
            self.assertEqual(report["frame_count"], 0 if cancellation == "setup" else 1)
            self.assertEqual(models[0].predict_crop.call_count, report["frame_count"])
            self.assertEqual(models[1].predict_crop.call_count, report["frame_count"])
            np.testing.assert_array_equal(source, original)
            self.assertTrue(any(np.any(call.args[1] != original) for call in show.call_args_list))
            for call in show.call_args_list:
                self.assertFalse(np.shares_memory(call.args[1], source))
            capture.assert_called_once_with(0)
            create.assert_called_once()
            resize.assert_called_once_with("PAD model comparison", 960, 720)
            destroy.assert_called_once_with("PAD model comparison")
            camera.release.assert_called_once()

    def test_gui_cancellation_during_loading_never_opens_the_camera(self):
        marker, output = self.root/"loading.cancel", self.root/"loading.json"
        calls = []

        def load(path, device):
            calls.append(path)
            if len(calls) == 2:
                marker.touch()
            return trained_model()

        with patch("tools.check_liveness_camera.LivenessDetector", side_effect=load), \
             patch("tools.check_liveness_camera.FaceLocator"), \
             patch("tools.check_liveness_camera.cv2.VideoCapture") as camera, redirect_stdout(io.StringIO()):
            main(self.arguments(output)+["--cancel-file", str(marker)])
        camera.assert_not_called()
        self.assertFalse(output.exists())

    def test_immediate_gui_cancellation_is_normal_and_loads_no_models_or_camera(self):
        marker, output = self.root/"immediate.cancel", self.root/"immediate.json"
        marker.touch()
        with patch("tools.check_liveness_camera.LivenessDetector") as load, \
             patch("tools.check_liveness_camera.cv2.VideoCapture") as camera, redirect_stdout(io.StringIO()):
            main(self.arguments(output)+["--cancel-file", str(marker)])
        load.assert_not_called()
        camera.assert_not_called()
        self.assertFalse(output.exists())

    def test_gui_cancellation_saves_partial_counts_and_releases_the_camera(self):
        marker, output = self.root/"measurement.cancel", self.root/"measurement.json"
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((80, 100, 3), dtype=np.uint8))

        def processor(locator, gate):
            def observe(rgb):
                result = gate.analyze(rgb)
                marker.touch()
                return result
            return SimpleNamespace(process=observe)

        with patch("tools.check_liveness_camera.LivenessDetector", side_effect=[trained_model(), trained_model()]), \
             patch("tools.check_liveness_camera.FaceLocator"), \
             patch("tools.check_liveness_camera.FrameProcessor", side_effect=processor), \
             patch("tools.check_liveness_camera.cv2.VideoCapture", return_value=camera), \
             patch("tools.check_liveness_camera.sha256", return_value="a"*64), redirect_stdout(io.StringIO()):
            main(self.arguments(output)+["--cancel-file", str(marker), "--frames", "10"])
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertTrue(report["operator_cancelled"])
        self.assertEqual(report["cancelled_during"], "measurement")
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["models"]["baseline"]["scored_crops"], 1)
        self.assertEqual(report["models"]["candidate"]["scored_crops"], 1)
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
