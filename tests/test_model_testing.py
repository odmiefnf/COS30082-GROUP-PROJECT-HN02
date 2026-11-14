"""Model-test reports, process lifecycle and camera interlocks; no real camera."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from attendance_app.model_testing import ComparisonRun, export_report, load_report, report_rows


def comparison_report(condition="glasses", presentation="live-person"):
    return {
        "condition_supplied_by_operator": condition,
        "presentation_type_supplied_by_operator": presentation,
        "frame_count": 10,
        "frame_statuses": {"scored": 4, "no_face": 3, "multiple_faces": 2, "image_quality": 1},
        "started_utc": "2026-11-09T09:00:00+00:00", "operator_cancelled": False,
        "same_original_rgb_crop_for_both_models": True, "images_or_video_saved": False,
        "profiles_or_attendance_written": False, "thresholds_retuned_on_camera": False,
        "models": {
            name: {"checkpoint": f"{name}.pt", "sha256": character*64,
                   "threshold": threshold, "threshold_source": "validation",
                   "scored_crops": 4, "passed": passed, "not_passed": 4-passed}
            for name, character, threshold, passed in (("baseline", "a", .3, 3), ("candidate", "b", .7, 1))
        },
    }


class ReportChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_correct_decisions_follow_the_operator_scenario_and_only_scored_crops(self):
        for condition, presentation, expected in (("glasses", "live-person", (3, 1)),
                ("no-glasses", "live-person", (3, 1)), ("screen-replay", "phone-still-photo", (1, 3)),
                ("screen-replay", "recorded-video", (1, 3)), ("print", "printed-photo", (1, 3))):
            with self.subTest(condition=condition, presentation=presentation):
                rows = report_rows(comparison_report(condition, presentation))
                self.assertEqual(tuple(row["correct"] for row in rows), expected)
                self.assertEqual([row["scored_crops"] for row in rows], [4, 4])
                self.assertEqual([row["acceptance_rate"] for row in rows], [.75, .25])
                self.assertTrue(all(row["correct"]+row["incorrect"] == 4 for row in rows))

    def test_no_observations_have_no_acceptance_percentage_and_unknown_truth_is_not_guessed(self):
        report = comparison_report("unspecified", None)
        report.update(frame_count=0, frame_statuses={}, operator_cancelled=True)
        for model in report["models"].values():
            model.update(scored_crops=0, passed=0, not_passed=0)
        rows = report_rows(report)
        self.assertTrue(all(row["acceptance_rate"] is None and row["correct"] is None
                            and row["incorrect"] is None for row in rows))

    def test_inconsistent_or_unsafe_reports_are_rejected(self):
        cases = []
        for key, value in (("frame_count", True), ("frame_count", -1),
                           ("started_utc", "2026-11-09"), ("operator_cancelled", "false"),
                           ("thresholds_retuned_on_camera", True), ("images_or_video_saved", True),
                           ("profiles_or_attendance_written", True),
                           ("same_original_rgb_crop_for_both_models", False),
                           ("condition_supplied_by_operator", "invented"),
                           ("presentation_type_supplied_by_operator", "printed-photo"),
                           ("frame_statuses", {"scored": 4})):
            report = comparison_report()
            report[key] = value
            cases.append(report)
        for key, value in (("threshold", float("nan")), ("threshold", True), ("threshold", 1.1),
                           ("threshold_source", "camera"), ("sha256", "not-a-hash"),
                           ("checkpoint", ""), ("passed", 5), ("passed", True),
                           ("scored_crops", 3)):
            report = comparison_report()
            report["models"]["candidate"][key] = value
            cases.append(report)
        cases.extend(([], {"models": {}}, {**comparison_report(), "models": {"baseline": {}}}))
        for report in cases:
            with self.subTest(report=report), self.assertRaises(ValueError):
                report_rows(report)

    def test_same_frame_can_pair_distinct_crop_policies(self):
        report = comparison_report()
        report['same_original_rgb_crop_for_both_models'] = False
        report['same_original_rgb_frame_for_both_models'] = True
        report['models']['candidate']['crop_policy'] = 'context-2x'
        self.assertEqual([row['crop_policy'] for row in report_rows(report)], ['tight', 'context-2x'])
        report['same_original_rgb_frame_for_both_models'] = False
        with self.assertRaises(ValueError):
            report_rows(report)

    def test_json_roundtrip_preserves_evidence_and_exports_refuse_overwrite(self):
        report = comparison_report()
        path = self.root/"report.json"
        export_report(path, report)
        self.assertEqual(load_report(path), report)
        for suffix in (".json", ".csv"):
            target = self.root/("existing"+suffix)
            target.write_text("historical evidence", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_report(target, report)
            self.assertEqual(target.read_text(encoding="utf-8"), "historical evidence")

    def test_legacy_report_without_cancellation_metadata_is_not_labelled_uncancelled(self):
        report = comparison_report()
        del report["operator_cancelled"]
        source = self.root/"legacy.json"
        source.write_text(json.dumps(report), encoding="utf-8")
        self.assertEqual(load_report(source), report)
        target = self.root/"legacy.csv"
        export_report(target, report)
        with target.open(encoding="utf-8-sig", newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["operator_cancelled"], "not-recorded")

    def test_csv_is_spreadsheet_safe_and_keeps_scope_thresholds_and_cancellation(self):
        report = comparison_report("screen-replay", "phone-still-photo")
        report["operator_cancelled"] = True
        report["models"]["baseline"]["checkpoint"] = '=HYPERLINK("bad")'
        before = deepcopy(report)
        target = self.root/"trial.csv"
        export_report(target, report)
        with target.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["checkpoint"].startswith("'="))
        self.assertEqual((rows[0]["correct"], rows[0]["incorrect"]), ("1", "3"))
        self.assertEqual(rows[0]["threshold"], "0.3")
        self.assertEqual(rows[0]["operator_cancelled"], "True")
        self.assertIn("not population accuracy", rows[0]["scope"])
        self.assertEqual(report, before)

    def test_nonfinite_json_invalid_json_and_wrong_export_suffix_are_rejected(self):
        path = self.root/"bad.json"
        for content in ("{", "[]", json.dumps({**comparison_report(), "unused": float("nan")})):
            path.write_text(content, encoding="utf-8")
            with self.subTest(content=content[:25]), self.assertRaises(ValueError):
                load_report(path)
        with self.assertRaises(ValueError):
            export_report(self.root/"bad.txt", comparison_report())
        self.assertFalse((self.root/"bad.txt").exists())


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.baseline, self.candidate = self.root/"baseline.pt", self.root/"candidate.pt"
        self.baseline.write_bytes(b"test checkpoint placeholder")
        self.candidate.write_bytes(b"test checkpoint placeholder")

    def job(self, **changes):
        arguments = dict(project=self.root, baseline=self.baseline, candidate=self.candidate,
                         scenario="Phone still photo", camera=2, seconds=15.)
        arguments.update(changes)
        return ComparisonRun(**arguments)

    def test_invalid_setup_never_starts_a_process(self):
        with patch("attendance_app.model_testing.subprocess.Popen") as spawn:
            for changes in ({"candidate": self.root/"missing.pt"}, {"scenario": "made-up"},
                            {"camera": -1}, {"seconds": 0}, {"seconds": float("inf")}, {"seconds": 121}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    self.job(**changes)
        spawn.assert_not_called()

    def test_runner_uses_the_existing_tool_fixed_thresholds_and_unique_reports(self):
        process = Mock()
        process.poll.return_value = None
        job = self.job()
        with patch("attendance_app.model_testing.subprocess.Popen", return_value=process) as spawn:
            job.start()
        command = spawn.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "tools.check_liveness_camera"])
        self.assertEqual(command[command.index("--camera")+1], "2")
        self.assertEqual(command[command.index("--presentation-type")+1], "phone-still-photo")
        self.assertIn("--preview", command)
        self.assertNotIn("--threshold", command)
        self.assertEqual(spawn.call_args.kwargs["cwd"], self.root)
        self.assertNotEqual(job.output, self.job().output)
        self.assertIsNone(job.poll())
        job.cancel()
        self.assertTrue(job.cancel_file.is_file())
        self.assertTrue(job.cancel_requested)
        report = comparison_report("screen-replay", "phone-still-photo")
        report["operator_cancelled"] = True
        job.output.write_text(json.dumps(report), encoding="utf-8")
        process.poll.return_value = 0
        result, message = job.poll()
        self.assertEqual(result, report)
        self.assertIn("Partial trial", message)
        self.assertFalse(job.cancel_file.exists())
        self.assertIsNone(job.process)
        self.assertIsNone(job.log)

    def test_cancel_during_loading_and_process_failures_release_handles(self):
        for code in (0, 1):
            job, process = self.job(), Mock()
            process.poll.return_value = None
            with patch("attendance_app.model_testing.subprocess.Popen", return_value=process):
                job.start()
            job.cancel()
            log = job.log
            log.write("Traceback from fixture\nFixture model-load failure")
            process.poll.return_value = code
            if code:
                with self.assertRaisesRegex(RuntimeError, "^Fixture model-load failure$"):
                    job.poll()
            else:
                report, message = job.poll()
                self.assertIsNone(report)
                self.assertIn("before opening", message)
            self.assertTrue(log.closed)
            self.assertFalse(job.cancel_file.exists())
            self.assertIsNone(job.process)

    def test_spawn_failure_closes_log_and_creates_no_evidence(self):
        job = self.job()
        with patch("attendance_app.model_testing.subprocess.Popen", side_effect=OSError("Fixture launch error")), \
             self.assertRaises(OSError):
            job.start()
        self.assertIsNone(job.log)
        self.assertFalse(job.output.exists())


class DialogChecks(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)

    def test_model_dialog_locks_other_camera_owners_and_preview_never_captures(self):
        from attendance_app.gui import AttendanceWindow
        app = AttendanceWindow(self.root, SimpleNamespace(preview=True, camera=0,
            session="model-test-fixture", device="cpu", database=Path("unused-model-tests.sqlite3")))
        app.models_ready = True
        with patch("cv2.VideoCapture") as camera, patch("attendance_app.model_testing.subprocess.Popen") as spawn:
            app.open_model_tests()
            dialog = app.model_test_dialog
            dialog.window.withdraw()
            self.assertTrue(app.device_locked())
            self.assertEqual(str(app.start_button.cget("state")), "disabled")
            self.assertEqual(str(app.camera_selector.cget("state")), "disabled")
            app.start()
            app.test_camera()
            app.open_profiles()
            self.assertFalse(app.camera_starting)
            self.assertFalse(app.probe_busy)
            self.assertIsNone(app.profiles_dialog)
            dialog.candidate.set("not-read.pt")
            dialog.ready.set(True)
            dialog.start()
            self.assertIsNone(dialog.job)
            self.assertEqual(str(dialog.start_button.cget("state")), "disabled")
            dialog.close()
        camera.assert_not_called()
        spawn.assert_not_called()
        self.assertFalse(app.device_locked())
        self.assertEqual(str(app.start_button.cget("state")), "normal")

    def test_report_table_and_close_wait_for_the_camera_runner(self):
        from attendance_app.model_test_dialog import ModelTestDialog
        closed = Mock()
        dialog = ModelTestDialog(self.root, Path.cwd(), Path("baseline.pt"), 0, closed)
        dialog.window.withdraw()
        self.addCleanup(lambda: dialog.window.destroy() if dialog.window.winfo_exists() else None)
        report = comparison_report("screen-replay", "recorded-video")
        report["operator_cancelled"] = True
        dialog.show_report(report)
        first = dialog.tree.item(dialog.tree.get_children()[0], "values")
        self.assertEqual(first[2:7], ("4", "3", "1", "1", "3"))
        self.assertIn("Captured 10 · One face 5 · Scored 4", dialog.summary.get())
        self.assertIn("Partial trial", dialog.summary.get())
        self.assertEqual(str(dialog.csv_button.cget("state")), "normal")
        dialog.job = Mock(cancel_requested=False, output=Path("not-written.json"))
        dialog.job.poll.return_value = (None, "Cancelled before opening the camera.")
        job = dialog.job
        dialog.close()
        job.cancel.assert_called_once()
        closed.assert_not_called()
        self.assertFalse(dialog.closed)
        dialog.poll()
        closed.assert_called_once()
        self.assertTrue(dialog.closed)


if __name__ == "__main__":
    unittest.main()
