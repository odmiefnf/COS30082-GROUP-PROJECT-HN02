"""Build a submission ZIP from an explicit source/model/evidence allowlist."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "COS30082_Face_Attendance"
SOURCE_DIRS = ("anti_spoofing", "attendance_app", "emotion_recognition", "face_verification", "tools", "tests")
MODEL_DIRS = ("classification", "triplet", "liveness_lcc", "liveness_camera_aug", "liveness_camera_matched", "emotion_fer")
REQUIRED = (
    "attendance_pipeline.py", "attendance_app/gui.py", "attendance_app/service.py",
    "attendance_app/store.py", "attendance_app/video_stream.py", "face_verification/train.py",
    "anti_spoofing/train.py", "emotion_recognition/train.py", "README.md",
    "requirements.txt", "requirements-webcam.txt", "start_attendance.cmd",
    "reports/COS30082_Group_Report.docx", "results/identity_app_config.json",
    "results/liveness_app_config.json", "results/comparison/comparison.json",
    "results/liveness_lcc_test/metrics.json", "results/liveness_camera_matched/calibration.json",
    "results/liveness_camera_matched/test_regression_metrics.json", "results/emotion_fer_test/metrics.json",
    "results/readiness_20261108/status.json", "results/readiness_20261108/full_tests.log",
) + tuple(f"results/{name}/best_model.pt" for name in MODEL_DIRS)
RESULT_DIRS = (
    *MODEL_DIRS, "comparison", "liveness_lcc_test", "liveness_camera_aug_test", "emotion_fer_test",
    "app_upgrade_20261030",
)
RESULT_NAMES = {
    "config.json", "history.csv", "train_history.csv", "calibration.json", "classification_test.json",
    "training.log", "evaluation.log", "experiment_status.json", "matched_calibration.log",
    "test_regression_metrics.json", "attendance_integration.json", "attendance_integration.log",
    "comparison.json", "comparison.csv", "metadata.json", "pair_scores.csv", "roc_curve.svg",
    "roc_points.csv", "metrics.json", "predictions.csv", "confusion_matrix.csv", "inference_check.json",
    "inference_examples.json", "inference_consistency.json", "cli_example.json",
    "reference_import_integration.json", "phase5_report_status.json", "phase5_full_tests.log",
    *(f"phase{phase}_status.json" for phase in range(1, 6)),
    *(f"phase{phase}_full_tests.log" for phase in range(2, 5)),
}
ROOT_RESULTS = (
    "identity_app_config.json", "liveness_app_config.json", "identity_experiment_summary.json",
    "attendance_integration.json", "manual_review_integration.json", "gpu_environment.txt",
    "section6_status.json", "section7_status.json", "section8_status.json",
)
EVIDENCE_FILES = (
    *(f"app_performance_20261106/{name}" for name in (
        "baseline_cpu.json", "baseline_cpu_live.json", "cuda_4threads_live.json",
        "full_tests.log", "full_tests_final.log", "gui_latest_cpu.json",
        "latest_cpu_live.json", "model_device_synthetic.json",
    )),
    *(f"preview_layout_20261106/{name}" for name in (
        "layout_geometry.json", "layout_tests.log", "layout_tests_final.log",
        "regression_tests.log", "single_view_tests.log",
    )),
    *(f"readiness_20261108/{name}" for name in (
        "attendance_integration.json", "full_tests.log", "layout_geometry.json",
        "manual_review_integration.json", "model_startup.json", "package_validation.json",
        "pad_training.log", "relocation_check.json", "report_build.log", "report_render.log",
        "report_render_final.log", "report_status.json", "status.json",
    )),
    "liveness_balanced_20261109/experiment_plan.json",
    "liveness_balanced_20261109/experiment_status.json",
    "pad_webcam_20261109/final_model_decision.json",
    "model_tests_feature_1826ff45/model_testing_final.log",
    "identity_open_set_20261114/metrics.json",
    "pad_context_20261114/metrics.json",
    "readiness_20261114/full_tests.log",
    "readiness_20261114/model_startup.json",
    "readiness_20261114/report_status.json",
)
CANDIDATE = "results/liveness_regularized_20261108"
SKIP_PARTS = {"__pycache__", "backup", "backups", "archive", "archives"}


def submission_files(root):
    root = Path(root).resolve()
    missing = [name for name in REQUIRED if not (root/name).is_file() or not (root/name).stat().st_size]
    if missing:
        raise FileNotFoundError("Missing submission essentials: " + ", ".join(missing))
    selected = {root/name for name in REQUIRED}
    for directory in SOURCE_DIRS:
        selected.update(path for path in (root/directory).rglob("*.py")
                        if not SKIP_PARTS.intersection(path.relative_to(root).parts))
    selected.update((root/"docs").glob("*.md"))
    selected.update(root/"results"/name for name in ROOT_RESULTS if (root/"results"/name).is_file())
    selected.update(root/"results"/name for name in EVIDENCE_FILES if (root/"results"/name).is_file())
    for directory in RESULT_DIRS:
        for path in (root/"results"/directory).rglob("*"):
            relative = path.relative_to(root/"results"/directory)
            if not path.is_file() or SKIP_PARTS.intersection(relative.parts):
                continue
            if directory != "comparison" and len(relative.parts) != 1:
                continue
            if path.name in RESULT_NAMES:
                selected.add(path)
    trials = root/"results/app_upgrade_20261030/evaluation/trials"
    for name in ("glasses_01.json", "phone_photo_01.json"):
        if (trials/name).is_file():
            selected.add(trials/name)
    candidate = root/CANDIDATE
    if (candidate/"experiment_status.json").is_file():
        status = json.loads((candidate/"experiment_status.json").read_text(encoding="utf-8"))
        if status.get("status") == "completed":
            selected.update(candidate/name for name in (
                "best_model.pt", "config.json", "calibration.json", "history.csv",
                "experiment_plan.json", "experiment_status.json") if (candidate/name).is_file())
    for path in selected:
        # Refuse links/junctions rather than copying an unexpected target's data.
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Linked files cannot be packaged: {path.relative_to(root)}")
    return sorted(selected)


def build_package(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.suffix.lower() != ".zip":
        raise ValueError("The submission output must be a ZIP file.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    files = submission_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name+".", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "archive_prefix": PREFIX,
                "scope": "Source, trained checkpoints, report and allowlisted evaluation evidence; no datasets or personal gallery",
                "files": {}}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                relative = path.relative_to(root).as_posix()
                digest, size = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(f"{PREFIX}/{relative}", "w") as destination:
                    for block in iter(lambda: source.read(1024*1024), b""):
                        destination.write(block)
                        digest.update(block)
                        size += len(block)
                manifest["files"][relative] = {"bytes": size, "sha256": digest.hexdigest()}
            archive.writestr(f"{PREFIX}/SUBMISSION_MANIFEST.json", json.dumps(manifest, indent=2)+"\n")
        # Windows rename refuses existing targets; POSIX rename would overwrite.
        if os.name == "nt":
            temporary.rename(output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with output.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return {"output": str(output), "files": len(files), "bytes": output.stat().st_size,
            "sha256": digest.hexdigest(), "archive_prefix": PREFIX}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_package(ROOT, args.output), indent=2))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Submission package was not created: {exc}\n")


if __name__ == "__main__":
    main()
