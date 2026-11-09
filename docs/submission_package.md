# Build the submission ZIP

Run from `grp_project` after the final report and readiness tests are complete:

```powershell
.\.venv-gpu\Scripts\python.exe -m tools.package_submission --output reports/COS30082_Submission.zip
```

The command refuses to replace an existing archive and fails if essential
source files, the final group report, runtime configurations, reported metrics,
six trained checkpoints or the readiness status/test log are missing. Choose a
new output filename when creating another version. It builds a temporary ZIP
beside the destination and publishes the completed file atomically without
overwriting another file. Windows uses rename; other platforms require a
filesystem supporting hard links.

The archive has one `COS30082_Face_Attendance/` folder containing:

- Python source under `anti_spoofing`, `attendance_app`, `emotion_recognition`,
  `face_verification`, `tools` and `tests`, plus `attendance_pipeline.py`.
- README, requirements, Windows launcher, Markdown documentation and the final
  English group report. Its figures are embedded in the DOCX.
- Best checkpoints for classification, triplet, baseline PAD, augmented PAD,
  camera-matched PAD and expression classification.
- Allowlisted configurations, histories, calibration, metrics, predictions and
  ROC files from the reported model directories; recorded integration evidence,
  two aggregate webcam trials and the fixed readiness/performance/layout folders.
- The isolated `liveness_regularized_20261108` candidate only when its
  `experiment_status.json` declares `status: completed`. Inclusion does not
  promote it to the default application model.
- `SUBMISSION_MANIFEST.json`, which records the byte count and SHA256 of every
  included file except the manifest itself. The command also prints the ZIP's
  byte count and SHA256.

Raw datasets, downloads, caches, virtual environments, temporary files, personal
gallery/attendance databases, backups, archived reports and credential files are
excluded. The result-directory allowlist is in `tools/package_submission.py`;
do not put private files in the fixed readiness/performance evidence folders.
Those folders permit only JSON, logs, CSV/text and generated PNG/SVG evidence.
There are no camera images or videos in the recorded evaluation evidence.

After extracting, create a Python environment, install the documented
requirements and run `python -m attendance_app.gui`. The personal gallery is
intentionally absent; register demo participants in a new local database. The
app can run with its included trained models without the training datasets.
Training and re-evaluating the original datasets require obtaining them through
their documented sources. Original experiment metadata may retain historical
machine paths; the active identity configuration must use its relative
checkpoint path.

The DOCX can be opened directly. Rebuilding/rendering the report uses the
authoring runtime described by its tools and is not required to run the app.
The ZIP does not replace each student's individual report, peer evaluation or
the required continuous video link, and this command does not submit to Canvas.
