# Submission readiness

Reviewed on 8 November 2026 against the supplied six-page project brief. This is
a working checklist of evidence and outstanding tasks, not a predicted mark.
Group name, member names and student IDs remain blank as requested.

The 14 November update adds a separate [1:N gallery evaluation](identity_open_set.md)
and a [paired PAD crop study](pad_context_crop.md). The app threshold matched
2 of 20 known probes correctly and assigned 25 of 300 gallery outsiders to a
profile. The wide PAD crop lowered genuine rejection on reused LCC-FASD images
while accepting two additional attack crops. Fresh webcam trials for the wide
model and identities unseen during encoder training remain open.

**9 November PAD follow-up:** the class-balanced candidate improves validation
spoof rejection at the same genuine-rejection limit, but fails the predeclared
matched-point ACER target. An authorized paired glasses trial also rejected
more genuine crops with the candidate (56/86 passed versus 75/86 for the current
model). The new phone-photo trial accepted 134/134 crops with the current model
and 101/134 with the candidate. Neither demonstrates reliable phone-photo
protection. The candidate was not promoted; the default model is unchanged.
See [the experiment and paired comparison](pad_balanced_experiment.md).
The report/package and historical evidence below keep their dated scope.

## Requirement-to-evidence map

Paths below are relative to `grp_project`.

| Brief | Required outcome | Existing implementation and evidence |
| --- | --- | --- |
| 1, 2.1 | Train a CNN embedding model using classification and metric learning; compare both | `face_verification/train.py`, `model.py`; completed `results/classification` and `results/triplet` runs, each 10 epochs on 380,638 training images / 4,000 IDs. Both start from random weights. |
| 2.2, 5 | ROC/AUC on `verification_pairs_val.txt` | `face_verification/evaluate.py`; `results/comparison/comparison.json`, `comparison.csv`, pair scores and ROC artifacts. All 8,805 original pairs are retained: 4,315 same-person and 4,490 different-person. |
| 2.3 | Research and justify the similarity metric; testing both common metrics is encouraged | Both cosine and Euclidean were evaluated. On unit embeddings they give equivalent rankings because squared Euclidean distance is `2 - 2*cosine`. The report explains the choice and cites embedding research. |
| 3 | Train and integrate an effective anti-spoofing model | `anti_spoofing/train.py`, `infer.py`; historical baseline and current app checkpoints, validation calibration and `results/liveness_camera_matched/test_regression_metrics.json`. The measured phone-photo failure remains an important limitation. |
| 4, 6 | Train a public-data expression model and display real-time results | `emotion_recognition/train.py`, `infer.py`; FER2013 manifests and `results/emotion_fer_test/metrics.json`. Seven visible expression labels; expressions never authorize attendance. |
| 6 | Webcam recognition, registered name, unknown-person registration and immediate liveness/expression feedback | `attendance_app/gui.py`, `camera.py`, `service.py`, `recognition.py`, `store.py`; GUI, pipeline and capture regression tests. The camera is started explicitly. |
| 7 | Relevant individual innovative feature, explained and demonstrated | Appearance templates, traceable manual review, profile/reference management, history/CSV and diagnostics exist. Each student must identify their own actual contribution; a feature is not automatically attributed to every member. |
| 8 | Group report, no more than 12 pages, including Methodology, Results and Discussion | `reports/COS30082_Group_Report.docx` has 12 pages. `results/readiness_20261114/report_status.json` records the rendered 14 November version. Check pagination again after filling member details. |
| 9 | One continuous 5-10 minute demonstration; comparable speaking time; speaker faces visible; accessible hosted link | `docs/09_video_checklist.md` gives a rehearsal plan. Recording, access check and submission remain group tasks. |
| 10 | Each student submits an individual report of at most 5 pages | Outline in `docs/interview_preparation.md`. Personal contribution statements are not filled in on anyone's behalf. |
| 11 | Source code, group report and video link submitted by one member; individual report and peer evaluation by each member | Final checklist below. The brief does not supply a numeric calendar due date; check Canvas. |
| 12 | Explain implementation and methods at the Week 12 tutorial interview | Questions and evidence pointers in `docs/interview_preparation.md`. Peer evaluation and interview performance can adjust marks. |

Fully pretrained models used without additional training are prohibited by the
brief; partial fine-tuning is allowed. The current identity, PAD and expression
CNNs were trained from scratch. OpenCV Haar is a disclosed face-localization
component, not the learned identity verifier. Keep that distinction in the report
and presentation.

The marking table's merged cells allocate marks as follows: face recognition 5,
anti-spoofing 5, emotion detection 5, and UI plus group report together 5
(group subtotal 20); innovation plus individual report together 10; interview
10. Total: 40. UI and report are not separate five-mark items. Better
presentation cannot guarantee a particular mark.

## Numbers to use accurately

| Item | Result and scope |
| --- | --- |
| Identity verification | Classification AUC 0.825601; triplet AUC 0.859969 on the original verification validation pairs. Cosine and negative-Euclidean AUC agree. |
| Classification closed-set test | 39.4625% on 8,000 images. This is not the same task as pair verification. |
| App identity operating point | Cosine threshold 0.8695402108345607; validation pair FPR 0.9800% and TPR 9.7335%. The very low recall must be disclosed. |
| Identity overlap | 667 verification validation images match training pixels. Results follow the assignment pair protocol, but are not an independent verification test. |
| Current PAD offline regression | AUC 0.929084, APCER 19.10%, BPCER 11.33%, ACER 15.22% on 7,522 filtered, reused test predictions. Threshold was selected with validation, not that test or webcam trial. |
| Current PAD validation | APCER 33.94%, BPCER 13.58%, ACER 23.76%. Genuine rejection is worse than the historical baseline on validation; improvement is not universal. |
| Phone still photo | One operator-labelled trial: 237/237 scored crops passed PAD; all 237 identity probes were unknown; zero scratch attendance records. Identity, not PAD, prevented attendance in that trial. |
| Genuine with glasses | One trial: 232/243 PAD-scored crops passed; 175/232 identity probes matched the confirmed profile; one correct scratch attendance record. |
| Expression | Filtered FER2013 PrivateTest: 58.36% accuracy, macro-F1 0.4984, balanced accuracy 50.91%, 3,271 images. No webcam expression accuracy has been measured. |

Frames within one trial are correlated. Do not report them as independent people,
an attack-population error rate, deployment accuracy, or guaranteed FAR/FRR.
Use the final software-test log for the current test count; a passing test is not
a correctly recognized person.

## Portable run check

The identity configuration now references `triplet/best_model.pt` relative to its
own directory. Its checkpoint SHA256, threshold, validation scores and original
pair-file provenance are unchanged. Keep the configuration and checkpoint in the
same relative directory structure when copying the app. The checkpoint must
match the gallery fingerprint; do not exchange encoders to make an error vanish.

A copied-directory check loaded the real trained encoder and produced a finite,
unit-length 128-dimensional embedding from a synthetic RGB crop while running
from another working directory. Its SHA256 and threshold matched the originals;
no gallery or camera was accessed. This verifies checkpoint-path portability,
not a fresh dependency installation on a second physical computer.

On another Windows computer, with Python 3.11 available, run from the copied
project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-webcam.txt
.\.venv\Scripts\python.exe -m attendance_app.gui --device cpu --database data/attendance/assessment_demo.sqlite3 --session COS30082-assessment-demo
```

That demo database starts empty and is separate from the user's existing gallery.
Registration is demonstrated with the participant's permission. A GPU is not
required to run the demo. Dataset downloads are needed to reproduce training,
not to run inference with the supplied trained checkpoints.

Before distributing a source package, include source, dependency files, the
needed trained checkpoints/configurations and readable evaluation evidence.
Keep the personal gallery, attendance SQLite files, virtual environments,
temporary files and bulk training downloads out of the submission package.
Packing files is not evidence that a clean installation has succeeded: launch
the copied application and check every model component before recording.

## Final handover checklist

- [ ] Fill group/member names and student IDs in the group report.
- [ ] Confirm actual contributions with each member; map an implemented feature
  and relevant evidence to each person's explanation.
- [x] Rebuilt/rendered the 8 November report and reviewed all 12 pages. Recheck
  after adding member details.
- [ ] Verify the package can load models on a copied path without a private
  gallery, using the documented dependency setup.
- [x] Full software suite: 243 tests passed. See `results/readiness_20261114/full_tests.log`.
- [ ] Complete independent genuine/unknown-person and spoof trials when
  participants and materials are available. Current pending conditions include
  no-glasses, an unregistered live person, recorded video and printed photo.
- [ ] Rehearse and record the unedited 5-10 minute video, covering all project
  functions and each member's genuine contribution.
- [ ] Check the hosted link from an account that does not own the video.
- [ ] One member submits source, group report and the video link through Canvas.
- [ ] Each member submits their own report (at most 5 pages) and peer evaluation.
- [ ] Each member can explain the model, metric, evidence and limitations during
  the Week 12 tutorial interview.

No video has been recorded or submitted by this checklist, and no personal
contributions, final mark, or unperformed experiment is asserted.
