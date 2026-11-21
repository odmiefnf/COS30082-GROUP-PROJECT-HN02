# Submission readiness

Reviewed on 21 November 2026 against the supplied six-page project brief. This is
a working checklist of evidence and outstanding tasks, not a predicted mark.
The group report identifies Group HN02 and the six supplied member names and student IDs.

**21 November update:** the app now uses three ImageNet-initialised, fine-tuned
models: ResNet-18 ArcFace for identity, MobileNetV3 trained on LCC-FASD
plus our own webcam recordings for liveness, and MobileNetV3 for expression. The
from-scratch models remain in `results/` as baselines. The group report was
rewritten for these models and rebuilt at 12 pages. Faces are located with
YuNet; up to 10 faces get name tags, and only the largest face is liveness
checked and recorded.

Earlier notes (the 9 and 14 November PAD experiments, the 1:N gallery study and
the phone-photo trials) remain in [pad_balanced_experiment.md](pad_balanced_experiment.md),
[pad_context_crop.md](pad_context_crop.md) and [identity_open_set.md](identity_open_set.md)
with their dated scope; their numbers belong to the earlier models.

## Requirement-to-evidence map

Paths below are relative to `grp_project`.

| Brief | Required outcome | Existing implementation and evidence |
| --- | --- | --- |
| 1, 2.1 | Train a CNN embedding model using classification and metric learning; compare both | `face_verification/train.py`, `model.py`. Four runs, each 10 epochs on 380,638 training images / 4,000 IDs: compact CNN from scratch (`results/classification`, `results/triplet`) and ImageNet ResNet-18 fine-tuned (`results/classification_resnet18`, `results/triplet_resnet18`), both objectives per backbone, plus ResNet-18 ArcFace as an additional approach (`results/arcface_resnet18`, the app model). |
| 2.2, 5 | ROC/AUC on `verification_pairs_val.txt` | `face_verification/evaluate.py`; `results/comparison_resnet18/comparison.json` compares all four on all 8,805 original pairs (4,315 same-person, 4,490 different-person). The earlier two-model comparison remains in `results/comparison`. |
| 2.3 | Research and justify the similarity metric; testing both common metrics is encouraged | Both cosine and Euclidean were evaluated. On unit embeddings they give equivalent rankings because squared Euclidean distance is `2 - 2*cosine`. The report explains the choice and cites embedding research. |
| 3 | Train and integrate an effective anti-spoofing model | `anti_spoofing/train.py`, `infer.py`; app model `results/liveness_mobilenetv3_webcam_20261120` (config in `results/liveness_app_config.json`), held-out webcam test in `..._on_test`, LCC-FASD test in `..._lcc_test`. Webcam data from `tools/capture_pad_data.py` (crop 1.4x, split by session). |
| 4, 6 | Train a public-data expression model and display real-time results | `emotion_recognition/train.py`, `infer.py`; app model `results/emotion_fer_mnv3`, test `results/emotion_fer_mnv3_test/metrics.json`; baseline `results/emotion_fer_test`. Seven labels with icons; expressions never authorize attendance. |
| 6 | Webcam recognition, registered name, unknown-person registration and immediate liveness/expression feedback | `attendance_app/web.py`, `web_ui.py`, `gui.py`, `camera.py`, `service.py`, `recognition.py`, `store.py`; guided 10-shot registration with per-shot liveness, appearance mode, profile deletion, multi-face name tags. The camera is started explicitly. |
| 7 | Relevant individual innovative feature, explained and demonstrated | Appearance templates, traceable manual review, profile/reference management, history/CSV, diagnostics, multi-face display and the webcam PAD data workflow exist. Each student must identify their own actual contribution; a feature is not automatically attributed to every member. |
| 8 | Group report, no more than 12 pages, including Methodology, Results and Discussion | `reports/COS30082_Group_Report.docx`, built by `tools/build_group_report.py` on 21 November 2026; Microsoft Word reports 12 pages and every page was reviewed visually. |
| 9 | One continuous 5-10 minute demonstration; comparable speaking time; speaker faces visible; accessible hosted link | `docs/09_video_checklist.md` gives a rehearsal plan. Recording, access check and submission remain group tasks. |
| 10 | Each student submits an individual report of at most 5 pages | Outline in `docs/interview_preparation.md`. Personal contribution statements are not filled in on anyone's behalf. |
| 11 | Source code, group report and video link submitted by one member; individual report and peer evaluation by each member | Final checklist below. The brief does not supply a numeric calendar due date; check Canvas. |
| 12 | Explain implementation and methods at the Week 12 tutorial interview | Questions and evidence pointers in `docs/interview_preparation.md`. Peer evaluation and interview performance can adjust marks. |

Fully pretrained models used without additional training are prohibited by the
brief; partial fine-tuning is allowed. The three app networks start from ImageNet
weights and are fine-tuned on the task data (ResNet-18: stem frozen; MobileNetV3:
first four blocks frozen); the from-scratch baselines are kept for comparison.
YuNet is a disclosed, pretrained face-localization component, not one of the
trained networks. Keep that distinction in the report and presentation.

The marking table's merged cells allocate marks as follows: face recognition 5,
anti-spoofing 5, emotion detection 5, and UI plus group report together 5
(group subtotal 20); innovation plus individual report together 10; interview
10. Total: 40. UI and report are not separate five-mark items. Better
presentation cannot guarantee a particular mark.

## Numbers to use accurately

| Item | Result and scope |
| --- | --- |
| Identity verification | Cosine AUC on the original pairs (`results/comparison_identity_20261121`): compact classification 0.825601, compact triplet 0.859969, ResNet-18 classification 0.913943, ResNet-18 triplet 0.914668, ResNet-18 ArcFace 0.914747. Cosine and negative-Euclidean AUC agree. |
| App identity operating point | ResNet-18 ArcFace, cosine threshold 0.509131; validation pair FPR 0.98% and TPR 32.98% (1,423/4,315). ResNet-18 classification gave TPR 26.63% and the first compact triplet 9.73%. About two thirds of same-identity pairs are still rejected. |
| Classification closed-set test | ArcFace 69.79%, ResNet-18 classification 60.00%, compact 39.46% on 8,000 images. This is not the same task as pair verification. |
| 1:N gallery test | 20 known identities, 300 outsiders: ArcFace finds 8/20 and misidentifies 25/300 at its threshold (classification 6/20 and 21/300); with thresholds calibrated to 1% outsider matches, 3/20 and 4/300 (classification 4/20 and 5/300). Small samples; all identities appeared in encoder training. |
| Identity overlap | 667 verification validation images match training pixels. Results follow the assignment pair protocol, but are not an independent verification test. |
| PAD on LCC-FASD test (7,522) | Scratch baseline ACER 26.05%; MobileNetV3 LCC-only ACER 5.72% (BPCER 10.16%); app model ACER 8.88% (APCER 1.35%, BPCER 16.41%). Thresholds chosen on validation only. |
| PAD on held-out webcam sessions | App model: 0/52 phone-photo and 0/116 phone-video crops accepted, 96/96 genuine passed. LCC-only model: 29/52, 67/116 and 91/96. One person, room and day; the same photo/video content appears in train and test sessions; frames are correlated. |
| Live incident | The LCC-only model scored a phone photo in a dark room at 0.85 (threshold 0.041) and recorded attendance; this motivated the webcam data. |
| Expression | Filtered FER2013 PrivateTest, 3,271 images: MobileNetV3 64.20% accuracy, macro-F1 0.6284, balanced accuracy 61.47%; EmotionCNN 58.36%, 0.4984, 50.91%. No webcam expression accuracy has been measured. |
| Latency (RTX 3060 laptop) | Median single calls: YuNet 25 ms on a 960x720 frame (CPU), liveness 13 ms, identity 6 ms, expression 13 ms (`results/app_update_20261120/pipeline_metrics.json`). Not end-to-end FPS. |

Frames within one session are correlated. Do not report them as independent people,
an attack-population error rate, deployment accuracy, or guaranteed FAR/FRR.
Use the final software-test log for the current test count; a passing test is not
a correctly recognized person.

## Portable run check

`results/identity_app_config.json` references `arcface_resnet18/best_model.pt`
relative to its own directory, with the checkpoint SHA256, threshold and pair-file
provenance. Keep the configuration and checkpoint in the same relative directory
structure when copying the app. The checkpoint must match the gallery fingerprint;
a database created with another encoder is refused, so register again after an
encoder change rather than exchanging encoders to make an error vanish.

On another Windows computer, with Python 3.11 available, run from the copied
project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-webcam.txt
.\.venv\Scripts\python.exe -m attendance_app.web --open --device cpu --database data/attendance/assessment_demo.sqlite3 --session COS30082-assessment-demo
```

`requirements.txt` includes torchvision 0.20.1, which all three app networks need.
That demo database starts empty and is separate from the user's existing gallery.
Registration is demonstrated with the participant's permission. A GPU is not
required to run the demo. Dataset downloads are needed to reproduce training,
not to run inference with the supplied trained checkpoints.

Before distributing a source package, include source, dependency files, the
needed trained checkpoints/configurations and readable evaluation evidence.
Keep the personal gallery, attendance SQLite files, webcam PAD recordings,
virtual environments, temporary files and bulk training downloads out of the
submission package. `tools/package_submission.py` built a 172 MB package with all
three app checkpoints on 21 November 2026. Packing files is not evidence that a
clean installation has succeeded: launch the copied application and check every
model component before recording.

## Final handover checklist

- [x] Add Group HN02, member names and student IDs to the group report.
- [ ] Confirm actual contributions with each member; map an implemented feature
  and relevant evidence to each person's explanation.
- [x] Rebuilt the report for the current models on 21 November 2026; Word reports 12 pages and all pages were reviewed.
- [x] Verify the package can load models on a copied path without a private
  gallery. On 21 November 2026 the 292-file, 172 MB package was extracted into a new
  folder; the web app loaded all three models and YuNet on CPU with an empty database,
  and 267 tests passed inside the copy (`results/readiness_20261121/package_check.json`), before the cleanup below.
  A clean `pip install` on another computer has not been tried.
- [x] Full software suite: 269 tests passed after the 21 November cleanup (old liveness experiments, unused results and the temporary LCC launcher removed; one obsolete package test dropped) and the confirmed-face lock (three new tests). See `results/readiness_20261121/full_tests.log`.
- [ ] Re-register profiles for the new identity encoder and try the app live:
  phone photo in a dark room, genuine face with and without glasses, several people in view.
- [ ] Complete independent spoof trials with more people, other phones and printed
  photos. The brief names printed images, which are untested: run
  `tools\record_print_attack.cmd` (stores crops in `data/pad_webcam_print`, outside
  training, and scores them with the app model).
- [ ] Rehearse and record the unedited 5-10 minute video, covering all project
  functions and each member's genuine contribution. Script: `docs/video_script.md`.
- [ ] Check the hosted link from an account that does not own the video.
- [ ] One member submits source, group report and the video link through Canvas.
- [ ] Each member submits their own report (at most 5 pages) and peer evaluation.
- [ ] Each member can explain the model, metric, evidence and limitations during
  the Week 12 tutorial interview.

No video has been recorded or submitted by this checklist, and no personal
contributions, final mark, or unperformed experiment is asserted.
