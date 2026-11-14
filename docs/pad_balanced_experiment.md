# Class-balanced PAD candidate — 9 November 2026

The offline training change is complete. The new model improves validation
spoof rejection at the current genuine-rejection limit, but does **not** meet
every predeclared criterion. The subsequent paired glasses and phone-photo
trials also do not support promotion. The candidate is retained as a failed
experiment; the app default is unchanged. No webcam was opened during offline
training. The authorized trials below reproduced the current model's
phone-photo failure, which remains unresolved.

## Controlled training change

The existing LivenessCNN was trained from random weights for 10 complete epochs
on 8,093 LCC-FASD training images: 6,900 spoof and 1,193 live. The only change
from the previous camera-augmentation training recipe is inverse-frequency
training cross entropy, with weights `N / (2 * class_count)`. This prevents the
majority spoof class from dominating the training objective. It is a training
choice, not a guarantee of generalization or calibrated probabilities.

Architecture, RGB 112 by 112 preprocessing, batch 64, Adam learning rate 0.0005,
weight decay 0.0001, seed 42 and train-only camera augmentation are unchanged.
Validation loss is unweighted. Counts and weights come only from the training
manifest. Validation uses all 2,948 images; no test or webcam images were used
for training, epoch selection or threshold calibration.

The plan was saved before training in
`results/liveness_balanced_20261109/experiment_plan.json`. Epoch 10 was selected
by minimum validation ACER, with unweighted validation loss breaking ties.

## Two explicitly different operating points

At its minimum-ACER threshold, `0.8448503820911647`, the candidate achieves validation
ACER **18.14%**, APCER **11.09%** and BPCER **25.19%**. The previous augmented
model's minimum ACER was 19.34%, with BPCER 21.73%. The candidate's lower ACER
therefore comes with more genuine rejections at this point. This is not the
point prepared for the glasses comparison.

The second checkpoint contains the **same weights**, calibrated on validation
to minimize spoof acceptance while keeping BPCER at most the current app's
13.5802%. Its fixed threshold is `0.18372472011345659`.

| Same validation protocol | Current app | Matched candidate |
| --- | ---: | ---: |
| Genuine rejected / 405 (BPCER) | 55 / 405 (13.58%) | 55 / 405 (13.58%) |
| Spoofs accepted / 2,543 (APCER) | 863 / 2,543 (33.94%) | 759 / 2,543 (29.85%) |
| ACER | 23.76% | 21.71% |
| AUC | 0.874515 | 0.883648 |

This is 104 fewer accepted spoof images at the same validation genuine-rejection
limit. These are development results on reused validation data, not independent
webcam accuracy or a measured improvement for eyeglasses specifically.

The predeclared matched-point requirements were APCER below 33.9363%, BPCER at
most 13.5802%, and ACER below 19.3384%, plus a fresh camera acceptance check.
The **ACER requirement failed** at the matched point; the subsequent camera
trials also do not meet the acceptance requirement. Metrics from the two
thresholds must not be combined to claim a pass.
The original plan is retained unchanged and the app default remains
`results/liveness_camera_matched/best_model.pt`.

## Locked-threshold regression and software checks

After calibration was saved, the matched checkpoint was evaluated once on the
same filtered 7,522-image test set used in earlier development. This is a
**reused-test regression**, not a new independent test. Neither weights nor
threshold were retuned from its results. Evaluation subject IDs are unavailable,
so subject separation cannot be verified.

| Reused test set | Current app | Matched candidate |
| --- | ---: | ---: |
| Spoofs accepted (APCER) | 19.10% | 17.56% |
| Genuine rejected (BPCER) | 11.33% | 15.23% |
| ACER | 15.22% | 16.40% |
| AUC | 0.929084 | 0.912709 |

The candidate accepts fewer spoof samples, but rejects more genuine samples and
has worse ACER/AUC on this reused set. This is a regression, not an overall win,
and is another reason to keep the existing default. The camera trials below
probe the real camera/material gap; they do not hide the failed offline
criterion or establish protection against all attacks.

The 207-test suite passed, followed by all six paired-camera tests after the
framing-preview and no-face-path fixes (213 distinct tests across the runs).
Camera inputs in these tests were
mocked. The actual candidate also ran through the three-model attendance service
on 24 dataset photos using a temporary database: no real camera, personal
gallery or production attendance was accessed. That check confirms integration,
not recognition accuracy. Test logs, hashes and decision are retained with the
experiment evidence.

## Reproduction

Run from the project directory with the installed environment and local data.
Always choose new output directories to preserve evidence.

```powershell
.\.venv-gpu\Scripts\python.exe -m anti_spoofing.train --train-manifest data/lcc_fasd_manifests/train.csv --val-manifest data/lcc_fasd_manifests/val.csv --output-dir results/liveness_balanced_reproduction --crop-mode full --epochs 10 --batch-size 64 --learning-rate 0.0005 --weight-decay 0.0001 --camera-augmentation --balanced-loss --checkpoint-selection val-acer --device cuda --num-threads 4 --seed 42 --cache-images
.\.venv-gpu\Scripts\python.exe -m tools.calibrate_liveness_camera --baseline results/liveness_camera_matched/best_model.pt --candidate results/liveness_balanced_reproduction/best_model.pt --validation data/lcc_fasd_manifests/val.csv --output-dir results/liveness_balanced_matched_reproduction --match-rate bpcer
```

`tools/calibrate_liveness_camera.py` retains its original APCER-matching default.
The explicit BPCER mode supports the security comparison without increasing the
validation genuine-rejection cap. Both modes respect tied scores and threshold
endpoints. Calibration refuses an output containing existing model/evidence.

## First authorized webcam comparison and next trials

The participant confirmed readiness while wearing glasses. The first completed
paired trial lasted 15.015 seconds and captured 447 frames: 437 had no detected
face, one had multiple face detections and nine were scored. On exactly those
nine crops, the current model passed 9/9 and the matched candidate passed 3/9.
No images/video were saved, no gallery/attendance was used and thresholds were
unchanged. The immutable report is
`results/pad_webcam_20261109/glasses_01.json`.

This is too little usable face coverage to assess stability or population
genuine rejection. It does show six candidate rejections on crops the baseline
accepted. The model remains unpromoted. The repeat below uses a visible framing
preview. The cause of the 437 missed detections cannot be inferred from
aggregate counts alone.

The participant then confirmed a repeat with the single-window preview. After
10.016 seconds of unscored alignment, the 15.062-second measurement captured
299 frames: 205 no-face, eight multiple-face detections and 86 scored crops.
The current model passed **75/86 (87.21%)**; the candidate passed
**56/86 (65.12%)**. The report is
`results/pad_webcam_20261109/glasses_02.json`. Neither threshold was changed.
This second genuine-labelled trial also argues against promoting the candidate.
Detection coverage is still limited, and these correlated crops from one person
must not be reported as independent-person accuracy.

The participant next confirmed readiness to present a still photo of their own
face on a phone, with instructions to keep their real face outside the camera
view. After 10.016 seconds of unscored framing, the 15.015-second measurement
captured 251 frames: 134 were scored and 117 had multiple face detections.
On the same 134 scored crops, the current model accepted **134/134 (100%)**;
the candidate accepted **101/134 (75.37%)** and rejected **33/134 (24.63%)**.
The report is `results/pad_webcam_20261109/phone_photo_01.json`.

The 117 multiple-face frames were excluded before PAD; they are **not** PAD
rejections. Without saved images, the detector output cannot establish whether
multiple real faces were present. This remains one operator-labelled phone
presentation with correlated frames, not an independent-population APCER.
Identity and attendance were not evaluated by this paired PAD-only tool.

**Decision: do not promote the candidate.** It still accepts most phone-photo
crops and rejected more genuine glasses crops than the current model. The
current model also fails the phone-photo trial; keeping it as the app default
preserves existing behavior and is not a claim of adequate anti-spoofing.
Both thresholds remain locked. Further work must investigate the camera/data
gap and validate any new training method on fresh genuine and attack trials.
No images/video were saved, and no personal gallery or attendance was accessed.

The model paths are fixed before capture. Do not tune either threshold from
the trial. The paired tool scores both models on exactly the same original RGB
face crop, after the app's face-count and image-quality checks. It saves aggregate
numbers only, uses no gallery/database, and refuses to overwrite an existing
JSON report. Captures require the participant to be ready; no camera was started
as part of offline development.

Start with a real face wearing glasses, then without glasses, then a still face
photo on a phone with the real face outside the camera view. Recorded video and
printed photos are separate attack conditions if the materials are available.
Repeat conditions with repositioning and similar lighting; frames within a
trial are correlated. A small classroom trial does not establish population
APCER/BPCER or certify resistance to all presentation attacks.

The optional `--preview` uses one camera and one window. It provides ten seconds
of unscored alignment, followed by the measured interval with face boxes,
detection status and scored-crop count. Overlays are drawn only on a display
copy after inference. Q/Escape stops the trial and marks it operator-cancelled.
The preview window and camera are released when the process finishes.

Example **for use only after participant readiness is confirmed**:

```powershell
.\.venv-gpu\Scripts\python.exe -m tools.check_liveness_camera --baseline results/liveness_camera_matched/best_model.pt --candidate results/liveness_balanced_matched_20261109/best_model.pt --condition glasses --preview --seconds 15 --frames 1000 --output results/pad_webcam_20261109/glasses_repeat_new.json
.\.venv-gpu\Scripts\python.exe -m tools.check_liveness_camera --baseline results/liveness_camera_matched/best_model.pt --candidate results/liveness_balanced_matched_20261109/best_model.pt --condition screen-replay --presentation-type phone-still-photo --preview --seconds 15 --frames 1000 --output results/pad_webcam_20261109/phone_photo_repeat_new.json
```

The full app trial tool also accepts `--liveness-checkpoint` to test a candidate
with a temporary gallery snapshot. Its default stays unchanged. A diagnostic
camera comparison alone does not approve automatic deployment of this candidate;
the failed offline requirement and any genuine/attack regressions must remain
visible in the decision.

The English group report and submission ZIP retain their reviewed 8 November
snapshot. The 9 November experiment and camera decisions are recorded here;
include this newer evidence when preparing the final submission report/package.
