# Interview and individual-report preparation

Use this guide to explain the actual implementation and evidence. It is not a
completed individual report or a script claiming that every member wrote every
component. Group HN02, names and student IDs are recorded in the group report; individual contribution assignments remain
blank until the group supplies them.

## Questions every member should be able to answer

1. **What is the difference between classification and verification?**
   Classification predicts one of the 4,000 training IDs. Verification compares
   two embeddings, including images of identities that need not be training
   classes. The attendance app compares a probe with enrolled appearance
   templates and may return unknown or ambiguous.
2. **How were the two required identity models trained?**
   Each objective was trained on two backbones, all with RGB 64 x 64 input, a
   128-dimensional embedding and ten complete epochs. The compact FaceEncoder
   starts from random weights (Adam 0.001, no augmentation). The ResNet-18
   encoder starts from ImageNet weights with its max pool removed and stem
   frozen (Adam 0.0003 with a cosine schedule, horizontal flips, mixed
   precision). Classification uses cross-entropy and a 4,000-class head. Triplet
   learning uses anchor/positive/negative images and a margin of 0.2 with
   unsquared L2 distance; it samples random triplets, not hard mining.
3. **What is the triplet loss trying to achieve?**
   `max(0, ||a-p||2 - ||a-n||2 + 0.2)` encourages the positive to be closer to
   the anchor than the negative by the margin. Labels define the relationships;
   this implementation is not a label-free training method.
4. **Why compare cosine and Euclidean distance?**
   On L2-normalized embeddings, squared Euclidean distance equals
   `2 - 2*cosine`. Both were evaluated and rank the pairs equivalently, so their
   equal AUC values are expected, not two independent confirmations.
5. **How were the best identity checkpoints and model selected?**
   Classification epochs are selected by validation loss and triplet epochs by
   training loss (compact 9 and 10, ResNet-18 8 and 10; ArcFace 9). On all 8,805
   original pairs the AUCs are 0.825601 and 0.859969 (compact) and 0.913943,
   0.914668 and 0.914747 (ResNet-18 classification, triplet, ArcFace). The app
   uses ResNet-18 ArcFace: the three ResNet AUCs differ by less than 0.001, but at
   1% pair FPR ArcFace recognises 32.98% of same-identity pairs, classification
   26.63% and triplet 12.44%. Pair labels did not update weights.
5a. **What does ArcFace change?** The same ResNet-18 and classification setup, but
   the head compares the embedding with 4,000 normalised class weights by cosine,
   adds an angular margin of 0.3 to the true class and scales by 32 before
   cross-entropy. Same-person embeddings are pushed into tighter angular clusters,
   which helps most at strict thresholds. The margin is used only in training.
6. **Does an AUC of 0.91 mean 91% attendance accuracy?**
   No. AUC summarizes pair-score ranking over thresholds. At the app's cosine
   threshold 0.509131, validation pair FPR is 0.98% and TPR 32.98%, so about
   two thirds of genuine validation pairs are still rejected. In the 1:N
   gallery test the app finds 8 of 20 known people and misidentifies 25 of 300
   outsiders. Appearance templates and manual review compensate in operation.
7. **What data leakage or evaluation limits exist?**
   The assignment's original validation protocol contains 667 images matching
   training pixels. The pair file was preserved and that overlap disclosed.
   Additional filtered PAD/FER evaluation sets remove specified duplicates but
   do not establish full independence by participant, video or capture device.
8. **How is PAD trained and evaluated?**
   The app model is ImageNet MobileNetV3-Large with its first four blocks
   frozen, 224 px input on a 1.4x square crop around the face, AdamW with a
   cosine schedule, class-balanced sampling and camera-oriented augmentation.
   It is trained on LCC-FASD plus 872 of our own webcam crops (oversampled x8),
   split by whole recording session. A validation threshold (0.5927, Youden J)
   gates identity, enrolment and attendance. APCER is spoof acceptance, BPCER
   genuine rejection, ACER their mean; the score is not a security guarantee.
9. **Why train on your own webcam data, and what did it cost?**
   The LCC-only MobileNetV3 had LCC-FASD ACER 5.72%, yet in live use it scored a
   phone photo in a dark room at 0.85 against a 0.041 threshold and recorded
   attendance. On held-out webcam sessions it accepted 29/52 photo and 67/116
   video crops. The app model accepts 0/52 and 0/116 and passes 96/96 genuine
   crops, but its LCC-FASD ACER rises to 8.88% (BPCER 16.41%): it is specialised
   to our camera. The webcam evidence is one person, room and day, with the same
   photo and video content in training and test sessions, so it is not an
   attack-population error rate. Two attack sessions were excluded after review
   because they captured the operator's real face.
10. **What can expression classification tell us?**
    The app's fine-tuned MobileNetV3 predicts one of seven visible FER2013
    labels. Filtered PrivateTest accuracy is 64.20% and macro-F1 0.6284 (the
    from-scratch EmotionCNN: 58.36% and 0.4984); disgust and fear improved most
    but fear and sad remain near F1 0.5. It does not establish internal mood,
    mental health or work performance, and it never authorizes attendance.
11. **Why are image quality, face count and freshness checked, and how are several faces handled?**
    YuNet finds every face; each scored face must be large, clear and reasonably
    exposed. With several faces, the largest is the main face: only it is
    liveness checked, shows expression and can be recorded. Other faces get a
    name tag (green if registered) but no liveness check, so they are never
    recorded. Registration, appearance capture and review still need exactly one
    face. Stale observations reset or invalidate evidence.
12. **How can a new person be registered without retraining the classifier?**
    Guided registration captures ten poses; each shot is liveness checked on its
    own frame and only live shots (at least three) are stored as embeddings. The
    app matches embeddings instead of adding a class to the classifier head.
    Appearance templates (for example with and without glasses) are added through
    the same flow and refused if the shots clearly match another person.
12a. **Is using ImageNet weights allowed?**
    The brief prohibits pretrained models used without further training. All
    three app networks are fine-tuned on the task data, and the from-scratch
    baselines are reported for comparison. YuNet is pretrained face
    localisation, disclosed as preprocessing, not one of the trained networks.
13. **How is duplicate attendance prevented?**
    Automatic attendance needs three consecutive valid recognized observations.
    SQLite enforces one record per person/session. Expressions are descriptive
    feedback and do not change that decision.
14. **What does manual review improve, and what does it not improve?**
    An operator may explicitly verify a live unknown/ambiguous person and record
    a separate manual source, reviewer and reason. Transactions preserve record
    provenance. The CNN stays unknown; manual approval does not raise its cosine
    score or count as improved model accuracy. The demo has no authenticated
    administrator or tamper-proof audit service.
15. **What remains before making deployment claims?**
    Independent genuine/unknown-person and attack trials with more participants,
    appearances, devices and lighting; better PAD effectiveness; improved
    low-FPR identity recall; and measured operator-review performance. Existing
    software tests verify paths and invariants, not deployment accuracy.

## Evidence to have open during practice

| Topic | Files to explain |
| --- | --- |
| Identity data, model and loss | `face_verification/data.py`, `model.py`, `train.py` |
| ROC, pair counts and threshold | `results/comparison_identity_20261121/comparison.csv`, `results/identity_app_config.json`, `tools/calibrate_identity_app.py` |
| 1:N gallery test | `results/identity_open_set_20261121_arcface_resnet18/metrics.json`, `tools/evaluate_identity_open_set.py` |
| PAD training and operating point | `anti_spoofing/train.py`, `results/liveness_app_config.json`, `results/liveness_mobilenetv3_webcam_20261120/calibration.json` |
| Webcam PAD data and results | `tools/capture_pad_data.py`, `tools/record_pad_sessions.cmd`, `results/liveness_mobilenetv3_webcam_20261120_on_test/metrics.json` |
| Expression training and per-class errors | `emotion_recognition/model.py`, `train.py`, `results/emotion_fer_mnv3_test/metrics.json` |
| Live pipeline and attendance | `attendance_app/camera.py`, `service.py`, `store.py`, `web.py`, `video_stream.py` |
| Additional appearances/manual review | `attendance_app/templates.py`, `review.py`, `reference_import.py`, and the corresponding tests |

Use the final code and test log when stating the current behavior or test count.
Do not memorize an old screenshot or architecture paragraph if the pipeline has
changed. Be able to trace one result from camera input through gates to storage.

## English individual-report outline (at most five pages)

This allocation is a writing guide. Keep the final submitted document within
the brief's five-page cap and check its actual rendered page count. Do not add
an appendix that makes the submission longer than the cap.

| Suggested space | Section and prompts |
| --- | --- |
| Page 1 | **Identity and Contribution Summary.** Name: ______; Student ID: ______; Group: ______. State your actual responsibilities, exact files/experiments you contributed to, and how they connect to the group system. Distinguish work you implemented, reviewed and tested. |
| Page 2 | **Methods Used.** Explain the relevant model or algorithm, inputs/outputs, loss, training scheme and important hyperparameters. Explain why the approach was suitable and cite sources. Describe only the components relevant to your own contribution. |
| Page 3 | **Implementation and Evaluation.** Explain your change and its test protocol, baseline, measured results and failure cases. Link the evidence to the claim. Distinguish software fixtures, reused dataset evaluation and human webcam trials. |
| Page 4 | **Innovation and Challenges.** If you personally contributed an innovative feature, explain the user problem, rationale, method, implementation and demonstration. Explain a concrete challenge, diagnosis, resolution and validation. Do not invent ownership or an experiment. |
| Page 5 | **Reflection, Limitations and References.** Explain what your evidence supports, what remains untested and one justified next improvement. Include concise references. Adjust the earlier sections to fit the actual material; five pages are a maximum, not a target to fill. |

Useful claim structure:

> I contributed [specific work] in [files/experiment]. The problem was [observed
> behavior]. I changed [method] because [reason]. I evaluated it using [protocol
> and evidence]. The result was [measured result with denominator]. This does not
> establish [relevant limitation].

Leave the brackets unfilled until they can be replaced with the student's real
work. The group should agree the contribution record before writing separate
reports. Neither this guide nor the source code assigns authorship to a member.

## Practical rehearsal

- Explain your contribution without reading the source aloud, then show the
  exact function and one meaningful test supporting it.
- Describe one failure honestly and how you would design the next evaluation.
- Prepare the required continuous 5-10 minute group demonstration, with similar
  speaking time and each speaker's face visible; see `09_video_checklist.md`.
- Confirm each member submits their own report and peer evaluation, and checks
  the actual due date on Canvas. No numeric deadline is stated in the supplied
  PDF.
