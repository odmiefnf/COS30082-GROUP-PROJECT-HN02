# Interview and individual-report preparation

Use this guide to explain the actual implementation and evidence. It is not a
completed individual report or a script claiming that every member wrote every
component. Names, student IDs, group details and contribution assignments remain
blank until the group supplies them.

## Questions every member should be able to answer

1. **What is the difference between classification and verification?**
   Classification predicts one of the 4,000 training IDs. Verification compares
   two embeddings, including images of identities that need not be training
   classes. The attendance app compares a probe with enrolled appearance
   templates and may return unknown or ambiguous.
2. **How were the two required identity models trained?**
   Both use the same compact FaceEncoder, RGB 64 x 64 input and 128-dimensional
   embedding, random initialization, Adam at 0.001 and ten complete epochs.
   Classification uses cross-entropy and a 4,000-class head. Triplet learning
   uses anchor/positive/negative images and a margin of 0.2 with unsquared L2
   distance. Positives are different images of the same training ID; negatives
   have a different ID. The baseline samples random triplets, not hard mining.
3. **What is the triplet loss trying to achieve?**
   `max(0, ||a-p||2 - ||a-n||2 + 0.2)` encourages the positive to be closer to
   the anchor than the negative by the margin. Labels define the relationships;
   this implementation is not a label-free training method.
4. **Why compare cosine and Euclidean distance?**
   On L2-normalized embeddings, squared Euclidean distance equals
   `2 - 2*cosine`. Both were evaluated and rank the pairs equivalently, so their
   equal AUC values are expected, not two independent confirmations.
5. **How were the best identity checkpoints and model selected?**
   Classification selected epoch 9 by validation loss; triplet selected epoch
   10 by training loss. The app then selected triplet because its AUC was
   0.859969 versus classification's 0.825601 on all 8,805 original verification
   validation pairs. Pair labels did not update network weights.
6. **Does an AUC of 0.86 mean 86% attendance accuracy?**
   No. AUC summarizes pair-score ranking over thresholds. At the app's strict
   cosine threshold 0.8695402108345607, validation pair FPR is approximately
   0.98% and TPR only 9.73%. Most genuine validation pairs are rejected there.
   A 1:N gallery and webcam domain require separate evaluation.
7. **What data leakage or evaluation limits exist?**
   The assignment's original validation protocol contains 667 images matching
   training pixels. The pair file was preserved and that overlap disclosed.
   Additional filtered PAD/FER evaluation sets remove specified duplicates but
   do not establish full independence by participant, video or capture device.
8. **How is PAD trained and evaluated?**
   A separate two-class LivenessCNN was trained from random weights on
   LCC-FASD data. The current recipe includes camera-oriented augmentation.
   A validation-calibrated live-score threshold gates identity, enrollment and
   attendance. APCER is spoof acceptance, BPCER is genuine rejection, and ACER
   is their mean. AUC measures ranking; the softmax score is not a security
   guarantee.
9. **What happened in the phone-photo trial?**
   All 237 scored photo crops passed PAD. Identity was unknown for all of them,
   preventing a scratch attendance record. This is a PAD failure for that
   material, not proof that PAD blocked the attack. It is one correlated trial,
   not a population estimate. Printed photos and recorded video were not tested
   in that trial.
10. **What can expression classification tell us?**
    The own-trained FER2013 CNN predicts one of seven visible expression labels:
    angry, disgust, fear, happy, sad, surprise and neutral. Filtered PrivateTest
    accuracy is 58.36% and macro-F1 is 0.4984. Weak classes such as fear and
    disgust need attention. It does not establish a person's internal mood,
    mental health or work performance, and it never authorizes attendance.
11. **Why are image quality, face count and freshness checked?**
    The app requires one sufficiently large, clear, reasonably exposed face.
    Missing/multiple faces and stale observations reset or invalidate evidence.
    Capture sequence numbers prevent a preview refresh being counted as another
    recognition observation; inference completion does not make an old frame
    fresh again.
12. **How can a new person be registered without retraining the classifier?**
    Registration stores a name and three distinct, consistent live embeddings,
    spaced at least 0.45 seconds apart. The app performs embedding matching,
    rather than appending a class to the old classifier head. Appearance
    templates can represent changes such as glasses within one identity.
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
| ROC, pair counts and threshold | `results/comparison/comparison.csv`, `results/identity_app_config.json`, `tools/calibrate_identity_app.py` |
| PAD training and operating point | `anti_spoofing/train.py`, `results/liveness_camera_matched/calibration.json`, `test_regression_metrics.json` |
| Expression training and per-class errors | `emotion_recognition/train.py`, `results/emotion_fer_test/metrics.json` |
| Live pipeline and attendance | `attendance_app/camera.py`, `service.py`, `store.py`, `video_stream.py` |
| Additional appearances/manual review | `attendance_app/templates.py`, `review.py`, `reference_import.py`, and the corresponding tests |
| Honest webcam evidence | `results/app_upgrade_20261030/evaluation/trials/glasses_01.json` and `phone_photo_01.json` |

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
