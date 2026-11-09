# PAD regularization experiment

Completed on 8 November 2026 using the existing training code. The experiment
tested whether stronger weight decay and a lower learning rate would improve
the validation operating point. It did not meet its predeclared promotion
criteria, so the application's checkpoint and threshold remain unchanged.

LivenessCNN was initialized randomly and trained for 20 complete epochs on the
same 8,093 LCC-FASD training crops and 2,948 validation crops. The recipe used
RGB 112 by 112, batch 64, Adam learning rate 0.0002, weight decay 0.001, seed 42,
and the existing train-only lighting, horizontal-flip and upper-face-occlusion
augmentation. No test images or webcam images were used for selection.

Epoch 6 was selected by minimum validation ACER with validation loss breaking
ties. The selected threshold was 0.8717164423, calibrated on all validation
images. This score scale cannot be compared directly with another model's
threshold.

| Validation measurement | Result | Predeclared requirement |
| --- | ---: | ---: |
| ACER | 19.82% | Below 19.34% |
| APCER | 16.67% | At most 33.94% |
| BPCER | 22.96% | At most 13.58% |
| AUC | 0.843840 | Reported descriptively |

Both the ACER and genuine-rejection criteria failed. The new candidate was not
evaluated on the reused test set or on a webcam, and was not promoted. The
original app model remains `results/liveness_camera_matched/best_model.pt`.
The known phone-photo PAD failure remains unresolved by this experiment.

The plan was saved before training in
`results/liveness_regularized_20261108/experiment_plan.json`. The selected
checkpoint, configuration, history, calibration and decision are retained in
that directory; the full log is `results/readiness_20261108/pad_training.log`.
Validation has already been reused during development, so these results do not
establish independent performance on phones, prints or replay attacks.

```powershell
.\.venv-gpu\Scripts\python.exe -m anti_spoofing.train --train-manifest data/lcc_fasd_manifests/train.csv --val-manifest data/lcc_fasd_manifests/val.csv --output-dir results/liveness_new_experiment --crop-mode full --epochs 20 --batch-size 64 --learning-rate 0.0002 --weight-decay 0.001 --camera-augmentation --checkpoint-selection val-acer --device cuda --num-threads 4 --seed 42 --cache-images
```

Use a new output directory when reproducing the experiment. Further promotion
requires a justified validation improvement and independent operator-labelled
genuine and attack trials, including the appearances and materials intended for
the demonstration.
