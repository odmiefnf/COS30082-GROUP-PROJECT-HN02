"""Match a trained candidate to the baseline VALIDATION spoof-acceptance bound."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anti_spoofing.data import LivenessDataset
from anti_spoofing.evaluate import _sha256, load_liveness_model, predict_dataset
from anti_spoofing.metrics import liveness_metrics
from face_verification.metrics import compute_roc


def select_at_apcer(labels, scores, maximum_apcer):
    """Choose validation TPR under a fixed spoof bound; ties prefer lower APCER."""
    maximum_apcer = float(maximum_apcer)
    if not np.isfinite(maximum_apcer) or not 0 <= maximum_apcer <= 1:
        raise ValueError("APCER bound must be finite and in [0,1].")
    liveness_metrics(labels, scores, .5)  # Validate class support, labels and scores.
    fpr, tpr, thresholds = compute_roc(labels, scores)
    eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
                              & (fpr <= maximum_apcer+1e-12))
    if not len(eligible):
        raise ValueError("No finite threshold meets the validation APCER bound.")
    chosen = max(eligible, key=lambda i: (tpr[i], -fpr[i], thresholds[i]))
    threshold = float(thresholds[chosen])
    return threshold, liveness_metrics(labels, scores, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
    payload = torch.load(args.candidate, map_location="cpu", weights_only=True)
    expected = payload["config"]["calibration_manifest_sha256"]
    if (_sha256(args.validation) != expected
            or baseline["config"]["calibration_manifest_sha256"] != expected):
        raise ValueError("Both models must use this exact original validation manifest.")
    baseline_stats = baseline["training"]["calibration"]["validation_metrics"]
    maximum = baseline_stats["APCER"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_liveness_model(args.candidate, device=device)
    dataset = LivenessDataset(args.validation, image_size=config["image_size"], crop_mode=config["crop_mode"])
    labels, scores = predict_dataset(model, dataset, 128, device)
    threshold, metrics = select_at_apcer(labels, scores, maximum)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir/"best_model.pt"
    calibration = {
        "threshold": threshold, "threshold_source": "validation", "threshold_calibrated": True,
        "selection_method": "maximum validation TPR subject to APCER <= baseline validation APCER; lower APCER then higher threshold break ties",
        "maximum_validation_apcer": maximum, "baseline_validation_metrics": baseline_stats,
        "baseline_checkpoint_sha256": _sha256(args.baseline), "validation_metrics": metrics,
        "scope": "validation operating point; not an independent deployment error guarantee",
        "validation_samples_used": len(dataset), "validation_batches_capped": False,
        "test_data_used": False, "camera_data_used": False, "selected_checkpoint": str(output.resolve()),
    }
    payload["config"] = {**payload["config"], "threshold": threshold}
    payload["training"] = {**payload["training"], "calibration": calibration,
        "threshold_selection": calibration["selection_method"],
        "derived_from_checkpoint": str(args.candidate.resolve()),
        "derived_from_checkpoint_sha256": _sha256(args.candidate),
        "weight_changes_during_recalibration": False}
    torch.save(payload, output)
    saved = torch.load(output, map_location="cpu", weights_only=True)
    assert all(torch.equal(value, saved["model_state"][name]) for name, value in payload["model_state"].items())
    (args.output_dir/"calibration.json").write_text(json.dumps(calibration, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(calibration, indent=2), flush=True)


if __name__ == "__main__":
    main()
