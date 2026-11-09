"""Select the trained encoder and a conservative validation operating point for the demo."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from face_verification.metrics import compute_roc
from face_verification.image_cache import sha256


def calibrate(comparison_path, output, max_fpr=.01):
    comparison_path, output = Path(comparison_path).resolve(), Path(output).resolve()
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be in [0,1].")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    candidates = []
    for model in comparison["models"]:
        if model["checkpoint_smoke_test"] is not False:
            raise ValueError("App calibration requires uncapped real training checkpoints.")
        metric = next(m for m in model["metrics"] if m["metric"] == "cosine_similarity")
        candidates.append((metric["auc"], model))
    auc, model = max(candidates, key=lambda pair: pair[0])
    with Path(model["artifacts"]["pair_scores_csv"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["cosine_similarity"]) for r in rows])
    fpr, tpr, thresholds = compute_roc(labels, scores)
    candidates = [i for i in range(len(thresholds)) if np.isfinite(thresholds[i])
                  and -1 <= thresholds[i] <= 1 and fpr[i] <= max_fpr]
    if not candidates:
        raise ValueError("No finite cosine threshold meets the requested validation FPR.")
    index = max(candidates, key=lambda i: (tpr[i], thresholds[i]))
    checkpoint = Path(model["checkpoint"]).resolve()
    if sha256(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since evaluation.")
    try:
        checkpoint_reference = Path(os.path.relpath(checkpoint, output.parent)).as_posix()
    except ValueError:
        checkpoint_reference = str(checkpoint)  # Windows cannot relativize different drives.
    report = {
        "checkpoint": checkpoint_reference, "checkpoint_sha256": model["checkpoint_sha256"],
        "method": model["method"], "image_size": model["image_size"],
        "metric": "cosine", "threshold": float(thresholds[index]), "ambiguity_margin": .05,
        "threshold_source": "verification_validation", "max_validation_pair_fpr": max_fpr,
        "observed_validation_pair_fpr": float(fpr[index]), "observed_validation_pair_tpr": float(tpr[index]),
        "selected_model_validation_auc": auc, "pairs_file": comparison["pairs_file"],
        "pairs_file_sha256": comparison["pairs_file_sha256"], "pair_count": len(rows),
        "selection": "Highest original verification-validation cosine AUC; highest TPR with pair FPR <= target; strict threshold on ties.",
        "scope": "In-sample validation operating point; 667 verification validation images match train pixels. "
                 "Pair FPR is not a guarantee for 1:N gallery recognition. Classroom demo needs separate webcam calibration.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=Path("results/comparison/comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("results/identity_app_config.json"))
    parser.add_argument("--max-fpr", type=float, default=.01)
    args = parser.parse_args(argv)
    print(json.dumps(calibrate(args.comparison, args.output, args.max_fpr), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
