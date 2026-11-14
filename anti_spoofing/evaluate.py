"""Evaluate liveness using a threshold calibrated on validation data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from face_verification.metrics import compute_roc

from .data import LivenessDataset, bbox_sidecar_path
from .metrics import liveness_metrics
from .model import LivenessCNN


def _device(choice: str | torch.device) -> torch.device:
    if isinstance(choice, torch.device):
        device = choice
    else:
        device = torch.device("cuda" if choice == "auto" and torch.cuda.is_available() else
                              "cpu" if choice == "auto" else choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu or auto.")
    return device


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest_provenance(manifest: str | Path, records: list[dict], crop_mode: str) -> dict:
    """Fingerprint the actual manifest, images and used bounding-box sidecars."""
    manifest = Path(manifest).expanduser().resolve()
    samples = []
    subjects: set[str] = set()
    for record in records:
        image = Path(record["image_path"]).expanduser().resolve()
        subject = str(record["subject_id"]).strip()
        if not subject and record.get("subject_id_source") != "unavailable":
            raise ValueError(f"Empty subject ID for {image} must be explicitly marked unavailable.")
        sample = {
            "image_path": str(image),
            "image_sha256": _sha256(image),
            "label": int(record["label"]),
            "subject_id": subject,
            "source_label": int(record["source_label"]),
            "subject_id_source": record.get("subject_id_source", "provided"),
        }
        if crop_mode == "bbox":
            sidecar = bbox_sidecar_path(image).resolve()
            sample.update({"bbox_path": str(sidecar), "bbox_sha256": _sha256(sidecar)})
        samples.append(sample)
        if subject:
            subjects.add(subject)
    return {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "sample_count": len(samples),
        "live_count": sum(sample["label"] == 1 for sample in samples),
        "spoof_count": sum(sample["label"] == 0 for sample in samples),
        "subject_count": len(subjects),
        "subject_ids": sorted(subjects),
        "subject_ids_complete": all(bool(sample["subject_id"]) for sample in samples),
        "unknown_subject_samples": sum(not sample["subject_id"] for sample in samples),
        "crop_mode": crop_mode,
        "samples": samples,
    }


def _overlap_report(reference: dict, candidate: dict) -> dict:
    reference_samples = reference.get("samples", [])
    candidate_samples = candidate.get("samples", [])
    reference_paths = {os.path.normcase(str(Path(row["image_path"]).resolve())) for row in reference_samples}
    candidate_paths = {os.path.normcase(str(Path(row["image_path"]).resolve())) for row in candidate_samples}
    reference_hashes = {row["image_sha256"] for row in reference_samples}
    candidate_hashes = {row["image_sha256"] for row in candidate_samples}
    reference_subjects = {row["subject_id"] for row in reference_samples if row.get("subject_id")} | {value for value in reference.get("subject_ids", []) if value}
    candidate_subjects = {row["subject_id"] for row in candidate_samples if row.get("subject_id")} | {value for value in candidate.get("subject_ids", []) if value}
    return {
        "subject_ids": sorted(reference_subjects & candidate_subjects),
        "image_paths": sorted(reference_paths & candidate_paths),
        "image_sha256": sorted(reference_hashes & candidate_hashes),
    }


def _has_overlap(report: dict) -> bool:
    return any(bool(report[key]) for key in ("subject_ids", "image_paths", "image_sha256"))


def _validate_held_out(training: dict, candidate: dict, allow_development: bool) -> dict:
    provenance = training.get("provenance")
    def complete_split(split: str) -> bool:
        if not isinstance(provenance, dict) or not isinstance(provenance.get(split), dict):
            return False
        samples = provenance[split].get("samples")
        return isinstance(samples, list) and bool(samples) and all(
            isinstance(sample, dict) and all(
                isinstance(sample.get(key), str) and bool(sample[key])
                for key in ("image_path", "image_sha256", "subject_id")
            )
            for sample in samples
        )

    complete = all(complete_split(split) for split in ("train", "validation"))
    if not complete:
        if not allow_development:
            raise ValueError(
                "Checkpoint lacks complete train/validation image and subject provenance. "
                "Held-out test separation cannot be verified. Use a checkpoint produced by "
                "anti_spoofing.train, or explicitly use --allow-development-evaluation for an "
                "in-sample/development report."
            )
        return {"status": "unverified", "reason": "checkpoint lacks train/validation provenance",
                "evaluation_scope": "development / in-sample (explicit override); not an unbiased test estimate"}
    overlaps = {split: _overlap_report(provenance[split], candidate) for split in ("train", "validation")}
    subjects_complete = bool(candidate.get("samples")) and all(
        bool(sample.get("subject_id")) for sample in candidate["samples"]
    )
    has_overlap = any(_has_overlap(report) for report in overlaps.values())
    if has_overlap and not allow_development:
        counts = "; ".join(
            f"{split}: {len(report['subject_ids'])} subjects, {len(report['image_paths'])} paths, "
            f"{len(report['image_sha256'])} image-content hashes"
            for split, report in overlaps.items()
        )
        raise ValueError(
            "Evaluation data overlaps recorded training/validation data (" + counts + "). "
            "Use a disjoint test manifest. To inspect these development samples explicitly, "
            "pass --allow-development-evaluation; that report will be labelled in-sample."
        )
    return {
        "status": ("overlap explicitly allowed" if has_overlap else
                   "disjoint" if subjects_complete else "image-disjoint; evaluation subject IDs unavailable"),
        "checks": ["subject ID" if subjects_complete else "available subject IDs only",
                   "absolute image path", "image-content SHA256"],
        "subject_disjoint_verified": subjects_complete and not has_overlap,
        "overlaps": overlaps,
        "evaluation_scope": (
            "development / in-sample (explicit override); not an unbiased test estimate"
            if allow_development else
            "held-out test; disjoint from recorded train and validation data" if subjects_complete else
            "held-out evaluation by image path/content; evaluation subject separation cannot be verified"
        ),
    }


def load_liveness_model(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[LivenessCNN, dict]:
    """Load a liveness CNN in eval mode, preserving config and training metadata.

    Uncalibrated checkpoints may be loaded for inspection. Evaluation and
    acceptance decisions must require a validation-calibrated threshold.
    """
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Liveness checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not load liveness checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"{path}: expected a checkpoint dict containing config and model_state.")
    config = dict(payload["config"])
    for field in ("feature_dim", "image_size"):
        value = config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Checkpoint config.{field} must be a positive integer.")
    if config.get("crop_mode") not in ("bbox", "full"):
        raise ValueError("Checkpoint config.crop_mode must be bbox or full.")
    if type(config.get("context_crop", False)) is not bool or (
            config.get("context_crop") and config["crop_mode"] != "full"):
        raise ValueError("Context checkpoint requires a boolean context_crop and full input crops.")
    if (type(config.get("live_label")) is not int or type(config.get("spoof_label")) is not int
            or config.get("live_label") != 1 or config.get("spoof_label") != 0):
        raise ValueError("Checkpoint label mapping must be canonical: 1=live, 0=spoof.")
    threshold = config.get("threshold")
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Checkpoint threshold must be finite and within [0, 1].")
    if not isinstance(payload.get("model_state"), dict):
        raise ValueError("Checkpoint model_state must be the complete LivenessCNN state dict.")
    model = LivenessCNN(feature_dim=config["feature_dim"])
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Liveness checkpoint weights do not match config: {exc}") from exc
    model.to(_device(device))
    model.eval()
    if not isinstance(payload.get("training", {}), dict):
        raise ValueError("Checkpoint training provenance must be a dictionary.")
    config["training"] = payload.get("training", {})
    config["checkpoint_format_version"] = payload.get("format_version")
    return model, config


def _calibrated_threshold(config: dict) -> float:
    if config.get("threshold") is None or config.get("threshold_source") != "validation" or not config.get("threshold_calibrated", True):
        raise ValueError(
            "This checkpoint has no validation-calibrated live threshold. Use best_model.pt "
            "after training/calibration has completed; evaluation never calibrates on its input manifest."
        )
    return float(config["threshold"])


def predict_dataset(
    model: LivenessCNN,
    dataset: LivenessDataset,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered ``(labels, P(live))`` arrays with inference-only execution."""
    if batch_size < 1 or len(dataset) == 0:
        raise ValueError("Prediction requires a positive batch size and a nonempty dataset.")
    device = _device(device)
    model.to(device)
    model.eval()
    labels_out, scores_out = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for images, labels in loader:
            if not torch.isfinite(images).all().item():
                raise ValueError("Liveness input contains nonfinite pixels.")
            logits = model(images.to(device))
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (len(labels), 2):
                raise ValueError("LivenessCNN must return two raw logits per image, ordered [spoof, live].")
            if not torch.isfinite(logits).all().item():
                raise ValueError("LivenessCNN produced nonfinite logits.")
            if not torch.all((labels == 0) | (labels == 1)).item():
                raise ValueError("Dataset returned invalid labels; expected canonical 0=spoof, 1=live.")
            scores = torch.softmax(logits.to(torch.float64), dim=1)[:, 1]
            labels_out.append(labels.cpu().numpy().astype(np.int64))
            scores_out.append(scores.cpu().numpy())
    return np.concatenate(labels_out), np.concatenate(scores_out)


def _write_roc_svg(path: Path, fpr: np.ndarray, tpr: np.ndarray, stats: dict, scope: str) -> None:
    left, top, width, height = 90, 75, 700, 410
    points = " ".join(f"{left + x * width:.4f},{top + (1 - y) * height:.4f}" for x, y in zip(fpr, tpr))
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="650" viewBox="0 0 850 650" role="img" aria-labelledby="title desc">',
        '<title id="title">Liveness ROC curve</title>',
        '<desc id="desc">Live is the positive class. The marked operating point uses the saved validation threshold.</desc>',
        '<rect width="850" height="650" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172033">',
        '<text x="90" y="32" font-size="23" font-weight="bold">Liveness ROC</text>',
        f'<text x="90" y="55" font-size="12">{html.escape(scope[:110])}</text>',
    ]
    for i in range(6):
        value = i / 5
        x, y = left + value * width, top + (1 - value) * height
        svg.extend((
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + height}" stroke="#e2e8f0"/>',
            f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#e2e8f0"/>',
            f'<text x="{x}" y="{top + height + 24}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 15}" y="{y + 4}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ))
    svg.extend((
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top}" stroke="#a0acbb" stroke-dasharray="6 6"/>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#65758c"/>',
        f'<polyline points="{points}" fill="none" stroke="#176ab3" stroke-width="2.5"/>',
        f'<circle cx="{left + stats["APCER"] * width}" cy="{top + stats["BPCER"] * height}" r="6" fill="#d26913" stroke="white" stroke-width="2"/>',
        '<text x="440" y="535" text-anchor="middle" font-size="14">False positive rate: spoof accepted as live (APCER)</text>',
        '<text x="28" y="280" transform="rotate(-90 28 280)" text-anchor="middle" font-size="14">True positive rate: live accepted (1 - BPCER)</text>',
        f'<text x="90" y="572" font-size="15">AUC = {stats["AUC"]:.6f}; saved live threshold = {stats["threshold"]:.6f}</text>',
        f'<text x="90" y="597" font-size="13">Orange point: APCER {stats["APCER"]:.4f}, BPCER {stats["BPCER"]:.4f}, ACER {stats["ACER"]:.4f}</text>',
        '<text x="90" y="624" font-size="12">Threshold fixed from validation; no calibration or threshold selection on this evaluation manifest.</text>',
        '</g></svg>',
    ))
    path.write_text("\n".join(svg), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict:
    """Evaluate a manifest once with the checkpoint's unchanged live threshold."""
    device = _device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(args.num_threads, os.cpu_count() or 1))
    checkpoint = args.checkpoint.expanduser().resolve()
    model, config = load_liveness_model(checkpoint, device)
    threshold = _calibrated_threshold(config)
    dataset = LivenessDataset(args.manifest, image_size=config["image_size"], crop_mode=config["crop_mode"])
    # Validate class support before any expensive network inference.
    liveness_metrics([int(row["label"]) for row in dataset.records], np.zeros(len(dataset)), threshold)
    provenance = _manifest_provenance(args.manifest, dataset.records, config["crop_mode"])
    leakage = _validate_held_out(config["training"], provenance, args.allow_development_evaluation)
    labels, scores = predict_dataset(model, dataset, args.batch_size, device)
    stats = liveness_metrics(labels, scores, threshold)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for record, label, score in zip(dataset.records, labels, scores):
        predicted = int(score >= threshold)
        predictions.append({
            "image_path": str(record["image_path"]), "subject_id": str(record["subject_id"]),
            "source_label": int(record["source_label"]), "label": int(label),
            "live_score": float(score), "predicted_label": predicted,
            "decision": "live" if predicted else "spoof", "threshold": threshold,
        })
    _write_csv(output_dir / "predictions.csv", predictions)
    fpr, tpr, thresholds = compute_roc(labels, scores)
    _write_csv(output_dir / "roc_points.csv", [
        {"fpr": float(x), "tpr": float(y), "live_score_threshold": float(t)}
        for x, y, t in zip(fpr, tpr, thresholds)
    ])
    _write_roc_svg(output_dir / "roc_curve.svg", fpr, tpr, stats, leakage["evaluation_scope"])
    training = config["training"]
    smoke = bool(training.get("smoke_test") or training.get("max_batches") is not None)
    report = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_config": config,
        "evaluation_manifest": provenance,
        "metrics": stats,
        "threshold": threshold, "threshold_source": "validation",
        "threshold_retuned_on_evaluation": False,
        "evaluation_scope": leakage["evaluation_scope"], "leakage_check": leakage,
        "training_smoke_test": smoke, "training_max_batches": training.get("max_batches"),
        "training_scope_note": (
            "Smoke-test checkpoint; scores do not represent a completed training experiment."
            if smoke else "Training scope is recorded in checkpoint_config.training."
        ),
        "label_definition": {"0": "spoof", "1": "live"},
        "source_label_definition": {"0": "live", "1": "spoof"},
        "score_definition": "softmax(logits)[1] = P(live)",
        "crop_mode": config["crop_mode"], "image_size": config["image_size"],
        "batch_size": args.batch_size, "device": str(device),
        "num_threads": torch.get_num_threads(),
        "artifacts": {name: str(output_dir / name) for name in ("predictions.csv", "metrics.json", "roc_points.csv", "roc_curve.svg")},
    }
    _write_json(output_dir / "metrics.json", report)
    print(f"Liveness evaluation: AUC={stats['AUC']:.6f}, APCER={stats['APCER']:.4f}, "
          f"BPCER={stats['BPCER']:.4f}, ACER={stats['ACER']:.4f}", flush=True)
    print(f"Scope: {leakage['evaluation_scope']}", flush=True)
    if smoke:
        print(f"Training scope: {report['training_scope_note']}", flush=True)
    print(f"Saved predictions, metrics and ROC artifacts to {output_dir}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=_positive_int, default=4)
    parser.add_argument("--allow-development-evaluation", action="store_true",
                        help="Explicitly allow train/validation overlap; label the report development/in-sample.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evaluate(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Liveness evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_liveness_model", "predict_dataset", "evaluate", "build_parser", "main"]
