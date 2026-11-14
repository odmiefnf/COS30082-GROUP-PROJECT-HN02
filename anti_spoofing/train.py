"""Train a liveness CNN from scratch, then calibrate on validation data only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import islice
import math
import os
from pathlib import Path
import platform
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import LivenessDataset, _path_key, load_manifest, validate_manifest_splits
from .evaluate import (
    _device, _has_overlap, _manifest_provenance, _overlap_report, _positive_int,
    _sha256, _write_json, predict_dataset,
)
from .metrics import calibrate_threshold
from .model import LivenessCNN


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def _epoch(
    model: LivenessCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
    augment=None,
) -> dict:
    model.train(optimizer is not None)
    total_loss, correct, count, batches = 0.0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in islice(loader, max_batches):
            images, labels = images.to(device), labels.to(device)
            if not torch.isfinite(images).all().item() or not torch.all((labels == 0) | (labels == 1)).item():
                raise ValueError("Training batch has nonfinite pixels or invalid canonical liveness labels.")
            if augment is not None and optimizer is not None:
                images = augment(images)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (len(labels), 2):
                raise ValueError("LivenessCNN must return two raw logits per image: [spoof, live].")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("LivenessCNN produced nonfinite logits; check the inputs and learning rate.")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Liveness training loss became nonfinite; check the inputs and learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_count = len(labels)
            total_loss += loss.item() * batch_count
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += batch_count
            batches += 1
    if count == 0:
        raise ValueError("No liveness batches were available.")
    return {"loss": float(total_loss / count), "accuracy": float(correct / count),
            "samples": count, "batches": batches}


def _save_checkpoint(path: Path, model: LivenessCNN, config: dict, training: dict) -> None:
    payload = {
        "format_version": 1, "task": "anti_spoofing",
        "model_state": {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()},
        "config": config, "training": training,
    }
    # Write directly, as Windows sandbox temporary directories can deny a
    # rename even when the explicitly requested checkpoint path is writable.
    torch.save(payload, path)


def _code_provenance() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    names = ("train.py", "evaluate.py", "metrics.py", "data.py", "model.py", "augmentation.py")
    return {name: _sha256(directory / name) for name in names if (directory/name).is_file()}


def train(args: argparse.Namespace) -> Path:
    """Train/select on train/val only and return the calibrated best checkpoint."""
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must lie in [0, 2**32 - 1].")
    if args.image_size < 32:
        raise ValueError("image-size must be at least 32 for spatial downsampling.")
    context_crop = getattr(args, "context_crop", False)
    if context_crop and args.crop_mode != "full":
        raise ValueError("Context training expects already prepared 2x crops and --crop-mode full.")
    seed_everything(args.seed)
    device = _device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(args.num_threads, os.cpu_count() or 1))
    train_manifest = args.train_manifest.expanduser().resolve()
    val_manifest = args.val_manifest.expanduser().resolve()
    split_validation = validate_manifest_splits(train_manifest, val_manifest)
    training_dataset = LivenessDataset(train_manifest, image_size=args.image_size, crop_mode=args.crop_mode)
    validation_dataset = LivenessDataset(val_manifest, image_size=args.image_size, crop_mode=args.crop_mode)
    print(f"Fingerprinting {len(training_dataset)} training and {len(validation_dataset)} validation images...", flush=True)
    provenance = {
        "train": _manifest_provenance(train_manifest, training_dataset.records, args.crop_mode),
        "validation": _manifest_provenance(val_manifest, validation_dataset.records, args.crop_mode),
    }
    content_overlap = _overlap_report(provenance["train"], provenance["validation"])
    if _has_overlap(content_overlap):
        raise ValueError(
            "Training and validation must be disjoint by subject, image path and image-content SHA256. "
            f"Found {len(content_overlap['subject_ids'])} shared subjects, "
            f"{len(content_overlap['image_paths'])} shared paths and "
            f"{len(content_overlap['image_sha256'])} shared image-content hashes."
        )
    cache_enabled = bool(getattr(args, "cache_images", False))
    cache_bytes = 0
    if cache_enabled:
        print("Caching train/validation tensors in CPU RAM; preprocessing is unchanged.", flush=True)
        cache_bytes = training_dataset.cache_in_memory() + validation_dataset.cache_in_memory()
        print(f"Cached tensors occupy {cache_bytes / 1024**3:.3f} GiB.", flush=True)
    loader_generator = torch.Generator().manual_seed(args.seed)
    oversample_manifest = getattr(args, "oversample_manifest", None)
    oversample_factor = float(getattr(args, "oversample_factor", 1.))
    sampler, oversampled = None, 0
    if oversample_manifest is not None:
        # Draw a small in-domain subset (e.g. webcam crops) more often without duplicating rows.
        boosted = {_path_key(record["image_path"]) for record in load_manifest(oversample_manifest)}
        weights = [oversample_factor if _path_key(record["image_path"]) in boosted else 1.
                   for record in training_dataset.records]
        oversampled = sum(weight != 1. for weight in weights)
        if not oversampled:
            raise ValueError("No training record matches --oversample-manifest.")
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True,
                                        generator=loader_generator)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=sampler is None,
                                 sampler=sampler, num_workers=0, generator=loader_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LivenessCNN(feature_dim=args.feature_dim).to(device)
    class_counts = [sum(record["label"] == label for record in training_dataset.records) for label in (0, 1)]
    balanced_loss = bool(getattr(args, "balanced_loss", False))
    class_weights = [len(training_dataset) / (2 * count) for count in class_counts] if balanced_loss else None
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights is not None else None)
    validation_criterion = nn.CrossEntropyLoss()
    camera_augmentation = bool(getattr(args, "camera_augmentation", False))
    weight_decay = float(getattr(args, "weight_decay", 0.))
    selection = getattr(args, "checkpoint_selection", "val-loss")
    augment = None
    if camera_augmentation:
        from .augmentation import augment_camera_batch
        augment = augment_camera_batch
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=weight_decay, amsgrad=False)
    model_config = {
        "feature_dim": args.feature_dim, "image_size": args.image_size, "crop_mode": args.crop_mode,
        "context_crop": context_crop,
        "live_label": 1, "spoof_label": 0,
        "threshold": None, "threshold_source": None, "threshold_calibrated": False,
    }
    recipe = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "initialization": "random_from_scratch", "pretrained": False,
        "epochs_requested": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
        "learning_rate": args.learning_rate, "device": str(device), "num_threads": torch.get_num_threads(),
        "max_batches": args.max_batches, "smoke_test": args.max_batches is not None,
        "epoch_training_capped": args.max_batches is not None,
        "epoch_validation_capped": args.max_batches is not None,
        "calibration_validation_capped": False,
        "checkpoint_selection": ("minimum validation ACER; validation loss breaks ties" if selection == "val-acer"
                                 else "minimum validation loss; earliest epoch on equal loss"),
        "threshold_selection": "maximum validation Youden J (minimum validation ACER)",
        "test_data_read": False,
        "loss": {"name": "CrossEntropyLoss", "input": "raw_logits", "class_order": ["spoof", "live"],
                 "training_class_counts": {"0": class_counts[0], "1": class_counts[1]},
                 "class_weights": class_weights,
                 "weights_source": "training_manifest_inverse_frequency" if balanced_loss else None,
                 "validation_loss_unweighted": True, "label_smoothing": 0.0, "reduction": "mean"},
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate, "betas": [0.9, 0.999],
                      "eps": 1e-8, "weight_decay": weight_decay, "amsgrad": False},
        "augmentation": {"enabled": camera_augmentation, "training_only": True,
                         "operations": (["horizontal_flip", "moderate_brightness_contrast", "small_upper_face_occlusion"]
                                        if camera_augmentation else []),
                         "same_distribution_for_live_and_spoof": True,
                         "validation_and_inference_unchanged": True},
        "oversampling": {"manifest": str(oversample_manifest) if oversample_manifest else None,
                         "factor": oversample_factor if oversample_manifest else None,
                         "records": oversampled, "with_replacement": sampler is not None},
        "image_cache": {"enabled": cache_enabled, "bytes": cache_bytes,
                        "location": "CPU RAM", "preprocessing_changed": False},
        "preprocessing": {"image_size": args.image_size, "crop_mode": args.crop_mode,
                          "channels": "RGB", "normalization": "[-1, 1]"},
        "label_definition": {"0": "spoof", "1": "live"},
        "source_label_definition": {"0": "live", "1": "spoof"},
        "source_label_conversion": "canonical_label = 1 - official_source_label",
        "split_validation": split_validation,
        "provenance": provenance,
        "source_code_sha256": _code_provenance(),
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__)},
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").unlink(missing_ok=True)
    _write_json(output_dir / "config.json", {"model": model_config, "training": recipe})
    best_path, last_path = output_dir / "best_model.pt", output_dir / "last_model.pt"
    best_loss, best_epoch, best_rank = math.inf, 0, (math.inf, math.inf)
    print(f"Training liveness CNN from scratch on {device}; labels: 0=spoof, 1=live.", flush=True)
    if args.max_batches is not None:
        print(f"Smoke test: training/selection validation capped at {args.max_batches} batches per epoch; "
              "final threshold calibration still uses the complete validation manifest.", flush=True)
    columns = ["epoch", "train_loss", "train_accuracy", "train_samples", "train_batches", "val_loss", "val_accuracy", "val_samples", "val_batches", "best_epoch"]
    if selection == "val-acer":
        columns += ["val_acer", "val_apcer", "val_bpcer"]
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            training_stats = _epoch(model, training_loader, criterion, device, optimizer, args.max_batches, augment)
            validation_stats = _epoch(model, validation_loader, validation_criterion, device, None, args.max_batches)
            operating_metrics = None
            if selection == "val-acer":
                val_labels, val_scores = predict_dataset(model, validation_dataset, args.batch_size, device)
                operating_metrics = calibrate_threshold(val_labels, val_scores)["validation_metrics"]
                rank = (operating_metrics["ACER"], validation_stats["loss"])
            else:
                rank = (validation_stats["loss"], 0.)
            improved = rank < best_rank
            if improved:
                best_loss, best_epoch = validation_stats["loss"], epoch
                best_rank = rank
            training = {**recipe, "epoch": epoch, "best_epoch": best_epoch, "best_val_loss": best_loss,
                        "train_epoch_metrics": training_stats, "validation_epoch_metrics": validation_stats}
            if operating_metrics is not None:
                training["validation_operating_metrics"] = operating_metrics
                training["best_validation_acer"] = best_rank[0]
            if improved:
                _save_checkpoint(best_path, model, model_config, training)
            _save_checkpoint(last_path, model, model_config, training)
            history_row = {"epoch": epoch, **{f"train_{key}": value for key, value in training_stats.items()},
                           **{f"val_{key}": value for key, value in validation_stats.items()}, "best_epoch": best_epoch}
            if operating_metrics is not None:
                history_row.update(val_acer=operating_metrics["ACER"], val_apcer=operating_metrics["APCER"],
                                   val_bpcer=operating_metrics["BPCER"])
            writer.writerow(history_row)
            handle.flush()
            print(f"Epoch {epoch}/{args.epochs}: train_loss={training_stats['loss']:.6f}, "
                  f"val_loss={validation_stats['loss']:.6f}, val_accuracy={validation_stats['accuracy']:.4f}; "
                  f"best epoch={best_epoch}", flush=True)
            if operating_metrics is not None:
                print(f"Validation ACER={operating_metrics['ACER']:.4f}, "
                      f"APCER={operating_metrics['APCER']:.4f}, BPCER={operating_metrics['BPCER']:.4f}", flush=True)
    # Reload selected weights, not last-epoch weights, before complete validation calibration.
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"], strict=True)
    model.to(device)
    labels, live_scores = predict_dataset(model, validation_dataset, args.batch_size, device)
    calibration = calibrate_threshold(labels, live_scores)
    calibration.update({
        "selected_checkpoint": str(best_path), "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_provenance": provenance["validation"],
        "validation_samples_used": int(labels.size), "validation_batches_capped": False,
        "training_smoke_test": args.max_batches is not None, "training_max_batches": args.max_batches,
        "test_data_used": False,
    })
    calibrated_config = {**model_config, "threshold": calibration["threshold"],
                         "threshold_source": "validation", "threshold_calibrated": True,
                         "calibration_manifest_sha256": provenance["validation"]["manifest_sha256"]}
    best_training = {**best["training"], "completed_epochs": args.epochs,
                     "calibration": calibration, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    _save_checkpoint(best_path, model, calibrated_config, best_training)
    _write_json(output_dir / "calibration.json", calibration)
    _write_json(output_dir / "config.json", {"model": calibrated_config, "training": best_training,
                                              "best_checkpoint": str(best_path), "last_checkpoint": str(last_path)})
    print(f"Calibrated threshold {calibration['threshold']:.6f} on ALL {labels.size} validation images "
          f"using best epoch {best_epoch}; saved {best_path}", flush=True)
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--image-size", type=_positive_int, default=112)
    parser.add_argument("--feature-dim", type=_positive_int, default=128)
    parser.add_argument("--crop-mode", choices=("bbox", "full"), default="bbox")
    parser.add_argument("--context-crop", action="store_true",
                        help="Declare prepared 2x context crops used with --crop-mode full.")
    parser.add_argument("--learning-rate", type=_positive_float, default=0.001)
    parser.add_argument("--weight-decay", type=_nonnegative_float, default=0.)
    parser.add_argument("--camera-augmentation", action="store_true",
                        help="Train-only mild lighting and small occlusion augmentation for both labels.")
    parser.add_argument("--balanced-loss", action="store_true",
                        help="Weight training cross entropy by inverse training class frequency; validation stays unweighted.")
    parser.add_argument("--checkpoint-selection", choices=("val-loss", "val-acer"), default="val-loss")
    parser.add_argument("--oversample-manifest", type=Path,
                        help="Training rows listed here are drawn --oversample-factor times as often.")
    parser.add_argument("--oversample-factor", type=_positive_float, default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=_positive_int, default=4)
    parser.add_argument("--cache-images", action="store_true",
                        help="Cache preprocessed train/validation tensors in CPU RAM to avoid repeated decoding.")
    parser.add_argument("--max-batches", type=_positive_int, default=None,
                        help="Explicit smoke test: cap training/selection batches, never full-validation calibration.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Liveness training failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["seed_everything", "train", "build_parser", "main"]
