"""Train a new face encoder using identity classification or triplet loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IMAGE_EXTENSIONS, EmptyFaceDatasetError, FaceImageDataset, TripletFaceDataset
from .model import FaceClassifier, FaceEncoder


def seed_everything(seed: int) -> None:
    """Seed all random generators used by the training pipeline."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("classification", "triplet"), default="classification")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root containing classification_data/, or classification_data/ itself.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--image-size", type=_positive_int, default=112)
    parser.add_argument("--embedding-dim", type=_positive_int, default=128)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=_positive_int, default=4,
                        help="Maximum CPU compute threads (default: 4; use 1 for tiny smoke tests).")
    parser.add_argument("--max-batches", type=_positive_int, default=None,
                        help="Cap training and validation batches per epoch for explicit smoke tests.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Classification only: evaluate the selected best checkpoint on all known-ID test images once.")
    parser.add_argument("--image-cache", type=Path, help="Optional checksum-verified uint8 cache from image_cache.")
    return parser


def _classification_root(data_root: Path) -> Path:
    """Resolve the classification split directory without reading other datasets."""
    data_root = data_root.expanduser().resolve()
    if (data_root / "classification_data").is_dir():
        root = data_root / "classification_data"
    elif (data_root / "train_data").is_dir():
        root = data_root
    else:
        raise FileNotFoundError(
            f"Cannot find classification_data/train_data under {data_root}. "
            "Pass the dataset root or the classification_data directory."
        )
    if not (root / "train_data").is_dir():
        raise FileNotFoundError(f"Training split does not exist: {root / 'train_data'}")
    return root


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu or auto.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                        "cpu" if name == "auto" else name)


def _classification_epoch(
    model: FaceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> dict[str, float | int]:
    """Run a weighted classification epoch with or without parameter updates."""
    model.train(optimizer is not None)
    total_loss, correct, count, batches = 0.0, 0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in islice(loader, max_batches):
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Classification loss became non-finite; check the inputs and learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_count = labels.size(0)
            total_loss += loss.item() * batch_count
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_count
            batches += 1
    if count == 0:
        raise ValueError("No classification batches were available.")
    return {"loss": total_loss / count, "accuracy": correct / count,
            "samples": count, "batches": batches}


def _triplet_epoch(
    model: FaceEncoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None,
) -> dict[str, float | int]:
    """Train using normalized anchor, positive, and negative embeddings."""
    model.train()
    total_loss, count, batches = 0.0, 0, 0
    for anchors, positives, negatives in islice(loader, max_batches):
        anchors, positives, negatives = (batch.to(device) for batch in (anchors, positives, negatives))
        optimizer.zero_grad(set_to_none=True)
        # One forward pass gives BatchNorm a common batch for all three roles.
        embeddings = model(torch.cat((anchors, positives, negatives), dim=0))
        anchor_embeddings, positive_embeddings, negative_embeddings = embeddings.chunk(3, dim=0)
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        if not torch.isfinite(loss):
            raise RuntimeError("Triplet loss became non-finite; check the inputs and learning rate.")
        loss.backward()
        optimizer.step()
        batch_count = anchors.size(0)
        total_loss += loss.item() * batch_count
        count += batch_count
        batches += 1
    if count == 0:
        raise ValueError("No triplet batches were available.")
    return {"loss": total_loss / count, "samples": count, "batches": batches}


def _save_checkpoint(
    path: Path, encoder: FaceEncoder, config: dict[str, Any], metadata: dict[str, Any],
    classifier: nn.Module | None = None,
) -> None:
    """Save encoder weights, plus a separate optional identity classifier head."""
    checkpoint = {
        "format_version": 1,
        "model_state": {name: tensor.detach().cpu().clone()
                        for name, tensor in encoder.state_dict().items()},
        "config": {key: config[key] for key in ("embedding_dim", "image_size", "method")},
        "training": metadata,
    }
    if classifier is not None:
        checkpoint["classifier_state"] = {
            name: tensor.detach().cpu().clone() for name, tensor in classifier.state_dict().items()
        }
    torch.save(checkpoint, path)


def _split_summary(root: Path, class_to_idx: dict[str, int]) -> dict[str, Any]:
    """Count known and excluded identity images, including empty directories."""
    summary: dict[str, Any] = {
        "root": str(root), "status": "missing_split",
        "included_identities": [], "included_identity_count": 0,
        "unknown_identities": [], "unknown_identity_count": 0,
        "excluded_identities": [], "excluded_identity_count": 0,
        "empty_identities": [], "empty_identity_count": 0,
        "total_identity_count": 0, "total_image_count": 0,
        "included_image_count": 0, "excluded_image_count": 0,
    }
    if not root.is_dir():
        return summary
    identity_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    counts = {
        directory.name: sum(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                            for path in directory.rglob("*"))
        for directory in identity_dirs
    }
    included = [name for name, count in counts.items() if name in class_to_idx and count > 0]
    unknown = [name for name in counts if name not in class_to_idx]
    empty = [name for name, count in counts.items() if count == 0]
    summary.update({
        "status": "available" if included else "no_known_images",
        "included_identities": included, "included_identity_count": len(included),
        "unknown_identities": unknown, "unknown_identity_count": len(unknown),
        "excluded_identities": unknown, "excluded_identity_count": len(unknown),
        "empty_identities": empty, "empty_identity_count": len(empty),
        "total_identity_count": len(counts), "total_image_count": sum(counts.values()),
        "included_image_count": sum(counts[name] for name in included),
        "excluded_image_count": sum(counts[name] for name in unknown),
    })
    return summary


def _evaluate_classification_test(
    model: FaceClassifier, best_path: Path, test_root: Path, output_dir: Path,
    image_size: int, batch_size: int, device: torch.device,
) -> dict[str, Any]:
    """Evaluate a selected encoder/head once without influencing model selection."""
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.encoder.load_state_dict(checkpoint["model_state"])
    model.classifier.load_state_dict(checkpoint["classifier_state"])
    class_to_idx = checkpoint["training"]["class_to_idx"]
    summary = _split_summary(test_root, class_to_idx)
    report = {
        "metric": "closed_set_known_identity_classification_accuracy",
        "checkpoint": str(best_path), "selected_epoch": checkpoint["training"]["epoch"],
        "checkpoint_selection": checkpoint["training"]["checkpoint_selection"],
        "used_for_selection": False, "test_batches_capped": False,
        "training_smoke_test": checkpoint["training"]["smoke_test"],
        "class_to_idx": class_to_idx, "split": summary,
        "status": summary["status"], "metrics": None,
    }
    if summary["included_image_count"]:
        test_dataset = FaceImageDataset(test_root, image_size, augment=False, class_to_idx=class_to_idx)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        report["metrics"] = _classification_epoch(
            model, test_loader, nn.CrossEntropyLoss(), device, None, max_batches=None,
        )
        report["status"] = "evaluated"
        print(f"Final known-ID classification test (best epoch {report['selected_epoch']}): "
              f"accuracy={report['metrics']['accuracy']:.4f}, "
              f"samples={report['metrics']['samples']}, "
              f"excluded IDs={summary['excluded_identity_count']}.", flush=True)
    else:
        print(f"Classification test skipped: {summary['status']} at {test_root}.", flush=True)
    (output_dir / "classification_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train(args: argparse.Namespace) -> Path:
    """Train on train_data and return the selected encoder checkpoint path."""
    if args.evaluate_test and args.method != "classification":
        raise ValueError("--evaluate-test is supported only with --method classification.")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("seed must be in the range 0 to 2**32 - 1.")
    if args.image_size < 32:
        raise ValueError("image-size must be at least 32 for the CNN's spatial downsampling.")
    seed_everything(args.seed)
    device = _device(args.device)
    torch.set_num_threads(min(args.num_threads, os.cpu_count() or 1))
    if device.type == "cpu":
        # Large default thread pools are slower than a few threads for small CNN batches.
        torch.set_num_threads(min(args.num_threads, os.cpu_count() or 1))
    split_root = _classification_root(args.data_root)
    validation_dataset = None
    validation_summary = None
    if args.method == "classification":
        training_dataset = FaceImageDataset(split_root / "train_data", args.image_size, augment=False)
        if len(training_dataset.class_to_idx) < 2:
            raise ValueError("Classification training requires at least two identities in train_data.")
        validation_path = split_root / "val_data"
        validation_summary = _split_summary(validation_path, training_dataset.class_to_idx)
        if validation_path.is_dir():
            try:
                validation_dataset = FaceImageDataset(
                    validation_path, args.image_size, augment=False,
                    class_to_idx=training_dataset.class_to_idx,
                )
            except EmptyFaceDatasetError as exc:
                print(f"Validation skipped: {exc}", flush=True)
        else:
            print(f"Validation skipped: no val_data directory at {validation_path}.", flush=True)
        model = FaceClassifier(len(training_dataset.class_to_idx), args.embedding_dim).to(device)
        encoder = model.encoder
        criterion = nn.CrossEntropyLoss()
    else:
        training_dataset = TripletFaceDataset(split_root / "train_data", args.image_size, seed=args.seed)
        model = FaceEncoder(args.embedding_dim).to(device)
        encoder = model
        criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    if getattr(args, "image_cache", None) is not None:
        from .image_cache import ImageCache
        for dataset in (training_dataset, validation_dataset):
            if dataset is not None:
                dataset.image_cache = ImageCache(args.image_cache, dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
    loader_generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=0, generator=loader_generator)
    validation_loader = (DataLoader(validation_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)
                         if validation_dataset is not None else None)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Reusing a run directory must not leave a previous run's test result beside new weights.
    (output_dir / "classification_test.json").unlink(missing_ok=True)
    selection_metric = "val_loss" if validation_loader is not None else "train_loss"
    config = {
        "method": args.method, "embedding_dim": args.embedding_dim,
        "image_size": args.image_size, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "seed": args.seed, "device": str(device), "max_batches": args.max_batches,
        "num_threads": torch.get_num_threads(),
        "image_cache": (training_dataset.image_cache.provenance if training_dataset.image_cache is not None else None),
        "smoke_test": args.max_batches is not None,
        "initialization": "random_from_scratch", "pretrained": False,
        "train_root": str(split_root / "train_data"),
        "training_examples": len(training_dataset),
        "class_to_idx": training_dataset.class_to_idx,
        "num_classes": len(training_dataset.class_to_idx),
        "validation_examples": len(validation_dataset) if validation_dataset is not None else 0,
        "validation_split": validation_summary,
        "checkpoint_selection": selection_metric,
        "triplet_margin": 0.2 if args.method == "triplet" else None,
        "evaluate_test_requested": args.evaluate_test,
        "classification_test_report": "classification_test.json" if args.evaluate_test else None,
        "augmentation": {
            "train": {"enabled": False, "operations": []},
            "validation": {"enabled": False}, "test": {"enabled": False},
        },
        "comparison_controls": "Both methods use the same encoder, preprocessing, seed, "
                               "and requested epochs; training augmentation is disabled for both.",
        "loss": ({"name": "CrossEntropyLoss", "input": "raw_identity_logits",
                  "label_smoothing": 0.0, "reduction": "mean"}
                 if args.method == "classification" else
                 {"name": "TripletMarginLoss", "input": "L2_normalized_embeddings",
                  "margin": 0.2, "p": 2, "eps": 1e-6, "swap": False, "reduction": "mean"}),
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate,
                      "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0, "amsgrad": False},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Training {args.method} from scratch on {device}: "
          f"{len(training_dataset)} examples; checkpoint selection = {selection_metric}.", flush=True)
    if args.max_batches is not None:
        print(f"Smoke test: capped at {args.max_batches} batches per split per epoch.", flush=True)

    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
                  "train_samples", "val_samples", "train_batches", "val_batches", "learning_rate"]
    best_loss, best_epoch = math.inf, 0
    best_path = output_dir / "best_model.pt"
    with (output_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fieldnames)
        writer.writeheader()
        history_file.flush()
        for epoch in range(1, args.epochs + 1):
            if args.method == "triplet":
                training_dataset.set_epoch(epoch - 1)
                train_metrics = _triplet_epoch(model, training_loader, criterion, device,
                                               optimizer, args.max_batches)
            else:
                train_metrics = _classification_epoch(model, training_loader, criterion,
                                                      device, optimizer, args.max_batches)
            val_metrics = (_classification_epoch(model, validation_loader, criterion,
                                                 device, None, args.max_batches)
                           if validation_loader is not None else None)
            selected_loss = float(val_metrics["loss"] if val_metrics is not None else train_metrics["loss"])
            improved = selected_loss < best_loss
            if improved:
                best_loss, best_epoch = selected_loss, epoch
            row = {
                "epoch": epoch, "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics.get("accuracy", ""),
                "val_loss": val_metrics["loss"] if val_metrics else "",
                "val_accuracy": val_metrics["accuracy"] if val_metrics else "",
                "train_samples": train_metrics["samples"],
                "val_samples": val_metrics["samples"] if val_metrics else 0,
                "train_batches": train_metrics["batches"],
                "val_batches": val_metrics["batches"] if val_metrics else 0,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            history_file.flush()
            metadata = {
                **config, "epoch": epoch, "best_epoch": best_epoch,
                "best_selection_loss": best_loss, "metrics": row,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "verification_pairs_used": False, "test_data_used": False,
                "test_data_used_for_selection": False,
            }
            if improved:
                _save_checkpoint(best_path, encoder, config, metadata,
                                 model.classifier if args.method == "classification" else None)
            _save_checkpoint(output_dir / "last_model.pt", encoder, config, metadata,
                             model.classifier if args.method == "classification" else None)
            details = f"train_loss={float(train_metrics['loss']):.6f}"
            if "accuracy" in train_metrics:
                details += f" train_accuracy={float(train_metrics['accuracy']):.4f}"
            if val_metrics is not None:
                details += (f" val_loss={float(val_metrics['loss']):.6f}"
                            f" val_accuracy={float(val_metrics['accuracy']):.4f}")
            print(f"Epoch {epoch}/{args.epochs}: {details}" + (" [best]" if improved else ""), flush=True)
    print(f"Saved best encoder from epoch {best_epoch}: {best_path}", flush=True)
    if args.evaluate_test:
        _evaluate_classification_test(model, best_path, split_root / "test_data", output_dir,
                                      args.image_size, args.batch_size, device)
    return best_path


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and report actionable training errors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Training error: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
