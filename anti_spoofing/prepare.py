"""Prepare subject-disjoint CelebA-Spoof CSVs without copying images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .data import MANIFEST_COLUMNS, _binary_label, _path_key, validate_manifest_splits, validate_record_splits


def sha256_file(path: str | Path) -> str:
    """Hash a source annotation or manifest for reproducible provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON image key: {key}")
        result[key] = value
    return result


def _official_record(raw_path: str, raw_label: Any, root: Path, expected_split: str | None) -> dict[str, Any]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Official annotation image paths must be nonempty strings.")
    raw_path = raw_path.strip()
    relative = PurePosixPath(raw_path.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(raw_path).drive or ".." in relative.parts:
        raise ValueError(f"Annotation path must stay inside data-root: {raw_path}")
    parts = relative.parts
    if (len(parts) < 5 or parts[0].lower() != "data" or
            parts[1].lower() not in ("train", "test") or parts[3].lower() not in ("live", "spoof")):
        raise ValueError(f"Expected Data/train|test/<subject>/live|spoof/image in annotation: {raw_path}")
    if expected_split is not None and parts[1].lower() != expected_split:
        raise ValueError(f"Expected official {expected_split} image, found {raw_path}")
    source_label = _binary_label(raw_label, raw_path)
    expected_label = 0 if parts[3].lower() == "live" else 1
    if source_label != expected_label:
        raise ValueError(f"Official source label {source_label} conflicts with {parts[3]} directory: {raw_path}")
    image_path = (root / Path(*parts)).resolve()
    if not image_path.is_relative_to(root):
        raise ValueError(f"Annotation path escapes data-root: {raw_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Annotated image does not exist: {image_path}")
    return {"image_path": str(image_path), "label": 1 - source_label,
            "subject_id": parts[2], "source_label": source_label}


def parse_official_labels(
    label_path: str | Path, data_root: str | Path, expected_split: str | None = None,
) -> list[dict[str, Any]]:
    """Read official JSON index 43, or text ``relative_path official_label`` rows."""
    label_path = Path(label_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Official annotation file does not exist: {label_path}")
    if expected_split not in (None, "train", "test"):
        raise ValueError("expected_split must be train, test, or None.")
    text = label_path.read_text(encoding="utf-8-sig")
    rows = []
    if label_path.suffix.lower() == ".json":
        try:
            labels = json.loads(text, object_pairs_hook=_unique_json_object)
        except ValueError as exc:
            raise ValueError(f"Invalid annotation JSON {label_path}: {exc}") from exc
        if not isinstance(labels, dict):
            raise ValueError(f"{label_path}: expected an object mapping relative image paths to attribute lists.")
        for path, attributes in labels.items():
            if not isinstance(attributes, list) or len(attributes) < 44:
                raise ValueError(f"{label_path}: {path} needs at least 44 attributes (official label at index 43).")
            rows.append((path, attributes[43]))
    else:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                raise ValueError(f"{label_path}:{line_number}: expected relative_path official_label.")
            rows.append((parts[0], parts[1]))
    records = []
    seen = {}
    for raw_path, raw_label in rows:
        record = _official_record(raw_path, raw_label, root, expected_split)
        key = _path_key(record["image_path"])
        if key in seen:
            kind = "conflicting" if seen[key] != record else "duplicate"
            raise ValueError(f"{label_path}: {kind} annotation image: {record['image_path']}")
        seen[key] = record
        records.append(record)
    if not records:
        raise ValueError(f"No labeled image records in {label_path}")
    return sorted(records, key=lambda record: record["image_path"])


def split_official_train(
    records: list[dict[str, Any]], val_fraction: float = 0.2, seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split whole subjects while retaining both canonical labels in each part."""
    if not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_subject.setdefault(record["subject_id"], []).append(record)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        raise ValueError("At least two official training subjects are required for a subject-disjoint validation split.")
    random.Random(seed).shuffle(subjects)
    n_val = max(1, min(len(subjects) - 1, int(round(len(subjects) * val_fraction))))
    groups: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for subject in subjects:
        labels = {record["label"] for record in by_subject[subject]}
        mask = sum(1 << label for label in labels)
        if mask not in groups:
            raise ValueError(f"Subject {subject} has invalid canonical labels.")
        groups[mask].append(subject)
    initial = set(subjects[:n_val])
    preferred = {mask: sum(subject in initial for subject in group) for mask, group in groups.items()}
    # Select class-support counts exactly, keeping the random subject order inside each group.
    a, b, c = (len(groups[mask]) for mask in (1, 2, 3))
    counts = None
    for both_count in sorted(range(max(0, n_val - a - b), min(c, n_val) + 1),
                             key=lambda count: (abs(count - preferred[3]), count)):
        low = max(0, n_val - both_count - b)
        high = min(a, n_val - both_count)
        if both_count == 0:
            low, high = max(low, 1), min(high, n_val - 1)
        if c - both_count == 0:
            low = max(low, n_val - both_count - b + 1)
            high = min(high, a - 1)
        if low <= high:
            spoof_only = min(high, max(low, preferred[1]))
            counts = {1: spoof_only, 2: n_val - both_count - spoof_only, 3: both_count}
            break
    if counts is None:
        raise ValueError("Cannot retain both live/spoof classes in train and validation with this subject split size. "
                         "Add subjects covering both classes or change --val-fraction.")
    val_subjects = {subject for mask, count in counts.items() for subject in groups[mask][:count]}
    train = [record for record in records if record["subject_id"] not in val_subjects]
    val = [record for record in records if record["subject_id"] in val_subjects]
    validate_record_splits({"train": train, "val": val})
    return train, val


def prepare_manifests(
    data_root: str | Path, output_dir: str | Path, train_labels: str | Path | None = None,
    test_labels: str | Path | None = None, val_fraction: float = 0.2, seed: int = 42,
) -> dict[str, Any]:
    """Prepare manifests using official train for train/val and official test for test."""
    root = Path(data_root).expanduser().resolve()
    train_labels = Path(train_labels).expanduser().resolve() if train_labels else root / "metas/intra_test/train_label.json"
    test_labels = Path(test_labels).expanduser().resolve() if test_labels else root / "metas/intra_test/test_label.json"
    official_train = parse_official_labels(train_labels, root, "train")
    test = parse_official_labels(test_labels, root, "test")
    validate_record_splits({"official_train": official_train, "official_test": test})
    train, val = split_official_train(official_train, val_fraction=val_fraction, seed=seed)
    splits = {"train": train, "val": val, "test": test}
    summary = validate_record_splits(splits)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
    validate_manifest_splits(output_dir / "train.csv", output_dir / "val.csv", output_dir / "test.csv")
    summary.update({
        "dataset": "CelebA-Spoof", "dataset_root": str(root), "seed": seed,
        "split_strategy": "subject_disjoint_validation_from_official_train",
        "requested_val_subject_fraction": val_fraction,
        "actual_val_subject_fraction": summary["splits"]["val"]["subjects"] / len({record["subject_id"] for record in official_train}),
        "test_used_for_splitting_or_tuning": False,
        "label_conversion": "canonical_label = 1 - official_source_label",
        "bbox_convention": "image-stem_BB.txt: x,y,w,h,score in224x224; scale to stored image dimensions; no EXIF transpose",
        "source_files": {
            "official_train": {"path": str(train_labels), "sha256": sha256_file(train_labels)},
            "official_test": {"path": str(test_labels), "sha256": sha256_file(test_labels)},
        },
        "manifests": {name: {"path": str(output_dir / f"{name}.csv"),
                             "sha256": sha256_file(output_dir / f"{name}.csv")}
                      for name in splits},
    })
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        summary = prepare_manifests(args.data_root, args.output_dir, args.train_labels,
                                    args.test_labels, args.val_fraction, args.seed)
    except (OSError, ValueError) as exc:
        print(f"Preparation error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary["splits"], indent=2), flush=True)
    print(f"Prepared manifests: {args.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
