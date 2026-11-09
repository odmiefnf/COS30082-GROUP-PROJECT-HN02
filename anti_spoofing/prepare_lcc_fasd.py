"""Prepare LCC-FASD manifests from its official directories, without copying images.

Training/development retain IDs parsed from filenames. To keep those IDs
disjoint, every training image whose numeric ID occurs in development is
excluded, with the complete exclusion recorded. Development is kept intact;
byte-identical training/validation content is removed from evaluation.
Anonymous evaluation names such as real_0.png keep their
identity explicitly unavailable; a split name is never invented as an ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator

from .data import MANIFEST_COLUMNS, validate_manifest_splits, validate_record_splits


SPLIT_DIRECTORIES = {
    "train": "LCC_FASD_training",
    "val": "LCC_FASD_development",
    "test": "LCC_FASD_evaluation",
}
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
REFERENCE_LOADER_URL = "https://github.com/kprokofi/light-weight-face-anti-spoofing/blob/master/datasets/lcc_fasd.py"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path, context: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{context} escapes its dataset/split root, including through a symlink: {path} -> {resolved}")
    return resolved


def _named_directory(parent: Path, name: str) -> Path | None:
    matches = [child for child in parent.iterdir() if child.name.casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous directory name {name!r} under {parent}: {matches}")
    if not matches:
        return None
    candidate = matches[0]
    _inside(candidate, parent.resolve(), "Dataset directory")
    if not candidate.is_dir():
        raise ValueError(f"Expected a directory at {candidate}.")
    return candidate


def resolve_data_root(data_root: str | Path) -> Path:
    """Resolve either the split-containing root or one enclosing LCC_FASD folder.

    Multiple plausible layouts are rejected rather than silently choosing one.
    A recognized wrapper or split directory may not redirect outside its root.
    """
    requested = Path(data_root).expanduser().resolve()
    if not requested.is_dir():
        raise FileNotFoundError(f"LCC-FASD dataset root does not exist: {requested}")

    def present(root: Path) -> bool:
        return any(child.name.casefold() in {value.casefold() for value in SPLIT_DIRECTORIES.values()}
                   for child in root.iterdir())

    wrapper = _named_directory(requested, "LCC_FASD")
    candidates = ([requested] if present(requested) else [])
    if wrapper is not None and present(wrapper):
        candidates.append(wrapper)
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous LCC-FASD layout under {requested}: official split directories occur "
            "both directly and inside LCC_FASD. Pass the intended dataset directory explicitly."
        )
    root = (candidates[0] if candidates else wrapper or requested).resolve()
    for official_name in SPLIT_DIRECTORIES.values():
        split_path = _named_directory(root, official_name)
        if split_path is None:
            raise FileNotFoundError(
                f"Missing official LCC-FASD split: {root / official_name}. "
                "Pass the directory containing all three LCC_FASD_training, "
                "LCC_FASD_development and LCC_FASD_evaluation folders, or their single LCC_FASD wrapper."
            )
        _inside(split_path, root, "Official split directory")
    return root


def extract_subject_id(filename: str | Path) -> str:
    """Return the unique canonical filename ID (``_id001_`` becomes ``id1``).

    The same ID remains the same across splits. Unknown and multiply specified
    IDs are errors; this helper never supplies anonymous evaluation IDs.
    """
    # Accept both filename separator conventions independently of the host OS.
    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(basename).stem
    matches = re.findall(r"(?=_id([0-9]+)_)", stem, flags=re.IGNORECASE)
    if len(matches) != 1:
        detail = "no _id<digits>_ token" if not matches else "multiple _id<digits>_ tokens"
        raise ValueError(f"Cannot determine the true subject ID from {basename!r}: {detail}.")
    return "id" + (matches[0].lstrip("0") or "0")


def _prefix_label(stem: str) -> int | None:
    lowered = stem.casefold()
    if lowered.startswith("real"):
        return 1
    if lowered.startswith("spoof"):
        return 0
    return None


def _label_for_image(relative: Path, split: str) -> int:
    folder_labels = {1 if part.casefold() == "real" else 0
                     for part in relative.parts[:-1] if part.casefold() in ("real", "spoof")}
    if len(folder_labels) > 1:
        raise ValueError(f"Ambiguous class folders real/spoof for {relative} in {split}.")
    prefix = _prefix_label(relative.stem)
    if folder_labels:
        label = next(iter(folder_labels))
        if prefix is not None and prefix != label:
            raise ValueError(f"Filename class prefix conflicts with its real/spoof folder: {relative}.")
        return label
    if split == "test" and len(relative.parts) == 1 and prefix is not None:
        return prefix
    raise ValueError(
        f"Cannot determine the class for {relative} in {split}. Use real/spoof class folders; "
        "only official evaluation also supports flat filenames starting with real or spoof."
    )


def _iter_images(split_path: Path, root: Path) -> Iterator[tuple[Path, Path]]:
    """Walk deterministically, checking directory links before following them."""
    boundary = _inside(split_path, root, "Official split directory")
    pending = [split_path]
    visited: set[str] = set()
    while pending:
        directory = pending.pop()
        resolved_directory = _inside(directory, root, "Image directory")
        _inside(directory, boundary, "Image directory")
        key = os.path.normcase(str(resolved_directory))
        if key in visited:
            raise ValueError(f"Directory alias or symlink cycle duplicates an official image directory: {directory}")
        visited.add(key)
        for child in sorted(directory.iterdir(), key=lambda path: (path.name.casefold(), path.name)):
            resolved = _inside(child, root, "Dataset entry")
            _inside(child, boundary, "Dataset entry")
            if child.is_dir():
                pending.append(child)
            elif child.is_file() and child.suffix.casefold() in IMAGE_EXTENSIONS:
                yield child, resolved


def read_split_records(data_root: str | Path, split: str) -> list[dict[str, Any]]:
    """Read one official split, including explicitly anonymous evaluation records.

    Known identities have ``subject_id_source='filename'``. Only evaluation
    files named real_<number> or spoof_<number> may have an empty subject ID
    and ``subject_id_source='unavailable'``.
    """
    if split not in SPLIT_DIRECTORIES:
        raise ValueError("split must be train, val, or test.")
    root = resolve_data_root(data_root)
    split_path = _named_directory(root, SPLIT_DIRECTORIES[split])
    assert split_path is not None  # resolve_data_root has already checked every official split.
    records = []
    for image, resolved in _iter_images(split_path, root):
        relative = image.relative_to(split_path)
        label = _label_for_image(relative, split)
        anonymous = (split == "test" and re.fullmatch(r"(?:real|spoof)_[0-9]+", image.stem, flags=re.IGNORECASE))
        if anonymous:
            subject, source = "", "unavailable"
        else:
            subject, source = extract_subject_id(image.name), "filename"
        if image.is_symlink():
            # A file alias may not change the apparent class/ID of its target.
            target_relative = resolved.relative_to(split_path.resolve())
            target_label = _label_for_image(target_relative, split)
            if target_label != label:
                raise ValueError(f"Image symlink changes the class label: {image} -> {resolved}")
            if source == "filename" and extract_subject_id(resolved.name) != subject:
                raise ValueError(f"Image symlink changes the subject ID: {image} -> {resolved}")
        records.append({"image_path": str(resolved), "label": label,
                        "subject_id": subject, "source_label": 1 - label,
                        "subject_id_source": source})
    if not records:
        raise ValueError(f"No supported LCC-FASD images were found in {split_path}.")
    return sorted(records, key=lambda record: record["image_path"])


def _summary(records: list[dict[str, Any]]) -> dict:
    subjects = {record["subject_id"] for record in records if record["subject_id"]}
    return {"samples": len(records), "live": sum(record["label"] == 1 for record in records),
            "spoof": sum(record["label"] == 0 for record in records), "subjects": len(subjects),
            "identity_unavailable_samples": sum(record["subject_id_source"] == "unavailable" for record in records)}


def _id_sort(subject: str) -> tuple[int, str]:
    digits = subject.removeprefix("id")
    return len(digits), digits


def prepare_manifests(data_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Prepare CSVs and hashed provenance using the documented conservative protocol.

    The official development set remains intact. Numeric IDs
    shared by official training/development cause ALL corresponding training
    records to be excluded. This is a documented protocol modification, not a
    random repartition. Training content duplicated in validation is excluded;
    evaluation content already in train/validation is excluded from evaluation.
    Evaluation never determines which images are retained for training.
    Unknown evaluation identity is never reported verified.
    """
    root = resolve_data_root(data_root)
    official = {split: read_split_records(root, split) for split in SPLIT_DIRECTORIES}
    original_splits = {split: _summary(records) for split, records in official.items()}
    development_ids = {record["subject_id"] for record in official["val"]}
    overlap_ids = {record["subject_id"] for record in official["train"]} & development_ids
    excluded = [record for record in official["train"] if record["subject_id"] in overlap_ids]
    splits = {
        "train": [record for record in official["train"] if record["subject_id"] not in overlap_ids],
        "val": official["val"], "test": official["test"],
    }
    if not splits["train"]:
        raise ValueError("No training images remain after excluding all training IDs overlapping development. "
                         "Check the complete official dataset; no random replacement split will be invented.")
    inventory = []
    for split, records in official.items():
        for record in records:
            image = Path(record["image_path"])
            inventory.append({
                "official_split": split, "relative_path": image.relative_to(root).as_posix(),
                "label": record["label"], "source_label": record["source_label"],
                "subject_id": record["subject_id"], "subject_id_source": record["subject_id_source"],
                "image_sha256": _sha256(image), "bytes": image.stat().st_size,
                "included_in_manifest": not (split == "train" and record["subject_id"] in overlap_ids),
            })
    hashes = {str((root / row["relative_path"]).resolve()): row["image_sha256"] for row in inventory}
    labels_by_hash: dict[str, set[int]] = {}
    for row in inventory:
        labels_by_hash.setdefault(row["image_sha256"], set()).add(row["label"])
    if any(len(labels) > 1 for labels in labels_by_hash.values()):
        raise ValueError("Identical image bytes have conflicting class labels; resolve the source annotations first.")
    val_hashes = {hashes[row["image_path"]] for row in splits["val"]}
    train_content_excluded = [row for row in splits["train"] if hashes[row["image_path"]] in val_hashes]
    splits["train"] = [row for row in splits["train"] if hashes[row["image_path"]] not in val_hashes]
    development_hashes = val_hashes | {hashes[row["image_path"]] for row in splits["train"]}
    test_content_excluded = [row for row in splits["test"] if hashes[row["image_path"]] in development_hashes]
    splits["test"] = [row for row in splits["test"] if hashes[row["image_path"]] not in development_hashes]
    retained = {row["image_path"] for records in splits.values() for row in records}
    for row in inventory:
        path = str((root / row["relative_path"]).resolve())
        row["included_in_manifest"] = path in retained
        row["exclusion_reason"] = (None if path in retained else
                                   "training filename ID overlaps validation" if row["official_split"] == "train" and row["subject_id"] in overlap_ids else
                                   "training bytes duplicate validation" if row["official_split"] == "train" else
                                   "evaluation bytes duplicate train/validation")
    summary = validate_record_splits(splits)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    columns = tuple(dict.fromkeys((*MANIFEST_COLUMNS, "subject_id_source")))
    for split, records in splits.items():
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    # Re-read actual outputs so the existing loader validates the exact CSV schema.
    validate_manifest_splits(output / "train.csv", output / "val.csv", output / "test.csv")
    inventory_path = output / "source_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    unknown_test = any(record["subject_id_source"] == "unavailable" for record in splits["test"])
    summary.update({
        "dataset": "LCC-FASD", "dataset_root": str(root),
        "official_split_mapping": SPLIT_DIRECTORIES,
        "split_strategy": "official_splits_with_train_ids_overlapping_development_removed_and_cross_split_content_deduplicated",
        "protocol_modified": True,
        "official_development_and_evaluation_preserved": not bool(test_content_excluded),
        "official_development_preserved": True,
        "official_evaluation_subset_after_content_deduplication": bool(test_content_excluded),
        "random_repartition": False,
        "test_used_for_splitting_or_tuning": False,
        "original_splits": original_splits,
        "actual_splits": {split: _summary(records) for split, records in splits.items()},
        "excluded_train_overlap": {
            "subject_ids": sorted(overlap_ids, key=_id_sort), "samples": len(excluded),
            "records": excluded,
            "rule": "exclude every official training record whose normalized filename ID occurs in official development",
        },
        "excluded_train_content_overlap": {
            "samples": len(train_content_excluded), "records": train_content_excluded,
            "rule": "exclude training bytes already present in validation; evaluation never filters training",
        },
        "excluded_test_content_overlap": {
            "samples": len(test_content_excluded), "records": test_content_excluded,
            "rule": "exclude evaluation bytes already present in retained training or validation; no model scores used",
        },
        "subject_id_convention": "unique _id<digits>_ filename token; normalize leading zeroes; never prefix with split",
        "subject_id_source_values": {"filename": "verified filename token", "unavailable": "anonymous evaluation filename; empty subject_id"},
        "identity_verification": {
            "train_val": "normalized filename IDs disjoint after documented training exclusions",
            "test": "unavailable: anonymous evaluation filenames; do not claim identity-disjoint test results" if unknown_test
                    else "filename IDs available and checked against train/validation",
        },
        "crop_mode": "full", "bbox_required": False,
        "preprocessing_note": "Use the supplied face crop as the full input; no CelebA-Spoof bounding-box sidecar is used.",
        "canonical_labels": {"0": "SPOOF", "1": "LIVE (real)"},
        "source_labels": {"0": "real", "1": "spoof"},
        "label_conversion": "canonical_label = 1 - source_label",
        "source_files": {"image_inventory": {"path": str(inventory_path), "sha256": _sha256(inventory_path),
                                             "samples": len(inventory), "bytes": sum(row["bytes"] for row in inventory)},
                         "preparation_module": {"path": str(Path(__file__).resolve()), "sha256": _sha256(__file__)}},
        "reference_loader": REFERENCE_LOADER_URL,
        "manifests": {split: {"path": str(output / f"{split}.csv"), "sha256": _sha256(output / f"{split}.csv")}
                      for split in splits},
        "counts_note": "Counts are measured from these extracted files; no paper or mirror totals are assumed.",
    })
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing the three official split folders, or one LCC_FASD wrapper.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = prepare_manifests(args.data_root, args.output_dir)
    except (OSError, ValueError) as exc:
        print(f"LCC-FASD preparation failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(summary["actual_splits"], indent=2), flush=True)
    print(f"Excluded {summary['excluded_train_overlap']['samples']} training images whose IDs overlap development.", flush=True)
    print(f"Excluded identical content: {summary['excluded_train_content_overlap']['samples']} training and "
          f"{summary['excluded_test_content_overlap']['samples']} evaluation images.", flush=True)
    print(f"Prepared LCC-FASD manifests: {args.output_dir.expanduser().resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SPLIT_DIRECTORIES", "resolve_data_root", "extract_subject_id", "read_split_records",
           "prepare_manifests", "build_parser", "main"]
