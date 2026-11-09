"""Decode/hash an official identity ZIP directly, avoiding repeated tiny-file I/O."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import zipfile

from PIL import Image, ImageOps

from .audit_data import sha256_file
from .data import IMAGE_EXTENSIONS
from .evaluate import _tokens, _HEADER_LABELS


def describe_overlaps(groups: list[dict], validation_images: set[str]) -> dict:
    combinations = Counter("+".join(sorted({row["split"] for row in group["images"]})) for group in groups)
    matched = {}
    for split in ("train_data", "val_data", "test_data"):
        names = {row["path"] for group in groups
                 if any(row["split"] == split for row in group["images"])
                 for row in group["images"] if row["split"] == "verification_data" and row["path"] in validation_images}
        matched[split] = len(names)
    return {"duplicate_group_split_combinations": dict(combinations),
            "verification_validation_images_matching_classification": matched}


def audit_archive(archive_path: str | Path, output_dir: str | Path) -> dict:
    archive_path = Path(archive_path).resolve()
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"archive": str(archive_path), "archive_sha256": sha256_file(archive_path),
              "images_decoded": 0, "unreadable_images": [], "errors": [],
              "original_pairs_preserved": True, "pairs_removed": 0,
              "subject_disjoint_verified": False}
    with zipfile.ZipFile(archive_path) as archive:
        entries = {entry.filename: entry for entry in archive.infolist() if not entry.is_dir()}
        if len(entries) != len([entry for entry in archive.infolist() if not entry.is_dir()]):
            raise ValueError("Duplicate ZIP member names.")
        candidates = []
        for name in entries:
            if PurePosixPath(name).name == "verification_pairs_val.txt":
                prefix = name[:-len("verification_pairs_val.txt")]
                if any(key.startswith(prefix + "classification_data/") for key in entries):
                    candidates.append(prefix)
        if len(candidates) != 1:
            raise ValueError("Expected exactly one official identity dataset root in ZIP.")
        prefix = candidates[0]
        members = {name[len(prefix):]: entry for name, entry in entries.items() if name.startswith(prefix)}
        image_members = {name: entry for name, entry in members.items()
                         if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS}
        counts = defaultdict(Counter); roles = {}; groups = defaultdict(list)
        for name in image_members:
            parts = PurePosixPath(name).parts
            if parts[0] == "classification_data" and len(parts) >= 4 and parts[1] in ("train_data", "val_data", "test_data"):
                split, identity = parts[1], parts[2]
                counts[split][identity] += 1
            elif parts[0] == "verification_data":
                split, identity = "verification_data", None
            else:
                raise ValueError(f"Unrecognized image placement: {name}")
            roles[name] = (split, identity)
        if set(counts) != {"train_data", "val_data", "test_data"}:
            raise ValueError("ZIP must contain all three classification splits.")
        def resolve(reference):
            reference = reference.replace("\\", "/")
            path = PurePosixPath(reference)
            if path.is_absolute() or ".." in path.parts or ":" in reference:
                raise ValueError("Pair references must remain within the dataset ZIP.")
            candidates = [str(path), "verification_data/" + str(path)]
            root_name = PurePosixPath(prefix.rstrip("/")).name if prefix else None
            if root_name and path.parts[0] == root_name:
                candidates.append("/".join(path.parts[1:]))
            matches = {name for name in candidates if name in image_members}
            if len(matches) != 1:
                raise ValueError(f"Missing or ambiguous verification reference: {reference}")
            return next(iter(matches))
        raw_pairs = archive.read(members["verification_pairs_val.txt"])
        pair_counts = Counter(); seen = {}; repeats = 0; conflicts = 0; self_positive = 0; self_negative = 0
        used_images = set(); first = True
        for line_number, line in enumerate(raw_pairs.decode("utf-8-sig").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", ";", "// ")):
                continue
            fields = _tokens(stripped)
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"Pair row {line_number} must have two paths and one label.")
            header = re.sub("[^a-z]", "", fields[2].lower())
            if first and header in _HEADER_LABELS:
                first = False; continue
            first = False
            label = float(fields[2])
            if label not in (0., 1.):
                raise ValueError(f"Pair row {line_number} has a nonbinary label.")
            left, right = resolve(fields[0]), resolve(fields[1])
            key = tuple(sorted((left, right)))
            if key in seen:
                repeats += 1; conflicts += seen[key] != int(label)
            seen[key] = int(label); pair_counts[int(label)] += 1
            used_images.update((left, right))
            if left == right:
                self_positive += int(label) == 1; self_negative += int(label) == 0
        if not pair_counts[0] or not pair_counts[1]:
            report["errors"].append("Verification AUC requires both pair labels.")
        if conflicts or self_negative:
            report["errors"].append("Contradictory pair labels or a negative self-pair.")
        report["verification"] = {"pairs": sum(pair_counts.values()), "same_person": pair_counts[1],
                                  "different_person": pair_counts[0], "unique_images": len(used_images),
                                  "repeated_unordered_pairs": repeats, "conflicting_pairs": conflicts,
                                  "self_positive_pairs": self_positive, "self_negative_pairs": self_negative,
                                  "pair_file_sha256": hashlib.sha256(raw_pairs).hexdigest()}
        report["classification"] = {split: {"images": sum(ids.values()), "identities": len(ids),
                                             "identities_with_two_or_more_images": sum(count >= 2 for count in ids.values()),
                                             "unknown_training_identities": sorted(set(ids) - set(counts["train_data"]))}
                                    for split, ids in counts.items()}
        with (output / "image_hashes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["archive_path", "split", "identity", "file_sha256", "pixel_sha256", "width", "height"])
            for index, (name, entry) in enumerate(image_members.items(), 1):
                split, identity = roles[name]
                try:
                    encoded = archive.read(entry)  # ZipExtFile checks the source CRC.
                    with Image.open(io.BytesIO(encoded)) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                        fingerprint = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode() + image.tobytes()).hexdigest()
                    groups[fingerprint].append((name, split, identity))
                    writer.writerow([name, split, identity, hashlib.sha256(encoded).hexdigest(), fingerprint, image.width, image.height])
                    report["images_decoded"] += 1
                except (OSError, ValueError) as exc:
                    report["unreadable_images"].append({"path": name, "error": str(exc)})
                if index % 10000 == 0:
                    print(f"Decoded archive images {index}/{len(image_members)}", flush=True)
        overlaps = []; conflicts_by_id = []
        for fingerprint, rows in groups.items():
            split_names = {row[1] for row in rows}; identities = {row[2] for row in rows if row[2] is not None}
            value = {"pixel_sha256": fingerprint, "images": [{"path": name, "split": split, "identity": identity} for name, split, identity in rows]}
            if len(split_names) > 1:
                overlaps.append(value)
            if len(identities) > 1:
                conflicts_by_id.append(value)
        duplicate_path = output / "duplicates.json"
        duplicate_path.write_text(json.dumps({"cross_split": overlaps, "different_identity_labels": conflicts_by_id}, indent=2) + "\n", encoding="utf-8")
        report.update(describe_overlaps(overlaps, used_images))
        if report["unreadable_images"]:
            report["errors"].append("Some source ZIP images cannot be decoded.")
        warning = bool(overlaps or conflicts_by_id or repeats or self_positive)
        report.update(archive_files=len(entries), image_files=len(image_members),
                      all_images_decoded=report["images_decoded"] == len(image_members),
                      decoded_duplicate_groups_cross_split=len(overlaps),
                      decoded_duplicate_groups_with_different_ids=len(conflicts_by_id),
                      duplicates_report=str(duplicate_path),
                      seconds=round(time.monotonic() - started, 1),
                      audit_status="not_ready" if report["errors"] else "ready_with_warnings" if warning else "ready",
                      note="Archive decoded directly; original images, IDs and verification pair rows are not filtered. On-disk CRC-preserving extraction is checked separately.")
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/identity_archive_audit"))
    args = parser.parse_args(argv)
    try:
        report = audit_archive(args.archive, args.output_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Identity archive audit failed: {exc}\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
