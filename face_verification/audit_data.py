"""Audit the assignment identity dataset without changing images or pair rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageOps

from .check_data import inspect_data
from .data import FaceImageDataset, IMAGE_EXTENSIONS
from .evaluate import read_verification_pairs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_root: str | Path, decode_images: bool = True) -> dict:
    root = Path(data_root).expanduser().resolve()
    report = inspect_data(root)
    report.update(audit_status="not_ready", images_decoded=0, unreadable_images=[], warnings=[],
                  pair_file_modified=False, pairs_removed=0, subject_disjoint_verified=False)
    required = [root / "classification_data" / name for name in ("train_data", "val_data", "test_data")]
    required += [root / "verification_data", root / "verification_pairs_val.txt"]
    for path in required:
        if not path.exists():
            message = f"Section 5 requires: {path}"
            if message not in report["errors"]:
                report["errors"].append(message)
    if report["errors"]:
        return report
    pair_file = root / "verification_pairs_val.txt"
    before = sha256_file(pair_file)
    pairs = read_verification_pairs(pair_file, root)
    seen_pairs = {}; repeated = 0; conflicts = []; self_positive = 0; self_negative = 0
    for left, right, label in pairs:
        key = tuple(sorted((os.path.normcase(str(left)), os.path.normcase(str(right)))))
        if key in seen_pairs:
            repeated += 1
            if seen_pairs[key] != label:
                conflicts.append({"images": list(key), "labels": [seen_pairs[key], label]})
        seen_pairs[key] = label
        if left == right:
            self_positive += label == 1
            self_negative += label == 0
    report["pair_integrity"] = {
        "sha256": before, "rows": len(pairs), "repeated_unordered_pairs": repeated,
        "conflicting_pairs": conflicts, "self_positive_pairs": self_positive,
        "self_negative_pairs": self_negative,
    }
    if conflicts or self_negative:
        report["errors"].append("Pair file contains contradictory labels or a self-pair labelled different-person.")
    if repeated or self_positive:
        report["warnings"].append("Repeated/self-positive pairs are preserved in the original evaluation protocol.")

    roles = defaultdict(list)
    for split in ("train_data", "val_data", "test_data"):
        dataset = FaceImageDataset(root / "classification_data" / split)
        for path, label in dataset.samples:
            roles[path].append({"split": split, "identity": dataset.classes[label]})
    pair_paths = {path for left, right, _ in pairs for path in (left, right)}
    verification_paths = {path for path in (root / "verification_data").rglob("*")
                          if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    for path in verification_paths:
        roles[path].append({"split": "verification_data", "identity": None,
                            "used_in_validation_pairs": path.resolve() in pair_paths})
    for path in pair_paths - {path.resolve() for path in verification_paths}:
        roles[path].append({"split": "verification_pairs_val", "identity": None})
    report["verification_images_present"] = len(verification_paths)
    report["verification_images_used_in_pairs"] = len(pair_paths)
    unsafe = set()
    for path in roles:
        if not path.resolve().is_relative_to(root):
            report["errors"].append(f"Referenced image escapes the dataset root: {path}")
            unsafe.add(path)
    report["unique_images_referenced"] = len(roles)
    report["includes_unlabelled_verification_images"] = True
    path_overlap = [str(path) for path, entries in roles.items() if len({row["split"] for row in entries}) > 1]
    report["path_overlap"] = path_overlap
    groups = defaultdict(list)
    if decode_images:
        for index, (path, entries) in enumerate(sorted(roles.items()), 1):
            if path in unsafe:
                continue
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    pixels = image.tobytes()
                    header = f"{image.width}x{image.height}:RGB:".encode("ascii")
                    fingerprint = hashlib.sha256(header + pixels).hexdigest()
                groups[fingerprint].append({"image_path": str(path), "roles": entries})
                report["images_decoded"] += 1
            except (OSError, ValueError) as exc:
                report["unreadable_images"].append({"path": str(path), "error": str(exc)})
            if index % 10000 == 0:
                print(f"Audited {index}/{len(roles)} identity images", flush=True)
        overlaps = []; conflicts_by_identity = []
        for fingerprint, rows in groups.items():
            split_names = {role["split"] for row in rows for role in row["roles"]}
            identities = {role["identity"] for row in rows for role in row["roles"] if role["identity"] is not None}
            if len(split_names) > 1:
                overlaps.append({"pixel_sha256": fingerprint, "splits": sorted(split_names), "images": rows})
            if len(identities) > 1:
                conflicts_by_identity.append({"pixel_sha256": fingerprint, "identities": sorted(identities), "images": rows})
        report["decoded_pixel_overlap"] = overlaps
        report["identical_pixels_under_different_ids"] = conflicts_by_identity
        if overlaps or conflicts_by_identity:
            report["warnings"].append("Identical images cross splits or identity labels; review before claiming independent verification performance.")
    report["all_referenced_images_decoded"] = decode_images and report["images_decoded"] == len(roles)
    if report["unreadable_images"]:
        report["errors"].append("Some identity images cannot be decoded.")
    if before != sha256_file(pair_file):
        raise RuntimeError("The official pair file changed during the read-only audit.")
    report["audit_status"] = "ready_with_warnings" if not report["errors"] and report["warnings"] else "ready" if not report["errors"] else "not_ready"
    report["note"] = "Official files/pairs are never rewritten or filtered. Hashes identify exact decoded duplicates, not unknown identities or near-duplicate photos."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/face_recognition"))
    parser.add_argument("--output", type=Path, default=Path("data/identity_audit.json"))
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.data_root, decode_images=not args.structure_only)
    except (OSError, ValueError) as exc:
        report = {"audit_status": "not_ready", "errors": [str(exc)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_status": report["audit_status"], "errors": report["errors"], "report": str(args.output.resolve())}, ensure_ascii=True, indent=2))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
