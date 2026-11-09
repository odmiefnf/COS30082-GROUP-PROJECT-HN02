"""Check the identity dataset and verification trials before training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def inspect_data(data_root: str | Path) -> dict:
    """Report readiness for both baselines without modifying any dataset files."""
    from .data import FaceImageDataset
    from .evaluate import read_verification_pairs

    root = Path(data_root).expanduser().resolve()
    report = {"data_root": str(root), "status": "not_ready", "errors": [], "splits": {}}
    train = None
    for split in ("train_data", "val_data", "test_data"):
        folder = root / "classification_data" / split
        if not folder.is_dir():
            report["splits"][split] = {"present": False, "path": str(folder)}
            if split == "train_data":
                report["errors"].append(f"Missing training folder: {folder}")
            continue
        try:
            dataset = FaceImageDataset(folder)
            counts = Counter(label for _, label in dataset.samples)
            report["splits"][split] = {
                "present": True,
                "image_count": len(dataset),
                "identity_count": len(dataset.classes),
                "identities_with_two_or_more_images": sum(n >= 2 for n in counts.values()),
                "empty_identity_folders": list(dataset.empty_classes),
            }
            if split == "train_data":
                train = dataset
                if len(dataset.classes) < 2:
                    report["errors"].append("Classification training needs at least two identities.")
                if sum(n >= 2 for n in counts.values()) < 2:
                    report["errors"].append(
                        "Triplet training needs at least two identities with two images each."
                    )
            elif train is not None:
                known = set(train.classes)
                report["splits"][split]["unknown_training_identities"] = sorted(
                    set(dataset.classes) - known
                )
        except (OSError, ValueError) as exc:
            report["errors"].append(f"{split}: {exc}")

    pairs_file = root / "verification_pairs_val.txt"
    try:
        pairs = read_verification_pairs(pairs_file, root)
        labels = Counter(int(label) for _, _, label in pairs)
        report["verification"] = {
            "pairs_file": str(pairs_file),
            "pair_count": len(pairs),
            "same_person_pairs": labels[1],
            "different_person_pairs": labels[0],
            "unique_images": len({path for left, right, _ in pairs for path in (left, right)}),
        }
        if not labels[0] or not labels[1]:
            report["errors"].append("ROC/AUC requires both same-person and different-person pairs.")
    except (OSError, ValueError) as exc:
        report["errors"].append(f"verification: {exc}")
    report["status"] = "ready" if not report["errors"] else "not_ready"
    report["note"] = "This checks structure and paths, not every image's decode or dataset quality."
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/face_recognition"))
    parser.add_argument("--output", type=Path, help="Optional diagnostic JSON output.")
    args = parser.parse_args(argv)
    report = inspect_data(args.data_root)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
