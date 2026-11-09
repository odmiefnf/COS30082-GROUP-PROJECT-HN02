"""Diagnose liveness manifests or raw CelebA-Spoof inputs without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .data import (bbox_sidecar_path, load_liveness_image, load_manifest,
                   validate_manifest_splits, validate_record_splits)
from .prepare import parse_official_labels


def _inspect_records(records: list[dict[str, Any]], crop_mode: str, inspect_images: int) -> dict[str, Any]:
    missing = ([str(bbox_sidecar_path(record["image_path"])) for record in records
                if not bbox_sidecar_path(record["image_path"]).is_file()]
               if crop_mode == "bbox" else [])
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} bounding-box sidecar(s); first entries: {missing[:10]}")
    inspected = 0
    for record in records[:inspect_images]:
        load_liveness_image(record["image_path"], crop_mode=crop_mode)
        inspected += 1
    return {"samples": len(records), "bbox_sidecar_presence_checked": crop_mode == "bbox",
            "images_decoded": inspected, "crop_mode": crop_mode}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="Root containing Data/ and metas/.")
    parser.add_argument("--manifest-dir", type=Path, help="Directory containing train.csv, val.csv, test.csv.")
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--test-labels", type=Path)
    parser.add_argument("--crop-mode", choices=("bbox", "full"), default="bbox")
    parser.add_argument("--inspect-images", type=int, default=0,
                        help="Decode up to this many images per split (default: 0; always check paths/sidecar presence).")
    args = parser.parse_args(argv)
    if args.data_root is None and args.manifest_dir is None:
        parser.error("provide --data-root and/or --manifest-dir; no dataset is bundled or downloaded")
    if args.inspect_images < 0:
        parser.error("--inspect-images must be nonnegative")
    if (args.train_labels or args.test_labels) and args.data_root is None:
        parser.error("--train-labels/--test-labels require --data-root")
    report: dict[str, Any] = {}
    try:
        if args.data_root is not None:
            root = args.data_root.expanduser().resolve()
            train_labels = args.train_labels or root / "metas/intra_test/train_label.json"
            test_labels = args.test_labels or root / "metas/intra_test/test_label.json"
            train = parse_official_labels(train_labels, root, "train")
            test = parse_official_labels(test_labels, root, "test")
            report["official_splits"] = validate_record_splits({"official_train": train, "official_test": test})
            report["official_checks"] = {
                "official_train": _inspect_records(train, args.crop_mode, args.inspect_images),
                "official_test": _inspect_records(test, args.crop_mode, args.inspect_images),
            }
        if args.manifest_dir is not None:
            directory = args.manifest_dir.expanduser().resolve()
            paths = {name: directory / f"{name}.csv" for name in ("train", "val", "test")}
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing manifests: " + ", ".join(missing) + ". Run anti_spoofing.prepare_lcc_fasd (LCC-FASD) or anti_spoofing.prepare (CelebA-Spoof) after obtaining the data.")
            report["manifest_splits"] = validate_manifest_splits(paths["train"], paths["val"], paths["test"])
            report["manifest_checks"] = {
                name: _inspect_records(load_manifest(path), args.crop_mode, args.inspect_images)
                for name, path in paths.items()
            }
    except (OSError, ValueError) as exc:
        print(f"Data check failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(json.dumps(report, indent=2), flush=True)
    print("Local data checks passed. No data was copied, created, or downloaded.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
