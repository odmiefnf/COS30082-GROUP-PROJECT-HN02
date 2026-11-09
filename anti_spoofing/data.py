"""Liveness manifests and raw-coordinate face preprocessing.

Canonical model labels are 1=LIVE and 0=SPOOF. Official CelebA-Spoof labels
use the inverse convention, 0=live and 1=spoof. Images are never EXIF-transposed:
sidecar coordinates and image pixels both use the stored image orientation.
"""

from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LIVE = 1
SPOOF = 0
MANIFEST_COLUMNS = ("image_path", "label", "subject_id", "source_label")


def _binary_label(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) not in ("0", "1"):
        raise ValueError(f"{context}: expected an integer label 0 or 1, got {value!r}.")
    return int(value)


def _image_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("image_size must be a positive integer.")
    return value


def _path_key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _check_path_label(path: Path, label: int, context: str) -> None:
    for part in reversed(path.parts[:-1]):
        if part.lower() in ("live", "real", "spoof"):
            expected = LIVE if part.lower() in ("live", "real") else SPOOF
            if label != expected:
                raise ValueError(f"{context}: canonical label {label} conflicts with {part!r} directory: {path}")
            break
    if "lcc_fasd_evaluation" in {part.casefold() for part in path.parts}:
        prefix = path.stem.lower().split("_", 1)[0]
        if prefix in ("real", "spoof") and label != (LIVE if prefix == "real" else SPOOF):
            raise ValueError(f"{context}: canonical label conflicts with evaluation filename: {path}")


def load_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Read canonical records, rejecting missing images and inconsistent labels."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    records: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(MANIFEST_COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"{manifest_path}: CSV requires columns {', '.join(MANIFEST_COLUMNS)}.")
        for line_number, row in enumerate(reader, start=2):
            context = f"{manifest_path}:{line_number}"
            if None in row or any(row.get(column) is None for column in MANIFEST_COLUMNS):
                raise ValueError(f"{context}: malformed CSV row.")
            label = _binary_label(row["label"].strip(), context)
            source_label = _binary_label(row["source_label"].strip(), context)
            if label != 1 - source_label:
                raise ValueError(f"{context}: canonical label must equal 1 - official source_label.")
            subject_id = row["subject_id"].strip()
            subject_source = (row.get("subject_id_source") or "provided").strip()
            if subject_source not in ("provided", "filename", "unavailable"):
                raise ValueError(f"{context}: unsupported subject_id_source {subject_source!r}.")
            raw_path = row["image_path"].strip()
            if not raw_path or (not subject_id and subject_source != "unavailable"):
                raise ValueError(f"{context}: image_path and subject_id must be nonempty unless the subject source is explicitly unavailable.")
            if subject_id and subject_source == "unavailable":
                raise ValueError(f"{context}: an unavailable subject ID must be empty, not fabricated.")
            path = Path(raw_path).expanduser()
            path = (path if path.is_absolute() else manifest_path.parent / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{context}: image does not exist: {path}")
            _check_path_label(path, label, context)
            # Where official path structure is present, do not trust a changed subject column.
            parts = path.parts
            parts_lower = {part.casefold() for part in parts}
            for index in range(len(parts) - 4):
                if parts[index].lower() == "data" and parts[index + 1].lower() in ("train", "test"):
                    if parts[index + 2] != subject_id:
                        raise ValueError(f"{context}: subject_id conflicts with image path: {path}")
                    break
            if parts_lower & {"lcc_fasd_training", "lcc_fasd_development", "lcc_fasd_evaluation"}:
                ids = re.findall(r"(?:^|_)id(\d+)(?=_|$)", path.stem, flags=re.IGNORECASE)
                if ids and (len(ids) != 1 or subject_id != f"id{int(ids[0])}"):
                    raise ValueError(f"{context}: subject_id conflicts with LCC-FASD filename: {path}")
                if not ids and ("lcc_fasd_evaluation" not in parts_lower or subject_source != "unavailable"):
                    raise ValueError(f"{context}: LCC-FASD training/development require a filename subject ID: {path}")
            record = {"image_path": str(path), "label": label,
                      "subject_id": subject_id, "source_label": source_label,
                      "subject_id_source": subject_source}
            key = _path_key(path)
            if key in seen:
                kind = "conflicting" if seen[key] != record else "duplicate"
                raise ValueError(f"{context}: {kind} image record: {path}")
            seen[key] = record
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no image records: {manifest_path}")
    return records


def _records_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    return {"samples": len(records), "live": sum(record["label"] == LIVE for record in records),
            "spoof": sum(record["label"] == SPOOF for record in records),
            "subjects": len({record["subject_id"] for record in records if record["subject_id"]}),
            "samples_without_subject_id": sum(not record["subject_id"] for record in records)}


def validate_record_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Check paths and known subjects; allow explicitly anonymous evaluation only."""
    owners_by_path: dict[str, str] = {}
    owners_by_subject: dict[str, str] = {}
    summaries = {}
    missing_subject_ids = False
    for split, records in splits.items():
        if {record["label"] for record in records} != {SPOOF, LIVE}:
            raise ValueError(f"{split}: both canonical classes 0=SPOOF and 1=LIVE are required.")
        for record in records:
            path_key = _path_key(record["image_path"])
            if path_key in owners_by_path:
                raise ValueError(f"Duplicate image path in {owners_by_path[path_key]} and {split}: {record['image_path']}")
            owners_by_path[path_key] = split
            subject = record["subject_id"]
            if not subject:
                if split not in ("test", "evaluation") or record.get("subject_id_source") != "unavailable":
                    raise ValueError(f"{split}: training and validation require known subject IDs.")
                missing_subject_ids = True
                continue
            if subject in owners_by_subject and owners_by_subject[subject] != split:
                raise ValueError(f"Subject overlap between {owners_by_subject[subject]} and {split}: {subject}")
            owners_by_subject[subject] = split
        summaries[split] = _records_summary(records)
    return {"splits": summaries, "disjoint_paths": True,
            "disjoint_subjects": None if missing_subject_ids else True,
            "subject_identity_check": "evaluation subject IDs unavailable" if missing_subject_ids else "complete",
            "canonical_labels": {"0": "SPOOF", "1": "LIVE"},
            "official_source_labels": {"0": "LIVE", "1": "SPOOF"}}


def validate_manifest_splits(
    train_path: str | Path, val_path: str | Path, test_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate manifest paths with no image or identity overlap across splits."""
    splits = {"train": load_manifest(train_path), "val": load_manifest(val_path)}
    if test_path is not None:
        splits["test"] = load_manifest(test_path)
    return validate_record_splits(splits)


def bbox_sidecar_path(path: str | Path) -> Path:
    """Return CelebA-Spoof's image-stem_BB.txt sidecar path."""
    path = Path(path)
    return path.with_name(path.stem + "_BB.txt")


def read_celeba_bbox(
    bbox_path: str | Path, image_width: int, image_height: int,
) -> tuple[float, float, float, float]:
    """Read a *_BB.txt file as scaled/clipped stored-image pixel xywh."""
    sidecar = Path(bbox_path)
    width, height = image_width, image_height
    if (isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1):
        raise ValueError("image_width and image_height must be positive integers.")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing bounding-box sidecar: {sidecar}; use full mode explicitly only if intended.")
    try:
        values = [float(value) for value in sidecar.read_text(encoding="utf-8-sig").split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read bounding box {sidecar}: {exc}") from exc
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{sidecar}: expected five finite values x y w h score in a 224x224 reference frame.")
    x, y, box_width, box_height, _score = values
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"{sidecar}: bounding-box width and height must be positive.")
    left, top = max(0.0, x * width / 224.0), max(0.0, y * height / 224.0)
    right = min(float(width), (x + box_width) * width / 224.0)
    bottom = min(float(height), (y + box_height) * height / 224.0)
    if right <= left or bottom <= top:
        raise ValueError(f"{sidecar}: bounding box has no area inside the stored image.")
    return left, top, right - left, bottom - top


def crop_liveness_image(
    image: Image.Image | np.ndarray, bbox: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Return the RGB face crop used by both liveness and identity inference.

    ``bbox`` uses stored-image pixel (x,y,width,height) coordinates and is clipped
    to the image. Without a box the caller supplies an already cropped face.
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy image must be RGB uint8 with shape (H, W, 3).")
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image or an RGB uint8 NumPy array.")
    image = image.convert("RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image dimensions must be positive.")
    if bbox is not None:
        try:
            values = tuple(float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox must contain four finite pixel values x,y,width,height.") from exc
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("bbox must contain four finite pixel values x,y,width,height.")
        x, y, width, height = values
        if width <= 0 or height <= 0:
            raise ValueError("bbox width and height must be positive.")
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right, bottom = min(image.width, math.ceil(x + width)), min(image.height, math.ceil(y + height))
        if right <= left or bottom <= top:
            raise ValueError("Bounding box has no area inside the image.")
        image = image.crop((left, top, right, bottom))
    return image


def preprocess_liveness_image(
    image: Image.Image | np.ndarray, image_size: int = 112,
    bbox: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """Return float32 RGB CHW in [-1,1]; no box means an already-cropped face."""
    image_size = _image_size(image_size)
    image = crop_liveness_image(image, bbox=bbox)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def load_liveness_image(path: str | Path, image_size: int = 112, crop_mode: str = "bbox") -> torch.Tensor:
    """Load stored-orientation pixels, strictly cropping the official bbox by default."""
    if crop_mode not in ("bbox", "full"):
        raise ValueError("crop_mode must be 'bbox' or the explicit alternative 'full'.")
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Liveness image does not exist: {path}")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read liveness image {path}: {exc}") from exc
    bbox = (read_celeba_bbox(bbox_sidecar_path(path), image.width, image.height)
            if crop_mode == "bbox" else None)
    return preprocess_liveness_image(image, image_size=image_size, bbox=bbox)


class LivenessDataset(Dataset):
    """Return ``(face_tensor, canonical_label)`` from a validated CSV manifest."""

    def __init__(self, manifest_path: str | Path, image_size: int = 112, crop_mode: str = "bbox") -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.image_size = _image_size(image_size)
        if crop_mode not in ("bbox", "full"):
            raise ValueError("crop_mode must be 'bbox' or 'full'.")
        self.crop_mode = crop_mode
        self.samples = load_manifest(self.manifest_path)
        self.records = self.samples
        self.labels = [record["label"] for record in self.samples]
        self._cached_images: list[torch.Tensor] | None = None

    def cache_in_memory(self) -> int:
        """Cache identical preprocessed tensors once; return their byte size."""
        if self._cached_images is None:
            cached = []
            for index, record in enumerate(self.samples, 1):
                cached.append(load_liveness_image(record["image_path"], self.image_size, self.crop_mode))
                if index % 1000 == 0:
                    print(f"Cached {index}/{len(self.samples)} images from {self.manifest_path.name}", flush=True)
            self._cached_images = cached
        return sum(image.numel() * image.element_size() for image in self._cached_images)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.samples[index]
        if self._cached_images is not None:
            return self._cached_images[index], record["label"]
        return load_liveness_image(record["image_path"], self.image_size, self.crop_mode), record["label"]
