"""Image-folder datasets for the face-verification training pipeline.

An identity is a direct subdirectory of ``root``; supported images may be nested
inside that directory. No face detector or pretrained model is used here, so
inputs should already contain the face crops intended for training.
"""

from __future__ import annotations

import random
import os
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
)


class EmptyFaceDatasetError(ValueError):
    """A dataset directory contains no supported images for selected identities."""


def _validate_image_size(image_size: int) -> int:
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    return image_size


def _discover_images(root: str | Path) -> tuple[Path, dict[str, list[Path]]]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")
    identity_dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)
    if not identity_dirs:
        raise EmptyFaceDatasetError(
            f"No identity subdirectories found in {root}. "
            "Expected root/person_id/image.jpg."
        )
    def image_paths(directory: Path):
        # DirEntry retains Windows directory metadata. Avoid one additional
        # stat call per tiny image in the 380k-image training split.
        pending = [directory]
        while pending:
            current = pending.pop()
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_EXTENSIONS:
                        yield Path(entry.path)
    images = {
        directory.name: sorted(
            image_paths(directory),
            key=lambda path: path.relative_to(directory).as_posix(),
        )
        for directory in identity_dirs
    }
    return root, images


def _read_rgb(path: str | Path) -> Image.Image:
    path = Path(path)
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read face image {path}: {exc}") from exc


def _to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def load_face_image(path: str | Path, image_size: int = 112) -> torch.Tensor:
    """Read an image as a contiguous float32 RGB tensor, CHW in [-1, 1].

    EXIF orientation is applied before resizing to a square. Grayscale and RGBA
    images are converted to three RGB channels. Failed reads identify the path.
    """
    image_size = _validate_image_size(image_size)
    return _to_tensor(_read_rgb(path), image_size)


def preprocess_face_image(image: Image.Image | np.ndarray, image_size: int = 112) -> torch.Tensor:
    """Resize an already oriented RGB face crop from a camera or another module.

    Unlike the file loader, this accepts a crop whose orientation has already
    been resolved. NumPy input must be uint8 RGB, never OpenCV's BGR order.
    """
    image_size = _validate_image_size(image_size)
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a nonempty HWC uint8 RGB face image.")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Face image must be nonempty.")
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image) or image.width == 0 or image.height == 0:
        raise ValueError("Expected a nonempty PIL image or HWC uint8 RGB array.")
    return _to_tensor(image.convert("RGB"), image_size)


class FaceImageDataset(Dataset):
    """Deterministically enumerate identity folders and return ``(image, label)``.

    By default, labels follow sorted nonempty identity names. Supplying the
    training ``class_to_idx`` preserves those labels for a validation split;
    unknown identity directories are skipped, warned about, and recorded in
    ``unknown_classes``. ``empty_classes`` records folders without images.

    Augmentation uses Python's global RNG (horizontal flips and subtle brightness
    changes), so seed ``random`` and the DataLoader generator for reproducibility.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 112,
        augment: bool = False,
        class_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        self.image_size = _validate_image_size(image_size)
        self.augment = augment
        self.image_cache = None
        self.root, images = _discover_images(root)
        self.empty_classes = tuple(name for name, paths in images.items() if not paths)

        if class_to_idx is None:
            self.class_to_idx = {
                name: index
                for index, name in enumerate(name for name, paths in images.items() if paths)
            }
        else:
            self.class_to_idx = dict(class_to_idx)
            indices = list(self.class_to_idx.values())
            if (
                not self.class_to_idx
                or any(not isinstance(name, str) for name in self.class_to_idx)
                or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
                or sorted(indices) != list(range(len(indices)))
            ):
                raise ValueError(
                    "class_to_idx must map identity names to unique, contiguous "
                    "integer labels starting at zero."
                )

        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.__getitem__)
        self.unknown_classes = (
            tuple(name for name in images if name not in self.class_to_idx)
            if class_to_idx is not None
            else ()
        )
        if self.unknown_classes:
            warnings.warn(
                f"Skipping identity directories absent from class_to_idx in {self.root}: "
                + ", ".join(self.unknown_classes),
                UserWarning,
                stacklevel=2,
            )
        self.samples = [
            (path, self.class_to_idx[name])
            for name, paths in images.items()
            if name in self.class_to_idx
            for path in paths
        ]
        if not self.samples:
            raise EmptyFaceDatasetError(
                f"No valid images for the selected identities in {self.root}. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        if self.image_cache is not None:
            return self.image_cache.load(path), label
        image = _read_rgb(path)
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
        return _to_tensor(image, self.image_size), label


class TripletFaceDataset(Dataset):
    """Return a distinct same-person positive and a different-person negative.

    There is one triplet per eligible anchor image. At least two identities must
    each contain two images; singleton identities may also supply negatives.
    Sampling is deterministic for ``(seed, epoch, index)`` and independent of
    DataLoader worker scheduling. Call ``set_epoch`` before creating each epoch's
    iterator (use nonpersistent DataLoader workers for epoch changes to propagate).
    This class reads only the supplied root and applies no augmentation.
    """

    def __init__(self, root: str | Path, image_size: int = 112, seed: int = 42) -> None:
        self.image_size = _validate_image_size(image_size)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer.")
        self.seed = seed
        self.epoch = 0
        self.image_cache = None
        dataset = FaceImageDataset(root, image_size=image_size)
        self.root = dataset.root
        self.class_to_idx = dataset.class_to_idx
        self.classes = dataset.classes
        self.samples = dataset.samples
        self._by_label: dict[int, list[Path]] = {}
        for path, label in self.samples:
            self._by_label.setdefault(label, []).append(path)
        eligible_labels = {label for label, paths in self._by_label.items() if len(paths) >= 2}
        if len(eligible_labels) < 2:
            raise ValueError(
                f"Triplet training requires at least two identities with at least "
                f"two images each in {self.root}; found {len(eligible_labels)}. "
                "Additional singleton identities can be used as negatives."
            )
        self.anchors = [(path, label) for path, label in self.samples if label in eligible_labels]
        self.eligible_anchor_count = len(self.anchors)
        self._labels = sorted(self._by_label)
        self._label_positions = {label: index for index, label in enumerate(self._labels)}
        self._image_positions = {path: index for paths in self._by_label.values()
                                 for index, path in enumerate(paths)}

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible new set of triplets for a nonnegative epoch."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer.")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, label = self.anchors[index]
        # Normalize a valid negative Python index to the same triplet as its
        # equivalent nonnegative index; list indexing above enforces bounds.
        index %= len(self.anchors)
        rng = random.Random(f"{self.seed}:{self.epoch}:{index}")
        paths = self._by_label[label]
        positive_index = rng.randrange(len(paths) - 1)
        positive_index += positive_index >= self._image_positions[anchor_path]
        positive_path = paths[positive_index]
        negative_index = rng.randrange(len(self._labels) - 1)
        negative_index += negative_index >= self._label_positions[label]
        negative_label = self._labels[negative_index]
        negative_path = rng.choice(self._by_label[negative_label])
        load = (self.image_cache.load if self.image_cache is not None else
                lambda path: load_face_image(path, self.image_size))
        return load(anchor_path), load(positive_path), load(negative_path)
