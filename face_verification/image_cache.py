"""Build an optional uint8 training cache from the unchanged source ZIP."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageOps
import torch

from .data import IMAGE_EXTENSIONS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cache(archive, output, image_size=64):
    """Keep every classification image, original labels and PIL resize semantics."""
    if image_size < 32:
        raise ValueError("image_size must be at least 32.")
    archive, output = Path(archive).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Cache already exists; select a new output directory.")
    report = {"version": 1, "archive": str(archive), "archive_sha256": sha256(archive),
              "image_size": image_size, "layout": "NCHW uint8 RGB", "splits": {}}
    with zipfile.ZipFile(archive) as source:
        for split in ("train_data", "val_data", "test_data"):
            selected = []
            for name in source.namelist():
                parts = Path(name).parts
                if ("classification_data" in parts and Path(name).suffix.lower() in IMAGE_EXTENSIONS):
                    start = parts.index("classification_data")
                    if len(parts) == start + 4 and parts[start + 1] == split:
                        selected.append(("/".join(parts[start + 2:]), name))
            selected.sort()
            if not selected:
                raise ValueError(f"ZIP contains no images in {split}.")
            path = output / f"{split}.npy"
            if path.exists():
                raise ValueError(f"Refusing to overwrite {path}")
            pixels = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8,
                                               shape=(len(selected), 3, image_size, image_size))
            for index, (_, name) in enumerate(selected):
                with Image.open(io.BytesIO(source.read(name))) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB").resize(
                        (image_size, image_size), Image.Resampling.BILINEAR)
                    pixels[index] = np.asarray(rgb).transpose(2, 0, 1)
                if (index + 1) % 50000 == 0:
                    print(f"Caching {split}: {index + 1}/{len(selected)}", flush=True)
            pixels.flush()
            del pixels
            report["splits"][split] = {"file": path.name, "sha256": sha256(path),
                                        "paths": [relative for relative, _ in selected]}
            print(f"Cached {split}: {len(selected)} images", flush=True)
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ImageCache:
    """Read-only cache checked against the exact dataset order and byte checksum."""
    def __init__(self, directory, dataset):
        directory = Path(directory).resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if dataset.image_size != manifest["image_size"] or getattr(dataset, "augment", False):
            raise ValueError("Cache requires matching image size and disabled augmentation.")
        split = manifest["splits"][dataset.root.name]
        expected = [path.relative_to(dataset.root).as_posix() for path, _ in dataset.samples]
        if expected != split["paths"]:
            raise ValueError("Cache paths do not exactly match the dataset.")
        path = (directory / split["file"]).resolve()
        if path.parent != directory or sha256(path) != split["sha256"]:
            raise ValueError("Cache file path or checksum does not match.")
        self.pixels = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.pixels.dtype != np.uint8 or self.pixels.shape != (len(expected), 3, dataset.image_size, dataset.image_size):
            raise ValueError("Invalid cache shape or dtype.")
        self.indices = {path: index for index, (path, _) in enumerate(dataset.samples)}
        self.provenance = {"manifest": str(directory / "manifest.json"),
                           "archive_sha256": manifest["archive_sha256"], "cache_sha256": split["sha256"]}

    def load(self, path):
        # Copy into float32 without ever mutating the memory-mapped source.
        array = np.array(self.pixels[self.indices[path]], dtype=np.float32)
        return torch.from_numpy(array / 127.5 - 1.0)

    def close(self):
        self.pixels._mmap.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args(argv)
    build_cache(args.archive, args.output_dir, args.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
