"""Check exact preprocessing and triplet sampling for the optional large-data cache."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch

from face_verification.data import FaceImageDataset, TripletFaceDataset, load_face_image
from face_verification.image_cache import build_cache, ImageCache


class IdentityCacheChecks(unittest.TestCase):
    def test_cache_is_exact_and_rejects_wrong_mapping_or_tampering(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as source:
                for split in ("train_data", "val_data", "test_data"):
                    for label in ("a", "b"):
                        for index in range(2):
                            path = root / "classification_data" / split / label / f"{index}.png"
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(np.random.default_rng(index).integers(0, 256, (43, 39, 3), dtype=np.uint8)).save(path)
                            source.write(path, path.relative_to(root).as_posix())
            build_cache(archive, root / "cache", 32)
            dataset = FaceImageDataset(root / "classification_data/train_data", 32)
            expected = [dataset[index][0] for index in range(len(dataset))]
            dataset.image_cache = ImageCache(root / "cache", dataset)
            for index in range(len(dataset)):
                self.assertTrue(torch.equal(expected[index], dataset[index][0]))
            triplets = TripletFaceDataset(dataset.root, 32)
            original = [triplets[index] for index in range(len(triplets))]
            triplets.image_cache = ImageCache(root / "cache", triplets)
            for index in range(len(triplets)):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(original[index], triplets[index])))
            for epoch in range(3):
                triplets.set_epoch(epoch)
                for index, (anchor_path, label) in enumerate(triplets.anchors):
                    import random
                    rng = random.Random(f"42:{epoch}:{index}")
                    positive = rng.choice([p for p in triplets._by_label[label] if p != anchor_path])
                    negative_label = rng.choice([other for other in triplets._labels if other != label])
                    negative = rng.choice(triplets._by_label[negative_label])
                    actual = triplets[index]
                    for value, path in zip(actual, (anchor_path, positive, negative)):
                        self.assertTrue(torch.equal(value, load_face_image(path, 32)))
            dataset.samples.reverse()
            with self.assertRaisesRegex(ValueError, "paths"):
                ImageCache(root / "cache", dataset)
            dataset.samples.reverse()
            with (root / "cache/train_data.npy").open("r+b") as stream:
                stream.seek(-1, 2); stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "checksum"):
                ImageCache(root / "cache", dataset)
            dataset.image_cache.close()
            triplets.image_cache.close()


if __name__ == "__main__":
    unittest.main()
