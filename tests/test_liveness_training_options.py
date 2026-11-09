"""Training-only augmentation and validation-based candidate selection."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anti_spoofing.train import _epoch, build_parser, train


class CameraTrainingChecks(unittest.TestCase):
    def test_augmentation_runs_only_with_training_optimizer(self):
        torch.set_num_threads(2)
        images = torch.zeros((4, 3, 2, 2))
        labels = torch.tensor([0, 1, 0, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=4)
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
        augment = Mock(side_effect=lambda batch: batch+.1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        _epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer, None, augment)
        augment.assert_called_once()
        augment.reset_mock()
        _epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"), None, None, augment)
        augment.assert_not_called()

    def test_augmented_candidate_selects_and_calibrates_on_complete_validation(self):
        torch.set_num_threads(2)
        base = Path(__file__).resolve().parents[1]/"tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(123)
            for split in ("train", "val"):
                rows = []
                for index in range(4):
                    label = index % 2
                    path = root/f"{split}_{index}.png"
                    Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                    rows.append({"image_path": str(path), "label": label,
                                 "subject_id": f"{split}-{index}", "source_label": 1-label})
                with (root/f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            args = build_parser().parse_args([
                "--train-manifest", str(root/"train.csv"), "--val-manifest", str(root/"val.csv"),
                "--output-dir", str(root/"out"), "--image-size", "32", "--feature-dim", "8",
                "--epochs", "2", "--batch-size", "2", "--crop-mode", "full", "--device", "cpu",
                "--num-threads", "2", "--camera-augmentation", "--checkpoint-selection", "val-acer",
                "--weight-decay", "0.0001", "--cache-images",
            ])
            checkpoint = train(args)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            recipe = payload["training"]
            self.assertTrue(recipe["augmentation"]["enabled"])
            self.assertFalse(recipe["smoke_test"])
            self.assertEqual(recipe["initialization"], "random_from_scratch")
            self.assertEqual(recipe["completed_epochs"], 2)
            self.assertEqual(recipe["optimizer"]["weight_decay"], .0001)
            self.assertEqual(recipe["calibration"]["validation_samples_used"], 4)
            self.assertFalse(recipe["calibration"]["test_data_used"])
            with (root/"out/history.csv").open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
            best = min(history, key=lambda row: (float(row["val_acer"]), float(row["val_loss"])))
            self.assertEqual(recipe["epoch"], int(best["epoch"]))


if __name__ == "__main__":
    unittest.main()
