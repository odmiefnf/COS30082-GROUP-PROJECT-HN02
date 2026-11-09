"""Calibration selects a completed encoder and respects tied-score false positives."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from face_verification.image_cache import sha256
from tools.calibrate_identity_app import calibrate


class IdentityCalibrationChecks(unittest.TestCase):
    def test_conservative_threshold_and_fingerprint(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temporary:
            root = Path(temporary)
            checkpoint = root / "test_only.pt"
            checkpoint.write_bytes(b"Software fixture; not a model")
            pairs = root / "pairs.csv"
            with pairs.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["label", "cosine_similarity"])
                writer.writerows([(1, .95), (1, .9), (0, .9), (0, .1)])
            model = {"checkpoint_smoke_test": False, "method": "classification", "image_size": 64,
                     "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                     "metrics": [{"metric": "cosine_similarity", "auc": .875}],
                     "artifacts": {"pair_scores_csv": str(pairs)}}
            comparison = root / "comparison.json"
            source = {"models": [model], "pairs_file": "original_pairs.txt", "pairs_file_sha256": "a"*64}
            comparison.write_text(json.dumps(source))
            result = calibrate(comparison, root/"app.json", 0.)
            self.assertEqual(result["checkpoint"], "test_only.pt")
            self.assertEqual(result["threshold"], .95)
            self.assertEqual(result["observed_validation_pair_fpr"], 0.)
            self.assertEqual(result["observed_validation_pair_tpr"], .5)
            moved = root/"relocated"
            moved.mkdir()
            shutil.copy2(root/"app.json", moved/"app.json")
            shutil.copy2(checkpoint, moved/checkpoint.name)
            copied = json.loads((moved/"app.json").read_text(encoding="utf-8"))
            self.assertEqual(sha256(moved/copied["checkpoint"]), copied["checkpoint_sha256"])
            with patch("tools.calibrate_identity_app.os.path.relpath", side_effect=ValueError("different drives")):
                fallback = calibrate(comparison, root/"fallback.json", 0.)
            self.assertEqual(fallback["checkpoint"], str(checkpoint.resolve()))
            self.assertEqual({k: v for k, v in fallback.items() if k != "checkpoint"},
                             {k: v for k, v in result.items() if k != "checkpoint"})
            checkpoint.write_bytes(b"Modified fixture")
            with self.assertRaisesRegex(ValueError, "changed"):
                calibrate(comparison, root/"app.json", 0.)
            source["models"][0]["checkpoint_smoke_test"] = True
            comparison.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "uncapped"):
                calibrate(comparison, root/"app.json")


if __name__ == "__main__":
    unittest.main()
