"""Compare tight and 2x PAD crops on identical LCC-FASD source images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
from threading import local

import cv2
import numpy as np
from PIL import Image

from anti_spoofing.data import MANIFEST_COLUMNS, load_manifest
from anti_spoofing.evaluate import _sha256, _write_json, build_parser as evaluation_parser, evaluate
from anti_spoofing.train import build_parser as training_parser, train
from attendance_app.camera import FaceLocator, FrameProcessor


LIMITS = {"train": 720, "val": 250, "test": 250}
POLICIES = ("tight", "context-2x")
SEED = 42


class CropPair:
    def reset_tracking(self, reason=None):
        pass

    def analyze(self, crop, *, context_crop):
        return {"status": "paired", "crops": (crop, context_crop)}


def prepare_pairs(manifest_dir, work):
    """Keep a sample only if the normal face detector and quality checks pass."""
    cv2.setNumThreads(1)
    worker = local()
    rng = np.random.default_rng(SEED)
    inventory = {}
    columns = (*MANIFEST_COLUMNS, "subject_id_source")
    for split, limit in LIMITS.items():
        manifest = manifest_dir / f"{split}.csv"
        records = load_manifest(manifest)
        selected = []
        for label in (0, 1):
            options = [record for record in records if record["label"] == label]
            selected.extend(options[index] for index in rng.choice(len(options), min(limit, len(options)), replace=False))

        def write_pair(item):
            index, record = item
            if not hasattr(worker, "processor"):
                worker.processor = FrameProcessor(FaceLocator(), CropPair(), pad_context=True)
            with Image.open(record["image_path"]) as source:
                result = worker.processor.process(np.asarray(source.convert("RGB")))
            paths = []
            if result["status"] == "paired":
                for policy, crop in zip(POLICIES, result["crops"]):
                    label_dir = "live" if record["label"] else "spoof"
                    path = work / policy / split / label_dir / f"{index:04d}.png"
                    crop.save(path)
                    paths.append(path)
            return result["status"], paths

        for policy in POLICIES:
            for label_dir in ("spoof", "live"):
                (work / policy / split / label_dir).mkdir(parents=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            processed = list(pool.map(write_pair, enumerate(selected)))
        saved = {"live": 0, "spoof": 0}
        excluded = {}
        for record, (status, _) in zip(selected, processed):
            if status == "paired":
                saved["live" if record["label"] else "spoof"] += 1
            else:
                excluded[status] = excluded.get(status, 0) + 1
        if min(saved.values()) == 0:
            raise ValueError(f"{split} did not yield both classes after quality checks.")
        for position, policy in enumerate(POLICIES):
            with (work / policy / f"{split}.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for record, (_, paths) in zip(selected, processed):
                    if paths:
                        writer.writerow({**record, "image_path": paths[position].relative_to(work / policy).as_posix()})
        inventory[split] = {"manifest_sha256": _sha256(manifest), "selected": len(selected),
                            "scored_per_arm": saved, "excluded_before_training_or_scoring": excluded}
        print(f"{split}: {saved}, excluded {excluded}", flush=True)
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/lcc_fasd_manifests"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    inventory = prepare_pairs(args.manifest_dir, work)
    arms = {}
    for policy in POLICIES:
        directory = work / policy
        recipe = ["--train-manifest", str(directory / "train.csv"), "--val-manifest", str(directory / "val.csv"),
                  "--output-dir", str(directory / "model"), "--crop-mode", "full", "--epochs", "10",
                  "--batch-size", "64", "--learning-rate", "0.0005", "--weight-decay", "0.0001",
                  "--checkpoint-selection", "val-acer", "--camera-augmentation", "--cache-images",
                  "--seed", str(SEED), "--device", args.device]
        if policy == "context-2x":
            recipe.append("--context-crop")
        checkpoint = train(training_parser().parse_args(recipe))
        run = evaluate(evaluation_parser().parse_args([
            "--checkpoint", str(checkpoint), "--manifest", str(directory / "test.csv"),
            "--output-dir", str(directory / "evaluation"), "--device", args.device]))
        arms[policy] = {"checkpoint_sha256": _sha256(checkpoint),
                        "threshold": run["threshold"], "threshold_source": run["threshold_source"],
                        "test": run["metrics"], "scope": run["evaluation_scope"]}
    _write_json(output / "metrics.json", {
        "status": "completed", "scope": "Paired development comparison on a subset of reused LCC-FASD images.",
        "limits_per_class": LIMITS, "seed": SEED, "epochs": 10, "image_size": 112,
        "protocol": "Same source images, detector, face quality gate, training recipe and seed. Each arm calibrates on validation.",
        "limitations": ["The source images are already cropped; no missing scene pixels were added.",
                        "Evaluation subject IDs and attack types are unavailable for this LCC split.",
                        "No fresh webcam or device-held-out results were measured."],
        "inventory": inventory, "arms": arms, "candidate_promoted": False,
    })


if __name__ == "__main__":
    main()
