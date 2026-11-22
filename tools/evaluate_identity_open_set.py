"""Measure 1:N face matching on an offline gallery built from official test images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import copy
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random
import zipfile

import torch
from PIL import Image, ImageOps

from attendance_app.recognition import FaceRecognizer
from face_verification.image_cache import sha256


@dataclass(frozen=True)
class AuditedImage:
    path: str
    file_sha256: str
    pixel_sha256: str


@dataclass(frozen=True)
class RankedProbe:
    expected: str | None
    best: str
    cosine: float
    gap: float


def audited_test_pool(index: Path, audit: dict) -> tuple[dict[str, list[AuditedImage]], dict]:
    """Exclude each test RGB digest shared with any other indexed image."""
    images = defaultdict(list)
    test_pixels = Counter()
    other_pixels = set()
    split_counts = Counter()
    with index.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"]
            split_counts[split] += 1
            pixel = row["pixel_sha256"]
            if split != "test_data":
                other_pixels.add(pixel)
                continue
            path = PurePosixPath(row["archive_path"])
            if (path.is_absolute() or len(path.parts) != 4 or ".." in path.parts
                    or path.parts[:3] != ("classification_data", split, row["identity"])):
                raise ValueError("Invalid classification test path in the archive audit index.")
            images[row["identity"]].append(AuditedImage(str(path), row["file_sha256"], pixel))
            test_pixels[pixel] += 1
    if (sum(split_counts.values()) != audit["images_decoded"]
            or any(split_counts[split] != audit["classification"][split]["images"]
                   for split in ("train_data", "val_data", "test_data"))):
        raise ValueError("The image index does not match the complete archive audit counts.")

    unique = {}
    excluded = 0
    for identity, rows in images.items():
        eligible = [row for row in rows if row.pixel_sha256 not in other_pixels
                    and test_pixels[row.pixel_sha256] == 1]
        excluded += len(rows) - len(eligible)
        if len(eligible) >= 2:
            unique[identity] = sorted(eligible, key=lambda row: row.path)
    return unique, {
        "audit_images_indexed": sum(split_counts.values()),
        "official_classification_test_images": split_counts["test_data"],
        "exact_pixel_duplicate_test_images_excluded": excluded,
        "eligible_identities_with_two_or_more_distinct_test_images": len(unique),
    }


def make_trials(pool, gallery_size: int, unknown_count: int, seed: int) -> dict:
    """Draw separate people for calibration and final test, then assign photo roles."""
    if gallery_size < 2 or unknown_count < 1:
        raise ValueError("A trial needs at least two gallery identities and one unknown probe.")
    rng = random.Random(seed)
    identities = sorted(pool)
    rng.shuffle(identities)
    size = gallery_size + unknown_count
    if len(identities) < 2 * size:
        raise ValueError("Not enough eligible identities for separate calibration and test trials.")
    trials = {}
    for split, offset in (("calibration", 0), ("test", size)):
        members = identities[offset:offset + size]
        gallery = {identity: rng.sample(pool[identity], 2) for identity in members[:gallery_size]}
        unknown = {identity: rng.choice(pool[identity]) for identity in members[gallery_size:]}
        trials[split] = (gallery, unknown)
    return trials


def score_trial(source, prefix, recognizer, trial):
    gallery, unknown = trial
    digest = hashlib.sha256()

    def embed(image: AuditedImage, role: str):
        encoded = source.read(prefix + image.path)
        if hashlib.sha256(encoded).hexdigest() != image.file_sha256:
            raise ValueError("A selected archive image does not match the audit byte digest.")
        with Image.open(io.BytesIO(encoded)) as file:
            rgb = ImageOps.exif_transpose(file).convert("RGB")
        pixel_hash = hashlib.sha256(
            f"{rgb.width}x{rgb.height}:RGB:".encode("ascii") + rgb.tobytes()
        ).hexdigest()
        if pixel_hash != image.pixel_sha256:
            raise ValueError("A selected archive image does not match the audit RGB digest.")
        digest.update(f"{role}\t{image.path}\t{image.file_sha256}\n".encode("utf-8"))
        return recognizer.embed(rgb)

    profiles = [{"id": identity, "name": identity, "embedding": embed(reference, "gallery")}
                for identity, (reference, _) in gallery.items()]
    # Read the app's exact top match and runner-up gap without applying a decision threshold.
    ranker = copy(recognizer)
    ranker.threshold = -1.
    ranker.margin = 0.
    probes = [(identity, query) for identity, (_, query) in gallery.items()]
    probes.extend((None, query) for query in unknown.values())
    ranked = []
    for expected, image in probes:
        match = ranker.match(embed(image, "known_probe" if expected else "unknown_probe"), profiles)
        ranked.append(RankedProbe(expected, match["person_id"], match["cosine"], match["gap"]))
    return ranked, digest.hexdigest()


def outcomes(probes: list[RankedProbe], threshold: float, margin: float) -> dict:
    """Count app decisions using its minimum cosine and different-person gap."""
    totals = dict.fromkeys(("known_correct", "known_misidentified", "known_unknown", "known_ambiguous",
                           "unknown_false_identifications", "unknown_unknown", "unknown_ambiguous"), 0)
    for probe in probes:
        if probe.cosine < threshold:
            key = "known_unknown" if probe.expected is not None else "unknown_unknown"
        elif probe.gap < margin:
            key = "known_ambiguous" if probe.expected is not None else "unknown_ambiguous"
        elif probe.expected is None:
            key = "unknown_false_identifications"
        else:
            key = "known_correct" if probe.best == probe.expected else "known_misidentified"
        totals[key] += 1
    known = sum(probe.expected is not None for probe in probes)
    unknown = len(probes) - known
    if not known or not unknown:
        raise ValueError("Evaluation requires both enrolled and unknown probes.")
    return {
        **totals, "known_probes": known, "unknown_probes": unknown,
        "correct_identification_rate": totals["known_correct"] / known,
        "unknown_false_identification_rate": totals["unknown_false_identifications"] / unknown,
    }


def select_threshold(probes: list[RankedProbe], margin: float, max_fpir: float) -> float:
    if not 0 <= max_fpir <= 1:
        raise ValueError("Maximum unknown false identification rate must be in [0,1].")
    candidates = []
    # Tied cosine scores enter together. Threshold 1 allows a no-accept operating point
    # when all scores are strictly below 1.
    for threshold in {1., *(probe.cosine for probe in probes)}:
        metrics = outcomes(probes, threshold, margin)
        if metrics["unknown_false_identification_rate"] <= max_fpir:
            candidates.append((metrics["known_correct"], -metrics["unknown_false_identifications"], threshold))
    if not candidates:
        raise ValueError("No valid cosine threshold meets the calibration false identification limit.")
    return max(candidates)[2]


def evaluate(config, archive_path, audit_dir, gallery_size=20, unknown_count=300, seed=42, max_fpir=.01):
    recognizer = FaceRecognizer(config, "cpu")
    training = recognizer.config["training"]
    archive_hash = sha256(archive_path)
    audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
    if (archive_hash != audit["archive_sha256"] or not audit["all_images_decoded"] or audit["errors"]
            or archive_hash != (training.get("image_cache") or {}).get("archive_sha256")):
        raise ValueError("A complete audit for the exact archive used to train this encoder is required.")
    index = audit_dir / "image_hashes.csv"
    pool, counts = audited_test_pool(index, audit)
    trials = make_trials(pool, gallery_size, unknown_count, seed)
    with zipfile.ZipFile(archive_path) as source:
        pair_files = [name for name in source.namelist() if name.endswith("/verification_pairs_val.txt")
                      or name == "verification_pairs_val.txt"]
        if len(pair_files) != 1:
            raise ValueError("Expected one official verification pair file in the archive.")
        prefix = pair_files[0][:-len("verification_pairs_val.txt")]
        if hashlib.sha256(source.read(pair_files[0])).hexdigest() != recognizer.settings["pairs_file_sha256"]:
            raise ValueError("The pair-calibration file does not match the app identity configuration.")
        calibration, cal_digest = score_trial(source, prefix, recognizer, trials["calibration"])
        candidate = select_threshold(calibration, recognizer.margin, max_fpir)
        test, test_digest = score_trial(source, prefix, recognizer, trials["test"])
    train_ids = training["class_to_idx"]
    scores = {}
    for split, probes, digest in (("calibration", calibration, cal_digest), ("test", test, test_digest)):
        gallery, unknown = trials[split]
        scores[split] = {
            "trial_sha256": digest,
            "gallery_identities": len(gallery),
            "unknown_identities": len(unknown),
            "identities_also_in_encoder_training": sum(person in train_ids for person in (*gallery, *unknown)),
            "current_app_threshold": outcomes(probes, recognizer.threshold, recognizer.margin),
            "gallery_calibrated_threshold": outcomes(probes, candidate, recognizer.margin),
        }
    return {
        "protocol": "Offline 1:N search using a simulated single-photo gallery from classification_data/test_data.",
        "scope": [
            "Calibration and test use disjoint dataset identity labels and different enrollment/probe images.",
            "Test images shared by decoded RGB digest with any other indexed archive image are excluded.",
            "The official classification test identities also occur in encoder training.",
            "Unknown means absent from this gallery, not unseen by the encoder.",
            "The offline test uses one photo per gallery identity, one probe per person, and no webcam or liveness.",
        ],
        "checkpoint_sha256": recognizer.fingerprint,
        "archive_sha256": archive_hash,
        "audit_index_sha256": sha256(index),
        "identity_config_sha256": sha256(config),
        "seed": seed,
        "counts": counts,
        "ambiguity_margin": recognizer.margin,
        "calibration_unknown_false_identification_limit": max_fpir,
        "threshold_selection": "Maximize correct known matches on calibration within the unknown false-identification limit, then prefer fewer unknown false matches and higher cosine threshold.",
        "current_app_threshold": recognizer.threshold,
        "gallery_calibrated_threshold": candidate,
        "splits": scores,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("results/identity_app_config.json"))
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/identity_kaggle_2020.zip"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/identity_archive_audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery-size", type=int, default=20)
    parser.add_argument("--unknown-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-fpir", type=float, default=.01)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Choose a new output file: {args.output}")
        torch.set_num_threads(4)
        report = evaluate(args.config, args.archive, args.audit_dir, args.gallery_size, args.unknown_count,
                          args.seed, args.max_fpir)
        content = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Open-set evaluation failed: {exc}\n")
    print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
