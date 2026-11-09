"""Evaluate trained face encoders on labelled verification pairs.

Example::

    python -m face_verification.evaluate --checkpoint runs/classification/best.pt \
        --checkpoint runs/triplet/best.pt --data-root data --output-dir results

This evaluates actual checkpoints and images. It does not manufacture results
when training data or checkpoints are absent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shlex
from typing import Any

import numpy as np
import torch

from .data import load_face_image
from .metrics import auc, compute_roc, select_validation_threshold
from .model import FaceEncoder


Pair = tuple[Path, Path, int]
_HEADER_LABELS = {
    "label", "labels", "same", "issame", "target", "match", "y",
    "sameidentity", "issameperson", "class",
}
_SCORE_DEFINITIONS = {
    "cosine_similarity": {
        "raw_value": "dot product of L2-normalized embeddings",
        "roc_score": "cosine_similarity",
        "accept_rule": "cosine_similarity >= threshold",
    },
    "euclidean_distance": {
        "raw_value": "Euclidean distance between L2-normalized embeddings",
        "roc_score": "negative_euclidean_distance",
        "accept_rule": "euclidean_distance <= threshold",
    },
}
_THRESHOLD_SCOPE = "validation-calibrated / in-sample; not an unbiased test estimate"


def _tokens(line: str) -> list[str]:
    # Disable escaping so Windows backslashes survive; quoted paths may have spaces.
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    lexer.escape = ""
    return list(lexer)


def _resolve_image_path(reference: str, data_root: Path, pairs_parent: Path) -> Path:
    """Resolve explicit paths without basename searches or arbitrary suffix matching."""
    native_reference = reference.replace("\\", os.sep).replace("/", os.sep)
    relative = Path(native_reference).expanduser()
    is_absolute = relative.is_absolute() or PureWindowsPath(reference).is_absolute()
    candidates: list[Path] = []
    if is_absolute:
        candidates.append(relative)
    else:
        candidates.extend((data_root / relative, pairs_parent / relative))
        parts = relative.parts
        first = parts[0].casefold() if parts else ""
        root_name = data_root.name.casefold()
        # A pair list may include the dataset folder name even when --data-root
        # already names that folder. Remove only this recognized leading prefix.
        if len(parts) > 1 and first == root_name:
            candidates.append(data_root.joinpath(*parts[1:]))
        if first not in {"verification_data", root_name} and root_name != "verification_data":
            candidates.append(data_root / "verification_data" / relative)
        if len(parts) > 1 and first == "verification_data" and root_name == "verification_data":
            candidates.append(data_root.joinpath(*parts[1:]))

    checked: dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        checked.setdefault(os.path.normcase(str(resolved)), resolved)
    matches = [path for path in checked.values() if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        locations = "\n  ".join(str(path) for path in matches)
        raise ValueError(
            f"Ambiguous image path {reference!r}; it matches multiple files:\n  {locations}\n"
            "Use an absolute path or an unambiguous dataset-relative path in the pair file."
        )
    attempted = "\n  ".join(str(path) for path in checked.values())
    raise FileNotFoundError(
        f"Image {reference!r} was not found. Checked:\n  {attempted}\n"
        "Check --data-root and the pair-file paths. Expected images below the dataset "
        "root, its verification_data folder, or the pair file's folder."
    )


def read_verification_pairs(pairs_path: str | Path, data_root: str | Path) -> list[Pair]:
    """Read ``image1 image2 label`` rows and return resolved existing image paths.

    Labels are 1 for the same person and 0 for different people. Blank lines,
    ``#`` comments (including trailing comments), whole-line ``;``/``//``
    comments, an optional three-column header, and quoted paths are accepted.
    Path resolution is explicit and rejects ambiguous or missing images.
    """
    pair_file = Path(pairs_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {root}")
    if not pair_file.is_file():
        raise FileNotFoundError(f"Verification pair file does not exist: {pair_file}")
    result: list[Pair] = []
    resolved_paths: dict[str, Path] = {}
    first_content = True
    with pair_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", ";")) or stripped == "//" or stripped.startswith("// "):
                continue
            try:
                fields = _tokens(stripped)
            except ValueError as exc:
                raise ValueError(f"{pair_file}:{line_number}: invalid quoting: {exc}") from exc
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(
                    f"{pair_file}:{line_number}: expected two image paths and a 0/1 label; "
                    f"found {len(fields)} columns. Quote paths containing spaces."
                )
            normalized_label = re.sub(r"[^a-z]", "", fields[2].casefold())
            if first_content and normalized_label in _HEADER_LABELS:
                first_content = False
                continue
            first_content = False
            try:
                number = float(fields[2])
            except ValueError as exc:
                raise ValueError(
                    f"{pair_file}:{line_number}: label {fields[2]!r} must be 0 or 1."
                ) from exc
            if not np.isfinite(number) or number not in (0.0, 1.0):
                raise ValueError(f"{pair_file}:{line_number}: label {fields[2]!r} must be 0 or 1.")
            try:
                for reference in fields[:2]:
                    if reference not in resolved_paths:
                        resolved_paths[reference] = _resolve_image_path(reference, root, pair_file.parent)
                left = resolved_paths[fields[0]]
                right = resolved_paths[fields[1]]
            except (OSError, ValueError) as exc:
                raise type(exc)(f"{pair_file}:{line_number}: {exc}") from exc
            result.append((left, right, int(number)))
    if not result:
        raise ValueError(f"No verification pairs were found in {pair_file}.")
    return result


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Checkpoint config.{field} must be a positive integer; got {value!r}.")
    return value


def load_encoder(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[FaceEncoder, dict]:
    """Load an encoder in eval mode plus its config and available training metadata.

    ``config['training']`` preserves checkpoint training metadata when present;
    ``smoke_test`` and ``max_batches`` are also exposed directly when recorded.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint_path}")
    device = _resolve_device(device) if isinstance(device, str) else device
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not read encoder checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError(f"{checkpoint_path}: expected a checkpoint dict with 'model_state' and 'config'.")
    config = dict(checkpoint["config"])
    config["checkpoint_format_version"] = checkpoint.get("format_version")
    if isinstance(checkpoint.get("training"), dict):
        config["training"] = checkpoint["training"]
        for key in ("max_batches", "smoke_test"):
            if key not in config and key in checkpoint["training"]:
                config[key] = checkpoint["training"][key]
    for key in ("max_batches", "smoke_test"):
        if key not in config and key in checkpoint:
            config[key] = checkpoint[key]
    embedding_dim = _positive_integer(config.get("embedding_dim"), "embedding_dim")
    _positive_integer(config.get("image_size"), "image_size")
    if not isinstance(config.get("method"), str) or not config["method"].strip():
        raise ValueError(f"{checkpoint_path}: config.method must be a nonempty string.")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"{checkpoint_path}: 'model_state' must contain the encoder-only state dict.")
    encoder = FaceEncoder(embedding_dim=embedding_dim)
    try:
        encoder.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{checkpoint_path}: encoder state does not match its config: {exc}") from exc
    encoder.to(device)
    encoder.eval()
    return encoder, config


def _normalized_embeddings(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float64).numpy()
    try:
        embeddings = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric embeddings with shape [D] or [N, D].") from exc
    if embeddings.ndim not in (1, 2) or embeddings.size == 0 or embeddings.shape[-1] == 0:
        raise ValueError(f"{name} must have nonempty shape [D] or [N, D].")
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name} contains nonfinite embeddings.")
    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError(f"{name} contains zero-norm or invalid embeddings.")
    return embeddings / norms


def score_embeddings(embedding1: Any, embedding2: Any) -> tuple[Any, Any]:
    """Return ``(cosine_similarity, euclidean_distance)`` after L2 normalization.

    Inputs may be NumPy arrays or PyTorch tensors of identical shape [D] or
    [N, D]. Vector inputs return two floats; batch inputs return two arrays.
    Larger cosine and smaller distance support the same-identity hypothesis.
    """
    left = _normalized_embeddings(embedding1, "embedding1")
    right = _normalized_embeddings(embedding2, "embedding2")
    if left.shape != right.shape:
        raise ValueError(f"Embedding shapes must match; got {left.shape} and {right.shape}.")
    cosine = np.clip(np.sum(left * right, axis=-1), -1.0, 1.0)
    distance = np.linalg.norm(left - right, axis=-1)
    if left.ndim == 1:
        return float(cosine), float(distance)
    return cosine, distance


def _checkpoint_tag(checkpoint_path: Path, method: str) -> str:
    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")[:40] or "encoder"

    digest = hashlib.sha256(os.path.normcase(str(checkpoint_path)).encode("utf-8")).hexdigest()[:10]
    return f"{slug(method)}-{slug(checkpoint_path.stem)}-{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_embeddings(
    encoder: FaceEncoder,
    paths: list[Path],
    image_size: int,
    embedding_dim: int,
    batch_size: int,
    device: torch.device,
) -> dict[Path, np.ndarray]:
    cache: dict[Path, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images: list[torch.Tensor] = []
            for path in batch_paths:
                try:
                    tensor = load_face_image(path, image_size=image_size)
                except Exception as exc:
                    raise ValueError(f"Could not load face image {path}: {exc}") from exc
                if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (3, image_size, image_size):
                    raise ValueError(f"Image loader returned an invalid CHW tensor for {path}.")
                if not torch.isfinite(tensor).all().item():
                    raise ValueError(f"Image loader returned nonfinite pixels for {path}.")
                images.append(tensor)
            output = encoder(torch.stack(images).to(device))
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (len(batch_paths), embedding_dim):
                raise ValueError(
                    f"Encoder must return [batch, {embedding_dim}] embeddings; "
                    f"got {getattr(output, 'shape', type(output))}."
                )
            embeddings = output.detach().to(device="cpu", dtype=torch.float64).numpy()
            normalized = _normalized_embeddings(embeddings, f"Encoder batch starting at {batch_paths[0]}")
            cache.update(zip(batch_paths, normalized))
    return cache


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    columns = fieldnames or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_roc_svg(path: Path, title: str, curves: list[dict], pair_count: int) -> None:
    left, top, width, height = 90, 75, 825, 440
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="700" viewBox="0 0 960 700" role="img" aria-labelledby="title desc">',
        f'<title id="title">{html.escape(title)}</title>',
        '<desc id="desc">ROC curves for cosine similarity and negative Euclidean distance. Larger scores mean the same identity.</desc>',
        '<rect width="960" height="700" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#172033">',
        f'<text x="90" y="32" font-size="21" font-weight="bold">{html.escape(title[:78])}</text>',
        f'<text x="90" y="55" font-size="13">{pair_count} verification pairs; positive label 1 = same identity</text>',
    ]
    for index in range(6):
        value = index / 5
        x = left + value * width
        y = top + (1 - value) * height
        svg.extend((
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + height}" stroke="#e3e8ef"/>',
            f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#e3e8ef"/>',
            f'<text x="{x}" y="{top + height + 24}" text-anchor="middle" font-size="12">{value:.1f}</text>',
            f'<text x="{left - 14}" y="{y + 4}" text-anchor="end" font-size="12">{value:.1f}</text>',
        ))
    svg.extend((
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top}" stroke="#9aa5b4" stroke-dasharray="6 6"/>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#637085"/>',
        f'<text x="{left + width / 2}" y="565" text-anchor="middle" font-size="15">False positive rate</text>',
        f'<text x="28" y="{top + height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 28 {top + height / 2})">True positive rate</text>',
    ))
    colors = ("#1467b3", "#d27012")
    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{left + float(x) * width:.4f},{top + (1 - float(y)) * height:.4f}"
            for x, y in zip(curve["fpr"], curve["tpr"])
        )
        dash = ' stroke-dasharray="9 5"' if index else ""
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        legend_y = 595 + index * 23
        label = f'{curve["metric"]}: AUC {curve["auc"]:.6f}'
        svg.extend((
            f'<line x1="90" y1="{legend_y - 4}" x2="125" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"{dash}/>',
            f'<text x="138" y="{legend_y}" font-size="14">{html.escape(label)}</text>',
        ))
    svg.extend((
        '<text x="90" y="653" font-size="12">L2 normalization makes cosine and Euclidean rankings equivalent (up to numerical ties).</text>',
        '<text x="90" y="675" font-size="12">Selected thresholds are validation-calibrated / in-sample, not unbiased test operating points.</text>',
        '</g></svg>',
    ))
    path.write_text("\n".join(svg), encoding="utf-8")


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    pairs: list[Pair],
    output_dir: str | Path,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict:
    """Encode unique images once, score pairs, and save one checkpoint's artifacts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    labels = np.asarray([label for _, _, label in pairs], dtype=np.int64)
    # Fail before expensive image loading when verification metrics are undefined.
    compute_roc(labels, np.zeros(labels.size))
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint_file}")
    device = _resolve_device(device) if isinstance(device, str) else device
    encoder, config = load_encoder(checkpoint_file, device)
    tag = _checkpoint_tag(checkpoint_file, config["method"])
    unique_paths = list(dict.fromkeys(path for left, right, _ in pairs for path in (left, right)))
    cache = _cache_embeddings(
        encoder, unique_paths, config["image_size"], config["embedding_dim"],
        batch_size, torch.device(device),
    )
    left = np.stack([cache[a] for a, _, _ in pairs])
    right = np.stack([cache[b] for _, b, _ in pairs])
    cosine, distance = score_embeddings(left, right)
    model_dir = Path(output_dir).expanduser().resolve() / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(model_dir / "pair_scores.csv", [
        {
            "image1": str(a), "image2": str(b), "label": label,
            "cosine_similarity": float(cosine[index]),
            "euclidean_distance": float(distance[index]),
            "negative_euclidean_distance": float(-distance[index]),
        }
        for index, (a, b, label) in enumerate(pairs)
    ])
    curves = []
    roc_rows = []
    metric_results = []
    for metric, scores in (("cosine_similarity", cosine), ("euclidean_distance", -distance)):
        fpr, tpr, thresholds = compute_roc(labels, scores)
        roc_auc = auc(fpr, tpr)
        selected = select_validation_threshold(labels, scores)
        raw_threshold = float(selected["score_threshold"])
        if metric == "euclidean_distance":
            raw_threshold = -raw_threshold
        metric_result = {
            "metric": metric,
            "auc": roc_auc,
            "score_definition": _SCORE_DEFINITIONS[metric]["roc_score"],
            "score_threshold": selected["score_threshold"],
            "raw_threshold": raw_threshold,
            "accept_rule": _SCORE_DEFINITIONS[metric]["accept_rule"],
            "threshold_scope": _THRESHOLD_SCOPE,
            "threshold_selection": selected,
        }
        metric_results.append(metric_result)
        curves.append({"metric": metric, "auc": roc_auc, "fpr": fpr, "tpr": tpr})
        for x, y, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append({
                "metric": metric,
                "fpr": float(x), "tpr": float(y),
                "score_threshold": float(threshold),
                "raw_threshold": float(threshold if metric == "cosine_similarity" else -threshold),
                "accept_rule": _SCORE_DEFINITIONS[metric]["accept_rule"],
            })
    _write_csv(model_dir / "roc_points.csv", roc_rows)
    _write_roc_svg(model_dir / "roc_curve.svg", f'ROC: {config["method"]} / {checkpoint_file.stem}', curves, len(pairs))
    result = {
        "checkpoint_tag": tag,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": _file_sha256(checkpoint_file),
        "method": config["method"],
        "embedding_dim": config["embedding_dim"],
        "image_size": config["image_size"],
        "checkpoint_config": config,
        "checkpoint_smoke_test": True if config.get("max_batches") is not None else config.get("smoke_test"),
        "training_max_batches": config.get("max_batches"),
        "training_scope_note": (
            "Smoke-test checkpoint: capped training; scores do not represent a completed training experiment."
            if config.get("smoke_test") or config.get("max_batches") is not None
            else "Checkpoint records uncapped training." if config.get("smoke_test") is False
            else "Checkpoint does not record whether training was a smoke test."
        ),
        "pair_count": len(pairs),
        "same_identity_pairs": int(labels.sum()),
        "different_identity_pairs": int(labels.size - labels.sum()),
        "unique_image_count": len(unique_paths),
        "device": str(device),
        "embedding_normalization": "per-image L2 normalization in float64 before scoring",
        "score_definitions": _SCORE_DEFINITIONS,
        "ranking_note": "For unit embeddings, squared Euclidean distance = 2 - 2*cosine similarity; rankings and AUC are equivalent up to numerical ties.",
        "threshold_scope": _THRESHOLD_SCOPE,
        "metrics": metric_results,
        "artifacts": {
            "pair_scores_csv": str(model_dir / "pair_scores.csv"),
            "roc_points_csv": str(model_dir / "roc_points.csv"),
            "roc_curve_svg": str(model_dir / "roc_curve.svg"),
            "metadata_json": str(model_dir / "metadata.json"),
        },
    }
    (model_dir / "metadata.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested, but CUDA is unavailable. Use --device cpu or auto.")
    return torch.device("cuda" if choice == "auto" and torch.cuda.is_available() else "cpu" if choice == "auto" else choice)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True, help="Encoder checkpoint; repeat to compare training methods.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset directory containing verification_data and/or the referenced images.")
    parser.add_argument("--pairs", type=Path, help="Pair file (default: DATA_ROOT/verification_pairs_val.txt).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-threads", type=int, default=4, help="CPU inference threads (default: 4).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive.")
        if args.num_threads < 1:
            raise ValueError("--num-threads must be positive.")
        root = args.data_root.expanduser().resolve()
        pair_file = (args.pairs or root / "verification_pairs_val.txt").expanduser().resolve()
        checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
        checkpoint_keys = [os.path.normcase(str(path)) for path in checkpoints]
        if len(set(checkpoint_keys)) != len(checkpoint_keys):
            raise ValueError("The same checkpoint was supplied more than once; provide each checkpoint once.")
        for path in checkpoints:
            if not path.is_file():
                raise FileNotFoundError(f"Encoder checkpoint does not exist: {path}")
        pairs = read_verification_pairs(pair_file, root)
        device = _resolve_device(args.device)
        if device.type == "cpu":
            torch.set_num_threads(args.num_threads)
        output_dir = args.output_dir.expanduser().resolve()
        results = []
        for checkpoint in checkpoints:
            print(f"Evaluating {checkpoint.name} on {len(pairs)} pairs ({device})...", flush=True)
            result = evaluate_checkpoint(checkpoint, pairs, output_dir, args.batch_size, device)
            if any(previous["checkpoint_tag"] == result["checkpoint_tag"] for previous in results):
                raise ValueError("Checkpoint output-name collision; move or rename one checkpoint before evaluation.")
            results.append(result)
            if result["checkpoint_smoke_test"]:
                print(f'  Training scope: {result["training_scope_note"]}', flush=True)
            for metric in result["metrics"]:
                print(f'  {result["method"]} / {metric["metric"]}: AUC={metric["auc"]:.6f}', flush=True)
        comparison_rows = []
        for result in results:
            for metric in result["metrics"]:
                selected = metric["threshold_selection"]
                comparison_rows.append({
                    "checkpoint_tag": result["checkpoint_tag"],
                    "method": result["method"],
                    "checkpoint": result["checkpoint"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "metric": metric["metric"],
                    "auc": metric["auc"],
                    "pair_count": result["pair_count"],
                    "same_identity_pairs": result["same_identity_pairs"],
                    "different_identity_pairs": result["different_identity_pairs"],
                    "checkpoint_smoke_test": result["checkpoint_smoke_test"],
                    "training_max_batches": result["training_max_batches"],
                    "score_definition": metric["score_definition"],
                    "score_threshold": metric["score_threshold"],
                    "raw_threshold": metric["raw_threshold"],
                    "accept_rule": metric["accept_rule"],
                    "validation_threshold_tpr": selected["tpr"],
                    "validation_threshold_fpr": selected["fpr"],
                    "validation_threshold_accuracy": selected["accuracy"],
                    "threshold_scope": _THRESHOLD_SCOPE,
                })
        _write_csv(output_dir / "comparison.csv", comparison_rows)
        summary = {
            "data_root": str(root),
            "pairs_file": str(pair_file),
            "pairs_file_sha256": _file_sha256(pair_file),
            "pair_count": len(pairs),
            "label_definition": {"1": "same identity", "0": "different identities"},
            "threshold_scope": _THRESHOLD_SCOPE,
            "score_definitions": _SCORE_DEFINITIONS,
            "models": results,
        }
        (output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Saved comparison.csv, comparison.json and per-checkpoint artifacts to {output_dir}", flush=True)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Pair", "read_verification_pairs", "load_encoder", "score_embeddings",
    "evaluate_checkpoint", "build_parser", "main",
]
