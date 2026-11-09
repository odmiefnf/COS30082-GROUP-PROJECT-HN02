"""Safely import an official identity ZIP/folder while preserving every byte."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import zipfile

from .audit_data import sha256_file


def find_dataset_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = []
    for pair_file in root.rglob("verification_pairs_val.txt"):
        parent = pair_file.parent
        if (parent / "classification_data").is_dir() and (parent / "verification_data").is_dir():
            candidates.append(parent.resolve())
    if len(set(candidates)) != 1:
        raise ValueError("Expected exactly one root with classification_data, verification_data and verification_pairs_val.txt.")
    return candidates[0]


def _safe_destination(output: Path, name: str, checked_parents: dict | None = None) -> Path:
    path = PurePosixPath(name.replace("\\", "/"))
    reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)}
    if (path.is_absolute() or not path.parts or ".." in path.parts or
            any(":" in part or part.endswith((" ", ".")) or part.split(".")[0].casefold() in reserved for part in path.parts)):
        raise ValueError(f"Unsafe archive/directory path: {name}")
    candidate = output / Path(*path.parts)
    if checked_parents is None:
        destination = candidate.resolve()
    else:
        # Most identity images share an ID directory. Resolve its boundary once,
        # rather than walking the same missing parents for every image.
        parent = candidate.parent
        if parent not in checked_parents:
            checked_parents[parent] = (parent.resolve(), parent.exists())
        resolved_parent, existed = checked_parents[parent]
        destination = resolved_parent / candidate.name
        if existed and candidate.is_symlink():
            destination = candidate.resolve()
    if not destination.is_relative_to(output):
        raise ValueError(f"Destination escapes output directory: {name}")
    return destination


def _write_stream(source, destination: Path, expected_hash: str | None = None, created_parents: set | None = None):
    if created_parents is None or destination.parent not in created_parents:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if created_parents is not None:
            created_parents.add(destination.parent)
    temporary = destination.with_name(destination.name + ".importing")
    with temporary.open("xb") as handle:
        shutil.copyfileobj(source, handle, length=1024 * 1024)
    if expected_hash is not None and sha256_file(temporary) != expected_hash:
        raise ValueError(f"Copy checksum mismatch: {destination}")
    if destination.exists():
        if sha256_file(destination) != sha256_file(temporary):
            raise FileExistsError(f"Existing different file was preserved: {destination}")
        temporary.unlink()
    else:
        temporary.replace(destination)


def import_dataset(source: str | Path, output_dir: str | Path) -> dict:
    source = Path(source).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output or (source.is_dir() and output.is_relative_to(source)):
        raise ValueError("Output must be separate from the source directory.")
    output.mkdir(parents=True, exist_ok=True)
    copied = 0; archive_hash = None
    checked_parents = {}; created_parents = set()
    started = time.monotonic()
    def progress(phase, total=None):
        state = {"status": phase, "source": str(source), "output_dir": str(output),
                 "files_processed": copied, "files_total": total,
                 "elapsed_seconds": round(time.monotonic() - started, 1)}
        (output / "import_status.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(state), flush=True)
    if source.is_file() and zipfile.is_zipfile(source):
        progress("checking_zip")
        archive_hash = sha256_file(source)
        with zipfile.ZipFile(source) as archive:
            planned = []; seen = set()
            for checked, entry in enumerate(archive.infolist(), 1):
                if entry.is_dir():
                    continue
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not accepted.")
                destination = _safe_destination(output, entry.filename, checked_parents)
                key = os.path.normcase(str(destination))
                if key in seen:
                    raise ValueError("ZIP entries collide at the same Windows path.")
                seen.add(key); planned.append((entry, destination))
                if checked % 50000 == 0:
                    print(f"Checked {checked}/{len(archive.infolist())} ZIP paths", flush=True)
            required_bytes = sum(entry.file_size for entry, _ in planned)
            if shutil.disk_usage(output).free < required_bytes + 1024 ** 3:
                raise OSError("Insufficient disk space for the extracted identity archive and 1 GiB reserve.")
            progress("extracting", len(planned))
            for entry, destination in planned:
                with archive.open(entry) as stream:
                    _write_stream(stream, destination, created_parents=created_parents)
                copied += 1
                if copied % 10000 == 0:
                    progress("extracting", len(planned))
    elif source.is_dir():
        original = find_dataset_root(source)
        for path in sorted(original.rglob("*")):
            if path.is_symlink() or not path.resolve().is_relative_to(original):
                raise ValueError("Source links escaping/aliasing the dataset are not accepted.")
            if not path.is_file():
                continue
            # Keep the original root name so pair paths prefixed by it still resolve.
            destination = _safe_destination(output, original.name + "/" + path.relative_to(original).as_posix(), checked_parents)
            with path.open("rb") as stream:
                _write_stream(stream, destination, sha256_file(path), created_parents)
            copied += 1
    else:
        raise ValueError("Source must be a ZIP or an extracted official dataset folder.")
    dataset_root = find_dataset_root(output)
    report = {"status": "imported_not_yet_audited", "source": str(source), "output_dir": str(output),
              "data_root": str(dataset_root), "files_processed": copied, "archive_sha256": archive_hash,
              "pair_file_sha256": sha256_file(dataset_root / "verification_pairs_val.txt"),
              "original_pair_bytes_preserved": True, "pairs_regenerated": False,
              "next_step": "Run python -m face_verification.audit_data --data-root DATA_ROOT"}
    (output / "import_status.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/face_recognition"))
    args = parser.parse_args(argv)
    try:
        report = import_dataset(args.source, args.output_dir)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Identity import failed: {exc}\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
