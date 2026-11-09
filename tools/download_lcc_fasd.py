"""Download public LCC-FASD v1, resume the ZIP, and verify safe extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
import zlib


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
DOWNLOADS = DATA / "downloads"
ARCHIVE = DOWNLOADS / "lcc-fasd.zip"
PARTIAL = DOWNLOADS / "lcc-fasd.zip.part"
DESTINATION = DATA / "LCC_FASD"
STATUS = DATA / "lcc_fasd_download_status.json"
URL = "https://www.kaggle.com/api/v1/datasets/download/faber24/lcc-fasd?datasetVersionNumber=1"
ARCHIVE_BYTES = 5_196_000_412
IMAGE_BYTES = 5_189_883_545
IMAGE_COUNT = 18_827
START = time.monotonic()


def report(state: str, **values) -> None:
    payload = {"dataset": "faber24/lcc-fasd", "version": 1, "status": state,
               "process_id": os.getpid(), "source": URL, "archive": str(ARCHIVE),
               "destination": str(DESTINATION), "elapsed_seconds": round(time.monotonic() - START, 1),
               "dataset_ready": state == "ready", **values}
    STATUS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


def download() -> None:
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size != ARCHIVE_BYTES:
            raise ValueError(f"Existing archive has an unexpected size: {ARCHIVE}")
        return
    last_progress = 0.0
    for attempt in range(1, 4):
        offset = PARTIAL.stat().st_size if PARTIAL.exists() else 0
        if offset == ARCHIVE_BYTES:
            PARTIAL.replace(ARCHIVE)
            return
        if offset > ARCHIVE_BYTES:
            raise ValueError(f"Partial archive is larger than the verified remote ZIP: {PARTIAL}")
        headers = {"User-Agent": "LCC-FASD academic project downloader", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=90) as response:
                if "zip" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Kaggle returned a non-ZIP response; authentication or download approval may be required.")
                if offset and response.status != 206:
                    raise ValueError("The server did not honor the resume range; partial archive was preserved.")
                if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("The server returned a different byte range; partial archive was preserved.")
                length = response.headers.get("Content-Length")
                if length and int(length) != ARCHIVE_BYTES - offset:
                    raise ValueError("ZIP length differs from the verified dataset version.")
                with PARTIAL.open("ab" if offset else "wb") as target:
                    while chunk := response.read(4 * 1024 * 1024):
                        if offset == 0 and chunk[:4] != b"PK\x03\x04":
                            raise ValueError("Download does not begin with a ZIP signature.")
                        target.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if now - last_progress >= 15:
                            target.flush()
                            report("downloading", downloaded_bytes=offset, total_bytes=ARCHIVE_BYTES,
                                   percent=round(100 * offset / ARCHIVE_BYTES, 2), attempt=attempt)
                            last_progress = now
            if offset != ARCHIVE_BYTES:
                raise OSError(f"Incomplete ZIP: {offset}/{ARCHIVE_BYTES} bytes.")
            PARTIAL.replace(ARCHIVE)
            return
        except (OSError, urllib.error.URLError) as exc:
            report("retrying", attempt=attempt, downloaded_bytes=offset, error=str(exc))
            if attempt == 3:
                raise
            time.sleep(3)


def extract() -> None:
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(ARCHIVE) as archive:
        files = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(files) != IMAGE_COUNT or sum(entry.file_size for entry in files) != IMAGE_BYTES:
            raise ValueError("Archive inventory differs from the verified v1 Kaggle file listing.")
        resolved_root = DESTINATION.resolve()
        targets = []
        seen = set()
        for entry in files:
            name = entry.filename.replace("\\", "/")
            path = PurePosixPath(name)
            if (path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "LCC_FASD"
                    or any(":" in part for part in path.parts)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"Unsafe archive entry: {entry.filename}")
            destination = (DATA / Path(*path.parts)).resolve()
            if not destination.is_relative_to(resolved_root):
                raise ValueError(f"Archive entry escapes the intended dataset directory: {entry.filename}")
            key = os.path.normcase(str(destination))
            if key in seen:
                raise ValueError(f"Duplicate archive destination: {entry.filename}")
            seen.add(key)
            targets.append((entry, destination))
        report("extracting", archive_sha256=digest.hexdigest(), files_total=len(files), files_verified=0)
        for index, (entry, destination) in enumerate(targets, 1):
            if destination.is_file():
                crc = 0
                with destination.open("rb") as existing:
                    for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if destination.stat().st_size == entry.file_size and (crc & 0xffffffff) == entry.CRC:
                    continue
                raise ValueError(f"An existing dataset file differs from the ZIP; preserved: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".extracting")
            with archive.open(entry) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            # ZipExtFile checks CRC at EOF before this replacement.
            temporary.replace(destination)
            if index % 2000 == 0:
                report("extracting", archive_sha256=digest.hexdigest(), files_total=len(files), files_verified=index)
    report("ready", archive_sha256=digest.hexdigest(), files_verified=IMAGE_COUNT,
           extracted_bytes=IMAGE_BYTES, downloaded_bytes=ARCHIVE_BYTES)


def main() -> int:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    needed = IMAGE_BYTES + (0 if ARCHIVE.exists() else ARCHIVE_BYTES) + 1024 ** 3
    if shutil.disk_usage(DATA).free < needed:
        raise OSError("Insufficient free disk space for the ZIP, extracted data and 1 GiB reserve.")
    report("starting", total_bytes=ARCHIVE_BYTES)
    try:
        download()
        extract()
    except Exception as exc:
        report("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
