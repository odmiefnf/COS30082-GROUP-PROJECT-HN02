"""Download the official public CelebA-Spoof split ZIP and safely extract it.

Uses only the Python standard library. Downloads resume from .part files and
the split archive is read directly, without creating a second combined ZIP.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import threading
import time
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
import zipfile


FOLDER_ID = "1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"
SOURCE_URL = "https://github.com/ZhangYuanhan-AI/CelebA-Spoof"
PROJECT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 CelebA-Spoof educational dataset downloader"
CHUNK = 4 * 1024 * 1024
GIB = 2**30
ACTIVE_STATUSES = frozenset({
    "preparing", "prepared", "downloading_final_part", "reading_archive_directory",
    "size_checked", "downloading", "validating_and_extracting", "extracting",
})


class DownloadQuotaError(RuntimeError):
    """The public source currently refuses downloads because of its quota."""


class DownloadCancelled(RuntimeError):
    """A different part failed; preserve this part for a later resume."""


def is_quota_page(page: str) -> bool:
    text = re.sub(r"<[^>]+>", " ", page).casefold()
    return any(marker in text for marker in (
        "quota exceeded", "download quota", "downloadquotaexceeded",
        "too many users have viewed or downloaded",
        "you can't view or download this file at this time",
    ))


def is_process_alive(pid: object) -> bool:
    """Query process state without sending a signal on Windows."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # Access denied means the process may be alive; refuse a competing
            # downloader conservatively rather than altering the active files.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)  # POSIX-only: check existence without delivering a signal.
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def assert_no_active_download(status_path: Path, current_pid: int | None = None):
    """Also protect the already-running original downloader, which has no lock."""
    if not status_path.is_file():
        return
    current_pid = os.getpid() if current_pid is None else current_pid
    try:
        state = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify existing download status {status_path}: {exc}") from exc
    pid = state.get("process_id")
    if state.get("status") in ACTIVE_STATUSES and pid != current_pid and is_process_alive(pid):
        raise RuntimeError(
            f"Another downloader is active (PID {pid}, status {state['status']}). "
            f"View {status_path} or use --status-only. Do not start a second download."
        )


class DownloadLocks:
    """OS locks cover both parts and status paths; released automatically on exit."""
    def __init__(self, status_path: Path, parts_directory: Path):
        self.status_path = status_path
        self.paths = (parts_directory / ".download.lock",
                      status_path.with_name(status_path.name + ".lock"))
        self.handles = []

    def acquire(self):
        assert_no_active_download(self.status_path)
        try:
            for path in self.paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+b")
                self.handles.append(handle)
                handle.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    raise RuntimeError(
                        f"Another downloader owns {path}. View {self.status_path}; "
                        "wait for that process instead of downloading into the same files."
                    ) from exc
                handle.seek(0)
                handle.truncate(0)
                handle.write(json.dumps({"process_id": os.getpid(),
                                         "status_path": str(self.status_path)}).encode("utf-8"))
                handle.truncate()
                handle.flush()
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        # Closing an OS-locked descriptor releases its lock even after an error.
        while self.handles:
            self.handles.pop().close()


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "form" and attributes.get("id") == "download-form":
            self.action = attributes.get("action")
        if tag == "input" and attributes.get("type") == "hidden" and attributes.get("name"):
            self.values[attributes["name"]] = attributes.get("value", "")


def opener():
    return build_opener(HTTPCookieProcessor(CookieJar()))


def open_binary(file_id: str, offset: int = 0, end: int | None = None):
    """Open a Drive binary response, following its public confirmation form."""
    client = opener()
    url = "https://drive.google.com/uc?" + urlencode({"export": "download", "id": file_id})
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity",
               "Range": f"bytes={offset}-{'' if end is None else end}"}
    for _ in range(3):
        try:
            response = client.open(Request(url, headers=headers), timeout=60)
        except HTTPError as exc:
            page = exc.read(256 * 1024).decode("utf-8", errors="replace")
            exc.close()
            if exc.code == 429 or (exc.code == 403 and is_quota_page(page)):
                raise DownloadQuotaError(
                    "Google Drive download quota exceeded; existing parts are preserved. "
                    "Retry only after the source permits downloads again."
                ) from exc
            raise
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return response
        page = response.read(256 * 1024).decode("utf-8", errors="replace")
        response.close()
        if is_quota_page(page):
            raise DownloadQuotaError(
                "Google Drive download quota exceeded; existing parts are preserved. "
                "Retry only after the source permits downloads again."
            )
        form = DownloadForm()
        form.feed(page)
        if form.action and urlparse(form.action).hostname == "drive.usercontent.google.com":
            url = form.action + "?" + urlencode(form.values)
        else:
            title = re.search(r"<title>(.*?)</title>", page, re.S)
            reason = re.sub(r"<[^>]+>", " ", page)
            reason = re.sub(r"\s+", " ", reason).strip()
            raise RuntimeError("Google Drive returned a non-download page: "
                               + (title.group(1) if title else reason[:300])
                               + ". It may require login or have a download quota limit.")
    raise RuntimeError("Google Drive confirmation did not produce a binary file.")


def response_size(response) -> int:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    if response.status == 200 and response.headers.get("Content-Length"):
        return int(response.headers["Content-Length"])
    raise RuntimeError("The server did not expose the archive part's total size.")


def file_size(path: Path) -> int | None:
    try:
        metadata = path.stat()
    except FileNotFoundError:
        return None
    return metadata.st_size if stat.S_ISREG(metadata.st_mode) else None


def part_inventory(parts: list[dict], folder: Path) -> dict:
    """Count actual local bytes without trusting a previous progress snapshot."""
    completed, partial, missing, invalid = [], [], [], []
    downloaded = 0
    for part in parts:
        final = folder / part["name"]
        temporary = final.with_name(final.name + ".part")
        final_size = file_size(final)
        if final_size == part["size"]:
            completed.append(part["name"])
            downloaded += final_size
            continue
        if final_size is not None:
            invalid.append(part["name"])
        size = file_size(temporary) or 0
        if size:
            partial.append({"name": part["name"], "bytes": size, "expected_bytes": part["size"]})
            downloaded += min(size, part["size"])
            if size > part["size"]:
                invalid.append(temporary.name)
        else:
            missing.append(part["name"])
    return {"downloaded_bytes": downloaded, "completed_part_count": len(completed),
            "partial_part_count": len(partial), "missing_part_count": len(missing),
            "partial_parts": partial, "missing_parts": missing, "invalid_parts": invalid,
            "archive_download_complete": len(completed) == len(parts) and not invalid}


def observed_status(status_path: Path) -> dict:
    """Return a live, read-only view; a dead process cannot look active."""
    state = json.loads(status_path.read_text(encoding="utf-8"))
    saved_status = state.get("status")
    alive = is_process_alive(state.get("process_id"))
    effective = "interrupted" if saved_status in ACTIVE_STATUSES and not alive else saved_status
    state.update({"status": effective, "saved_status": saved_status, "effective_status": effective,
                  "process_alive": alive, "observed_at_utc": datetime.now(timezone.utc).isoformat()})
    if effective == "interrupted":
        state.update({"dataset_ready": False, "download_bytes_per_second": 0,
                      "download_eta_seconds": None,
                      "observation_note": "Saved progress is stale: its downloader process is not running."})
    if state.get("dataset_ready") and not Path(state.get("target", "")).is_dir():
        state.update({"dataset_ready": False, "effective_status": "extracted_dataset_missing"})
    folder = Path(state.get("parts_directory", ""))
    manifest_path = folder / "manifest.json"
    if manifest_path.is_file():
        parts = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        state.update(part_inventory(parts, folder))
        state["download_percent"] = 100 * state["downloaded_bytes"] / sum(p["size"] for p in parts)
    return state


def probe_missing_part(parts: list[dict], folder: Path) -> dict:
    """Read one byte of the first incomplete official file; never save data."""
    for part in parts:
        final = folder / part["name"]
        if final.is_file() and final.stat().st_size == part["size"]:
            continue
        temporary = final.with_name(final.name + ".part")
        present = temporary.stat().st_size if temporary.is_file() else 0
        offset = min(present, part["size"] - 1)
        result = {"file": part["name"], "resume_offset": present,
                  "probe_byte_offset": offset, "dataset_ready": False}
        try:
            with open_binary(part["id"], offset, offset) as response:
                if (response.status != 206 or response_size(response) != part["size"]
                        or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                        or len(response.read(1)) != 1):
                    raise RuntimeError("Source did not return the requested valid archive byte.")
            return {**result, "probe_status": "download_available"}
        except DownloadQuotaError as exc:
            return {**result, "probe_status": "quota_exceeded", "error": str(exc)}
        except Exception as exc:
            return {**result, "probe_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    return {"probe_status": "all_archive_parts_present", "dataset_ready": False,
            "note": "Archive presence alone does not establish extraction or CRC verification."}


def discover_parts() -> list[dict]:
    """The embedded public folder view lists all parts, unlike its 50-row UI."""
    url = "https://drive.google.com/embeddedfolderview?" + urlencode({"id": FOLDER_ID})
    with opener().open(Request(url, headers={"User-Agent": USER_AGENT}), timeout=60) as response:
        page = response.read().decode("utf-8")
    parts = []
    for href, body in re.findall(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        match = re.search(r"CelebA_Spoof\.zip\.(\d+)", body)
        file_id = re.search(r"/file/d/([\w-]+)/", href)
        if match and file_id:
            parts.append({"name": match.group(0), "id": file_id.group(1),
                          "number": int(match.group(1))})
    parts.sort(key=lambda part: part["number"])
    if not parts or [part["number"] for part in parts] != list(range(1, len(parts) + 1)):
        raise RuntimeError("The official Drive folder did not contain a consecutive split archive.")
    with open_binary(parts[0]["id"], 0, 0) as response:
        ordinary_size = response_size(response)
        response.read(1)
    with open_binary(parts[-1]["id"], 0, 0) as response:
        last_size = response_size(response)
        response.read(1)
    # The official ZIP is split into uniform numbered parts; each binary
    # download independently verifies this size against its Content-Range.
    for part in parts:
        part["size"] = last_size if part is parts[-1] else ordinary_size
    return parts


class MultipartReader(io.RawIOBase):
    """A seekable concatenated view over split ZIP files; no duplicate archive."""
    def __init__(self, parts: list[dict], folder: Path):
        self.parts = parts
        self.folder = folder
        self.offsets = [0]
        for part in parts:
            self.offsets.append(self.offsets[-1] + part["size"])
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.offsets[-1] + offset
        if position < 0:
            raise ValueError("Negative archive seek.")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = self.offsets[-1] - self.position
        size = remaining if size is None or size < 0 else min(size, remaining)
        output = bytearray()
        while size > 0:
            index = bisect_right(self.offsets, self.position) - 1
            part = self.parts[index]
            local_offset = self.position - self.offsets[index]
            count = min(size, part["size"] - local_offset)
            with (self.folder / part["name"]).open("rb") as handle:
                handle.seek(local_offset)
                block = handle.read(count)
            if len(block) != count:
                raise IOError(f"Incomplete archive part: {part['name']}")
            output.extend(block)
            self.position += count
            size -= count
        return bytes(output)


def member_parts(info: zipfile.ZipInfo) -> list[str]:
    """Validate archive paths without filesystem lookups during the size scan."""
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    mode = (info.external_attr >> 16) & 0xFFFF
    if (path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts)
            or stat.S_ISLNK(mode)):
        raise ValueError(f"Unsafe archive member: {info.filename!r}")
    pieces = list(path.parts)
    if pieces and pieces[0].casefold() == "celeba_spoof":
        pieces.pop(0)
    return pieces


def safe_destination(root: Path, info: zipfile.ZipInfo) -> Path:
    """Also reject writes through pre-existing filesystem links outside root."""
    pieces = member_parts(info)
    destination = root.joinpath(*pieces).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes extraction directory: {info.filename!r}")
    return destination


class Progress:
    def __init__(self, status_path: Path, parts: list[dict], target: Path, folder: Path):
        self.path = status_path
        self.parts = parts
        self.target = target
        self.folder = folder
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.initial_bytes = sum(self.part_bytes(part) for part in parts)
        self.downloaded = {part["name"]: self.part_bytes(part) for part in parts}
        self.state = {"status": "preparing", "source": SOURCE_URL, "drive_folder_id": FOLDER_ID,
                      "process_id": os.getpid(),
                      "part_count": len(parts), "archive_bytes": sum(part["size"] for part in parts),
                      "target": str(target), "parts_directory": str(folder),
                      "started_at_utc": datetime.now(timezone.utc).isoformat(),
                      "dataset_ready": False}

    def part_bytes(self, part):
        completed = self.folder / part["name"]
        temporary = completed.with_name(completed.name + ".part")
        if completed.is_file() and completed.stat().st_size == part["size"]:
            return part["size"]
        return min(temporary.stat().st_size, part["size"]) if temporary.is_file() else 0

    def report(self, **updates):
        with self.lock:
            self.state.update(updates)
            inventory = part_inventory(self.parts, self.folder)
            self.state.update(inventory)
            downloaded = inventory["downloaded_bytes"]
            seconds = max(time.monotonic() - self.started, 0.001)
            rate = max(0, downloaded - self.initial_bytes) / seconds
            self.state.update({"downloaded_bytes": downloaded,
                               "download_percent": 100 * downloaded / self.state["archive_bytes"],
                               "download_bytes_per_second": rate,
                               "download_eta_seconds": ((self.state["archive_bytes"] - downloaded) / rate
                                                        if rate > 0 else None),
                               "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            temporary.replace(self.path)
            return dict(self.state)


def download_part(part: dict, folder: Path, progress: Progress,
                  stop: threading.Event | None = None):
    stop = threading.Event() if stop is None else stop
    destination = folder / part["name"]
    temporary = destination.with_name(destination.name + ".part")
    if destination.is_file() and destination.stat().st_size == part["size"]:
        return
    if destination.exists():
        raise ValueError(f"Completed archive part has unexpected size: {destination}")
    last_error = None
    for attempt in range(1, 11):
        if stop.is_set():
            raise DownloadCancelled("Another part failed; partial data was preserved.")
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > part["size"]:
            raise ValueError(f"Partial archive is larger than expected: {temporary}")
        if offset == part["size"]:
            temporary.replace(destination)
            return
        try:
            with open_binary(part["id"], offset) as response:
                if response_size(response) != part["size"]:
                    raise ValueError(f"Official file size changed: {part['name']}")
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError("Drive did not honor the requested byte range; refusing to append.")
                with temporary.open("ab") as handle:
                    while block := response.read(CHUNK):
                        if stop.is_set():
                            raise DownloadCancelled("Another part failed; partial data was preserved.")
                        if offset + len(block) > part["size"]:
                            raise RuntimeError("Server sent more bytes than the advertised archive size.")
                        handle.write(block)
                        offset += len(block)
                        with progress.lock:
                            progress.downloaded[part["name"]] = offset
                if offset != part["size"]:
                    raise IOError(f"Download ended early at {offset}/{part['size']} bytes.")
            temporary.replace(destination)
            print(f"Completed {part['name']} ({part['size'] / GIB:.3f} GiB)", flush=True)
            return
        except (DownloadQuotaError, DownloadCancelled):
            raise
        except Exception as exc:
            last_error = exc
            print(f"Retry {attempt}/10 for {part['name']}: {type(exc).__name__}: {exc}", flush=True)
            if attempt < 10 and stop.wait(min(2 ** attempt, 20)):
                raise DownloadCancelled("Another part failed; partial data was preserved.")
    raise RuntimeError(f"Failed {part['name']} after resumable retries: {last_error}")


def transfer_parts(parts: list[dict], folder: Path, progress: Progress, workers: int):
    """Keep only worker-count futures in flight and stop promptly on failure."""
    stop = threading.Event()
    executor = ThreadPoolExecutor(max_workers=workers)
    remaining = iter(parts)
    pending = {}

    def fill_available_workers():
        for _ in range(workers - len(pending)):
            part = next(remaining, None)
            if part is None:
                break
            pending[executor.submit(download_part, part, folder, progress, stop)] = part

    try:
        fill_available_workers()
        while pending:
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            failure = None
            failed_part = None
            for future in done:
                part = pending.pop(future)
                try:
                    future.result()
                except BaseException as exc:
                    if failure is None or isinstance(exc, DownloadQuotaError):
                        failure, failed_part = exc, part["name"]
            if failure is not None:
                stop.set()
                progress.report(status="quota_exceeded" if isinstance(failure, DownloadQuotaError) else "failed",
                                failed_part=failed_part, error=f"{type(failure).__name__}: {failure}",
                                dataset_ready=False)
                raise failure
            fill_available_workers()
            state = progress.report()
            eta = state["download_eta_seconds"]
            print(f"Downloading: {state['download_percent']:.2f}% "
                  f"({state['downloaded_bytes'] / GIB:.2f}/{state['archive_bytes'] / GIB:.2f} GiB), "
                  f"{state['download_bytes_per_second'] / 2**20:.2f} MiB/s, "
                  f"ETA {eta / 60:.1f} min" if eta is not None else "Waiting for transfer...", flush=True)
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        # Only already-running requests remain; they check stop between chunks
        # or retry waits. A network read is bounded by its 60-second timeout.
        executor.shutdown(wait=True, cancel_futures=True)


def archive_summary(parts, folder, target):
    """The last archive part contains its complete central directory."""
    with zipfile.ZipFile(MultipartReader(parts, folder)) as archive:
        total = 0
        file_count = 0
        for info in archive.infolist():
            member_parts(info)
            total += info.file_size
            file_count += not info.is_dir()
        return {"archive_entries": len(archive.infolist()), "extracted_files": file_count,
                "uncompressed_bytes": total}


def existing_crc(path: Path):
    import zlib
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def extract_archive(parts, folder, target, progress):
    target.mkdir(parents=True, exist_ok=True)
    count, written = 0, 0
    last_report = 0.0
    with zipfile.ZipFile(MultipartReader(parts, folder)) as archive:
        for info in archive.infolist():
            destination = safe_destination(target, info)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not (destination.is_file() and destination.stat().st_size == info.file_size
                    and existing_crc(destination) == info.CRC):
                temporary = destination.with_name(destination.name + ".extracting")
                with archive.open(info) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK)
                # zipfile checks the CRC when the complete member has been read.
                temporary.replace(destination)
            count += 1
            written += info.file_size
            if time.monotonic() - last_report > 15:
                progress.report(status="extracting", extracted_files_completed=count,
                                extracted_bytes_completed=written)
                print(f"Extracting: {count:,} files, {written / GIB:.2f} GiB", flush=True)
                last_report = time.monotonic()
    progress.report(status="complete", dataset_ready=True, extracted_files_completed=count,
                    extracted_bytes_completed=written, crc_validation="all extracted files verified")
    print(f"Dataset ready: {target} ({count:,} CRC-verified files)", flush=True)


def self_test():
    from unittest.mock import Mock, patch
    temporary_root = PROJECT / "data/tempdownloads"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="celeba_downloader_test_", dir=temporary_root) as directory:
        root = Path(directory)
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("CelebA_Spoof/Data/train/1/live/1.jpg", b"synthetic bytes" * 100)
            archive.writestr("CelebA_Spoof/metas/train.txt", b"Data/train/1/live/1.jpg 0\n")
        raw = payload.getvalue()
        parts = []
        for index, offset in enumerate(range(0, len(raw), 50), 1):
            block = raw[offset:offset + 50]
            name = f"test.zip.{index:03d}"
            (root / name).write_bytes(block)
            parts.append({"name": name, "size": len(block)})
        reader = MultipartReader(parts, root)
        assert reader.read() == raw
        reader.seek(-23, 2)
        assert reader.read(23) == raw[-23:]
        target = root / "out"
        with zipfile.ZipFile(MultipartReader(parts, root)) as archive:
            assert archive.read(archive.infolist()[0]) == b"synthetic bytes" * 100
            assert safe_destination(target, archive.infolist()[0]) == target / "Data/train/1/live/1.jpg"
        test_progress = Progress(root / "status.json", parts, target, root)
        extract_archive(parts, root, target, test_progress)
        assert (target / "Data/train/1/live/1.jpg").read_bytes() == b"synthetic bytes" * 100
        for member in ("../escape", "/absolute", "C:/escape", "folder/../../escape", "a:ads"):
            try:
                safe_destination(target, zipfile.ZipInfo(member))
            except ValueError:
                continue
            raise AssertionError(f"Unsafe path accepted: {member}")
        link = zipfile.ZipInfo("link")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        try:
            safe_destination(target, link)
        except ValueError:
            pass
        else:
            raise AssertionError("Symlink accepted")
        assert is_process_alive(os.getpid())
        active_status = root / "guard_status.json"
        active_status.write_text(json.dumps({"status": "downloading", "process_id": os.getpid()}),
                                 encoding="utf-8")
        try:
            assert_no_active_download(active_status, current_pid=os.getpid() + 1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Existing live downloader was not rejected")
        first_lock = DownloadLocks(active_status, root / "locked_parts").acquire()
        try:
            try:
                DownloadLocks(active_status, root / "locked_parts").acquire()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Competing file lock was not rejected")
        finally:
            first_lock.close()
        # A released lock must allow a later resumable invocation.
        released_lock = DownloadLocks(active_status, root / "locked_parts").acquire()
        released_lock.close()
        quota_page = b"<html><title>Google Drive - Quota exceeded</title></html>"
        response = Mock()
        response.headers = {"Content-Type": "text/html"}
        response.read.return_value = quota_page
        client = Mock()
        client.open.return_value = response
        with patch(f"{__name__}.opener", return_value=client):
            try:
                open_binary("fixture")
            except DownloadQuotaError:
                pass
            else:
                raise AssertionError("Quota HTML was not classified")
        assert client.open.call_count == 1
        client.open.side_effect = HTTPError("https://drive.google.com/", 403, "Forbidden", {},
                                           io.BytesIO(quota_page))
        with patch(f"{__name__}.opener", return_value=client):
            try:
                open_binary("fixture")
            except DownloadQuotaError:
                pass
            else:
                raise AssertionError("HTTP 403 quota was not classified")
        quota_part = {"name": "quota.zip.001", "id": "fixture", "size": 8}
        partial_file = root / "quota.zip.001.part"
        partial_file.write_bytes(b"abc")
        quota_progress = Progress(root / "quota_status.json", [quota_part], target, root)
        with patch(f"{__name__}.open_binary", side_effect=DownloadQuotaError("fixture quota")) as network:
            try:
                download_part(quota_part, root, quota_progress)
            except DownloadQuotaError:
                pass
            else:
                raise AssertionError("Quota must fail without retrying")
            network.assert_called_once_with("fixture", 3)
        assert partial_file.read_bytes() == b"abc"
        assert part_inventory([quota_part], root)["downloaded_bytes"] == 3
        bounded_parts = [{"name": f"queued.{index}", "id": "fixture", "size": 8} for index in range(6)]
        with patch(f"{__name__}.download_part", side_effect=DownloadQuotaError("fixture quota")) as worker:
            try:
                transfer_parts(bounded_parts, root, quota_progress, workers=1)
            except DownloadQuotaError:
                pass
            else:
                raise AssertionError("Quota must stop the transfer coordinator")
            assert worker.call_count == 1, "Queued parts must not start after a quota failure"
        assert json.loads(quota_progress.path.read_text())["status"] == "quota_exceeded"
        stale_status = root / "stale_status.json"
        stale_status.write_text(json.dumps({"status": "downloading", "process_id": 123,
                                           "dataset_ready": False}), encoding="utf-8")
        original_status = stale_status.read_bytes()
        with patch(f"{__name__}.is_process_alive", return_value=False):
            observed = observed_status(stale_status)
        assert observed["status"] == observed["effective_status"] == "interrupted"
        assert observed["download_eta_seconds"] is None and not observed["process_alive"]
        assert stale_status.read_bytes() == original_status, "Observing status must remain read-only"
        binary = Mock()
        binary.status = 206
        binary.headers = {"Content-Range": "bytes 3-3/8"}
        binary.read.return_value = b"x"
        binary.__enter__ = Mock(return_value=binary)
        binary.__exit__ = Mock(return_value=False)
        with patch(f"{__name__}.open_binary", return_value=binary) as network:
            probed = probe_missing_part([quota_part], root)
            network.assert_called_once_with("fixture", 3, 3)
        assert probed["probe_status"] == "download_available"
        assert partial_file.read_bytes() == b"abc", "Probe must not change partial files"
    print("Downloader self-test passed: multipart ZIP, safe paths, locks, quota fail-fast, "
          "bounded cancellation, read-only stale status, and one-byte probe.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=PROJECT / "data/CelebA_Spoof")
    parser.add_argument("--parts-dir", type=Path, default=PROJECT / "data/tempdownloads/CelebA_Spoof_parts")
    parser.add_argument("--status", type=Path, default=PROJECT / "data/download_status.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--last-part-only", action="store_true",
                        help="Download the final part and inspect exact extraction size before the full run.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--status-only", action="store_true",
                        help="Print the saved status without downloading, writing, or locking anything.")
    parser.add_argument("--probe-only", action="store_true",
                        help="Read one remote byte of the first incomplete part; do not save data or status.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8.")
    target, folder = args.target.resolve(), args.parts_dir.resolve()
    for path in (target, folder, args.status.resolve()):
        if not path.is_relative_to(PROJECT):
            parser.error("All outputs must stay within the project directory.")
    if args.status_only:
        if not args.status.is_file():
            print(f"No download status exists at {args.status}")
            return 2
        print(json.dumps(observed_status(args.status.resolve()), indent=2))
        return 0
    manifest_path = folder / "manifest.json"
    if args.probe_only:
        if not manifest_path.is_file():
            print(json.dumps({"probe_status": "failed", "error": "No saved manifest; prepare it first."}))
            return 2
        parts = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        result = probe_missing_part(parts, folder)
        print(json.dumps(result, indent=2))
        return 0 if result["probe_status"] in {"download_available", "all_archive_parts_present"} else 2
    progress = None
    locks = DownloadLocks(args.status.resolve(), folder)
    try:
        locks.acquire()
        folder.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            parts = manifest["files"]
        else:
            parts = discover_parts()
            manifest = {"source": SOURCE_URL, "folder_id": FOLDER_ID, "files": parts,
                        "size_verification": "first/last probed; every part checked on download"}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        progress = Progress(args.status.resolve(), parts, target, folder)
        progress.report(status="prepared")
        print(f"Official dataset: {len(parts)} parts, {sum(p['size'] for p in parts) / GIB:.3f} GiB compressed.", flush=True)
        if args.prepare_only:
            return 0
        if shutil.disk_usage(folder).free < parts[-1]["size"] + 2 * GIB:
            raise RuntimeError("Insufficient free space even for the final archive part.")
        progress.report(status="downloading_final_part")
        # The final part contains the full central directory: learn the exact
        # extraction size before committing to the remaining ~72 GiB download.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(download_part, parts[-1], folder, progress)
            while not future.done():
                wait([future], timeout=10)
                state = progress.report()
                print(f"Downloading final part: {state['downloaded_bytes'] / GIB:.3f} GiB; "
                      f"{state['download_bytes_per_second'] / 2**20:.2f} MiB/s", flush=True)
            future.result()
        progress.report(status="reading_archive_directory")
        summary = manifest.get("archive_summary") or archive_summary(parts, folder, target)
        manifest["archive_summary"] = summary
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        progress.report(status="size_checked", **summary)
        print(f"Extraction: {summary['uncompressed_bytes'] / GIB:.3f} GiB, "
              f"{summary['extracted_files']:,} files.", flush=True)
        if args.last_part_only:
            return 0
        remaining_download = sum(p["size"] - progress.part_bytes(p) for p in parts)
        required = remaining_download + summary["uncompressed_bytes"] + 5 * GIB
        free = shutil.disk_usage(folder).free
        progress.report(disk_free_bytes=free, disk_required_bytes=required)
        if free < required:
            raise RuntimeError(f"Need {required / GIB:.1f} GiB free for remaining parts, extraction, "
                               f"and headroom; only {free / GIB:.1f} GiB is available.")
        progress.report(status="downloading")
        transfer_parts(parts[:-1], folder, progress, args.workers)
        progress.report(status="validating_and_extracting")
        extract_archive(parts, folder, target, progress)
        return 0
    except BaseException as exc:
        if progress is not None:
            progress.report(status="quota_exceeded" if isinstance(exc, DownloadQuotaError) else
                            "interrupted" if isinstance(exc, (KeyboardInterrupt, DownloadCancelled)) else "failed",
                            error=f"{type(exc).__name__}: {exc}", dataset_ready=False)
        print(f"Download/extraction stopped: {type(exc).__name__}: {exc}", flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        locks.close()


if __name__ == "__main__":
    raise SystemExit(main())
